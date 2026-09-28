"""
Classification Service Client

HTTP client for communicating with the isolated classification service.
Provides retry logic and health checking. Uses httpx for both sync and async paths.
Updated to support the new 4-category relationship schema (dependency, expansion, contradiction, unrelated).
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import httpx

from config.settings import settings
from utils.logging_utils import get_logger
from utils.errors import DatabaseError

logger = get_logger(__name__)


@dataclass
class ClassificationRequest:
    """Request model for classification with new 4-category schema."""
    text_pairs: List[Dict[str, str]]  # List of {"id": "...", "text_a": "...", "text_b": "..."} objects
    batch_size: Optional[int] = None  # Uses settings.classification_batch_size if None


@dataclass
class ClassificationResult:
    """Individual classification result."""
    id: str
    relation_type: str  # dependency, expansion, contradiction, unrelated
    confidence_score: float
    text_a: str
    text_b: str
    debug_scores: Optional[Dict[str, float]] = None  # Added for debugging support
    metadata: Optional[Dict[str, Any]] = None  # Added for hybrid pipeline integration


@dataclass
class ClassificationResponse:
    """Response model for classification with new 4-category schema."""
    results: List[ClassificationResult]
    processing_time: float
    model_info: Dict[str, Any]


class ClassificationServiceClient:
    """
    Client for communicating with the isolated classification service.
    
    Features:
    - Retry logic with exponential backoff
    - Health checking and service availability monitoring
    - Batch processing support
    - Comprehensive error handling
    - Support for new 4-category relationship schema (dependency, expansion, contradiction, unrelated)
    """
    
    # Valid relationship types for the new 4-category schema (merged example + elaboration → expansion)
    VALID_RELATIONSHIP_TYPES = {
        'dependency', 'expansion', 'contradiction', 'unrelated'
    }
    
    def __init__(
        self,
        endpoint: str = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the classification service client.
        
        Args:
            endpoint: Classification service endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.endpoint = (endpoint or settings.classification_service_url).rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Service availability tracking
        self._service_available = True
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 30.0  # seconds

        # Singleton async client — reused across all requests to avoid TCP overhead
        self._async_client: Optional[httpx.AsyncClient] = None
        
        logger.info(
            f"Initialized ClassificationServiceClient: endpoint={self.endpoint}, "
            f"timeout={timeout}s, max_retries={max_retries}"
        )

    def _get_async_client(self) -> httpx.AsyncClient:
        """Return (or create) the singleton async httpx client with connection pooling.

        Note: When called from a thread-pool executor (sync wrapper inside a running
        event loop), the previous client may be closed. Always create a fresh one in
        that case.
        """
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._async_client
    
    def classify_relationships(
        self, 
        text_pairs: List[Dict[str, str]], 
        batch_size: Optional[int] = None
    ) -> 'ClassificationResponse':
        """Synchronous entry point — runs the async implementation."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(self.classify_relationships_async(text_pairs, batch_size))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, self.classify_relationships_async(text_pairs, batch_size)
                )
                return future.result()

    async def classify_relationships_async(
        self,
        text_pairs: List[Dict[str, str]],
        batch_size: Optional[int] = None,
    ) -> 'ClassificationResponse':
        """
        Async implementation. Classify relationships between text pairs.
        Handles batching internally to prevent OOM errors.
        """
        if not text_pairs:
            return ClassificationResponse(results=[], processing_time=0.0, model_info={})

        self._validate_text_pairs(text_pairs)

        if batch_size is None:
            batch_size = 30

        if len(text_pairs) <= batch_size:
            return await self._classify_batch(text_pairs)

        logger.info(f"Splitting {len(text_pairs)} pairs into batches of {batch_size}")

        all_results = []
        total_time = 0.0
        model_info = {}
        total_batches = (len(text_pairs) + batch_size - 1) // batch_size

        for batch_start in range(0, len(text_pairs), batch_size):
            batch_pairs = text_pairs[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_pairs)} pairs)")
            batch_response = await self._classify_batch(batch_pairs)
            all_results.extend(batch_response.results)
            total_time += batch_response.processing_time
            model_info = batch_response.model_info

        logger.info(f"Completed {len(text_pairs)} pairs in {total_batches} batches ({total_time:.2f}s)")
        return ClassificationResponse(results=all_results, processing_time=total_time, model_info=model_info)

    async def _classify_batch(self, text_pairs: List[Dict[str, str]]) -> 'ClassificationResponse':
        """Async: classify a single batch via the Docker service."""
        request_data: Dict[str, Any] = {"text_pairs": text_pairs}
        client = self._get_async_client()

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.monotonic()
                response = await client.post(
                    f"{self.endpoint}/v1/classify",
                    json=request_data,
                )
                response.raise_for_status()
                result = response.json()
                processing_time = time.monotonic() - start_time
                self._service_available = True

                results = [
                    ClassificationResult(
                        id=item["id"],
                        relation_type=item["relation_type"],
                        confidence_score=item["confidence_score"],
                        text_a=item["text_a"],
                        text_b=item["text_b"],
                        debug_scores=item.get("debug_scores"),
                    )
                    for item in result.get("results", [])
                ]
                return ClassificationResponse(
                    results=results,
                    processing_time=processing_time,
                    model_info=result.get("model_info", {}),
                )

            except httpx.TimeoutException:
                logger.warning(f"Classification timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt == self.max_retries:
                    self._service_available = False
                    raise DatabaseError("Classification service timeout after retries")
                await asyncio.sleep(2 ** attempt)

            except httpx.RequestError as e:
                logger.warning(f"Classification request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt == self.max_retries:
                    self._service_available = False
                    raise DatabaseError(f"Classification service connection failed: {e}")
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Unexpected error in classification request: {e}")
                if attempt == self.max_retries:
                    self._service_available = False
                    raise DatabaseError(f"Classification service error: {e}")
                await asyncio.sleep(2 ** attempt)

        raise DatabaseError("Classification service failed after all retries")
    
    def classify_relationships_legacy(
        self, 
        text_pairs: List[List[str]], 
        probe_type: str
    ) -> ClassificationResponse:
        """
        Legacy method for backward compatibility with old probe-based classification.
        
        Args:
            text_pairs: List of [text1, text2] pairs to classify
            probe_type: Type of probe relationship to classify (legacy)
            
        Returns:
            ClassificationResponse with predictions converted to new format
            
        Raises:
            DatabaseError: If classification fails after retries
        """
        # Convert legacy format to new format
        converted_pairs = []
        for i, (text_a, text_b) in enumerate(text_pairs):
            converted_pairs.append({
                "id": f"legacy_{i}",
                "text_a": text_a,
                "text_b": text_b
            })
        
        # Use new classification method
        response = self.classify_relationships(converted_pairs)
        
        # For legacy compatibility, also provide predictions list
        predictions = [result.confidence_score for result in response.results]
        
        # Create a legacy-compatible response that also includes new format
        legacy_response = ClassificationResponse(
            results=response.results,
            processing_time=response.processing_time,
            model_info=response.model_info
        )
        
        # Add legacy predictions field for backward compatibility
        legacy_response.predictions = predictions  # type: ignore[attr-defined]
        
        return legacy_response
    
    def _validate_text_pairs(self, text_pairs: List[Dict[str, str]]) -> None:
        """
        Validate the format of text pairs for the new schema.
        
        Args:
            text_pairs: List of text pair dictionaries to validate
            
        Raises:
            ValueError: If format is invalid
        """
        if not isinstance(text_pairs, list):
            raise ValueError("text_pairs must be a list")
        
        for i, pair in enumerate(text_pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"text_pairs[{i}] must be a dictionary")
            
            required_fields = {"id", "text_a", "text_b"}
            missing_fields = required_fields - set(pair.keys())
            if missing_fields:
                raise ValueError(f"text_pairs[{i}] missing required fields: {missing_fields}")
            
            for field in required_fields:
                if not isinstance(pair[field], str):
                    raise ValueError(f"text_pairs[{i}][{field}] must be a string")
    
    def is_healthy(self) -> bool:
        current_time = time.time()
        if (current_time - self._last_health_check) < self._health_check_interval:
            return self._service_available
        try:
            response = httpx.get(f"{self.endpoint}/health", timeout=5)
            self._service_available = response.status_code == 200
            self._last_health_check = current_time
            if not self._service_available:
                logger.warning(f"Classification service unhealthy: {response.status_code}")
        except Exception as e:
            logger.warning(f"Classification service health check failed: {e}")
            self._service_available = False
            self._last_health_check = current_time
        return self._service_available

    def wait_for_service(self, max_wait_time: int = 60) -> bool:
        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            if self.is_healthy():
                logger.info("Classification service is now available")
                return True
            logger.info("Waiting for classification service...")
            time.sleep(2)
        logger.error(f"Classification service did not become available within {max_wait_time}s")
        return False

    def get_service_info(self) -> Dict[str, Any]:
        try:
            response = httpx.get(f"{self.endpoint}/health", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            return {"error": f"Service returned {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}


# Global client instance
_global_client: Optional[ClassificationServiceClient] = None


def get_classification_client() -> ClassificationServiceClient:
    """
    Get the global classification service client instance.
    
    Returns:
        ClassificationServiceClient instance
    """
    global _global_client
    
    if _global_client is None:
        _global_client = ClassificationServiceClient()
    
    return _global_client


def classify_relationship(
    text_pairs: List[Dict[str, str]], 
    batch_size: Optional[int] = None  # Uses settings.classification_batch_size if None
) -> ClassificationResponse:
    """
    Convenience function for classifying relationships using the global client.
    
    Args:
        text_pairs: List of {"id": "...", "text_a": "...", "text_b": "..."} objects
        batch_size: Batch size for processing (optional)
        
    Returns:
        ClassificationResponse with results and metadata (4-category schema)
    """
    client = get_classification_client()
    return client.classify_relationships(text_pairs, batch_size)


def classify_relationship_legacy(
    text_pairs: List[List[str]], 
    probe_type: str
) -> ClassificationResponse:
    """
    Legacy convenience function for backward compatibility.
    
    Args:
        text_pairs: List of [text1, text2] pairs to classify
        probe_type: Type of probe relationship to classify (legacy)
        
    Returns:
        ClassificationResponse with predictions and metadata
    """
    client = get_classification_client()
    return client.classify_relationships_legacy(text_pairs, probe_type)


# Cleanup function for graceful shutdown
def cleanup_classification_client():
    """Clean up the global classification client."""
    global _global_client
    _global_client = None
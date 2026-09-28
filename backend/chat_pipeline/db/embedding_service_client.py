"""
Embedding Service Client for Service B communication.

This module provides a client for communicating with the containerized embedding service
that hosts the nomic-ai/nomic-embed-text-v2-moe model. It handles HTTP communication,
retry logic, proper prefix handling for optimal MoE performance,
and comprehensive metrics collection.
"""

import requests
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from utils.logging_utils import get_logger
from utils.errors import DatabaseError

logger = get_logger(__name__)


@dataclass
class EmbeddingRequest:
    """Request model for embedding generation."""
    text: str
    is_query: bool = False


@dataclass
class EmbeddingResponse:
    """Response model for embedding generation."""
    embedding: List[float]
    model: str
    processing_time: float


class EmbeddingServiceClient:
    """
    Client for communicating with Service B (embedding service).
    
    Features:
    - Automatic prefix handling for MoE model optimization
    - Retry logic with exponential backoff
    - Embedding dimension validation (768 dimensions)
    - Performance monitoring and logging
    
    Note: This client requires the embedding service to be available.
    No fallback to local models is provided.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:7997",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the embedding service client.
        
        Args:
            endpoint: Service B endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Service availability tracking
        self._service_available = True
        
        logger.info(
            f"Initialized EmbeddingServiceClient: endpoint={endpoint}, "
            f"timeout={timeout}s, max_retries={max_retries}"
        )
    
    def _add_prefix(self, text: str, is_query: bool) -> str:
        """
        Add appropriate prefix for MoE model optimization.
        
        This method is idempotent - applying the same prefix multiple times
        will not duplicate the prefix.
        
        Args:
            text: Input text
            is_query: Whether this is a query (True) or document (False)
            
        Returns:
            Text with appropriate prefix
        """
        query_prefix = "search_query: "
        document_prefix = "search_document: "
        
        if is_query:
            # Check if query prefix already exists
            if text.startswith(query_prefix):
                return text
            # Remove document prefix if it exists (switching types)
            if text.startswith(document_prefix):
                text = text[len(document_prefix):]
            return f"{query_prefix}{text}"
        else:
            # Check if document prefix already exists
            if text.startswith(document_prefix):
                return text
            # Remove query prefix if it exists (switching types)
            if text.startswith(query_prefix):
                text = text[len(query_prefix):]
            return f"{document_prefix}{text}"
    
    def _validate_embedding(self, embedding: List[float]) -> None:
        """
        Validate embedding dimensions and format.
        
        Args:
            embedding: Embedding vector to validate
            
        Raises:
            DatabaseError: If embedding is invalid
        """
        if not isinstance(embedding, list):
            raise DatabaseError(f"Embedding must be a list, got {type(embedding)}")
        
        if len(embedding) != 768:
            raise DatabaseError(
                f"Expected 768-dimensional embedding, got {len(embedding)} dimensions"
            )
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise DatabaseError("Embedding must contain only numeric values")
    
    def _request_embedding_with_retry(
        self,
        text: str,
        is_query: bool = False
    ) -> EmbeddingResponse:
        """
        Request embedding from service with retry logic and caching.
        
        Args:
            text: Text to embed
            is_query: Whether this is a query embedding
            
        Returns:
            EmbeddingResponse with embedding and metadata
            
        Raises:
            DatabaseError: If all retry attempts fail
        """
        prefixed_text = self._add_prefix(text, is_query)
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Prepare request payload — input must be a list per the OpenAI embeddings spec
                payload = {
                    "input": [prefixed_text],
                    "model": "nomic-ai/nomic-embed-text-v2-moe"
                }
                
                # Make request to embedding service
                response = requests.post(
                    f"{self.endpoint}/v1/embeddings",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    processing_time = time.time() - start_time
                    
                    # Extract embedding from response
                    if "data" in result and len(result["data"]) > 0:
                        embedding = result["data"][0]["embedding"]
                        model = result.get("model", "nomic-ai/nomic-embed-text-v2-moe")
                        
                        # Validate embedding
                        self._validate_embedding(embedding)
                        
                        # Mark service as available
                        self._service_available = True
                        
                        logger.debug(
                            f"Generated embedding: text_len={len(text)}, "
                            f"is_query={is_query}, time={processing_time:.3f}s"
                        )
                        
                        return EmbeddingResponse(
                            embedding=embedding,
                            model=model,
                            processing_time=processing_time
                        )
                    else:
                        raise DatabaseError("Invalid response format from embedding service")
                else:
                    raise DatabaseError(
                        f"Embedding service returned {response.status_code}: {response.text}"
                    )
                        
            except requests.exceptions.Timeout:
                logger.warning(
                    f"Embedding request timeout (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt == self.max_retries - 1:
                    self._service_available = False
                    raise DatabaseError("Embedding service timeout after all retries")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Embedding service connection error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    self._service_available = False
                    raise DatabaseError(f"Embedding service unavailable: {e}")
                    
            except Exception as e:
                logger.error(
                    f"Unexpected error in embedding request (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise DatabaseError(f"Embedding generation failed: {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.debug(f"Retrying embedding request in {wait_time}s...")
                time.sleep(wait_time)
        
        raise DatabaseError("All embedding service retry attempts failed")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            768-dimensional embedding vector
            
        Raises:
            DatabaseError: If embedding generation fails
        """
        response = self._request_embedding_with_retry(text, is_query=True)
        return response.embedding

    def embed_document(self, text: str) -> List[float]:
        """
        Generate embedding for document text.
        
        Args:
            text: Document text to embed
            
        Returns:
            768-dimensional embedding vector
            
        Raises:
            DatabaseError: If embedding generation fails
        """
        response = self._request_embedding_with_retry(text, is_query=False)
        return response.embedding

    def embed_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing.
        
        Args:
            texts: List of texts to embed
            is_query: Whether these are query embeddings
            batch_size: Batch size for processing (not used with direct API calls)
            
        Returns:
            List of 768-dimensional embedding vectors
            
        Raises:
            DatabaseError: If batch embedding generation fails
        """
        if not texts:
            return []
        
        logger.info(f"Processing batch of {len(texts)} embeddings (is_query={is_query})")
        
        embeddings = []
        
        for text in texts:
            try:
                if is_query:
                    embedding = self.embed_query(text)
                else:
                    embedding = self.embed_document(text)
                embeddings.append(embedding)
                    
            except Exception as e:
                logger.error(f"Failed to embed text in batch: {e}")
                raise DatabaseError(f"Batch embedding generation failed: {e}")
        
        logger.debug(f"Batch completed: {len(embeddings)} embeddings")
        
        return embeddings

    def health_check(self) -> Dict[str, Any]:
        """
        Check service health and availability.
        
        Returns:
            Health status information
        """
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=self.timeout)
            
            if response.status_code == 200:
                result = response.json()
                self._service_available = True
                
                return {
                    "status": "healthy",
                    "service_available": True,
                    "endpoint": self.endpoint,
                    **result
                }
            else:
                self._service_available = False
                return {
                    "status": "unhealthy",
                    "service_available": False,
                    "endpoint": self.endpoint,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            self._service_available = False
            return {
                "status": "unhealthy",
                "service_available": False,
                "endpoint": self.endpoint,
                "error": str(e)
            }


# Global client instance
_global_embedding_client: Optional[EmbeddingServiceClient] = None


def get_embedding_client() -> EmbeddingServiceClient:
    """
    Get or create the global embedding service client.
    
    Returns:
        EmbeddingServiceClient instance
    """
    global _global_embedding_client
    
    if _global_embedding_client is None:
        # Load configuration from settings
        try:
            from config.settings import get_settings
            settings = get_settings()
            
            # Use default endpoint for now, will be updated in configuration task
            endpoint = getattr(settings, 'embedding_service_url', 'http://localhost:7997')
            
            _global_embedding_client = EmbeddingServiceClient(
                endpoint=endpoint,
            )
            
            logger.info(f"Created global embedding service client: {endpoint}")
            
        except Exception as e:
            logger.warning(f"Failed to load embedding service settings: {e}")
            _global_embedding_client = EmbeddingServiceClient()
            logger.info("Created global embedding service client with default settings")
    
    return _global_embedding_client


def reset_embedding_client():
    """Reset the global embedding client (useful for testing)."""
    global _global_embedding_client
    _global_embedding_client = None
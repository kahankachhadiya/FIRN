"""
Embedding Service Client for Service B communication.

This module provides a client for communicating with the containerized embedding service
that hosts the nomic-ai/nomic-embed-text-v2-moe model. It handles HTTP communication,
retry logic, proper prefix handling for optimal MoE performance,
embedding caching for improved performance, and comprehensive metrics collection.
"""

import requests
import time
import hashlib
import json
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
    - Elasticsearch-based embedding caching for frequently requested texts
    
    Note: This client requires the embedding service to be available.
    No fallback to local models is provided.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:7997",
        timeout: int = 30,
        max_retries: int = 3,
        enable_cache: bool = True,
        cache_ttl: int = 3600  # 1 hour default TTL
    ):
        """
        Initialize the embedding service client.
        
        Args:
            endpoint: Service B endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            enable_cache: Whether to enable Elasticsearch-based embedding caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Service availability tracking
        self._service_available = True
        
        # Cache client (lazy initialization)
        self._cache_client = None
        
        logger.info(
            f"Initialized EmbeddingServiceClient: endpoint={endpoint}, "
            f"timeout={timeout}s, max_retries={max_retries}, "
            f"cache_enabled={enable_cache}"
        )
    
    def _get_cache_client(self):
        """Get or create Elasticsearch cache client."""
        if not self.enable_cache:
            return None
            
        if self._cache_client is None:
            try:
                from db.elasticsearch_client import _get_client
                self._cache_client = _get_client()
                logger.debug("Initialized Elasticsearch cache client for embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding cache: {e}")
                self.enable_cache = False
                return None
        
        return self._cache_client
    
    def _get_cache_key(self, text: str, is_query: bool) -> str:
        """
        Generate cache key for embedding.
        
        Args:
            text: Input text (with prefix applied)
            is_query: Whether this is a query embedding
            
        Returns:
            Cache key string
        """
        # Create a hash of the text for consistent key generation
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        prefix = "query" if is_query else "doc"
        return f"embedding:{prefix}:{text_hash}"

    def _get_cached_embedding(self, text: str, is_query: bool) -> Optional[List[float]]:
        """
        Retrieve embedding from cache if available.
        
        Args:
            text: Input text (with prefix applied)
            is_query: Whether this is a query embedding
            
        Returns:
            Cached embedding or None if not found
        """
        if not self.enable_cache:
            return None
            
        cache_client = self._get_cache_client()
        if cache_client is None:
            return None
        
        try:
            cache_key = self._get_cache_key(text, is_query)
            
            # Use Elasticsearch get operation to retrieve cached embedding
            # Store in a dedicated "embeddings_cache" index
            result = cache_client.get(index="embeddings_cache", id=cache_key, ignore=404)
            
            if result.get('found'):
                embedding = result['_source']['embedding']
                logger.debug(f"Cache hit for embedding: {cache_key}")
                return embedding
                
        except Exception as e:
            logger.warning(f"Failed to retrieve embedding from cache: {e}")
        
        return None

    def _cache_embedding(self, text: str, is_query: bool, embedding: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Input text (with prefix applied)
            is_query: Whether this is a query embedding
            embedding: Embedding vector to cache
        """
        if not self.enable_cache:
            return
            
        cache_client = self._get_cache_client()
        if cache_client is None:
            return
        
        try:
            cache_key = self._get_cache_key(text, is_query)
            
            # Ensure embeddings_cache index exists
            if not cache_client.indices.exists(index="embeddings_cache"):
                cache_client.indices.create(
                    index="embeddings_cache",
                    body={
                        "mappings": {
                            "properties": {
                                "embedding": {"type": "float"},
                                "created_at": {"type": "date"}
                            }
                        }
                    }
                )
            
            # Store embedding in Elasticsearch with TTL simulation via timestamp
            # Note: Elasticsearch doesn't have native TTL, but we can use timestamp
            # and periodic cleanup or ILM policies for production
            import datetime
            cache_client.index(
                index="embeddings_cache",
                id=cache_key,
                document={
                    "embedding": embedding,
                    "created_at": datetime.datetime.utcnow().isoformat()
                }
            )
            logger.debug(f"Cached embedding: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
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
        
        # Check cache first
        cached_embedding = self._get_cached_embedding(prefixed_text, is_query)
        if cached_embedding is not None:
            return EmbeddingResponse(
                embedding=cached_embedding,
                model="nomic-ai/nomic-embed-text-v2-moe",
                processing_time=0.0  # Cache hit, no processing time
            )
        
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
                        
                        # Cache the embedding
                        self._cache_embedding(prefixed_text, is_query, embedding)
                        
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
        cache_hits = 0
        
        for text in texts:
            try:
                if is_query:
                    embedding = self.embed_query(text)
                else:
                    embedding = self.embed_document(text)
                embeddings.append(embedding)
                
                # Check if it was a cache hit (processing_time would be 0.0)
                prefixed_text = self._add_prefix(text, is_query)
                if self._get_cached_embedding(prefixed_text, is_query) is not None:
                    cache_hits += 1
                    
            except Exception as e:
                logger.error(f"Failed to embed text in batch: {e}")
                raise DatabaseError(f"Batch embedding generation failed: {e}")
        
        cache_hit_rate = cache_hits / len(texts) if texts else 0.0
        logger.debug(f"Batch completed: {len(embeddings)} embeddings, cache hit rate: {cache_hit_rate:.2%}")
        
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
                    "cache_enabled": self.enable_cache,
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
            
            # Load cache configuration
            enable_cache = getattr(settings, 'enable_embedding_cache', True)
            cache_ttl = getattr(settings, 'embedding_cache_ttl', 3600)
            
            _global_embedding_client = EmbeddingServiceClient(
                endpoint=endpoint,
                enable_cache=enable_cache,
                cache_ttl=cache_ttl
            )
            
            logger.info(f"Created global embedding service client: {endpoint} (cache: {enable_cache})")
            
        except Exception as e:
            logger.warning(f"Failed to load embedding service settings: {e}")
            _global_embedding_client = EmbeddingServiceClient()
            logger.info("Created global embedding service client with default settings")
    
    return _global_embedding_client


def reset_embedding_client():
    """Reset the global embedding client (useful for testing)."""
    global _global_embedding_client
    _global_embedding_client = None
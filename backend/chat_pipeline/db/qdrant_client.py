"""
Qdrant database client for vector embeddings and semantic search.

This module provides functions for:
- Storing paragraph embeddings in the main collection
- Semantic search on paragraphs with metadata filtering

Optimized for low latency with:
- Connection pooling and keep-alive
- Batch operations support
- Efficient embedding caching
"""

from typing import List, Dict, Any, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    OptimizersConfigDiff,
    HnswConfigDiff,
    SearchParams,
)

from config.settings import settings
from config.production import get_production_config
from utils.errors import DatabaseError
from utils.logging_utils import get_logger
from utils.retry import with_retry, QDRANT_RETRY_CONFIG


logger = get_logger(__name__)


# Global Qdrant client (lazy initialization)
_qdrant_client: Optional[QdrantClient] = None
_collections_verified: bool = False


@with_retry(
    max_retries=QDRANT_RETRY_CONFIG.max_retries,
    base_delay=QDRANT_RETRY_CONFIG.base_delay,
    max_delay=QDRANT_RETRY_CONFIG.max_delay,
    retryable_exceptions=QDRANT_RETRY_CONFIG.retryable_exceptions
)
def _get_client() -> QdrantClient:
    """
    Get or create Qdrant client with environment-based configuration.
    
    All configuration parameters are loaded from environment variables via
    the production configuration system. No hardcoded values are used.
    
    Includes automatic retry logic with exponential backoff for connection failures.
    
    Returns:
        QdrantClient instance
        
    Raises:
        DatabaseError: If connection fails after retries
    """
    global _qdrant_client
    
    if _qdrant_client is None:
        try:
            # Load configuration from environment variables
            config = get_production_config()
            qdrant_config = config.qdrant_config
            
            # Create client with environment-based configuration
            _qdrant_client = QdrantClient(
                host=qdrant_config.host,
                port=qdrant_config.port,
                timeout=qdrant_config.connection_timeout,
                prefer_grpc=qdrant_config.prefer_grpc,
                https=qdrant_config.https,
                api_key=qdrant_config.password,  # API key stored in password field
            )
            logger.info(
                f"Connected to Qdrant at {qdrant_config.host}:{qdrant_config.port} "
                f"(timeout={qdrant_config.connection_timeout}s, prefer_grpc={qdrant_config.prefer_grpc})"
            )
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Qdrant: {e}") from e
    
    return _qdrant_client


@with_retry(
    max_retries=QDRANT_RETRY_CONFIG.max_retries,
    base_delay=QDRANT_RETRY_CONFIG.base_delay,
    max_delay=QDRANT_RETRY_CONFIG.max_delay,
    retryable_exceptions=QDRANT_RETRY_CONFIG.retryable_exceptions
)
def _ensure_collections_exist() -> None:
    """
    Ensure main collection exists in Qdrant.
    
    Creates collection if it doesn't exist with environment-based configuration:
    - Vector size: 768 (nomic-ai/nomic-embed-text-v2-moe dimension)
    - Distance metric: Cosine
    - HNSW index parameters from environment variables
    - Optimizer settings from environment variables
    
    All configuration parameters are loaded from environment variables via
    the production configuration system. No hardcoded values are used.
    
    Includes automatic retry logic with exponential backoff for failures.
    
    Raises:
        DatabaseError: If collection creation fails after retries
    """
    global _collections_verified
    
    # Skip if already verified this session
    if _collections_verified:
        return
    
    try:
        client = _get_client()
        
        # Load configuration from environment variables
        config = get_production_config()
        qdrant_config = config.qdrant_config
        
        # Check and create main collection
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        # HNSW config from environment variables
        hnsw_config = HnswConfigDiff(
            m=qdrant_config.hnsw_m,  # Number of edges per node (higher = better recall, more memory)
            ef_construct=qdrant_config.hnsw_ef_construct,  # Construction time accuracy (higher = better index)
        )
        
        # Optimizer config from environment variables
        optimizers_config = OptimizersConfigDiff(
            indexing_threshold=qdrant_config.indexing_threshold,  # Start indexing after N vectors
        )
        
        if settings.qdrant_main_collection not in collection_names:
            client.create_collection(
                collection_name=settings.qdrant_main_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
            )
            logger.info(
                f"Created Qdrant collection: {settings.qdrant_main_collection} "
                f"(hnsw_m={qdrant_config.hnsw_m}, ef_construct={qdrant_config.hnsw_ef_construct}, "
                f"indexing_threshold={qdrant_config.indexing_threshold})"
            )
        
        _collections_verified = True
            
    except Exception as e:
        raise DatabaseError(f"Failed to ensure Qdrant collections exist: {e}") from e


# Global embedding service client (lazy initialization)
_embedding_client = None


def _get_embedding_model():
    """
    Get or create the embedding service client.
    Uses lazy initialization to create the client only when needed.
    
    Returns:
        EmbeddingServiceClient instance for nomic-ai/nomic-embed-text-v2-moe
    """
    global _embedding_client
    
    if _embedding_client is None:
        try:
            from db.embedding_service_client import get_embedding_client
            
            logger.info("Initializing embedding service client...")
            _embedding_client = get_embedding_client()
            logger.info("Embedding service client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service client: {e}")
            raise DatabaseError(f"Embedding service client required but failed to initialize: {e}")
    
    return _embedding_client


def _build_qdrant_filter(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Build a Qdrant Filter from a resolved metadata_filter dict containing document_ids."""
    if not metadata_filter:
        return None
    doc_ids = metadata_filter.get("document_ids")
    if not doc_ids or not isinstance(doc_ids, list):
        return None
    from qdrant_client.models import MatchAny
    return Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=doc_ids))])  # type: ignore[arg-type]


async def _generate_embedding(text: str, is_query: bool = False) -> List[float]:
    """
    Generate embedding for text using the embedding service client.
    
    Args:
        text: Text to embed
        is_query: If True, use query prefix for better search accuracy
        
    Returns:
        768-dimensional embedding vector
        
    Raises:
        DatabaseError: If embedding generation fails
    """
    # Use the synchronous client since the embedding service is standalone
    return _generate_embedding_sync(text, is_query)


def _generate_embedding_sync(text: str, is_query: bool = False) -> List[float]:
    """
    Generate embedding for text using the embedding service client (synchronous version).
    
    Args:
        text: Text to embed
        is_query: If True, use query prefix for better search accuracy
        
    Returns:
        768-dimensional embedding vector
        
    Raises:
        DatabaseError: If embedding generation fails
    """
    try:
        client = _get_embedding_model()
        if is_query:
            return client.embed_query(text)
        else:
            return client.embed_document(text)
    except Exception as e:
        raise DatabaseError(f"Failed to generate embedding (sync): {e}") from e


async def _generate_embeddings_batch(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a batch (more efficient).
    
    Args:
        texts: List of texts to embed
        is_query: If True, use query prefix for better search accuracy
        
    Returns:
        List of 768-dimensional embedding vectors
        
    Raises:
        DatabaseError: If embedding generation fails
    """
    client = _get_embedding_model()
    
    try:
        embeddings = client.embed_batch(texts, is_query=is_query)
        return embeddings
    except Exception as e:
        raise DatabaseError(f"Failed to generate batch embeddings: {e}") from e


@with_retry(
    max_retries=QDRANT_RETRY_CONFIG.max_retries,
    base_delay=QDRANT_RETRY_CONFIG.base_delay,
    max_delay=QDRANT_RETRY_CONFIG.max_delay,
    retryable_exceptions=QDRANT_RETRY_CONFIG.retryable_exceptions
)
async def add_paragraph_embedding_async(
    chunk_id: str,
    text: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Generate and store paragraph embedding in the main Qdrant collection (async version).
    
    Includes automatic retry logic with exponential backoff for failures.
    
    Args:
        chunk_id: Unique chunk identifier (stored in payload)
        text: Paragraph text to embed
        metadata: Metadata to store with the vector (should include
                 document_id, page_number, chunk_index)
    
    Raises:
        DatabaseError: If embedding generation or storage fails after retries
    """
    try:
        _ensure_collections_exist()
        client = _get_client()
        
        # Generate embedding
        embedding = await _generate_embedding(text)
        
        # Generate UUID from chunk_id for Qdrant point ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        # Create point with metadata
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "chunk_id": chunk_id,
                **metadata
            }
        )
        
        # Store in main collection
        client.upsert(
            collection_name=settings.qdrant_main_collection,
            points=[point]
        )
        
        logger.debug(f"Stored paragraph embedding: {chunk_id}")
        
    except Exception as e:
        raise DatabaseError(f"Failed to add paragraph embedding for {chunk_id}: {e}") from e


def add_paragraph_embedding(
    chunk_id: str,
    text: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Generate and store paragraph embedding in the main Qdrant collection (synchronous version).
    
    This is a synchronous wrapper around the async version for backward compatibility.
    
    Args:
        chunk_id: Unique chunk identifier (stored in payload)
        text: Paragraph text to embed
        metadata: Metadata to store with the vector (should include
                 document_id, page_number, chunk_index)
    
    Raises:
        DatabaseError: If embedding generation or storage fails after retries
    """
    try:
        _ensure_collections_exist()
        client = _get_client()
        
        # Generate embedding synchronously
        embedding = _generate_embedding_sync(text)
        
        # Generate UUID from chunk_id for Qdrant point ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        # Create point with metadata
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "chunk_id": chunk_id,
                **metadata
            }
        )
        
        # Store in main collection
        client.upsert(
            collection_name=settings.qdrant_main_collection,
            points=[point]
        )
        
        logger.debug(f"Stored paragraph embedding: {chunk_id}")
        
    except Exception as e:
        raise DatabaseError(f"Failed to add paragraph embedding for {chunk_id}: {e}") from e


@with_retry(
    max_retries=QDRANT_RETRY_CONFIG.max_retries,
    base_delay=QDRANT_RETRY_CONFIG.base_delay,
    max_delay=QDRANT_RETRY_CONFIG.max_delay,
    retryable_exceptions=QDRANT_RETRY_CONFIG.retryable_exceptions
)
async def search_paragraphs(
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar paragraphs in the main Qdrant collection.
    
    All configuration parameters are loaded from environment variables via
    the production configuration system. Parameters can be overridden by
    passing explicit values.
    
    Includes automatic retry logic with exponential backoff for failures.
    
    Args:
        query: Query text to search for
        top_k: Maximum number of results to return (defaults to VECTOR_SEARCH_TOP_K from env)
        metadata_filter: Optional metadata filters (document_id, author, etc.)
        score_threshold: Minimum similarity score (0.0-1.0) to include in results (defaults to 0.0)
    
    Returns:
        List of search results, each containing:
        - chunk_id: Chunk identifier
        - score: Similarity score
        - metadata: Stored metadata (document_id, page_number, etc.)
    
    Raises:
        DatabaseError: If search fails after retries
    """
    try:
        _ensure_collections_exist()
        client = _get_client()
        
        # Load configuration from environment variables
        config = get_production_config()
        
        # Use environment-based defaults if not explicitly provided
        if top_k is None:
            top_k = config.retrieval_config.vector_search_top_k
        if score_threshold is None:
            score_threshold = 0.0
        
        # Generate query embedding with query prefix for better accuracy
        query_embedding = await _generate_embedding(query, is_query=True)

        # Build filter if provided
        qdrant_filter = _build_qdrant_filter(metadata_filter)

        # Perform ANN search with score threshold for better precision
        search_results = client.query_points(
            collection_name=settings.qdrant_main_collection,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold if score_threshold > 0 else None,
        ).points
        
        # Fallback: if ANN returns 0 results the HNSW index may not be built yet
        # (indexed_vectors_count < indexing_threshold). Retry with exact brute-force.
        if not search_results:
            logger.warning(
                "ANN vector search returned 0 results — HNSW index may not be built yet. "
                "Retrying with exact brute-force search."
            )
            search_results = client.query_points(
                collection_name=settings.qdrant_main_collection,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold if score_threshold > 0 else None,
                search_params=SearchParams(exact=True),
            ).points
            logger.info(f"Exact search fallback returned {len(search_results)} results")
        
        # Format results
        results = []
        for hit in search_results:
            results.append({
                "chunk_id": hit.payload.get("chunk_id"),
                "score": hit.score,
                "metadata": hit.payload
            })
        
        logger.debug(
            f"Searched paragraphs: query='{query[:30]}...', "
            f"top_k={top_k}, results={len(results)}"
        )
        
        return results
        
    except Exception as e:
        raise DatabaseError(f"Failed to search paragraphs: {e}") from e


async def search_paragraphs_with_client(
    client: QdrantClient,
    query: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar paragraphs using a provided Qdrant client (for connection pooling).
    
    This function allows using a specific Qdrant client instance from the connection
    pool manager for better resource management.
    
    Args:
        client: QdrantClient instance from connection pool
        query: Query text to search for
        top_k: Maximum number of results to return (defaults to VECTOR_SEARCH_TOP_K from env)
        metadata_filter: Optional metadata filters (document_id, author, etc.)
        score_threshold: Minimum similarity score (0.0-1.0) to include in results (defaults to 0.0)
    
    Returns:
        List of search results, each containing:
        - chunk_id: Chunk identifier
        - score: Similarity score
        - metadata: Stored metadata (document_id, page_number, etc.)
    
    Raises:
        DatabaseError: If search fails
    """
    try:
        # Load configuration from environment variables
        config = get_production_config()
        
        # Use environment-based defaults if not explicitly provided
        if top_k is None:
            top_k = config.retrieval_config.vector_search_top_k
        if score_threshold is None:
            score_threshold = 0.0
        
        # Generate query embedding with query prefix for better accuracy
        query_embedding = await _generate_embedding(query, is_query=True)

        # Build filter if provided
        qdrant_filter = _build_qdrant_filter(metadata_filter)

        # Perform ANN search with score threshold for better precision
        search_results = client.query_points(
            collection_name=settings.qdrant_main_collection,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold if score_threshold > 0 else None,
        ).points
        
        # Fallback: if ANN returns 0 results the HNSW index may not be built yet
        # (indexed_vectors_count < indexing_threshold). Retry with exact brute-force.
        if not search_results:
            logger.warning(
                "ANN vector search (with client) returned 0 results — HNSW index may not be built. "
                "Retrying with exact brute-force search."
            )
            search_results = client.query_points(
                collection_name=settings.qdrant_main_collection,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold if score_threshold > 0 else None,
                search_params=SearchParams(exact=True),
            ).points
            logger.info(f"Exact search fallback (with client) returned {len(search_results)} results")
        
        # Format results
        results = []
        for hit in search_results:
            results.append({
                "chunk_id": hit.payload.get("chunk_id"),
                "score": hit.score,
                "metadata": hit.payload
            })
        
        logger.debug(
            f"Searched paragraphs with client: query='{query[:30]}...', "
            f"top_k={top_k}, results={len(results)}"
        )
        
        return results
        
    except Exception as e:
        raise DatabaseError(f"Failed to search paragraphs with client: {e}") from e


async def generate_embedding(text: str, is_query: bool = False) -> List[float]:
    """
    Public function to generate embedding for text using the embedding service.
    
    This is the main entry point for embedding generation throughout the system.
    
    Args:
        text: Text to embed
        is_query: If True, use query prefix for better search accuracy
        
    Returns:
        768-dimensional embedding vector
        
    Raises:
        DatabaseError: If embedding generation fails
    """
    return await _generate_embedding(text, is_query)


async def generate_embeddings_batch(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Public function to generate embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to embed
        is_query: If True, use query prefix for better search accuracy
        
    Returns:
        List of 768-dimensional embedding vectors
        
    Raises:
        DatabaseError: If embedding generation fails
    """
    return await _generate_embeddings_batch(texts, is_query)

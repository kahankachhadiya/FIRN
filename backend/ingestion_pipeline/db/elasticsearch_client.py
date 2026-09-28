"""
Elasticsearch client for paragraph text and document metadata storage.

This module provides a drop-in replacement for redis_client.py, maintaining
the same interface while using Elasticsearch as the backend storage.
"""

import logging
from typing import Optional, Dict, Any, List
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from config.settings import get_settings
from utils.errors import DatabaseError
from utils.retry import with_retry, ELASTICSEARCH_RETRY_CONFIG

logger = logging.getLogger(__name__)

# Global connection pool and clients
_es_client: Optional[Elasticsearch] = None
_chunk_index: str = ""
_metadata_index: str = ""


def _get_client() -> Elasticsearch:
    """
    Get or create Elasticsearch client with connection pooling.
    
    The client uses built-in connection pooling from elasticsearch-py.
    Connection parameters are loaded from application settings.
    
    Returns:
        Elasticsearch client instance
        
    Raises:
        DatabaseError: If connection fails after retries
    """
    global _es_client, _chunk_index, _metadata_index
    
    if _es_client is not None:
        return _es_client
    
    try:
        settings = get_settings()
        
        # Store index names from settings
        _chunk_index = settings.elasticsearch_chunk_index
        _metadata_index = settings.elasticsearch_metadata_index
        
        # Create Elasticsearch client with connection pooling
        # Use compatibility mode for version 7/8 servers with version 9 client
        _es_client = Elasticsearch(
            hosts=[f"http://{settings.elasticsearch_host}:{settings.elasticsearch_port}"],
            # Connection pool settings
            max_retries=3,
            retry_on_timeout=True,
            # Timeout settings
            request_timeout=30,
            # Compatibility mode for version mismatch
            headers={"accept": "application/json", "content-type": "application/json"}
        )
        
        # Test connection
        if not _es_client.ping():
            raise DatabaseError("Failed to ping Elasticsearch cluster")
        
        # Ensure indices exist
        _ensure_indices_exist()
        
        # Register with graceful degradation manager
        _register_graceful_degradation()
        
        logger.info(
            f"Elasticsearch client initialized: {settings.elasticsearch_host}:{settings.elasticsearch_port}"
        )
        
        return _es_client
        
    except Exception as e:
        logger.error(f"Failed to initialize Elasticsearch client: {e}")
        raise DatabaseError(f"Failed to connect to Elasticsearch: {e}") from e


def _ensure_indices_exist() -> None:
    """
    Ensure Elasticsearch indices exist with appropriate mappings.
    
    Creates two indices:
    - chunks: For paragraph text storage with full-text search
    - documents: For document metadata storage
    
    Mappings:
    - chunks: { "text": "text" }
    - documents: { "metadata": "object" }
    
    Raises:
        DatabaseError: If index creation fails
    """
    try:
        client = _get_client() if _es_client is None else _es_client
        
        # Chunks index mapping for full-text search
        chunks_mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    }
                }
            }
        }
        
        # Documents index mapping for metadata storage
        documents_mapping = {
            "mappings": {
                "properties": {
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    }
                }
            }
        }
        
        # Create chunks index if it doesn't exist
        if not client.indices.exists(index=_chunk_index):
            client.indices.create(index=_chunk_index, body=chunks_mapping)
            logger.info(f"Created Elasticsearch index: {_chunk_index}")
        
        # Create documents index if it doesn't exist
        if not client.indices.exists(index=_metadata_index):
            client.indices.create(index=_metadata_index, body=documents_mapping)
            logger.info(f"Created Elasticsearch index: {_metadata_index}")
            
    except Exception as e:
        logger.error(f"Failed to ensure Elasticsearch indices exist: {e}")
        raise DatabaseError(f"Failed to create Elasticsearch indices: {e}") from e


def _register_graceful_degradation() -> None:
    """
    Register Elasticsearch component with graceful degradation manager.
    
    This registers the Elasticsearch component for health monitoring and
    provides fallback handlers when Elasticsearch is unavailable.
    """
    try:
        from utils.connection_pool import get_degradation_manager
        
        degradation_manager = get_degradation_manager()
        
        # Register component for graceful degradation
        # Health check function tests if Elasticsearch client can be obtained and pinged
        def elasticsearch_health_check() -> bool:
            try:
                client = _get_client()
                return client.ping()
            except Exception:
                return False
        
        degradation_manager.register_component(
            "elasticsearch",
            elasticsearch_health_check
        )
        
        # Register fallback handler for when Elasticsearch is unavailable
        def elasticsearch_fallback():
            logger.warning("Elasticsearch unavailable, returning empty results")
            return []
        
        degradation_manager.register_fallback(
            "elasticsearch",
            elasticsearch_fallback
        )
        
        logger.info("Registered Elasticsearch with graceful degradation manager")
        
    except Exception as e:
        logger.warning(f"Failed to register Elasticsearch with degradation manager: {e}")
        # Don't raise - graceful degradation registration is optional


@with_retry(
    max_retries=ELASTICSEARCH_RETRY_CONFIG.max_retries,
    base_delay=ELASTICSEARCH_RETRY_CONFIG.base_delay,
    max_delay=ELASTICSEARCH_RETRY_CONFIG.max_delay,
    retryable_exceptions=ELASTICSEARCH_RETRY_CONFIG.retryable_exceptions
)
def save_paragraph(chunk_id: str, text: str) -> None:
    """
    Save paragraph text to Elasticsearch with document ID CHUNK:{chunk_id}.
    
    Args:
        chunk_id: Unique chunk identifier
        text: Paragraph text content
        
    Raises:
        DatabaseError: If Elasticsearch operation fails after retries
    """
    try:
        client = _get_client()
        doc_id = f"CHUNK:{chunk_id}"
        
        client.index(
            index=_chunk_index,
            id=doc_id,
            document={"text": text}
        )
        
        logger.debug(f"Saved paragraph: {chunk_id}")
        
    except Exception as e:
        logger.error(f"Failed to save paragraph {chunk_id}: {e}")
        raise DatabaseError(f"Failed to save paragraph {chunk_id}: {e}") from e


@with_retry(
    max_retries=ELASTICSEARCH_RETRY_CONFIG.max_retries,
    base_delay=ELASTICSEARCH_RETRY_CONFIG.base_delay,
    max_delay=ELASTICSEARCH_RETRY_CONFIG.max_delay,
    retryable_exceptions=ELASTICSEARCH_RETRY_CONFIG.retryable_exceptions
)
def get_paragraph(chunk_id: str) -> Optional[str]:
    """
    Retrieve paragraph text from Elasticsearch by chunk ID.
    
    Args:
        chunk_id: Unique chunk identifier
        
    Returns:
        Paragraph text if found, None otherwise
        
    Raises:
        DatabaseError: If Elasticsearch operation fails after retries
    """
    try:
        client = _get_client()
        doc_id = f"CHUNK:{chunk_id}"
        
        result = client.get(index=_chunk_index, id=doc_id, ignore=404)
        
        if result.get('found'):
            return result['_source']['text']
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get paragraph {chunk_id}: {e}")
        raise DatabaseError(f"Failed to get paragraph {chunk_id}: {e}") from e


@with_retry(
    max_retries=ELASTICSEARCH_RETRY_CONFIG.max_retries,
    base_delay=ELASTICSEARCH_RETRY_CONFIG.base_delay,
    max_delay=ELASTICSEARCH_RETRY_CONFIG.max_delay,
    retryable_exceptions=ELASTICSEARCH_RETRY_CONFIG.retryable_exceptions
)
def save_paragraphs_batch(paragraphs: List[tuple]) -> None:
    """
    Save multiple paragraphs using Elasticsearch bulk API.
    
    Args:
        paragraphs: List of (chunk_id, text) tuples
        
    Raises:
        DatabaseError: If bulk operation fails
    """
    try:
        client = _get_client()
        
        # Prepare bulk actions
        actions = []
        for chunk_id, text in paragraphs:
            doc_id = f"CHUNK:{chunk_id}"
            actions.append({
                '_op_type': 'index',
                '_index': _chunk_index,
                '_id': doc_id,
                '_source': {'text': text}
            })
        
        # Execute bulk operation
        success, failed = bulk(client, actions, raise_on_error=False)
        
        if failed:
            logger.warning(f"Bulk save paragraphs: {success} succeeded, {len(failed)} failed")
            # Raise error if any operations failed
            raise DatabaseError(f"Bulk operation failed for {len(failed)} paragraphs")
        
        logger.debug(f"Saved {success} paragraphs in batch")
        
    except Exception as e:
        logger.error(f"Failed to save paragraphs batch: {e}")
        raise DatabaseError(f"Failed to save paragraphs batch: {e}") from e


@with_retry(
    max_retries=ELASTICSEARCH_RETRY_CONFIG.max_retries,
    base_delay=ELASTICSEARCH_RETRY_CONFIG.base_delay,
    max_delay=ELASTICSEARCH_RETRY_CONFIG.max_delay,
    retryable_exceptions=ELASTICSEARCH_RETRY_CONFIG.retryable_exceptions
)
def get_paragraphs_batch(chunk_ids: List[str]) -> Dict[str, Optional[str]]:
    """
    Retrieve multiple paragraphs using Elasticsearch mget API.
    
    Args:
        chunk_ids: List of chunk identifiers
        
    Returns:
        Dictionary mapping chunk_id to text (None if not found)
        
    Raises:
        DatabaseError: If mget operation fails
    """
    try:
        client = _get_client()
        
        # Prepare document IDs
        doc_ids = [f"CHUNK:{chunk_id}" for chunk_id in chunk_ids]
        
        # Execute mget operation
        result = client.mget(index=_chunk_index, body={'ids': doc_ids})
        
        # Build result dictionary
        paragraphs = {}
        for i, doc in enumerate(result['docs']):
            chunk_id = chunk_ids[i]
            if doc.get('found'):
                paragraphs[chunk_id] = doc['_source']['text']
            else:
                paragraphs[chunk_id] = None
        
        logger.debug(f"Retrieved {len(paragraphs)} paragraphs in batch")
        
        return paragraphs
        
    except Exception as e:
        logger.error(f"Failed to get paragraphs batch: {e}")
        raise DatabaseError(f"Failed to get paragraphs batch: {e}") from e


@with_retry(
    max_retries=ELASTICSEARCH_RETRY_CONFIG.max_retries,
    base_delay=ELASTICSEARCH_RETRY_CONFIG.base_delay,
    max_delay=ELASTICSEARCH_RETRY_CONFIG.max_delay,
    retryable_exceptions=ELASTICSEARCH_RETRY_CONFIG.retryable_exceptions
)
def save_document_metadata(document_id: str, metadata: Dict[str, Any]) -> None:
    """
    Save document metadata to Elasticsearch with document ID DOC:{document_id}.
    
    Preserves null values in metadata during JSON serialization.
    
    Args:
        document_id: Unique document identifier
        metadata: Document metadata dictionary (may contain None/null values)
        
    Raises:
        DatabaseError: If Elasticsearch operation fails after retries
    """
    try:
        client = _get_client()
        doc_id = f"DOC:{document_id}"
        
        # Elasticsearch automatically preserves null values in JSON
        # Store metadata as-is without any transformation
        client.index(
            index=_metadata_index,
            id=doc_id,
            document={"metadata": metadata}
        )
        
        logger.debug(f"Saved document metadata: {document_id}")
        
    except Exception as e:
        logger.error(f"Failed to save document metadata {document_id}: {e}")
        raise DatabaseError(f"Failed to save document metadata {document_id}: {e}") from e


@with_retry(
    max_retries=ELASTICSEARCH_RETRY_CONFIG.max_retries,
    base_delay=ELASTICSEARCH_RETRY_CONFIG.base_delay,
    max_delay=ELASTICSEARCH_RETRY_CONFIG.max_delay,
    retryable_exceptions=ELASTICSEARCH_RETRY_CONFIG.retryable_exceptions
)
def get_document_metadata(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve document metadata from Elasticsearch by document ID.
    
    Preserves null values in the returned metadata dictionary.
    
    Args:
        document_id: Unique document identifier
        
    Returns:
        Document metadata dictionary if found (may contain None values), None otherwise
        
    Raises:
        DatabaseError: If Elasticsearch operation fails after retries
    """
    try:
        client = _get_client()
        doc_id = f"DOC:{document_id}"
        
        result = client.get(index=_metadata_index, id=doc_id, ignore=404)
        
        if result.get('found'):
            # Return metadata as-is, Elasticsearch preserves null values
            return result['_source']['metadata']
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get document metadata {document_id}: {e}")
        raise DatabaseError(f"Failed to get document metadata {document_id}: {e}") from e


def list_all_document_metadata() -> List[Dict[str, Any]]:
    """
    Retrieve all document metadata records from Elasticsearch.

    Returns a list of dicts, each containing the document_id (stripped of the
    ``DOC:`` prefix) and the stored metadata fields.

    Returns:
        List of dicts with keys ``document_id`` and all metadata fields.

    Raises:
        DatabaseError: If Elasticsearch operation fails.
    """
    try:
        client = _get_client()

        result = client.search(
            index=_metadata_index,
            body={"query": {"match_all": {}}, "size": 10000},
        )

        documents = []
        for hit in result.get("hits", {}).get("hits", []):
            raw_id: str = hit.get("_id", "")
            document_id = raw_id[len("DOC:"):] if raw_id.startswith("DOC:") else raw_id
            metadata = hit.get("_source", {}).get("metadata", {}) or {}
            documents.append({"document_id": document_id, **metadata})

        return documents

    except Exception as e:
        logger.error(f"Failed to list all document metadata: {e}")
        raise DatabaseError(f"Failed to list all document metadata: {e}") from e


def delete_document_metadata(document_id: str) -> bool:
    """
    Delete a document metadata record from Elasticsearch.

    Args:
        document_id: Unique document identifier (without ``DOC:`` prefix).

    Returns:
        True if the document was deleted, False if it was not found.

    Raises:
        DatabaseError: If Elasticsearch operation fails.
    """
    try:
        client = _get_client()
        doc_id = f"DOC:{document_id}"

        result = client.delete(index=_metadata_index, id=doc_id, ignore=404)

        deleted = result.get("result") == "deleted"
        if deleted:
            logger.debug(f"Deleted document metadata: {document_id}")
        else:
            logger.debug(f"Document metadata not found for deletion: {document_id}")

        return deleted

    except Exception as e:
        logger.error(f"Failed to delete document metadata {document_id}: {e}")
        raise DatabaseError(f"Failed to delete document metadata {document_id}: {e}") from e


def _escape_special_characters(query: str) -> str:
    """
    Escape special characters in Elasticsearch query strings.

    Elasticsearch special characters that need escaping:
    + - = && || > < ! ( ) { } [ ] ^ " ~ * ? : \\ /

    Args:
        query: Raw search query string

    Returns:
        Query string with special characters escaped
    """
    special_chars = ['+', '-', '=', '&&', '||', '>', '<', '!', '(', ')', '{', '}',
                     '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/']

    escaped_query = query
    for char in special_chars:
        escaped_query = escaped_query.replace(char, f'\\{char}')

    return escaped_query


def search_text(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Full-text search using Elasticsearch match query.

    Searches the chunks index for paragraphs matching the query text.
    Results are ranked by relevance score in descending order.

    Args:
        query: Search query text
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching chunks with chunk_id and text, sorted by relevance.
        Returns empty list on error or if no matches found.
        Format: [{"chunk_id": "abc123", "text": "paragraph text..."}, ...]

    Note:
        This function does not raise exceptions. Errors are logged and
        an empty list is returned to allow graceful degradation.
    """
    try:
        client = _get_client()

        # Use match query for full-text search with standard analyzer
        # The match query automatically handles tokenization and relevance scoring
        search_body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": limit
        }

        # Execute search
        result = client.search(index=_chunk_index, body=search_body)

        # Extract and format results
        matches = []
        for hit in result['hits']['hits']:
            # Remove CHUNK: prefix from document ID to get chunk_id
            doc_id = hit['_id']
            chunk_id = doc_id.replace('CHUNK:', '') if doc_id.startswith('CHUNK:') else doc_id

            matches.append({
                'chunk_id': chunk_id,
                'text': hit['_source']['text']
            })

        logger.debug(f"Search returned {len(matches)} results for query: {query[:50]}")

        return matches

    except Exception as e:
        # Log warning but return empty list for graceful degradation
        logger.warning(f"Elasticsearch search failed for query '{query[:50]}': {e}")
        return []


async def search_text_async(query: str, limit: int = 10, connection_pool=None) -> List[Dict[str, Any]]:
    """
    Async full-text search using Elasticsearch with connection pooling.
    
    Performs asynchronous full-text search on the chunks index. This function
    uses the connection pool if provided, otherwise uses the default client.
    
    Args:
        query: Search query text
        limit: Maximum number of results (default: 10)
        connection_pool: Optional connection pool instance (currently unused,
                        as elasticsearch-py client has built-in connection pooling)
        
    Returns:
        List of matching chunks with chunk_id and text, sorted by relevance.
        Returns empty list on error or if no matches found.
        Format: [{"chunk_id": "abc123", "text": "paragraph text..."}, ...]
        
    Note:
        This function does not raise exceptions. Errors are logged and
        an empty list is returned to allow graceful degradation.
        
    Raises:
        DatabaseError: If Elasticsearch operation fails after retries
    """
    try:
        # Get the client (uses built-in connection pooling)
        client = _get_client()
        
        # Use match query for full-text search
        search_body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": limit
        }
        
        # Execute search (elasticsearch-py doesn't have native async support,
        # so we use the synchronous client in an async context)
        result = client.search(index=_chunk_index, body=search_body)
        
        # Extract and format results
        matches = []
        for hit in result['hits']['hits']:
            # Remove CHUNK: prefix from document ID to get chunk_id
            doc_id = hit['_id']
            chunk_id = doc_id.replace('CHUNK:', '') if doc_id.startswith('CHUNK:') else doc_id
            
            matches.append({
                'chunk_id': chunk_id,
                'text': hit['_source']['text']
            })
        
        logger.debug(f"Async search returned {len(matches)} results for query: {query[:50]}")
        
        return matches
        
    except Exception as e:
        # Log warning but return empty list for graceful degradation
        logger.warning(f"Async Elasticsearch search failed for query '{query[:50]}': {e}")
        return []


async def search_text_async_with_connection(es_connection, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Async full-text search using a provided Elasticsearch connection.
    
    Performs asynchronous full-text search using an existing Elasticsearch
    connection. This is useful when managing connections explicitly or when
    using connection pooling at a higher level.
    
    Args:
        es_connection: Active Elasticsearch connection/client instance
        query: Search query text
        limit: Maximum number of results (default: 10)
        
    Returns:
        List of matching chunks with chunk_id and text, sorted by relevance.
        Returns empty list on error or if no matches found.
        Format: [{"chunk_id": "abc123", "text": "paragraph text..."}, ...]
        
    Note:
        This function does not raise exceptions. Errors are logged and
        an empty list is returned to allow graceful degradation.
        
    Raises:
        DatabaseError: If Elasticsearch operation fails
    """
    try:
        # Use the provided connection
        client = es_connection
        
        # Get index name from global variable or settings
        chunk_index = _chunk_index if _chunk_index else get_settings().elasticsearch_chunk_index
        
        # Use match query for full-text search
        search_body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": limit
        }
        
        # Execute search with provided connection
        result = client.search(index=chunk_index, body=search_body)
        
        # Extract and format results
        matches = []
        for hit in result['hits']['hits']:
            # Remove CHUNK: prefix from document ID to get chunk_id
            doc_id = hit['_id']
            chunk_id = doc_id.replace('CHUNK:', '') if doc_id.startswith('CHUNK:') else doc_id
            
            matches.append({
                'chunk_id': chunk_id,
                'text': hit['_source']['text']
            })
        
        logger.debug(f"Async search (with connection) returned {len(matches)} results for query: {query[:50]}")
        
        return matches
        
    except Exception as e:
        # Log warning but return empty list for graceful degradation
        logger.warning(f"Async Elasticsearch search (with connection) failed for query '{query[:50]}': {e}")
        return []

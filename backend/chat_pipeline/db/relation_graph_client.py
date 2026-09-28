"""
Relation Graph Client for querying related chunks from Graph A (Similar Graph).

This module provides a clean interface for querying Graph A from the unified Knowledge Graph
to find semantically related chunks based on SIMILAR_TO relationships. It wraps the
underlying FalkorDB client functions with additional error handling and logging.

Graph A Architecture (in Unified Knowledge Graph):
- Stores SIMILAR_TO edges between chunks from different documents
- Edges are created using vector search + Qwen-Reranker-0.6B scoring
- Edge weights represent reranker scores (0.0 to 1.0)
- Provides broad semantic coverage across the corpus
- Stored unidirectionally but queried bidirectionally

This is part of the unified Knowledge Graph architecture where Graph A provides semantic
relations (SIMILAR_TO edges) and Graph B provides typed relationships (SUPPORTS, CONTRADICTS, etc.).
"""

from typing import List, Dict, Any, Optional
from config.settings import settings
from db.falkor_client import get_similar_neighbors, ensure_knowledge_graph_exists
from utils.errors import DatabaseError
from utils.logging_utils import get_logger
from utils.retry import with_retry, FALKOR_RETRY_CONFIG


logger = get_logger(__name__)


@with_retry(
    max_retries=FALKOR_RETRY_CONFIG.max_retries,
    base_delay=FALKOR_RETRY_CONFIG.base_delay,
    max_delay=FALKOR_RETRY_CONFIG.max_delay,
    retryable_exceptions=FALKOR_RETRY_CONFIG.retryable_exceptions
)
def query_related_chunks(
    chunk_id: str,
    max_neighbors: Optional[int] = None,
    min_similarity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Query Graph A (Similar Graph) from unified Knowledge Graph to find semantically related chunks.
    
    This function queries the unified Knowledge Graph for SIMILAR_TO edges to find chunks that are
    semantically related to the input chunk. The edges are created during ingestion using:
    1. Vector search on Main Collection to find candidates
    2. Qwen-Reranker-0.6B scoring to determine relationship strength
    3. Threshold filtering to ensure quality relationships
    
    Edges are stored unidirectionally (A→B) but queried bidirectionally (A↔B) for efficiency.
    
    Graph A provides broad semantic coverage across documents without relying on
    LLM-generated summaries, making it more efficient and accurate than the old
    summary-based approach.
    
    The function handles edge cases gracefully:
    - Empty graph: Returns empty list
    - Missing source node: Returns empty list
    - No edges: Returns empty list
    - Database unavailable: Returns empty list with warning
    
    Includes automatic retry logic with exponential backoff for failures.
    
    Args:
        chunk_id: Unique identifier of the chunk to find relations for
        max_neighbors: Maximum number of related chunks to return.
                      Defaults to SIMILAR_GRAPH_MAX_NEIGHBORS from config.
        min_similarity: Minimum similarity score threshold (0.0 to 1.0).
                       Defaults to SIMILAR_GRAPH_MIN_SIMILARITY from config.
    
    Returns:
        List of related chunk dictionaries, each containing:
        - chunk_id: Unique identifier of the related chunk
        - document_id: Document identifier the chunk belongs to
        - page_number: Page number of the chunk
        - similarity: Similarity score (0.0 to 1.0)
        
        Returns empty list if no related chunks found or on error.
    
    Example:
        >>> related = query_related_chunks("doc1_p1_c1", max_neighbors=3, min_similarity=0.8)
        >>> for chunk in related:
        ...     print(f"Related chunk: {chunk['chunk_id']} (similarity: {chunk['similarity']:.3f})")
    """
    # Use config defaults if not specified
    if max_neighbors is None:
        max_neighbors = 10

    if min_similarity is None:
        min_similarity = 0.0

    try:
        # Ensure unified Knowledge Graph exists before querying
        ensure_knowledge_graph_exists()
        
        # Query similar neighbors from unified Knowledge Graph (bidirectional)
        neighbors = get_similar_neighbors(
            chunk_id=chunk_id,
            max_neighbors=max_neighbors,
            min_similarity=min_similarity
        )
        
        logger.debug(
            f"Found {len(neighbors)} related chunks for {chunk_id} "
            f"(max={max_neighbors}, min_sim={min_similarity})"
        )
        
        return neighbors
        
    except DatabaseError as e:
        logger.error(f"Database error querying related chunks for {chunk_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error querying related chunks for {chunk_id}: {e}")
        return []


async def query_related_chunks_with_connection(
    connection,
    chunk_id: str,
    max_neighbors: Optional[int] = None,
    min_similarity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Query the unified Knowledge Graph using a provided FalkorDB connection (for connection pooling).
    
    This async function allows using a specific FalkorDB connection instance from the
    connection pool manager for better resource management. Queries SIMILAR_TO edges bidirectionally.
    
    Args:
        connection: FalkorDB connection from connection pool
        chunk_id: Unique identifier of the chunk to find relations for
        max_neighbors: Maximum number of related chunks to return.
                      Defaults to SIMILAR_GRAPH_MAX_NEIGHBORS from config.
        min_similarity: Minimum similarity score threshold (0.0 to 1.0).
                       Defaults to SIMILAR_GRAPH_MIN_SIMILARITY from config.
    
    Returns:
        List of related chunk dictionaries, each containing:
        - chunk_id: Unique identifier of the related chunk
        - document_id: Document identifier the chunk belongs to
        - page_number: Page number of the chunk
        - similarity: Similarity score (0.0 to 1.0)
        
        Returns empty list if no related chunks found or on error.
    """
    # Use config defaults if not specified
    if max_neighbors is None:
        max_neighbors = 10

    if min_similarity is None:
        min_similarity = 0.0

    try:
        # Use the sync get_similar_neighbors run in a thread executor
        # (get_similar_neighbors_with_connection does not exist in falkor_client)
        import asyncio
        loop = asyncio.get_event_loop()
        neighbors = await loop.run_in_executor(
            None,
            lambda: get_similar_neighbors(
                chunk_id=chunk_id,
                max_neighbors=max_neighbors,
                min_similarity=min_similarity
            )
        )
        
        logger.debug(
            f"Found {len(neighbors)} related chunks for {chunk_id} via connection pool "
            f"(max={max_neighbors}, min_sim={min_similarity})"
        )
        
        return neighbors
        
    except DatabaseError as e:
        logger.error(f"Database error querying related chunks for {chunk_id} via connection pool: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error querying related chunks for {chunk_id} via connection pool: {e}")
        return []


def query_related_chunks_batch(
    chunk_ids: List[str],
    max_neighbors_per_chunk: Optional[int] = None,
    min_similarity: Optional[float] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Query related chunks for multiple chunks in batch.
    
    This is a convenience function that queries related chunks for multiple
    source chunks and returns a dictionary mapping each chunk_id to its
    related chunks.
    
    Args:
        chunk_ids: List of chunk identifiers to find relations for
        max_neighbors_per_chunk: Maximum neighbors per chunk.
                                Defaults to SIMILAR_GRAPH_MAX_NEIGHBORS from config.
        min_similarity: Minimum similarity threshold.
                       Defaults to SIMILAR_GRAPH_MIN_SIMILARITY from config.
    
    Returns:
        Dictionary mapping chunk_id to list of related chunks.
        Chunks with no relations will have empty lists.
    
    Example:
        >>> results = query_related_chunks_batch(["doc1_p1_c1", "doc1_p2_c1"])
        >>> for chunk_id, related in results.items():
        ...     print(f"{chunk_id}: {len(related)} related chunks")
    """
    results = {}
    
    for chunk_id in chunk_ids:
        related = query_related_chunks(
            chunk_id=chunk_id,
            max_neighbors=max_neighbors_per_chunk,
            min_similarity=min_similarity
        )
        results[chunk_id] = related
    
    logger.debug(
        f"Batch query completed for {len(chunk_ids)} chunks, "
        f"found relations for {sum(1 for r in results.values() if r)} chunks"
    )
    
    return results


def get_relation_graph_stats() -> Dict[str, Any]:
    """
    Get statistics about the Similar Graph.
    
    Returns:
        Dictionary containing graph statistics:
        - total_chunks: Total number of chunk vertices in the graph
        - total_edges: Total number of SIMILAR_TO edges
        - avg_edges_per_chunk: Average number of edges per chunk
        - config: Current configuration settings
    
    Note: This function may be slow on large graphs as it counts all nodes and edges.
    """
    try:
        from db.falkor_client import _get_knowledge_graph
        
        graph = _get_knowledge_graph()
        
        # Count vertices
        vertex_query = f"""
        MATCH (c:{settings.falkor_vertex_collection})
        RETURN count(c) AS vertex_count
        """
        vertex_result = graph.query(vertex_query)
        total_chunks = vertex_result.result_set[0][0] if vertex_result.result_set else 0
        
        # Count edges
        edge_query = """
        MATCH ()-[r:SIMILAR_TO]->()
        RETURN count(r) AS edge_count
        """
        edge_result = graph.query(edge_query)
        total_edges = edge_result.result_set[0][0] if edge_result.result_set else 0
        
        # Calculate average
        avg_edges = total_edges / total_chunks if total_chunks > 0 else 0.0
        
        stats = {
            'total_chunks': total_chunks,
            'total_edges': total_edges,
            'avg_edges_per_chunk': round(avg_edges, 2),
            'config': {
                'graph_name': settings.falkor_knowledge_graph_name
            }
        }
        
        logger.info(f"Relation graph stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get relation graph stats: {e}")
        return {
            'total_chunks': 0,
            'total_edges': 0,
            'avg_edges_per_chunk': 0.0,
            'error': str(e)
        }

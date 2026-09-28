"""
Falkor DB database client for graph-based chunk relationships.

This module provides functions for:
- Creating and managing graph collections (vertices and edges)
- Adding chunk vertices to the graph
- Creating edges between chunks with similarity scores
- Querying neighbors of a chunk sorted by similarity

Optimized for low latency with:
- Connection reuse
- Batch operations support
- Indexed queries
"""

from typing import List, Dict, Any, Optional
from falkordb import FalkorDB

from config.settings import settings
from utils.errors import DatabaseError
from utils.logging_utils import get_logger
from utils.connection_pool import (
    register_connection_pool, get_connection_pool, ConnectionConfig,
    get_degradation_manager, with_retry, with_timeout
)


logger = get_logger(__name__)


# Global Falkor DB client (lazy initialization)
_falkor_client: Optional[FalkorDB] = None
_knowledge_graph = None  # Unified Knowledge Graph
_indexes_created: bool = False
_connection_pool_initialized: bool = False


def _initialize_connection_pools():
    """Initialize connection pools for FalkorDB."""
    global _connection_pool_initialized
    
    if _connection_pool_initialized:
        return
    
    def create_falkor_connection():
        """Create a new FalkorDB connection."""
        try:
            client = FalkorDB(
                host=settings.falkor_host,
                port=settings.falkor_port,
                username=settings.falkor_username if settings.falkor_username else None,
                password=settings.falkor_password if settings.falkor_password else None
            )
            return client
        except Exception as e:
            raise DatabaseError(f"Failed to create FalkorDB connection: {e}") from e
    
    def validate_falkor_connection(client):
        """Validate a FalkorDB connection."""
        try:
            # Try a simple query to test connection
            graph = client.select_graph("test_connection")
            graph.query("RETURN 1")
            return True
        except Exception:
            return False
    
    def close_falkor_connection(client):
        """Close a FalkorDB connection."""
        try:
            if hasattr(client, 'close'):
                client.close()
        except Exception as e:
            logger.debug(f"Error closing FalkorDB connection: {e}")
    
    # Register connection pool
    config = ConnectionConfig(
        max_connections=10,
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        timeout=30.0,
        connection_timeout=10.0
    )
    
    register_connection_pool(
        "falkor",
        create_falkor_connection,
        validate_falkor_connection,
        close_falkor_connection,
        config
    )
    
    # Register component for graceful degradation
    degradation_manager = get_degradation_manager()
    degradation_manager.register_component(
        "falkor_db",
        lambda: get_connection_pool("falkor").health_check() if get_connection_pool("falkor") else False
    )
    
    # Register fallback handlers
    degradation_manager.register_fallback(
        "falkor_db",
        lambda: logger.warning("FalkorDB unavailable, returning empty results")
    )
    
    _connection_pool_initialized = True
    logger.info("Initialized FalkorDB connection pools and graceful degradation")


@with_retry(max_retries=3, base_delay=1.0, retryable_exceptions=(DatabaseError,))
@with_timeout(30.0)
def _get_client() -> FalkorDB:
    """
    Get or create Falkor DB client with connection pooling and retry logic.
    
    Returns:
        FalkorDB instance
        
    Raises:
        DatabaseError: If connection fails after retries
        TimeoutError: If connection times out
    """
    global _falkor_client, _knowledge_graph
    
    _initialize_connection_pools()
    
    if _falkor_client is None:
        pool = get_connection_pool("falkor")
        if not pool:
            raise DatabaseError("FalkorDB connection pool not initialized")
        
        try:
            with pool.get_connection() as client:
                # Test the connection
                test_graph = client.select_graph("connection_test")
                test_graph.query("RETURN 1")
                
                # Store client and unified knowledge graph
                _falkor_client = client
                _knowledge_graph = client.select_graph(settings.falkor_knowledge_graph_name)
                
                logger.info(
                    f"Connected to Falkor DB at {settings.falkor_host}:{settings.falkor_port} "
                    f"(knowledge_graph={settings.falkor_knowledge_graph_name})"
                )
        except Exception as e:
            get_degradation_manager().mark_component_unhealthy("falkor_db")
            raise DatabaseError(f"Failed to connect to Falkor DB: {e}") from e
    
    return _falkor_client


@with_retry(max_retries=2, base_delay=0.5, retryable_exceptions=(DatabaseError,))
def _get_knowledge_graph():
    """
    Get the unified knowledge graph instance with retry logic.
    
    Returns:
        Knowledge Graph instance
        
    Raises:
        DatabaseError: If graph is not initialized
    """
    if _knowledge_graph is None:
        _get_client()
    return _knowledge_graph


def ensure_knowledge_graph_exists() -> None:
    """
    Ensure unified Knowledge Graph collections and indexes exist in Falkor DB.
    
    Creates indexes for efficient querying:
    - Vertex collection: chunks with chunk_id and document_id indexes
    - Edge types: SIMILAR_TO (Graph A) and probe types (Graph B)
    
    Raises:
        DatabaseError: If collection creation fails
    """
    global _indexes_created
    
    # Skip if already created this session
    if _indexes_created:
        return
    
    try:
        graph = _get_knowledge_graph()
        
        # Create indexes for vertices (chunks)
        try:
            query = f"CREATE INDEX FOR (c:{settings.falkor_vertex_collection}) ON (c.chunk_id)"
            graph.query(query)
            logger.info(f"Created index on {settings.falkor_vertex_collection}.chunk_id")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.debug(f"Vertex chunk_id index: {e}")
        
        # Create index on document_id for filtering
        try:
            query = f"CREATE INDEX FOR (c:{settings.falkor_vertex_collection}) ON (c.document_id)"
            graph.query(query)
            logger.info(f"Created index on {settings.falkor_vertex_collection}.document_id")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.debug(f"Vertex document_id index: {e}")
        
        # Create index for Graph A edges (SIMILAR_TO)
        try:
            query = "CREATE INDEX FOR ()-[r:SIMILAR_TO]-() ON (r.similarity)"
            graph.query(query)
            logger.info("Created index on SIMILAR_TO.similarity")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.debug(f"SIMILAR_TO edge index: {e}")
        
        # Create indexes for Graph B edges (probe types)
        probe_types = ["CONTRADICTS", "ELABORATES", "DEPENDS_ON"]
        for probe_type in probe_types:
            try:
                query = f"CREATE INDEX FOR ()-[r:{probe_type}]-() ON (r.confidence)"
                graph.query(query)
                logger.info(f"Created index on {probe_type}.confidence")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.debug(f"{probe_type} edge index: {e}")
        
        _indexes_created = True
        logger.info("Ensured unified Knowledge Graph indexes exist")
        
    except Exception as e:
        raise DatabaseError(f"Failed to ensure Knowledge Graph collections exist: {e}") from e


# Legacy function names for backward compatibility
def ensure_graph_exists() -> None:
    """Legacy function - redirects to ensure_knowledge_graph_exists()."""
    ensure_knowledge_graph_exists()


def ensure_probe_graph_exists() -> None:
    """Legacy function - redirects to ensure_knowledge_graph_exists()."""
    ensure_knowledge_graph_exists()




def add_chunk_vertex(
    chunk_id: str,
    document_id: str,
    page_number: int
) -> None:
    """
    Add a chunk vertex to the unified Knowledge Graph.
    
    Args:
        chunk_id: Unique chunk identifier
        document_id: Document identifier this chunk belongs to
        page_number: Page number of the chunk
    
    Raises:
        DatabaseError: If vertex creation fails
    """
    try:
        graph = _get_knowledge_graph()
        
        # Use MERGE to avoid duplicates (idempotent operation)
        query = f"""
        MERGE (c:{settings.falkor_vertex_collection} {{chunk_id: $chunk_id}})
        ON CREATE SET c.document_id = $document_id, c.page_number = $page_number
        ON MATCH SET c.document_id = $document_id, c.page_number = $page_number
        """
        
        params = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "page_number": page_number
        }
        
        graph.query(query, params)
        logger.debug(f"Added vertex: {chunk_id}")
        
    except Exception as e:
        raise DatabaseError(
            f"Failed to add chunk vertex {chunk_id} to Falkor DB: {e}"
        ) from e


def add_chunk_vertices_batch(vertices: List[tuple]) -> None:
    """
    Add multiple chunk vertices in batch (more efficient).
    
    Args:
        vertices: List of (chunk_id, document_id, page_number) tuples
    
    Raises:
        DatabaseError: If vertex creation fails
    """
    try:
        graph = _get_knowledge_graph()
        
        # Use UNWIND for batch insert
        query = f"""
        UNWIND $vertices AS v
        MERGE (c:{settings.falkor_vertex_collection} {{chunk_id: v.chunk_id}})
        ON CREATE SET c.document_id = v.document_id, c.page_number = v.page_number
        ON MATCH SET c.document_id = v.document_id, c.page_number = v.page_number
        """
        
        params = {
            "vertices": [
                {"chunk_id": v[0], "document_id": v[1], "page_number": v[2]}
                for v in vertices
            ]
        }
        
        graph.query(query, params)
        logger.debug(f"Added {len(vertices)} vertices in batch")
        
    except Exception as e:
        raise DatabaseError(f"Failed to add chunk vertices batch: {e}") from e


def add_chunk_relation(
    source_chunk_id: str,
    target_chunk_id: str,
    similarity: float
) -> None:
    """
    Add a SIMILAR_TO edge between two chunks in the unified Knowledge Graph.
    
    This creates Graph A edges with the SIMILAR_TO relationship type.
    
    Args:
        source_chunk_id: Source chunk identifier
        target_chunk_id: Target chunk identifier
        similarity: Similarity score between chunks (typically 0.0 to 1.0)
    
    Raises:
        DatabaseError: If edge creation fails
    """
    try:
        graph = _get_knowledge_graph()
        
        query = f"""
        MATCH (source:{settings.falkor_vertex_collection} {{chunk_id: $source_id}})
        MATCH (target:{settings.falkor_vertex_collection} {{chunk_id: $target_id}})
        MERGE (source)-[r:SIMILAR_TO]->(target)
        ON CREATE SET r.similarity = $similarity, r.source = 'graph_a'
        ON MATCH SET r.similarity = $similarity, r.source = 'graph_a'
        """
        
        params = {
            "source_id": source_chunk_id,
            "target_id": target_chunk_id,
            "similarity": similarity
        }
        
        graph.query(query, params)
        logger.debug(f"Added SIMILAR_TO edge: {source_chunk_id} -> {target_chunk_id}")
        
    except Exception as e:
        raise DatabaseError(
            f"Failed to add chunk relation {source_chunk_id} -> {target_chunk_id}: {e}"
        ) from e


def add_chunk_relations_batch(relations: List[tuple]) -> None:
    """
    Add multiple SIMILAR_TO edges in batch (more efficient).
    
    Args:
        relations: List of (source_chunk_id, target_chunk_id, similarity) tuples
    
    Raises:
        DatabaseError: If edge creation fails
    """
    try:
        graph = _get_knowledge_graph()
        
        # Process in smaller batches to avoid query size limits
        batch_size = 100
        for i in range(0, len(relations), batch_size):
            batch = relations[i:i + batch_size]
            
            query = f"""
            UNWIND $relations AS rel_data
            MATCH (source:{settings.falkor_vertex_collection} {{chunk_id: rel_data.source_id}})
            MATCH (target:{settings.falkor_vertex_collection} {{chunk_id: rel_data.target_id}})
            MERGE (source)-[r:SIMILAR_TO]->(target)
            SET r.similarity = rel_data.similarity, r.source = 'graph_a'
            """
            
            params = {
                "relations": [
                    {"source_id": rel[0], "target_id": rel[1], "similarity": rel[2]}
                    for rel in batch
                ]
            }
            
            graph.query(query, params)
        
        logger.debug(f"Added {len(relations)} SIMILAR_TO edges in batch")
        
    except Exception as e:
        raise DatabaseError(f"Failed to add chunk relations batch: {e}") from e


@with_timeout(15.0)
def get_neighbors(
    chunk_id: str,
    max_neighbors: int,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Get neighboring chunks connected by SIMILAR_TO edges (bidirectional query).
    
    Handles edge cases gracefully:
    - Empty graph: Returns empty list
    - Missing source node: Returns empty list
    - No edges: Returns empty list
    - Database unavailable: Returns empty list with warning
    
    Args:
        chunk_id: Chunk identifier to find neighbors for
        max_neighbors: Maximum number of neighbors to return
        min_similarity: Minimum similarity score threshold (default 0.0)
    
    Returns:
        List of neighbor dictionaries sorted by similarity descending.
        Returns empty list if no neighbors found or graph is empty.
    """
    degradation_manager = get_degradation_manager()
    
    def _execute_query():
        try:
            graph = _get_knowledge_graph()
            
            # First check if the source node exists (handles empty graph and missing node cases)
            check_query = f"""
            MATCH (source:{settings.falkor_vertex_collection} {{chunk_id: $chunk_id}})
            RETURN count(source) AS node_count
            """
            
            check_result = graph.query(check_query, {"chunk_id": chunk_id})
            
            # If no source node found, return empty list gracefully
            if not check_result.result_set or check_result.result_set[0][0] == 0:
                logger.info(f"Source node not found in graph: {chunk_id} - returning empty neighbors list")
                return []
            
            # Bidirectional query — fetch all edges first, then filter by similarity in Python.
            # NOTE: FalkorDB hangs when a WHERE clause filters on edge properties
            # inside an undirected pattern (a)-[r]-(b), so we avoid WHERE on r here
            # and apply the threshold in Python instead.
            query = f"""
            MATCH (source:{settings.falkor_vertex_collection} {{chunk_id: $chunk_id}})
            MATCH (source)-[r:SIMILAR_TO]-(neighbor:{settings.falkor_vertex_collection})
            RETURN neighbor.chunk_id AS chunk_id,
                   neighbor.document_id AS document_id,
                   neighbor.page_number AS page_number,
                   r.similarity AS similarity
            ORDER BY r.similarity DESC
            LIMIT $max_neighbors
            """
            
            params = {
                "chunk_id": chunk_id,
                "max_neighbors": max_neighbors * 4  # Fetch more; we'll filter below
            }
            
            result = graph.query(query, params)
            
            # Filter by min_similarity in Python (avoids FalkorDB edge-property WHERE hang)
            neighbors = []
            for record in result.result_set:
                sim = record[3]
                if sim is not None and float(sim) >= min_similarity:
                    neighbors.append({
                        "chunk_id": record[0],
                        "document_id": record[1],
                        "page_number": record[2],
                        "similarity": float(sim)
                    })
            
            # Re-sort and cap at max_neighbors
            neighbors.sort(key=lambda x: x["similarity"], reverse=True)
            neighbors = neighbors[:max_neighbors]
            
            logger.debug(f"Retrieved {len(neighbors)} neighbors for {chunk_id}")
            return neighbors
            
        except Exception as e:
            # Log the error but return empty list instead of raising
            # This handles cases where graph is empty or has connection issues
            logger.warning(f"Graph search failed for {chunk_id} (returning empty list): {e}")
            degradation_manager.mark_component_unhealthy("falkor_db")
            return []
    
    # Execute with graceful degradation
    result = degradation_manager.execute_with_fallback(
        "falkor_db",
        _execute_query,
        f"get_neighbors({chunk_id})"
    )
    
    return result if result is not None else []


def get_similar_neighbors(
    chunk_id: str,
    max_neighbors: int,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Get neighboring chunks connected by SIMILAR_TO edges (bidirectional query).
    Alias for get_neighbors() for backward compatibility.
    
    Args:
        chunk_id: Chunk identifier to find neighbors for
        max_neighbors: Maximum number of neighbors to return
        min_similarity: Minimum similarity score threshold
    
    Returns:
        List of neighbor dictionaries sorted by similarity descending
    """
    return get_neighbors(chunk_id, max_neighbors, min_similarity)


def add_similar_edge(
    source_chunk_id: str,
    target_chunk_id: str,
    similarity: float
) -> None:
    """
    Add SIMILAR_TO edge with similarity score to the unified Knowledge Graph.
    This is an alias for add_chunk_relation() for backward compatibility.
    
    Args:
        source_chunk_id: Source chunk identifier
        target_chunk_id: Target chunk identifier
        similarity: Similarity score between chunks (0.0 to 1.0)
    
    Raises:
        DatabaseError: If edge creation fails
    """
    add_chunk_relation(source_chunk_id, target_chunk_id, similarity)


def add_probe_edge(
    source_chunk_id: str,
    target_chunk_id: str,
    probe_type: str,
    confidence: float
) -> None:
    """
    Add a probe edge (Graph B) to the unified Knowledge Graph.
    
    Creates edges with probe relationship types: SUPPORTS, CONTRADICTS, 
    EXAMPLE_OF, ELABORATES, DEPENDS_ON.
    
    Args:
        source_chunk_id: Source chunk identifier
        target_chunk_id: Target chunk identifier
        probe_type: Probe relationship type (SUPPORTS, CONTRADICTS, etc.)
        confidence: Confidence score (0.0 to 1.0)
    
    Raises:
        DatabaseError: If edge creation fails
    """
    try:
        graph = _get_knowledge_graph()
        
        # Validate probe type
        valid_probe_types = ["CONTRADICTS", "ELABORATES", "DEPENDS_ON"]
        if probe_type not in valid_probe_types:
            raise ValueError(f"Invalid probe type: {probe_type}. Must be one of {valid_probe_types}")
        
        query = f"""
        MATCH (source:{settings.falkor_vertex_collection} {{chunk_id: $source_id}})
        MATCH (target:{settings.falkor_vertex_collection} {{chunk_id: $target_id}})
        MERGE (source)-[r:{probe_type}]->(target)
        ON CREATE SET r.confidence = $confidence, r.source = 'graph_b'
        ON MATCH SET r.confidence = $confidence, r.source = 'graph_b'
        """
        
        params = {
            "source_id": source_chunk_id,
            "target_id": target_chunk_id,
            "confidence": confidence
        }
        
        graph.query(query, params)
        logger.debug(f"Added {probe_type} edge: {source_chunk_id} -> {target_chunk_id} (confidence: {confidence})")
        
    except Exception as e:
        raise DatabaseError(
            f"Failed to add probe edge {source_chunk_id} -> {target_chunk_id}: {e}"
        ) from e


def get_probe_neighbors(
    chunk_id: str,
    probe_types: List[str],
    max_neighbors: int,
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Get neighboring chunks connected by probe edges (bidirectional query).
    
    Args:
        chunk_id: Chunk identifier to find neighbors for
        probe_types: List of probe types to query (SUPPORTS, CONTRADICTS, etc.)
        max_neighbors: Maximum number of neighbors to return
        min_confidence: Minimum confidence score threshold
    
    Returns:
        List of neighbor dictionaries with probe type and confidence
    """
    degradation_manager = get_degradation_manager()
    
    def _execute_query():
        try:
            graph = _get_knowledge_graph()
            
            # Check if source node exists
            check_query = f"""
            MATCH (source:{settings.falkor_vertex_collection} {{chunk_id: $chunk_id}})
            RETURN count(source) AS node_count
            """
            
            check_result = graph.query(check_query, {"chunk_id": chunk_id})
            
            if not check_result.result_set or check_result.result_set[0][0] == 0:
                logger.info(f"Source node not found in graph: {chunk_id}")
                return []
            
            # Bidirectional query for probe edges
            query = f"""
            MATCH (source:{settings.falkor_vertex_collection} {{chunk_id: $chunk_id}})
            MATCH (source)-[r]-(neighbor:{settings.falkor_vertex_collection})
            WHERE type(r) IN $probe_types AND r.confidence >= $min_confidence
            RETURN neighbor.chunk_id AS chunk_id,
                   neighbor.document_id AS document_id,
                   neighbor.page_number AS page_number,
                   r.confidence AS confidence,
                   type(r) AS probe_type
            ORDER BY r.confidence DESC
            LIMIT $max_neighbors
            """
            
            params = {
                "chunk_id": chunk_id,
                "probe_types": probe_types,
                "min_confidence": min_confidence,
                "max_neighbors": max_neighbors
            }
            
            result = graph.query(query, params)
            
            neighbors = [
                {
                    "chunk_id": record[0],
                    "document_id": record[1],
                    "page_number": record[2],
                    "confidence": record[3],
                    "probe_type": record[4]
                }
                for record in result.result_set
            ]
            
            logger.debug(f"Retrieved {len(neighbors)} probe neighbors for {chunk_id}")
            return neighbors
            
        except Exception as e:
            logger.warning(f"Probe graph search failed for {chunk_id}: {e}")
            degradation_manager.mark_component_unhealthy("falkor_db")
            return []
    
    result = degradation_manager.execute_with_fallback(
        "falkor_db",
        _execute_query,
        f"get_probe_neighbors({chunk_id})"
    )
    
    return result if result is not None else []


def clear_all_data() -> None:
    """
    Clear all data from unified Knowledge Graph (for testing/reset purposes).
    
    Raises:
        DatabaseError: If deletion fails
    """
    try:
        # Clear unified knowledge graph
        graph = _get_knowledge_graph()
        graph.query("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all data from unified Knowledge Graph")
        
    except Exception as e:
        raise DatabaseError(f"Failed to clear graph data: {e}") from e

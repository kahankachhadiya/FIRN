"""
Graph A Builder Module

This module implements the GraphABuilder class that creates semantic relationship edges
using vector search on the main collection with reranker scoring.

Graph A Architecture:
- Uses vector search on Main_Collection (Qdrant) to find semantically similar chunks
- Passes candidates through Reranker Service (Qwen-Reranker-0.6B) for precise scoring
- Creates SIMILAR_TO edges in FalkorDB with reranker scores as weights
- Provides broad semantic coverage across documents without relying on summaries

This replaces the old summary-based graph building approach with a more accurate
and efficient reranker-based method.
"""

from typing import Dict, Any, Optional, List
import asyncio
import aiohttp
import time

from db.qdrant_client import search_paragraphs
from db.falkor_client import ensure_graph_exists, add_chunk_vertex, add_similar_edge
from db.elasticsearch_client import get_paragraph
from config.settings import settings
from utils.logging_utils import get_logger
from utils.errors import DatabaseError

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reranker client — inlined from the deleted clients/reranker_client.py
# ---------------------------------------------------------------------------

class RerankerServiceError(DatabaseError):
    """Base exception for reranker service operations."""
    pass


class RerankerConnectionError(RerankerServiceError):
    """Raised when connection to the reranker service fails."""
    pass


class RerankerTimeoutError(RerankerServiceError):
    """Raised when a reranker service request times out."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class RerankerValidationError(RerankerServiceError):
    """Raised when input validation fails for reranker requests."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.field = field
        self.reason = reason


class RerankerResponseError(RerankerServiceError):
    """Raised when the reranker service returns an invalid or error response."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class RerankerServiceUnavailableError(RerankerServiceError):
    """Raised when the reranker service is unavailable after all retry attempts."""

    def __init__(
        self,
        message: str,
        retry_attempts: Optional[int] = None,
        last_error: Optional[str] = None,
    ):
        super().__init__(message)
        self.retry_attempts = retry_attempts
        self.last_error = last_error


class RerankerClient:
    """
    Client for communicating with the Qwen-Reranker-0.6B service.

    Used by GraphABuilder to score semantic relationships between chunks and
    create high-quality SIMILAR_TO edges in the knowledge graph.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:7997",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._service_available = True
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 30.0
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(
            f"Initialized RerankerClient: endpoint={self.endpoint}, "
            f"timeout={timeout}s, max_retries={max_retries}"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            try:
                current_loop = asyncio.get_running_loop()
                if self._session._loop != current_loop:  # type: ignore[attr-defined]
                    await self._session.close()
                    self._session = None
            except RuntimeError:
                await self._session.close()
                self._session = None

        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100, limit_per_host=30, ttl_dns_cache=300
            )
            timeout_config = aiohttp.ClientTimeout(
                total=self.timeout, connect=10.0, sock_read=self.timeout
            )
            self._session = aiohttp.ClientSession(
                connector=connector, timeout=timeout_config
            )
        return self._session

    def _validate_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        if not isinstance(candidates, list):
            raise RerankerValidationError(
                "candidates must be a list",
                field="candidates",
                reason=f"Expected list, got {type(candidates).__name__}",
            )
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise RerankerValidationError(
                    f"candidates[{i}] must be a dictionary",
                    field=f"candidates[{i}]",
                    reason=f"Expected dict, got {type(candidate).__name__}",
                )
            for required in ("chunk_id", "text"):
                if required not in candidate:
                    raise RerankerValidationError(
                        f"candidates[{i}] missing required field: {required}",
                        field=f"candidates[{i}].{required}",
                        reason=f"Required field '{required}' is missing",
                    )
            if not isinstance(candidate["chunk_id"], str) or not candidate["chunk_id"].strip():
                raise RerankerValidationError(
                    f"candidates[{i}]['chunk_id'] must be a non-empty string",
                    field=f"candidates[{i}].chunk_id",
                )
            if not isinstance(candidate["text"], str) or not candidate["text"].strip():
                raise RerankerValidationError(
                    f"candidates[{i}]['text'] must be a non-empty string",
                    field=f"candidates[{i}].text",
                )

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using the reranker service."""
        if not query or not query.strip():
            raise RerankerValidationError(
                "Query text cannot be empty", field="query",
                reason="Query must be a non-empty string",
            )
        if not candidates:
            return []
        self._validate_candidates(candidates)
        if top_k is not None and top_k < 1:
            raise RerankerValidationError(
                f"top_k must be >= 1, got {top_k}", field="top_k",
                reason="top_k must be a positive integer",
            )

        documents = [c["text"] for c in candidates]
        request_data: Dict[str, Any] = {"query": query, "documents": documents}
        if top_k is not None:
            request_data["top_k"] = top_k

        session = await self._get_session()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(
                    f"{self.endpoint}/v1/rerank",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                        except Exception as e:
                            raise RerankerResponseError(
                                f"Failed to parse JSON response: {e}",
                                status_code=200,
                                response_body=await response.text(),
                            )
                        if not isinstance(result, dict) or "results" not in result:
                            raise RerankerResponseError(
                                "Response missing 'results' field",
                                status_code=200,
                                response_body=str(result),
                            )
                        results = result["results"]
                        if not isinstance(results, list):
                            raise RerankerResponseError(
                                "Response 'results' field is not a list",
                                status_code=200,
                                response_body=str(result),
                            )
                        self._service_available = True
                        mapped = []
                        for r in results:
                            idx = r.get("index", -1)
                            if 0 <= idx < len(candidates):
                                merged = dict(candidates[idx])
                                merged["score"] = r.get("score", 0.0)
                                mapped.append(merged)
                        return mapped

                    elif 400 <= response.status < 500:
                        error_text = await response.text()
                        raise RerankerResponseError(
                            f"Reranker service returned client error {response.status}: {error_text}",
                            status_code=response.status,
                            response_body=error_text,
                        )
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"
                        if attempt == self.max_retries:
                            raise RerankerResponseError(
                                f"Reranker service returned error after {self.max_retries + 1} attempts",
                                status_code=response.status,
                                response_body=error_text,
                            )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                if attempt == self.max_retries:
                    self._service_available = False
                    raise RerankerTimeoutError(
                        f"Reranker service timeout after {self.max_retries + 1} attempts",
                        timeout_seconds=self.timeout,
                    )
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientConnectorError as e:
                last_error = f"Connection error: {e}"
                if attempt == self.max_retries:
                    self._service_available = False
                    raise RerankerConnectionError(
                        f"Failed to connect to reranker service at {self.endpoint}: {e}"
                    )
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                last_error = f"Client error: {e}"
                if attempt == self.max_retries:
                    self._service_available = False
                    raise RerankerConnectionError(f"Reranker service request failed: {e}")
                await asyncio.sleep(2 ** attempt)

            except (RerankerValidationError, RerankerResponseError):
                raise

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                if attempt == self.max_retries:
                    self._service_available = False
                    raise RerankerServiceError(f"Unexpected reranker service error: {e}")
                await asyncio.sleep(2 ** attempt)

        self._service_available = False
        raise RerankerServiceUnavailableError(
            f"Reranker service unavailable after {self.max_retries + 1} attempts",
            retry_attempts=self.max_retries + 1,
            last_error=last_error,
        )

    async def close(self):
        """Close the aiohttp session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()


# Global client instance
_global_reranker_client: Optional[RerankerClient] = None


def get_reranker_client(
    endpoint: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> RerankerClient:
    """Get the global reranker service client instance."""
    global _global_reranker_client
    if _global_reranker_client is None:
        endpoint = endpoint or getattr(settings, "reranker_service_url", "http://localhost:7997")
        max_retries = max_retries or getattr(settings, "service_retry_attempts", 3)
        _global_reranker_client = RerankerClient(
            endpoint=endpoint, timeout=30.0, max_retries=max_retries
        )
    return _global_reranker_client


# ---------------------------------------------------------------------------


class GraphABuilder:
    """
    Builds Graph A relationships using vector search + reranker.
    
    Graph A creates semantic relationship edges between chunks by:
    1. Performing vector search on Main Collection to find semantically similar candidates
    2. Passing candidates through Reranker Service (Qwen-Reranker-0.6B) for precise scoring
    3. Filtering by reranker threshold to ensure quality relationships
    4. Creating top_n unidirectional SIMILAR_TO edges in unified Knowledge Graph (queried bidirectionally)
    
    This approach provides more accurate semantic relationships than the old
    summary-based method, while maintaining broad coverage across documents.
    
    Key Features:
    - No dependency on LLM-generated summaries
    - Direct vector search on paragraph embeddings
    - Reranker-based scoring for precision
    - Configurable threshold and top_n limits
    - Graceful degradation on service failures
    """
    
    def __init__(
        self,
        reranker_client: Optional[RerankerClient] = None,
        top_n: int = 10,
        reranker_threshold: float = 0.6,
        max_candidates: Optional[int] = None
    ):
        """
        Initialize Graph A builder with configuration.
        
        Args:
            reranker_client: RerankerClient instance (creates new if None)
            top_n: Number of top edges to create per chunk
            reranker_threshold: Minimum reranker score for edge creation
            max_candidates: Maximum candidates to retrieve (uses settings.graph_a_reranker_candidates if None)
        """
        self.reranker_client = reranker_client
        if self.reranker_client is None:
            self.reranker_client = get_reranker_client()
        
        self.top_n = top_n
        self.reranker_threshold = reranker_threshold
        
        # Use configured max_candidates from settings if not provided
        if max_candidates is None:
            max_candidates = getattr(settings, 'graph_a_reranker_candidates', 35)
        self.max_candidates = max_candidates
        
        # Ensure graph exists
        ensure_graph_exists()
        
        # Statistics tracking
        self.stats = {
            "chunks_processed": 0,
            "total_candidates": 0,
            "candidates_after_filtering": 0,
            "edges_created": 0,
            "reranker_calls": 0,
            "reranker_failures": 0
        }
        
        logger.info(
            f"Initialized GraphABuilder: top_n={top_n}, "
            f"threshold={reranker_threshold}, "
            f"max_candidates={max_candidates}"
        )
    
    async def build_relations_for_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Build Graph A relations for a chunk.
        
        Process:
        1. Perform vector search on Main Collection
        2. Filter out self-matches and same-document matches
        3. Pass candidates to Reranker Service
        4. Filter by reranker_threshold
        5. Create top_n unidirectional SIMILAR_TO edges in unified Knowledge Graph
        
        Args:
            chunk_id: Unique chunk identifier
            chunk_text: Full text of the chunk for reranking
            document_id: Document ID (for filtering same-document matches)
            metadata: Optional metadata (page_number, etc.)
            
        Returns:
            Number of edges created
            
        Note:
            This method implements graceful degradation:
            - If reranker service fails, falls back to vector search scores
            - If individual edge creation fails, continues with remaining edges
            - Returns partial results rather than failing completely
        """
        if not chunk_text or not chunk_text.strip():
            logger.warning(f"Empty chunk text for {chunk_id}, skipping Graph A building")
            return 0
        
        start_time = time.time()
        edges_created = 0
        
        try:
            # Step 1: Vector search on Main Collection
            search_top_k = self.max_candidates
            
            logger.debug(
                f"Graph A: Vector search for chunk {chunk_id} "
                f"(top_k={search_top_k})"
            )
            
            try:
                search_results = await search_paragraphs(
                    query=chunk_text,
                    top_k=search_top_k,
                    score_threshold=0.0  # No threshold at search stage
                )
            except DatabaseError as e:
                logger.error(
                    f"Vector search failed for chunk {chunk_id}: {e}. "
                    f"Cannot build Graph A relations without search results."
                )
                self.stats["chunks_processed"] += 1
                return 0
            except Exception as e:
                logger.error(
                    f"Unexpected error in vector search for chunk {chunk_id}: {e}. "
                    f"Cannot build Graph A relations."
                )
                self.stats["chunks_processed"] += 1
                return 0
            
            self.stats["total_candidates"] += len(search_results)
            
            if not search_results:
                logger.debug(f"No vector search results for chunk {chunk_id}")
                self.stats["chunks_processed"] += 1
                return 0
            
            # Step 2: Filter out self-matches and same-document matches
            candidates = []
            for result in search_results:
                candidate_chunk_id = result.get("chunk_id")
                candidate_metadata = result.get("metadata", {})
                candidate_doc_id = candidate_metadata.get("document_id")
                
                # Skip self-matches
                if candidate_chunk_id == chunk_id:
                    logger.debug(f"Filtered self-match: {chunk_id}")
                    continue
                
                # Skip same-document matches
                if candidate_doc_id == document_id:
                    logger.debug(
                        f"Filtered same-document match: {chunk_id} -> {candidate_chunk_id}"
                    )
                    continue
                
                # Get candidate text from Redis
                try:
                    candidate_text = get_paragraph(candidate_chunk_id)
                    if not candidate_text:
                        logger.warning(
                            f"Could not retrieve text for candidate {candidate_chunk_id}, skipping"
                        )
                        continue
                except Exception as e:
                    logger.warning(
                        f"Error retrieving text for candidate {candidate_chunk_id}: {e}, skipping"
                    )
                    continue
                
                candidates.append({
                    "chunk_id": candidate_chunk_id,
                    "text": candidate_text,
                    "metadata": candidate_metadata
                })
            
            if not candidates:
                logger.debug(
                    f"No candidates after filtering for chunk {chunk_id} "
                    f"(filtered {len(search_results)} results)"
                )
                self.stats["chunks_processed"] += 1
                return 0
            
            logger.debug(
                f"Graph A: {len(candidates)} candidates after filtering "
                f"(from {len(search_results)} search results)"
            )
            
            # Step 3: Rerank candidates using Reranker Service
            try:
                self.stats["reranker_calls"] += 1
                
                reranked_results = await self.reranker_client.rerank(
                    query=chunk_text,
                    candidates=candidates,
                    top_k=None  # Get all results, we'll filter by threshold
                )
                
                logger.debug(
                    f"Graph A: Reranked {len(reranked_results)} candidates "
                    f"for chunk {chunk_id}"
                )
                
            except RerankerServiceError as e:
                logger.error(
                    f"Reranker service error for chunk {chunk_id}: {e}. "
                    f"Falling back to vector search scores."
                )
                self.stats["reranker_failures"] += 1
                
                # Fallback: use vector search scores
                reranked_results = [
                    {
                        "chunk_id": c["chunk_id"],
                        "score": search_results[i].get("score", 0.0),
                        "text": c["text"],
                        "metadata": c["metadata"]
                    }
                    for i, c in enumerate(candidates) if i < len(search_results)
                ]
            except Exception as e:
                logger.error(
                    f"Unexpected error in reranker for chunk {chunk_id}: {e}. "
                    f"Falling back to vector search scores."
                )
                self.stats["reranker_failures"] += 1
                
                # Fallback: use vector search scores
                reranked_results = [
                    {
                        "chunk_id": c["chunk_id"],
                        "score": search_results[i].get("score", 0.0),
                        "text": c["text"],
                        "metadata": c["metadata"]
                    }
                    for i, c in enumerate(candidates) if i < len(search_results)
                ]
            
            # Step 4: Filter by reranker threshold
            filtered_results = [
                r for r in reranked_results
                if r.get("score", 0.0) >= self.reranker_threshold
            ]
            
            self.stats["candidates_after_filtering"] += len(filtered_results)
            
            if not filtered_results:
                logger.debug(
                    f"No candidates above threshold {self.reranker_threshold} "
                    f"for chunk {chunk_id}"
                )
                self.stats["chunks_processed"] += 1
                return 0
            
            # Step 5: Create top_n edges in FalkorDB
            top_results = filtered_results[:self.top_n]
            
            # Ensure source vertex exists
            page_number = metadata.get("page_number", 0) if metadata else 0
            try:
                add_chunk_vertex(chunk_id, document_id, page_number)
            except DatabaseError as e:
                logger.error(
                    f"Failed to create source vertex for {chunk_id}: {e}. "
                    f"Cannot create Graph A edges."
                )
                self.stats["chunks_processed"] += 1
                return 0
            except Exception as e:
                logger.error(
                    f"Unexpected error creating source vertex for {chunk_id}: {e}. "
                    f"Cannot create Graph A edges."
                )
                self.stats["chunks_processed"] += 1
                return 0
            
            # Track edge creation failures
            edge_creation_failures = 0
            
            for result in top_results:
                target_chunk_id = result["chunk_id"]
                reranker_score = result["score"]
                target_metadata = result.get("metadata", {})
                target_doc_id = target_metadata.get("document_id", "unknown")
                target_page = target_metadata.get("page_number", 0)
                
                try:
                    # Ensure target vertex exists
                    add_chunk_vertex(target_chunk_id, target_doc_id, target_page)
                    
                    # Create unidirectional SIMILAR_TO edge (queried bidirectionally)
                    try:
                        # Only create forward edge (A→B)
                        # Bidirectional retrieval is handled by query pattern: MATCH (a)-[r:SIMILAR_TO]-(b)
                        add_similar_edge(chunk_id, target_chunk_id, reranker_score)
                        
                        edges_created += 1
                        
                        logger.debug(
                            f"Created SIMILAR_TO edge: {chunk_id} -> {target_chunk_id} "
                            f"(score: {reranker_score:.3f})"
                        )
                        
                    except DatabaseError as e:
                        edge_creation_failures += 1
                        logger.warning(
                            f"Failed to create edge {chunk_id} -> {target_chunk_id}: {e}. "
                            f"Continuing with remaining edges."
                        )
                    except Exception as e:
                        edge_creation_failures += 1
                        logger.warning(
                            f"Unexpected error creating edge {chunk_id} -> {target_chunk_id}: {e}. "
                            f"Continuing with remaining edges."
                        )
                        
                except DatabaseError as e:
                    edge_creation_failures += 1
                    logger.warning(
                        f"Failed to create target vertex for {target_chunk_id}: {e}. "
                        f"Skipping edge to this target."
                    )
                except Exception as e:
                    edge_creation_failures += 1
                    logger.warning(
                        f"Unexpected error with target vertex {target_chunk_id}: {e}. "
                        f"Skipping edge to this target."
                    )
            
            self.stats["edges_created"] += edges_created
            self.stats["chunks_processed"] += 1
            
            processing_time = time.time() - start_time
            
            # Log with appropriate level based on success
            if edges_created > 0:
                logger.info(
                    f"Graph A: Built {edges_created} edges for chunk {chunk_id} "
                    f"(candidates: {len(search_results)} -> {len(candidates)} -> "
                    f"{len(filtered_results)}, failures: {edge_creation_failures}, "
                    f"time: {processing_time:.2f}s)"
                )
            else:
                logger.warning(
                    f"Graph A: No edges created for chunk {chunk_id} "
                    f"(candidates: {len(search_results)} -> {len(candidates)} -> "
                    f"{len(filtered_results)}, failures: {edge_creation_failures}, "
                    f"time: {processing_time:.2f}s)"
                )
            
            return edges_created
            
        except Exception as e:
            logger.error(
                f"Unexpected error building Graph A relations for chunk {chunk_id}: {e}",
                exc_info=True
            )
            # Don't raise - return 0 to allow processing to continue
            self.stats["chunks_processed"] += 1
            return 0
    
    def build_relations_for_chunk_sync(
        self,
        chunk_id: str,
        chunk_text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Synchronous wrapper for build_relations_for_chunk.
        
        Args:
            chunk_id: Unique chunk identifier
            chunk_text: Full text of the chunk
            document_id: Document ID
            metadata: Optional metadata
            
        Returns:
            Number of edges created
        """
        # Always create a new event loop for synchronous calls to avoid "Event loop is closed" errors
        # This is necessary because ThreadPoolExecutor may reuse threads with closed loops
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(
                    self.build_relations_for_chunk(
                        chunk_id, chunk_text, document_id, metadata
                    )
                )
            )
            return future.result()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Graph A building.
        
        Returns:
            Dictionary with statistics:
            - chunks_processed: Number of chunks processed
            - total_candidates: Total candidates from vector search
            - candidates_after_filtering: Candidates after threshold filtering
            - edges_created: Total edges created
            - reranker_calls: Number of reranker service calls
            - reranker_failures: Number of reranker failures
            - avg_candidates_per_chunk: Average candidates per chunk
            - avg_edges_per_chunk: Average edges created per chunk
            - reranker_success_rate: Reranker success rate
        """
        stats = self.stats.copy()
        
        # Calculate averages
        if stats["chunks_processed"] > 0:
            stats["avg_candidates_per_chunk"] = float(  # type: ignore[assignment]
                stats["total_candidates"] / stats["chunks_processed"]
            )
            stats["avg_edges_per_chunk"] = float(  # type: ignore[assignment]
                stats["edges_created"] / stats["chunks_processed"]
            )
        else:
            stats["avg_candidates_per_chunk"] = 0.0  # type: ignore[assignment]
            stats["avg_edges_per_chunk"] = 0.0  # type: ignore[assignment]
        
        # Calculate reranker success rate
        if stats["reranker_calls"] > 0:
            stats["reranker_success_rate"] = float(  # type: ignore[assignment]
                (stats["reranker_calls"] - stats["reranker_failures"]) /
                stats["reranker_calls"]
            )
        else:
            stats["reranker_success_rate"] = 0.0  # type: ignore[assignment]
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "chunks_processed": 0,
            "total_candidates": 0,
            "candidates_after_filtering": 0,
            "edges_created": 0,
            "reranker_calls": 0,
            "reranker_failures": 0
        }
        logger.debug("Reset Graph A builder statistics")


# Convenience function for backward compatibility
async def build_graph_a_relations(
    chunk_id: str,
    chunk_text: str,
    document_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    top_n: Optional[int] = None,
    reranker_threshold: Optional[float] = None
) -> int:
    """
    Convenience function to build Graph A relations for a chunk.
    
    Args:
        chunk_id: Unique chunk identifier
        chunk_text: Full text of the chunk
        document_id: Document ID
        metadata: Optional metadata
        top_n: Optional override for top_n (uses settings default if None)
        reranker_threshold: Optional override for threshold (uses settings default if None)
        
    Returns:
        Number of edges created
    """
    # Use settings defaults if not provided
    if top_n is None:
        top_n = getattr(settings, 'graph_a_top_n', 10)
    if reranker_threshold is None:
        reranker_threshold = getattr(settings, 'graph_a_reranker_threshold', 0.6)
    
    builder = GraphABuilder(
        top_n=top_n,
        reranker_threshold=reranker_threshold
    )
    
    return await builder.build_relations_for_chunk(
        chunk_id, chunk_text, document_id, metadata
    )

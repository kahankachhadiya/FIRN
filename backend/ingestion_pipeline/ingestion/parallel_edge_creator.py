"""
Parallel Edge Creator for orchestrating Similar and Probe graph edge creation.

This module implements the ParallelEdgeCreator class that coordinates parallel
execution of Similar and Probe graph edge creation during document ingestion.

Graph A (Similar edges) uses vector search + reranker via graph_a_builder.py.
Graph B (Probe edges) uses the probe graph manager.
"""

import asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import time

from ingestion.probe_graph_manager import ProbeGraphManager
from ingestion.graph_a_builder import GraphABuilder
from config.settings import settings
from utils.logging_utils import get_logger
from utils.errors import DatabaseError

logger = get_logger(__name__)


class ParallelEdgeCreator:
    """
    Coordinates parallel edge creation for both Similar and Probe graphs.
    
    This class ensures that database operations (Redis, Qdrant, FalkorDB vertex creation)
    complete before edge creation begins, then executes Similar and Probe graph edge
    creation in parallel for optimal performance.
    
    Graph A (Similar edges): Uses vector search + reranker via GraphABuilder
    Graph B (Probe edges): Uses ProbeGraphManager
    """
    
    def __init__(self):
        """Initialize the ParallelEdgeCreator with GraphABuilder and ProbeGraphManager."""
        self.probe_manager = ProbeGraphManager()
        
        # Initialize Graph A Builder with settings from configuration
        graph_a_top_n = getattr(settings, 'graph_a_top_n', 10)
        graph_a_threshold = getattr(settings, 'graph_a_reranker_threshold', 0.6)
        graph_a_max_candidates = getattr(settings, 'graph_a_reranker_candidates', 35)
        
        self.graph_a_builder = GraphABuilder(
            top_n=graph_a_top_n,
            reranker_threshold=graph_a_threshold,
            max_candidates=graph_a_max_candidates
        )
        
        # Track failure statistics for monitoring and alerting
        self.failure_stats = {
            "graph_a_failures": 0,
            "graph_b_failures": 0,
            "total_chunks_processed": 0,
            "graph_a_success_count": 0,
            "graph_b_success_count": 0
        }
        
        logger.info(
            f"Initialized ParallelEdgeCreator with Graph A Builder "
            f"(top_n={graph_a_top_n}, threshold={graph_a_threshold}, "
            f"max_candidates={graph_a_max_candidates})"
        )
    
    def create_edges_parallel(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create edges in both Similar and Probe graphs in parallel.
        
        This method executes Similar graph edge creation (Graph A) and Probe graph 
        edge creation (Graph B) concurrently using ThreadPoolExecutor for optimal performance.
        
        Implements comprehensive error handling:
        - Tracks failure statistics for both graphs
        - Continues processing even if one graph fails
        - Logs detailed error information
        - Alerts when failure rate exceeds threshold
        
        Graph A uses vector search + reranker via GraphABuilder.
        Graph B uses probe graph manager.
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk (used for probe graph)
            document_id: Document ID to exclude same-document matches
            metadata: Additional metadata for the chunk
            chunk_text: Full chunk text for Graph A reranker (required for similar edges)
            
        Returns:
            Dict with edge counts for similar and each probe type:
            {
                'similar_edges': int,
                'probe_edges': {
                    'SUPPORTS': int,
                    'CONTRADICTS': int,
                    'EXAMPLE_OF': int,
                    'ELABORATES': int,
                    'DEPENDS_ON': int
                },
                'total_edges': int,
                'execution_time': float,
                'graph_a_failed': bool,
                'graph_b_failed': bool
            }
        """
        start_time = time.time()
        
        logger.debug(f"Starting parallel edge creation for chunk {chunk_id}")
        
        # Track chunk processing
        self.failure_stats["total_chunks_processed"] += 1
        
        # Use ThreadPoolExecutor to run both edge creation processes in parallel
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="EdgeCreator") as executor:
            # Submit both tasks to the executor
            similar_future = executor.submit(
                self._create_similar_edges,
                chunk_id, summary, document_id, metadata, chunk_text
            )
            
            probe_future = executor.submit(
                self._create_probe_edges,
                chunk_id, summary, document_id, chunk_text
            )
            
            # Wait for both tasks to complete and collect results
            similar_edges = 0
            probe_edges: Dict[str, int] = {}
            graph_a_failed = False
            graph_b_failed = False
            
            # Process completed futures as they finish
            future: Future[int]  # type: ignore[type-arg]
            for future in as_completed([similar_future, probe_future]):  # type: ignore[arg-type]
                try:
                    if future == similar_future:
                        similar_edges = future.result()
                        self.failure_stats["graph_a_success_count"] += 1
                        if similar_edges > 0:
                            logger.debug(f"Graph A edge creation completed: {similar_edges} edges")
                        else:
                            logger.debug(f"Graph A: no similar edges for chunk {chunk_id} (normal for some chunks)")
                    elif future == probe_future:
                        probe_edges = future.result()
                        total_probe_edges = sum(probe_edges.values())
                        # 0 edges is a legitimate outcome (e.g. reference-only chunks),
                        # only count as failure if the call itself raised an exception.
                        self.failure_stats["graph_b_success_count"] += 1
                        if total_probe_edges == 0:
                            logger.debug(f"Graph B: no probe edges for chunk {chunk_id} (normal for some chunks)")
                        else:
                            logger.debug(f"Graph B edge creation completed: {total_probe_edges} edges")
                except Exception as e:
                    logger.error(f"Edge creation task failed with exception: {e}", exc_info=True)
                    # Continue with other tasks even if one fails
                    if future == similar_future:
                        similar_edges = 0
                        graph_a_failed = True
                        self.failure_stats["graph_a_failures"] += 1
                    elif future == probe_future:
                        probe_edges = {
                            'dependency': 0,
                            'elaboration': 0,
                            'contradiction': 0,
                        }
                        graph_b_failed = True
                        self.failure_stats["graph_b_failures"] += 1
        
        execution_time = time.time() - start_time
        total_edges = similar_edges + sum(probe_edges.values())
        
        # Check failure rate and alert if threshold exceeded
        self._check_failure_threshold()
        
        result = {
            'similar_edges': similar_edges,
            'probe_edges': probe_edges,
            'total_edges': total_edges,
            'execution_time': execution_time,
            'graph_a_failed': graph_a_failed,
            'graph_b_failed': graph_b_failed
        }
        
        logger.info(
            f"Parallel edge creation completed for chunk {chunk_id}: "
            f"Graph A={similar_edges}{' (FAILED)' if graph_a_failed else ''}, "
            f"Graph B={sum(probe_edges.values())}{' (FAILED)' if graph_b_failed else ''}, "
            f"total={total_edges}, time={execution_time:.3f}s"
        )
        
        return result
    
    async def create_edges_parallel_async(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async version of create_edges_parallel for use in async contexts.
        
        Implements comprehensive error handling:
        - Tracks failure statistics for both graphs
        - Continues processing even if one graph fails
        - Logs detailed error information
        - Alerts when failure rate exceeds threshold
        
        Graph A uses vector search + reranker via GraphABuilder.
        Graph B uses probe graph manager.
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk (used for probe graph)
            document_id: Document ID to exclude same-document matches
            metadata: Additional metadata for the chunk
            chunk_text: Full chunk text for Graph A reranker (required for similar edges)
            
        Returns:
            Dict with edge counts for similar and each probe type
        """
        start_time = time.time()
        
        logger.debug(f"Starting async parallel edge creation for chunk {chunk_id}")
        
        # Track chunk processing
        self.failure_stats["total_chunks_processed"] += 1
        
        # Run both edge creation processes concurrently
        similar_task = asyncio.create_task(
            self._create_similar_edges_async(chunk_id, summary, document_id, metadata, chunk_text)
        )
        
        probe_task = asyncio.create_task(
            self._create_probe_edges_async(chunk_id, summary, document_id, chunk_text)
        )
        
        # Wait for both tasks to complete
        similar_edges = 0
        probe_edges = {}
        graph_a_failed = False
        graph_b_failed = False
        
        try:
            results = await asyncio.gather(
                similar_task, probe_task, return_exceptions=True
            )
            
            # Handle Graph A result
            similar_edges_result: int
            if isinstance(results[0], Exception):
                logger.error(f"Graph A edge creation failed: {results[0]}", exc_info=results[0])
                similar_edges_result = 0
                graph_a_failed = True
                self.failure_stats["graph_a_failures"] += 1
            else:
                similar_edges_result = results[0]  # type: ignore[assignment]
                self.failure_stats["graph_a_success_count"] += 1
                if similar_edges_result > 0:
                    logger.debug(f"Graph A edge creation completed: {similar_edges_result} edges")
                else:
                    logger.debug(f"Graph A: no similar edges for chunk {chunk_id} (normal for some chunks)")
            
            # Handle Graph B result
            probe_edges_result: Dict[str, int]
            if isinstance(results[1], Exception):
                logger.error(f"Graph B edge creation failed: {results[1]}", exc_info=results[1])
                probe_edges_result = {
                    'dependency': 0,
                    'elaboration': 0,
                    'contradiction': 0,
                }
                graph_b_failed = True
                self.failure_stats["graph_b_failures"] += 1
            else:
                probe_edges_result = results[1]  # type: ignore[assignment]
                total_probe_edges = sum(probe_edges_result.values())
                # 0 edges is a legitimate outcome (e.g. reference-only chunks),
                # only count as failure if the call itself raised an exception above.
                self.failure_stats["graph_b_success_count"] += 1
                if total_probe_edges == 0:
                    logger.debug(f"Graph B: no probe edges for chunk {chunk_id} (normal for some chunks)")
            
            similar_edges = similar_edges_result
            probe_edges = probe_edges_result
            
        except Exception as e:
            logger.error(f"Async edge creation failed: {e}", exc_info=True)
            similar_edges = 0
            probe_edges = {
                'dependency': 0,
                'elaboration': 0,
                'contradiction': 0,
            }
            graph_a_failed = True
            graph_b_failed = True
            self.failure_stats["graph_a_failures"] += 1
            self.failure_stats["graph_b_failures"] += 1
        
        execution_time = time.time() - start_time
        total_edges = similar_edges + sum(probe_edges.values())
        
        # Check failure rate and alert if threshold exceeded
        self._check_failure_threshold()
        
        result = {
            'similar_edges': similar_edges,
            'probe_edges': probe_edges,
            'total_edges': total_edges,
            'execution_time': execution_time,
            'graph_a_failed': graph_a_failed,
            'graph_b_failed': graph_b_failed
        }
        
        logger.info(
            f"Async parallel edge creation completed for chunk {chunk_id}: "
            f"Graph A={similar_edges}{' (FAILED)' if graph_a_failed else ''}, "
            f"Graph B={sum(probe_edges.values())}{' (FAILED)' if graph_b_failed else ''}, "
            f"total={total_edges}, time={execution_time:.3f}s"
        )
        
        return result
    
    def _create_similar_edges(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_text: Optional[str] = None
    ) -> int:
        """
        Create Similar graph edges (Graph A) using GraphABuilder.
        
        This method uses vector search + reranker to build semantic relationships
        between chunks. The summary parameter is deprecated and ignored.
        
        Implements comprehensive error handling:
        - Validates input parameters
        - Catches and logs all exceptions
        - Tracks failure statistics
        - Continues processing even on failure (graceful degradation)
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk (deprecated, not used)
            document_id: Document ID to exclude same-document matches
            metadata: Additional metadata for the chunk
            chunk_text: Full chunk text for reranker (required for Graph A)
            
        Returns:
            Number of similar edges created (0 on failure)
        """
        # Validate input parameters
        if not chunk_text or not chunk_text.strip():
            logger.warning(
                f"Graph A: No chunk text provided for chunk {chunk_id}. "
                f"Skipping similar edge creation."
            )
            return 0
        
        if not chunk_id or not chunk_id.strip():
            logger.error("Graph A: Invalid chunk_id provided (empty or None)")
            return 0
        
        if not document_id or not document_id.strip():
            logger.warning(
                f"Graph A: No document_id provided for chunk {chunk_id}. "
                f"Proceeding with 'unknown' document_id."
            )
            document_id = "unknown"
        
        try:
            # Use Graph A Builder to create semantic relationship edges
            edges_created = self.graph_a_builder.build_relations_for_chunk_sync(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                document_id=document_id,
                metadata=metadata
            )
            
            logger.debug(
                f"Graph A: Created {edges_created} similar edges for chunk {chunk_id}"
            )
            
            return edges_created
            
        except DatabaseError as e:
            # Database-specific errors (vector search, FalkorDB operations)
            logger.error(
                f"Graph A: Database error during edge creation for chunk {chunk_id}: {e}",
                exc_info=True
            )
            logger.warning(
                f"Graph A: Continuing ingestion despite database error for chunk {chunk_id}"
            )
            return 0
            
        except TimeoutError as e:
            # Timeout errors (reranker service, database operations)
            logger.error(
                f"Graph A: Timeout during edge creation for chunk {chunk_id}: {e}"
            )
            logger.warning(
                "Graph A: Reranker or database service may be slow or unresponsive"
            )
            return 0
            
        except ValueError as e:
            # Invalid parameter errors
            logger.error(
                f"Graph A: Invalid parameter during edge creation for chunk {chunk_id}: {e}"
            )
            return 0
            
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Graph A: Unexpected error during edge creation for chunk {chunk_id}: {e}",
                exc_info=True
            )
            logger.warning(
                f"Graph A: Continuing ingestion despite unexpected error for chunk {chunk_id}"
            )
            return 0
    
    def _create_probe_edges(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        chunk_text: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Create Probe graph edges using the ProbeGraphManager.
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk
            document_id: Document ID to exclude same-document matches
            chunk_text: Full chunk text for reranker validation (optional)
            
        Returns:
            Dict mapping probe type to number of edges created
        """
        try:
            # Use vector search top_k setting directly - no complex calculations
            from config.settings import settings
            max_candidates_per_probe = getattr(settings, 'vector_search_top_k_per_probe', 50)
            
            return self.probe_manager.build_probe_edges(
                chunk_id=chunk_id,
                summary=summary,
                document_id=document_id,
                chunk_text=chunk_text,
                max_candidates_per_probe=max_candidates_per_probe
            )
        except Exception as e:
            logger.error(f"Probe edge creation failed for chunk {chunk_id}: {e}")
            # Return zero counts for all probe types on failure
            return {
                'CONTRADICTS': 0,
                'ELABORATES': 0,
                'DEPENDS_ON': 0
            }
    
    async def _create_similar_edges_async(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_text: Optional[str] = None
    ) -> int:
        """
        Async wrapper for similar edge creation using Graph A Builder.
        
        Implements comprehensive error handling:
        - Validates input parameters
        - Catches and logs all exceptions
        - Tracks failure statistics
        - Continues processing even on failure (graceful degradation)
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk (deprecated, not used)
            document_id: Document ID to exclude same-document matches
            metadata: Additional metadata for the chunk
            chunk_text: Full chunk text for reranker (required for Graph A)
            
        Returns:
            Number of similar edges created (0 on failure)
        """
        # Validate input parameters
        if not chunk_text or not chunk_text.strip():
            logger.warning(
                f"Graph A (async): No chunk text provided for chunk {chunk_id}. "
                f"Skipping similar edge creation."
            )
            return 0
        
        if not chunk_id or not chunk_id.strip():
            logger.error("Graph A (async): Invalid chunk_id provided (empty or None)")
            return 0
        
        if not document_id or not document_id.strip():
            logger.warning(
                f"Graph A (async): No document_id provided for chunk {chunk_id}. "
                f"Proceeding with 'unknown' document_id."
            )
            document_id = "unknown"
        
        try:
            # Use Graph A Builder's async method directly
            edges_created = await self.graph_a_builder.build_relations_for_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                document_id=document_id,
                metadata=metadata
            )
            
            logger.debug(
                f"Graph A (async): Created {edges_created} similar edges for chunk {chunk_id}"
            )
            
            return edges_created
            
        except DatabaseError as e:
            # Database-specific errors (vector search, FalkorDB operations)
            logger.error(
                f"Graph A (async): Database error during edge creation for chunk {chunk_id}: {e}",
                exc_info=True
            )
            logger.warning(
                f"Graph A (async): Continuing ingestion despite database error for chunk {chunk_id}"
            )
            return 0
            
        except TimeoutError as e:
            # Timeout errors (reranker service, database operations)
            logger.error(
                f"Graph A (async): Timeout during edge creation for chunk {chunk_id}: {e}"
            )
            logger.warning(
                "Graph A (async): Reranker or database service may be slow or unresponsive"
            )
            return 0
            
        except ValueError as e:
            # Invalid parameter errors
            logger.error(
                f"Graph A (async): Invalid parameter during edge creation for chunk {chunk_id}: {e}"
            )
            return 0
            
        except asyncio.CancelledError:
            # Task cancellation
            logger.warning(
                f"Graph A (async): Edge creation cancelled for chunk {chunk_id}"
            )
            return 0
            
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Graph A (async): Unexpected error during edge creation for chunk {chunk_id}: {e}",
                exc_info=True
            )
            logger.warning(
                f"Graph A (async): Continuing ingestion despite unexpected error for chunk {chunk_id}"
            )
            return 0
    
    async def _create_probe_edges_async(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        chunk_text: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Async wrapper for probe edge creation.
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk
            document_id: Document ID to exclude same-document matches
            chunk_text: Full chunk text for reranker validation (optional)
            
        Returns:
            Dict mapping probe type to number of edges created
        """
        # Use vector search top_k setting directly - no complex calculations
        from config.settings import settings
        max_candidates_per_probe = getattr(settings, 'vector_search_top_k_per_probe', 50)
        
        return await self.probe_manager.build_probe_edges_async(
            chunk_id=chunk_id,
            summary=summary,
            document_id=document_id,
            chunk_text=chunk_text,
            max_candidates_per_probe=max_candidates_per_probe
        )
    
    def _check_failure_threshold(self, threshold: float = 0.1) -> None:
        """
        Check if failure rate exceeds threshold and log warning.
        
        This method monitors the failure rate for both Graph A and Graph B
        and logs warnings when the failure rate exceeds the specified threshold.
        
        Args:
            threshold: Failure rate threshold (default 0.1 = 10%)
        """
        total_processed = self.failure_stats["total_chunks_processed"]
        
        if total_processed < 10:
            # Don't check threshold until we have enough samples
            return
        
        # Check Graph A failure rate
        graph_a_failures = self.failure_stats["graph_a_failures"]
        graph_a_failure_rate = graph_a_failures / total_processed
        
        if graph_a_failure_rate > threshold:
            logger.warning(
                f"Graph A failure rate ({graph_a_failure_rate:.1%}) exceeds threshold ({threshold:.1%}). "
                f"Failures: {graph_a_failures}/{total_processed}. "
                f"Check reranker service health and database connectivity."
            )
        
        # Check Graph B failure rate
        graph_b_failures = self.failure_stats["graph_b_failures"]
        graph_b_failure_rate = graph_b_failures / total_processed
        
        if graph_b_failure_rate > threshold:
            logger.warning(
                f"Graph B failure rate ({graph_b_failure_rate:.1%}) exceeds threshold ({threshold:.1%}). "
                f"Failures: {graph_b_failures}/{total_processed}. "
                f"Check probe graph manager and classification service health."
            )
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """
        Get failure statistics for monitoring and debugging.
        
        Returns:
            Dictionary with failure statistics:
            - graph_a_failures: Number of Graph A failures
            - graph_b_failures: Number of Graph B failures
            - total_chunks_processed: Total chunks processed
            - graph_a_success_count: Number of successful Graph A operations
            - graph_b_success_count: Number of successful Graph B operations
            - graph_a_failure_rate: Graph A failure rate (0.0 to 1.0)
            - graph_b_failure_rate: Graph B failure rate (0.0 to 1.0)
            - graph_a_success_rate: Graph A success rate (0.0 to 1.0)
            - graph_b_success_rate: Graph B success rate (0.0 to 1.0)
        """
        total = self.failure_stats["total_chunks_processed"]
        
        if total == 0:
            return {
                **self.failure_stats,
                "graph_a_failure_rate": 0.0,
                "graph_b_failure_rate": 0.0,
                "graph_a_success_rate": 0.0,
                "graph_b_success_rate": 0.0
            }
        
        return {
            **self.failure_stats,
            "graph_a_failure_rate": self.failure_stats["graph_a_failures"] / total,
            "graph_b_failure_rate": self.failure_stats["graph_b_failures"] / total,
            "graph_a_success_rate": self.failure_stats["graph_a_success_count"] / total,
            "graph_b_success_rate": self.failure_stats["graph_b_success_count"] / total
        }
    
    def reset_failure_statistics(self) -> None:
        """Reset failure statistics counters."""
        self.failure_stats = {
            "graph_a_failures": 0,
            "graph_b_failures": 0,
            "total_chunks_processed": 0,
            "graph_a_success_count": 0,
            "graph_b_success_count": 0
        }
        logger.debug("Reset parallel edge creator failure statistics")
    
    def get_edge_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format edge creation statistics from results.
        
        Args:
            results: Results from create_edges_parallel
            
        Returns:
            Formatted statistics dictionary
        """
        probe_edges = results.get('probe_edges', {})
        
        return {
            'similar_edges': results.get('similar_edges', 0),
            'probe_edges_by_type': probe_edges,
            'total_probe_edges': sum(probe_edges.values()),
            'total_edges': results.get('total_edges', 0),
            'execution_time_seconds': results.get('execution_time', 0.0),
            'edges_per_second': (
                results.get('total_edges', 0) / results.get('execution_time', 1.0)
                if results.get('execution_time', 0) > 0 else 0
            ),
            'graph_a_failed': results.get('graph_a_failed', False),
            'graph_b_failed': results.get('graph_b_failed', False)
        }

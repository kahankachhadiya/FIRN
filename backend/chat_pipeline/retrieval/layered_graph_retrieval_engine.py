"""
Layered Graph Retrieval Engine for Two-Graph RAG Architecture.

This module implements the 12-stage retrieval pipeline with two-graph architecture:
1. Vector search (top VECTOR_SEARCH_TOP_K chunks)
2. Keyword search (top KEYWORD_SEARCH_TOP_K chunks)
3. Merge and deduplicate results
4. Select top chunks (VECTOR_TOP_CHUNKS + KEYWORD_TOP_CHUNKS)
5. Query Graph A (semantic relations using vector search + reranker scoring)
6. Collect all chunks and deduplicate
7. Cross-encoder reranking (select top RERANKED_TOP_CHUNKS)
8. Identify anchor chunk (top 1 from reranking)
9. Context expansion using metadata DB ordering (±1 adjacent chunks)
10. Query Graph B (probe graph with relationship types: SUPPORTS, CONTRADICTS, etc.)
11. Merge Graph A and Graph B into unified Knowledge Graph
12. Final chunk assembly and formatting

Two-Graph Architecture:
- Graph A: Semantic relations built using vector search on Main Collection + Qwen-Reranker-0.6B
  scoring. Creates SIMILAR_TO edges between semantically related chunks across documents.
- Graph B: Probe graph with typed relationships (SUPPORTS, CONTRADICTS, EXAMPLE_OF, ELABORATES,
  DEPENDS_ON) built using NLI classification. Unchanged from previous implementation.
- Knowledge Graph: Merged result of Graph A and Graph B with deduplication by chunk_id,
  keeping higher confidence scores. Each chunk includes graph_source metadata.

All configuration parameters are loaded from ProductionConfig with environment variable support.
"""

import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:
    from utils.conversation_logger import ConversationLogger

from db.metadata_client import get_adjacent_chunks
# Legacy reranker import removed - now using disambiguation system
from llm.query_generator import RAGCommand
from utils.logging_utils import get_logger
from utils.errors import DatabaseError
from config.production import ProductionConfig, get_production_config

logger = get_logger(__name__)


@dataclass
class EnhancedChunkOutput:
    """Enhanced chunk output with images embedded in text."""
    chunk_id: str
    document_id: str
    page_number: int
    document_path: str
    text: str  # Contains embedded image references like [Image: /path/to/image.png]
    origin: str  # anchor, relation, probe_supports, probe_contradicts, probe_example, probe_elaborates, probe_depends, context
    confidence: float


class LayeredGraphRetrievalEngine:
    """
    Core engine for two-graph RAG retrieval architecture.
    
    This engine orchestrates the retrieval pipeline using two complementary graph structures:
    - Graph A: Semantic similarity graph built with reranker scoring
    - Graph B: Probe graph with typed relationships (SUPPORTS, CONTRADICTS, etc.)
    
    The results from both graphs are merged into a unified Knowledge Graph with
    deduplication and confidence-based selection.
    """
    
    def __init__(
        self,
        conversation_logger: Optional["ConversationLogger"] = None,
        redis_connection_pool: Optional[Any] = None,
        production_config: Optional[ProductionConfig] = None
    ):
        """
        Initialize the layered graph retrieval engine.
        
        Args:
            conversation_logger: Optional conversation logger for debugging
            redis_connection_pool: Optional Redis connection pool for async operations
            production_config: Optional production configuration (will load from global if not provided)
        """
        self.conversation_logger = conversation_logger
        self.logger = logger
        self._redis_connection_pool = redis_connection_pool
        
        # Load configuration from ProductionConfig
        if production_config is not None:
            self.production_config = production_config
        else:
            try:
                self.production_config = get_production_config()
            except RuntimeError:
                # Fallback: create config from environment if not initialized
                self.production_config = ProductionConfig.from_env()
        
        # Extract pipeline parameters for easy access
        self.config = self.production_config.get_pipeline_parameters()
        
        self.logger.info("Layered Graph Retrieval Engine initialized with ProductionConfig")
        self.logger.debug(f"Pipeline configuration: {self.config}")
    
    def set_redis_connection_pool(self, connection_pool: Any) -> None:
        """
        Set the Redis connection pool for async keyword search.
        
        Args:
            connection_pool: RedisConnectionPool instance from ConnectionPoolManager
        """
        self._redis_connection_pool = connection_pool
        self.logger.info("Redis connection pool configured for keyword search")
    
    def set_connection_manager(self, connection_manager: Any) -> None:
        """
        Set the connection manager for database operations.
        
        Args:
            connection_manager: ConnectionPoolManager instance
        """
        self._connection_manager = connection_manager
        
        # Also set up individual connection pools for direct access
        if hasattr(connection_manager, 'redis_pool'):
            self.set_redis_connection_pool(connection_manager.redis_pool)
        
        self.logger.info("Connection manager configured for retrieval engine")
    
    def reload_config(self) -> None:
        """
        Reload configuration from ProductionConfig.
        
        This method allows hot-reloading of configuration parameters
        without restarting the engine.
        """
        try:
            # Try to get global config first
            try:
                self.production_config = get_production_config()
            except RuntimeError:
                # If global config not initialized, create from environment
                self.production_config = ProductionConfig.from_env()
            
            self.config = self.production_config.get_pipeline_parameters()
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            raise
    
    async def execute_pipeline(
        self,
        rag_command: RAGCommand,
        log: Optional[Any] = None,
        timeout_config: Optional[Dict[str, float]] = None
    ) -> List[EnhancedChunkOutput]:
        """
        Execute the 12-stage retrieval pipeline with two-graph architecture.

        Pipeline Stages:
        1. Vector search (top VECTOR_SEARCH_TOP_K chunks)
        2. Keyword search (top KEYWORD_SEARCH_TOP_K chunks)
        3. Merge and deduplicate results
        4. Select top chunks (VECTOR_TOP_CHUNKS + KEYWORD_TOP_CHUNKS)
        5. Query Graph A (semantic relations — top GRAPH_A_TOP_N neighbors)
        6. Collect all chunks and deduplicate
        7. Select top RERANKED_TOP_CHUNKS
        8. Identify anchor chunk (top 1)
        9. Context expansion (±CONTEXT_ADJACENT_CHUNKS)
        10. Query Graph B (probe graph with typed relationships)
        11. Merge Graph A and Graph B into unified Knowledge Graph
        12. Final chunk assembly and formatting
        """
        start_time = datetime.now()

        try:
            # Stage 1 & 2: Vector + Keyword Search (parallel)
            self.logger.info("Stage 1-2: Executing vector and keyword search")
            stage_start = datetime.now()

            # Resolve metadata filter to document_ids once before both searches
            resolved_filter: Optional[Dict[str, Any]] = None
            if rag_command.metadata_filter:
                from db.metadata_client import resolve_metadata_filter
                doc_ids = resolve_metadata_filter(rag_command.metadata_filter)
                if doc_ids is not None:
                    if not doc_ids:
                        self.logger.warning("Metadata filter matched 0 documents — returning empty results")
                        return []
                    resolved_filter = {"document_ids": doc_ids}
                    self.logger.info(f"Metadata filter resolved to {len(doc_ids)} document(s)")

            try:
                vector_results_raw, keyword_results_raw = await asyncio.gather(
                    self._vector_search(
                        rag_command.vector_query,
                        resolved_filter,
                        self.config['vector_search_top_k']
                    ),
                    self._keyword_search(
                        rag_command.keyword_query,
                        resolved_filter,
                        self.config['keyword_search_top_k']
                    ),
                    return_exceptions=True
                )

                vector_results: List[Dict[str, Any]] = [] if isinstance(vector_results_raw, Exception) else vector_results_raw  # type: ignore[assignment]
                keyword_results: List[Dict[str, Any]] = [] if isinstance(keyword_results_raw, Exception) else keyword_results_raw  # type: ignore[assignment]

                if isinstance(vector_results_raw, Exception):
                    self.logger.error(f"Vector search failed: {vector_results_raw}")
                if isinstance(keyword_results_raw, Exception):
                    self.logger.error(f"Keyword search failed: {keyword_results_raw}")

            except Exception as e:
                self.logger.error(f"Search stage failed: {e}")
                vector_results, keyword_results = [], []

            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.info(
                f"Stage 1-2 completed: {len(vector_results)} vector results, "
                f"{len(keyword_results)} keyword results ({stage_duration:.2f}s)"
            )

            if self.conversation_logger and log:
                self.conversation_logger.log_search_results(log, vector_results, keyword_results)
                self.conversation_logger.log_stage_timing(log, "vector_keyword_search", stage_duration)

            # Stage 3: Merge and Deduplicate
            self.logger.info("Stage 3: Merging and deduplicating results")
            stage_start = datetime.now()
            merged_results = self._merge_and_dedupe(vector_results, keyword_results)
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Stage 3 completed: {len(merged_results)} merged results ({stage_duration:.2f}s)")

            # Stage 4: Select Top Chunks for Relation Graph Query
            self.logger.info("Stage 4: Selecting top chunks for relation graph query")
            selected_chunks = self._select_top_chunks(
                vector_results,
                keyword_results,
                self.config['vector_top_chunks'],
                self.config['keyword_top_chunks']
            )
            self.logger.info(f"Stage 4 completed: {len(selected_chunks)} chunks selected")

            # Stage 5: Query Graph A (semantic relations)
            self.logger.info("Stage 5: Querying Graph A (semantic relations)")
            stage_start = datetime.now()
            try:
                graph_a_results = await self._query_graph_a(selected_chunks)
            except Exception as e:
                self.logger.error(f"Graph A query failed: {e}")
                graph_a_results = []
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Stage 5 completed: {len(graph_a_results)} Graph A results ({stage_duration:.2f}s)")

            # Stage 6: Collect All Chunks and Deduplicate
            self.logger.info("Stage 6: Collecting and deduplicating all chunks")
            all_chunks = merged_results + graph_a_results
            all_chunks = self._deduplicate_chunks(all_chunks)
            self.logger.info(f"Stage 6 completed: {len(all_chunks)} total chunks")

            # Stage 7: Select top chunks
            self.logger.info("Stage 7: Selecting top chunks")
            stage_start = datetime.now()
            reranked_chunks = all_chunks[:self.config['reranked_top_chunks']]
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Stage 7 completed: {len(reranked_chunks)} reranked chunks ({stage_duration:.2f}s)")

            if self.conversation_logger and log:
                self.conversation_logger.log_reranking(log, all_chunks, reranked_chunks)
                self.conversation_logger.log_stage_timing(log, "reranking", stage_duration)

            # Stage 8: Identify Anchor Chunk
            self.logger.info("Stage 8: Identifying anchor chunk")
            if not reranked_chunks:
                self.logger.warning("No chunks after reranking, returning empty results")
                return []
            anchor_chunk = reranked_chunks[0]
            anchor_chunk['origin'] = 'anchor'
            self.logger.info(f"Stage 8 completed: Anchor chunk identified: {anchor_chunk['chunk_id']}")

            # Stage 9: Context Expansion
            self.logger.info("Stage 9: Expanding context with adjacent chunks")
            stage_start = datetime.now()
            context_chunks = []
            if self.config['context_expansion_enabled']:
                try:
                    context_chunks = await self._expand_context(
                        anchor_chunk,
                        self.config['context_adjacent_chunks']
                    )
                except Exception as e:
                    self.logger.error(f"Context expansion failed: {e}")
                    context_chunks = []
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Stage 9 completed: {len(context_chunks)} context chunks ({stage_duration:.2f}s)")

            # Stage 10: Query Graph B (Probe Graph)
            self.logger.info("Stage 10: Querying Graph B (probe graph with typed relationships)")
            stage_start = datetime.now()
            try:
                graph_b_results = await self._query_probe_graph(anchor_chunk)
            except Exception as e:
                self.logger.error(f"Graph B query failed: {e}")
                graph_b_results = []
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.info(f"Stage 10 completed: {len(graph_b_results)} Graph B results ({stage_duration:.2f}s)")

            if self.conversation_logger and log:
                self.conversation_logger.log_graph_traversal(log, graph_a_results, graph_b_results)
                self.conversation_logger.log_context_expansion(log, context_chunks)
                self.conversation_logger.log_stage_timing(log, "graph_traversal", stage_duration)

            # Stage 11: Knowledge Graph = Graph B results from anchor
            self.logger.info("Stage 11: Merging Graph A and Graph B into Knowledge Graph")
            stage_start = datetime.now()
            knowledge_graph_chunks = graph_b_results
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.info(
                f"Stage 11 completed: {len(knowledge_graph_chunks)} Knowledge Graph chunks ({stage_duration:.2f}s)"
            )

            # Stage 12: Final Assembly
            self.logger.info("Stage 12: Assembling final chunks from Knowledge Graph")
            stage_start = datetime.now()
            final_chunks = await self._assemble_final_chunks_async(
                reranked_chunks,
                context_chunks,
                knowledge_graph_chunks
            )
            stage_duration = (datetime.now() - stage_start).total_seconds()
            total_duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Pipeline completed: {len(final_chunks)} final chunks "
                f"(assembly: {stage_duration:.2f}s, total: {total_duration:.2f}s)"
            )

            return final_chunks

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            if log and hasattr(log, 'errors'):
                log.errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Pipeline execution failed: {str(e)}",
                    "severity": "critical",
                    "stage": "pipeline_execution",
                    "error_type": type(e).__name__
                })
            raise

    async def _vector_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute vector search using Qdrant with proper error handling and connection pooling.
        
        Args:
            query: Vector search query
            metadata_filter: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            List of search results with chunk metadata
            
        Raises:
            DatabaseError: If vector search fails
        """
        try:
            # Use connection manager for Qdrant if available
            if hasattr(self, '_connection_manager') and self._connection_manager:
                try:
                    # Get Qdrant client from connection manager
                    qdrant_client = self._connection_manager.get_qdrant_client()
                    
                    # Use the client with connection pooling
                    from db.qdrant_client import search_paragraphs_with_client
                    results = await search_paragraphs_with_client(
                        client=qdrant_client,
                        query=query,
                        top_k=top_k,
                        metadata_filter=metadata_filter,
                        score_threshold=0.0
                    )
                    self.logger.debug(f"Vector search via connection manager: {len(results)} results")
                except Exception as e:
                    self.logger.warning(f"Connection manager vector search failed, using fallback: {e}")
                    # Fall through to fallback
                    results = None
            else:
                results = None
            
            # Fallback to standard search if connection manager failed or unavailable
            if results is None:
                from db.qdrant_client import search_paragraphs
                results = await search_paragraphs(
                    query=query,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                    score_threshold=0.0
                )
                self.logger.debug(f"Vector search via fallback: {len(results)} results")
            
            # Enrich results with text content from Elasticsearch (with connection pooling)
            enriched_results = []
            for result in results:
                chunk_id = result.get('chunk_id')
                if chunk_id:
                    # Try to get text from Elasticsearch with connection pooling fallback
                    text = await self._get_paragraph_with_fallback(chunk_id, result.get('metadata', {}))
                    
                    if text:
                        enriched_result = {
                            'chunk_id': chunk_id,
                            'text': text,
                            'similarity_score': result.get('score', 0.0),
                            'document_id': result.get('metadata', {}).get('document_id', ''),
                            'page_number': result.get('metadata', {}).get('page_number', 0),
                            'document_path': result.get('metadata', {}).get('document_path', ''),
                            'origin': 'vector_search'
                        }
                        enriched_results.append(enriched_result)
            
            return enriched_results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise DatabaseError(f"Vector search failed: {e}") from e
    
    async def _keyword_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute keyword search using Redis with proper error handling and connection pooling.
        
        Args:
            query: Keyword search query
            metadata_filter: Optional metadata filters (not used in current implementation)
            top_k: Number of results to return
            
        Returns:
            List of search results with chunk metadata
            
        Raises:
            DatabaseError: If keyword search fails
        """
        try:
            # Use synchronous Elasticsearch search (connection manager path removed —
            # get_redis_connection() is not available; sync fallback is reliable)
            results = None

            # Try dedicated pool if available
            if results is None and hasattr(self, '_redis_connection_pool') and self._redis_connection_pool:
                try:
                    from db.elasticsearch_client import search_text_async
                    results = await search_text_async(query=query, limit=top_k, connection_pool=self._redis_connection_pool)
                    self.logger.debug(f"Keyword search via dedicated pool: {len(results)} results")
                except Exception as e:
                    self.logger.debug(f"Dedicated pool keyword search failed, using sync: {e}")
                    results = None

            # Sync fallback (primary path)
            if results is None:
                from db.elasticsearch_client import search_text
                results = search_text(query=query, limit=top_k)
                self.logger.debug(f"Keyword search via sync: {len(results)} results")

            # Apply document_id filter if provided
            allowed_doc_ids = None
            if metadata_filter:
                allowed_doc_ids = set(metadata_filter.get("document_ids") or [])

            # Enrich results with metadata
            enriched_results = []
            for result in results:
                chunk_id = result.get('chunk_id')
                text = result.get('text', '')

                if chunk_id and text:
                    # Parse chunk_id to extract document_id and page_number
                    parts = chunk_id.split('_')
                    document_id = '_'.join(parts[:-2]) if len(parts) >= 3 else parts[0]
                    page_number = int(parts[-2][1:]) if len(parts) >= 2 and parts[-2].startswith('P') else 0

                    # Filter by document_ids if a metadata filter was applied
                    if allowed_doc_ids and document_id not in allowed_doc_ids:
                        continue

                    enriched_result = {
                        'chunk_id': chunk_id,
                        'text': text,
                        'similarity_score': 1.0,
                        'document_id': document_id,
                        'page_number': page_number,
                        'document_path': '',
                        'origin': 'keyword_search'
                    }
                    enriched_results.append(enriched_result)
            
            return enriched_results
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            raise DatabaseError(f"Keyword search failed: {e}") from e
    
    def _merge_and_dedupe(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from vector and keyword search.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            
        Returns:
            Merged and deduplicated list of chunks
        """
        seen_chunks = {}
        
        # Add vector results first (higher priority)
        for chunk in vector_results:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = chunk
        
        # Add keyword results (avoid duplicates)
        for chunk in keyword_results:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = chunk
        
        return list(seen_chunks.values())
    
    def _select_top_chunks(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        vector_top: int,
        keyword_top: int
    ) -> List[Dict[str, Any]]:
        """
        Select top N chunks from vector and keyword results for relation graph query.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            vector_top: Number of top chunks to select from vector results
            keyword_top: Number of top chunks to select from keyword results
            
        Returns:
            Selected chunks for relation graph query
        """
        selected = []
        seen_ids = set()
        
        # Top N from vector results
        for chunk in vector_results[:vector_top]:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_ids:
                selected.append(chunk)
                seen_ids.add(chunk_id)
        
        # Top N from keyword results (avoid duplicates)
        for chunk in keyword_results[:keyword_top]:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_ids:
                selected.append(chunk)
                seen_ids.add(chunk_id)
        
        return selected
    
    async def _query_graph_a(
        self,
        selected_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Query Graph A: fetch SIMILAR_TO neighbors from FalkorDB for each selected chunk,
        returning the top GRAPH_A_TOP_N results across all chunks."""
        from db.relation_graph_client import query_related_chunks

        top_n = self.config.get('graph_a_top_n', 10)
        graph_a_results: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for chunk in selected_chunks:
            chunk_id = chunk.get('chunk_id')
            if not chunk_id:
                continue
            try:
                neighbors = query_related_chunks(
                    chunk_id=chunk_id,
                    max_neighbors=top_n,
                )
                for neighbor in neighbors:
                    neighbor_id = neighbor.get('chunk_id')
                    if not neighbor_id or neighbor_id in seen_ids:
                        continue
                    seen_ids.add(neighbor_id)
                    text = await self._get_paragraph_with_fallback(
                        neighbor_id,
                        {'document_id': neighbor.get('document_id', ''),
                         'page_number': neighbor.get('page_number', 0)}
                    )
                    if text:
                        graph_a_results.append({
                            'chunk_id': neighbor_id,
                            'text': text,
                            'similarity_score': neighbor.get('similarity', 0.0),
                            'document_id': neighbor.get('document_id', ''),
                            'page_number': neighbor.get('page_number', 0),
                            'document_path': '',
                            'origin': 'graph_a',
                            'graph_source': 'graph_a',
                        })
            except Exception as e:
                self.logger.error(f"Graph A query failed for {chunk_id}: {e}")

        graph_a_results = graph_a_results[:top_n]
        self.logger.info(f"Graph A completed: {len(graph_a_results)} chunks (top {top_n})")
        return graph_a_results
    
    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate chunks by chunk_id.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            Deduplicated list of chunks
        """
        seen_chunks = {}
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = chunk
        
        return list(seen_chunks.values())
    
    async def _expand_context(
        self,
        anchor_chunk: Dict[str, Any],
        adjacent_count: int
    ) -> List[Dict[str, Any]]:
        """
        Expand context by retrieving ±N adjacent chunks with proper error handling.
        
        Args:
            anchor_chunk: Anchor chunk to expand context around
            adjacent_count: Number of adjacent chunks to retrieve (±N)
            
        Returns:
            List of context chunks with proper metadata and origin labels
            
        Raises:
            DatabaseError: If context expansion fails
        """
        try:
            # Extract anchor chunk information
            chunk_id = anchor_chunk.get('chunk_id', '')
            document_id = anchor_chunk.get('document_id', '')
            
            if not chunk_id:
                raise DatabaseError("Cannot expand context: anchor chunk has no chunk_id")
            
            # Use metadata client to get adjacent chunks
            adjacent_chunks = get_adjacent_chunks(
                chunk_id=chunk_id,
                document_id=document_id,
                adjacent_count=adjacent_count
            )
            
            # Enrich adjacent chunks with additional metadata from anchor
            context_chunks = []
            for chunk in adjacent_chunks:
                # Add document_path from anchor if not present
                if not chunk.get('document_path'):
                    chunk['document_path'] = anchor_chunk.get('document_path', '')
                
                # Add similarity score (0.0 for context chunks)
                if 'similarity_score' not in chunk:
                    chunk['similarity_score'] = 0.0
                
                context_chunks.append(chunk)
            
            self.logger.info(f"Context expansion succeeded: {len(context_chunks)} adjacent chunks")
            return context_chunks
            
        except Exception as e:
            self.logger.error(f"Context expansion failed: {e}")
            raise DatabaseError(f"Context expansion failed: {e}") from e
    
    async def _merge_knowledge_graph(
        self,
        graph_a_results: List[Dict[str, Any]],
        graph_b_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge Graph A and Graph B results into unified Knowledge Graph.
        
        Knowledge Graph Architecture:
        - Combines semantic relations (Graph A) with typed relationships (Graph B)
        - Provides comprehensive context from both broad semantic coverage and
          precise relationship information
        - Deduplicates by chunk_id to avoid redundancy
        - Preserves provenance through graph_source metadata
        
        Process:
        1. Combine both result lists
        2. Deduplicate by chunk_id (keep higher confidence score)
        3. Add graph_source metadata:
           - 'graph_a': Chunk only from Graph A (semantic relations)
           - 'graph_b': Chunk only from Graph B (probe relationships)
           - 'both': Chunk found in both graphs (highest confidence kept)
        4. Sort by confidence score descending
        5. Log merge statistics for monitoring
        
        Args:
            graph_a_results: Results from Graph A (semantic relations with reranker)
            graph_b_results: Results from Graph B (probe graph with typed relationships)
            
        Returns:
            Merged and deduplicated Knowledge Graph chunks with graph_source metadata
            
        Raises:
            Exception: If merge operation fails critically (falls back to concatenation)
        """
        try:
            self.logger.info(
                f"Merging Knowledge Graph: {len(graph_a_results)} Graph A chunks, "
                f"{len(graph_b_results)} Graph B chunks"
            )
            
            # Track chunks by chunk_id for deduplication
            chunk_map: Dict[str, Dict[str, Any]] = {}
            graph_a_ids = set()
            graph_b_ids = set()
            
            # Process Graph A results first
            for chunk in graph_a_results:
                chunk_id = chunk.get('chunk_id')
                if not chunk_id:
                    self.logger.debug("Skipping Graph A chunk with missing chunk_id")
                    continue
                
                graph_a_ids.add(chunk_id)
                
                # Get confidence score (use reranker_score or similarity_score)
                confidence = chunk.get('reranker_score', chunk.get('similarity_score', 0.0))
                
                # Check if chunk already exists in map (duplicate within Graph A)
                if chunk_id in chunk_map:
                    existing_chunk = chunk_map[chunk_id]
                    existing_confidence = existing_chunk.get('confidence', 0.0)
                    
                    # Keep the chunk with higher confidence
                    if confidence > existing_confidence:
                        # Current chunk has higher confidence, replace
                        chunk_copy = chunk.copy()
                        chunk_copy['confidence'] = confidence
                        chunk_copy['graph_source'] = 'graph_a'
                        chunk_map[chunk_id] = chunk_copy
                        
                        self.logger.debug(
                            f"Duplicate chunk {chunk_id} in Graph A, keeping version with "
                            f"higher confidence: {confidence:.3f} > {existing_confidence:.3f}"
                        )
                    else:
                        # Existing chunk has higher or equal confidence, keep it
                        self.logger.debug(
                            f"Duplicate chunk {chunk_id} in Graph A, keeping version with "
                            f"higher confidence: {existing_confidence:.3f} >= {confidence:.3f}"
                        )
                else:
                    # New chunk, add to map
                    chunk_copy = chunk.copy()
                    chunk_copy['confidence'] = confidence
                    chunk_copy['graph_source'] = 'graph_a'
                    
                    chunk_map[chunk_id] = chunk_copy
            
            # Process Graph B results
            for chunk in graph_b_results:
                chunk_id = chunk.get('chunk_id')
                if not chunk_id:
                    self.logger.debug("Skipping Graph B chunk with missing chunk_id")
                    continue
                
                graph_b_ids.add(chunk_id)
                
                # Get confidence score
                confidence = chunk.get('similarity_score', chunk.get('confidence', 0.0))
                
                # Check if chunk already exists from Graph A
                if chunk_id in chunk_map:
                    # Chunk exists in both graphs
                    existing_chunk = chunk_map[chunk_id]
                    existing_confidence = existing_chunk.get('confidence', 0.0)
                    
                    # Keep the chunk with higher confidence
                    if confidence > existing_confidence:
                        # Graph B has higher confidence, replace
                        chunk_copy = chunk.copy()
                        chunk_copy['confidence'] = confidence
                        chunk_copy['graph_source'] = 'both'  # Mark as from both graphs
                        chunk_map[chunk_id] = chunk_copy
                        
                        self.logger.debug(
                            f"Chunk {chunk_id} found in both graphs, keeping Graph B version "
                            f"(confidence: {confidence:.3f} > {existing_confidence:.3f})"
                        )
                    else:
                        # Graph A has higher or equal confidence, keep it but update source
                        existing_chunk['graph_source'] = 'both'
                        
                        self.logger.debug(
                            f"Chunk {chunk_id} found in both graphs, keeping Graph A version "
                            f"(confidence: {existing_confidence:.3f} >= {confidence:.3f})"
                        )
                else:
                    # New chunk from Graph B only
                    chunk_copy = chunk.copy()
                    chunk_copy['confidence'] = confidence
                    chunk_copy['graph_source'] = 'graph_b'
                    
                    chunk_map[chunk_id] = chunk_copy
            
            # Convert to list and sort by confidence descending
            merged_chunks = list(chunk_map.values())
            merged_chunks.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            
            # Calculate merge statistics
            total_input_chunks = len(graph_a_results) + len(graph_b_results)
            unique_chunks = len(merged_chunks)
            duplicates_removed = total_input_chunks - unique_chunks
            both_graphs_count = len(graph_a_ids & graph_b_ids)
            graph_a_only = len(graph_a_ids - graph_b_ids)
            graph_b_only = len(graph_b_ids - graph_a_ids)
            
            # Log detailed merge statistics
            self.logger.info(
                f"Knowledge Graph merge completed: {unique_chunks} unique chunks "
                f"({duplicates_removed} duplicates removed). "
                f"Sources: {graph_a_only} Graph A only, {graph_b_only} Graph B only, "
                f"{both_graphs_count} from both graphs"
            )
            
            # Log confidence distribution
            if merged_chunks:
                confidences = [c.get('confidence', 0.0) for c in merged_chunks]
                avg_confidence = sum(confidences) / len(confidences)
                max_confidence = max(confidences)
                min_confidence = min(confidences)
                
                self.logger.debug(
                    f"Knowledge Graph confidence stats: "
                    f"avg={avg_confidence:.3f}, max={max_confidence:.3f}, min={min_confidence:.3f}"
                )
            
            return merged_chunks
            
        except Exception as e:
            self.logger.error(f"Knowledge Graph merge failed: {e}", exc_info=True)
            
            # Attempt graceful degradation - return non-deduplicated results
            self.logger.warning(
                "Attempting graceful degradation: returning non-deduplicated results"
            )
            
            try:
                # Simple concatenation as fallback
                fallback_results = []
                
                for chunk in graph_a_results:
                    chunk_copy = chunk.copy()
                    chunk_copy['graph_source'] = 'graph_a'
                    fallback_results.append(chunk_copy)
                
                for chunk in graph_b_results:
                    chunk_copy = chunk.copy()
                    chunk_copy['graph_source'] = 'graph_b'
                    fallback_results.append(chunk_copy)
                
                self.logger.warning(
                    f"Graceful degradation successful: returning {len(fallback_results)} "
                    f"non-deduplicated chunks"
                )
                
                return fallback_results
                
            except Exception as fallback_error:
                self.logger.error(
                    f"Graceful degradation also failed: {fallback_error}",
                    exc_info=True
                )
                # Last resort: return empty list
                return []
    
    async def _query_probe_graph(
        self,
        anchor_chunk: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Query Graph B: fetch all probe edge types (CONTRADICTS, ELABORATES, DEPENDS_ON)
        for the anchor chunk using per-type top-N limits from config."""

        probe_chunks: List[Dict[str, Any]] = []
        chunk_id = anchor_chunk.get('chunk_id')
        if not chunk_id:
            return probe_chunks

        probe_limits = {
            'CONTRADICTS': self.config.get('graph_b_contradicts_top_n', 5),
            'ELABORATES':  self.config.get('graph_b_elaborates_top_n', 5),
            'DEPENDS_ON':  self.config.get('graph_b_depends_on_top_n', 5),
        }

        # Query each probe type separately with its own top-N limit
        for probe_type in probe_limits:
            if probe_type not in probe_limits:
                continue

            try:
                from db.falkor_client import get_probe_neighbors
                probe_neighbors = get_probe_neighbors(
                    chunk_id=chunk_id,
                    probe_types=[probe_type],
                    max_neighbors=probe_limits[probe_type],
                )
                self.logger.debug(f"Probe graph: {len(probe_neighbors)} neighbors for {probe_type}")
                
                # Enrich neighbors with text content (with connection pooling fallback)
                for neighbor in probe_neighbors:
                    neighbor_id = neighbor.get('chunk_id')
                    if neighbor_id:
                        # Try to get text with connection pooling fallback mechanisms
                        text = await self._get_paragraph_with_fallback(
                            neighbor_id,
                            {
                                'document_id': neighbor.get('document_id', ''),
                                'page_number': neighbor.get('page_number', 0)
                            }
                        )
                        
                        if text:
                            enriched_neighbor = {
                                'chunk_id': neighbor_id,
                                'text': text,
                                'similarity_score': neighbor.get('confidence', 0.0),
                                'document_id': neighbor.get('document_id', ''),
                                'page_number': neighbor.get('page_number', 0),
                                'document_path': '',
                                'origin': f'probe_{probe_type.lower()}',
                                'probe_type': probe_type,
                                'graph_source': 'graph_b'  # Add graph source for Knowledge Graph merging
                            }
                            probe_chunks.append(enriched_neighbor)
                
            except Exception as e:
                self.logger.error(f"Probe graph query failed for {probe_type}: {e}")
                continue  # Skip this probe type, try the next one
        
        return probe_chunks
    
    def _assemble_final_chunks(
        self,
        anchor_chunks: List[Dict[str, Any]],
        context_chunks: List[Dict[str, Any]],
        probe_chunks: List[Dict[str, Any]]
    ) -> List[EnhancedChunkOutput]:
        """
        Assemble final chunks with proper formatting and deduplication.
        Enriches document_path from Elasticsearch metadata so the LLM
        can emit proper source URLs.
        """
        all_chunks = anchor_chunks + context_chunks + probe_chunks

        # Build a document_id → path cache from Elasticsearch metadata
        doc_path_cache: dict = {}
        for chunk in all_chunks:
            doc_id = chunk.get('document_id', '')
            # Parse document_id from chunk_id if missing
            if not doc_id:
                chunk_id = chunk.get('chunk_id', '')
                parts = chunk_id.split('_P')
                if len(parts) >= 2:
                    doc_id = parts[0]
            if doc_id and doc_id not in doc_path_cache:
                try:
                    from db.elasticsearch_client import get_document_metadata
                    meta = get_document_metadata(doc_id)
                    if meta:
                        doc_path_cache[doc_id] = (
                            meta.get('path') or
                            meta.get('file_name') or
                            ''
                        )
                    else:
                        doc_path_cache[doc_id] = ''
                except Exception:
                    doc_path_cache[doc_id] = ''

        # Deduplicate by chunk_id (keep first occurrence)
        seen_chunks = set()
        final_chunks = []

        for chunk in all_chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)

                doc_id = chunk.get('document_id', '')
                # Parse document_id from chunk_id if missing (format: DOC_xxx_Pnn_Cnn)
                if not doc_id and chunk_id:
                    parts = chunk_id.split('_P')
                    if len(parts) >= 2:
                        doc_id = parts[0]

                doc_path = chunk.get('document_path') or doc_path_cache.get(doc_id, '')

                # Parse page_number from chunk_id if missing (format: DOC_xxx_Pnn_Cnn)
                page_number = chunk.get('page_number', 0)
                if not page_number and chunk_id:
                    try:
                        p_part = chunk_id.split('_P')[1].split('_C')[0]
                        page_number = int(p_part)
                    except (IndexError, ValueError):
                        page_number = 1

                enhanced_chunk = EnhancedChunkOutput(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    page_number=page_number,
                    document_path=doc_path,
                    text=chunk.get('text', ''),
                    origin=chunk.get('origin', 'unknown'),
                    confidence=chunk.get('similarity_score', 0.0)
                )
                final_chunks.append(enhanced_chunk)

        # Limit to max final chunks
        max_chunks = self.config['max_final_chunks']
        if len(final_chunks) > max_chunks:
            self.logger.warning(
                f"Final chunks ({len(final_chunks)}) exceeds max limit ({max_chunks}), "
                f"truncating to {max_chunks}"
            )
            final_chunks = final_chunks[:max_chunks]

        return final_chunks
    
    async def _merge_and_dedupe_async(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Async version of merge and deduplicate for timeout handling.
        
        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            
        Returns:
            Merged and deduplicated results
        """
        # Run the synchronous version in a thread pool to make it async
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._merge_and_dedupe, 
            vector_results, 
            keyword_results
        )
    
    async def _assemble_final_chunks_async(
        self,
        anchor_chunks: List[Dict[str, Any]],
        context_chunks: List[Dict[str, Any]],
        probe_chunks: List[Dict[str, Any]]
    ) -> List[EnhancedChunkOutput]:
        """
        Async version of final chunk assembly for timeout handling.
        
        Args:
            anchor_chunks: Anchor chunks
            context_chunks: Context expansion chunks
            probe_chunks: Probe graph chunks
            
        Returns:
            List of EnhancedChunkOutput objects
        """
        # Run the synchronous version in a thread pool to make it async
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._assemble_final_chunks,
            anchor_chunks,
            context_chunks,
            probe_chunks
        )
    
    async def _get_paragraph_with_fallback(
        self,
        chunk_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Get paragraph text with comprehensive fallback mechanisms and connection pooling.
        
        Fallback strategies (in priority order):
        1. Try connection manager Redis pool (preferred for production)
        2. Try Redis get_paragraph (legacy/fallback)
        3. Generate placeholder text from metadata
        
        Args:
            chunk_id: Chunk ID to retrieve
            metadata: Chunk metadata for fallback text generation
            
        Returns:
            Paragraph text or fallback text
        """
        # Strategy 1: Try Elasticsearch client (primary)
        try:
            from db.elasticsearch_client import get_paragraph
            text = get_paragraph(chunk_id)
            if text:
                self.logger.debug(f"Retrieved text for {chunk_id} via Elasticsearch")
                return text
        except Exception as e:
            self.logger.debug(f"Elasticsearch get_paragraph failed for {chunk_id}: {e}")
        
        # Strategy 3: Generate placeholder text from metadata
        document_id = metadata.get('document_id', 'unknown')
        page_number = metadata.get('page_number', 0)
        
        placeholder_text = (
            f"[Content temporarily unavailable for chunk {chunk_id}. "
            f"Document: {document_id}, Page: {page_number}. "
            f"The system is experiencing connectivity issues with the text storage. "
            f"Please try again later.]"
        )
        
        self.logger.warning(f"All text retrieval methods failed for {chunk_id}, using placeholder")
        return placeholder_text
    
    async def _fallback_generate_adjacent_chunks(
        self,
        chunk_id: str,
        document_id: str,
        adjacent_count: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback method to generate adjacent chunk IDs when metadata client fails.
        
        Args:
            chunk_id: Base chunk ID
            document_id: Document ID
            adjacent_count: Number of adjacent chunks to generate
            
        Returns:
            List of adjacent chunks (may be empty if generation fails)
        """
        adjacent_chunks = []
        
        try:
            # Parse chunk ID to extract page and chunk numbers
            # Expected format: DOC123_P5_C2
            parts = chunk_id.split('_')
            if len(parts) >= 3:
                doc_part = parts[0]
                page_part = parts[1]
                chunk_part = parts[2]
                
                if page_part.startswith('P') and chunk_part.startswith('C'):
                    page_num = int(page_part[1:])
                    chunk_num = int(chunk_part[1:])
                    
                    # Generate adjacent chunk IDs
                    for offset in range(-adjacent_count, adjacent_count + 1):
                        if offset == 0:  # Skip the anchor chunk itself
                            continue
                        
                        new_chunk_num = chunk_num + offset
                        if new_chunk_num > 0:  # Only positive chunk numbers
                            adjacent_id = f"{doc_part}_{page_part}_C{new_chunk_num}"
                            
                            # Try to get text for this chunk
                            text = await self._get_paragraph_with_fallback(
                                adjacent_id,
                                {'document_id': document_id, 'page_number': page_num}
                            )
                            
                            if text and not text.startswith('[Content temporarily unavailable'):
                                adjacent_chunks.append({
                                    'chunk_id': adjacent_id,
                                    'text': text,
                                    'document_id': document_id,
                                    'page_number': page_num,
                                    'document_path': '',
                                    'origin': 'context',
                                    'similarity_score': 0.0
                                })
        
        except Exception as e:
            self.logger.warning(f"Fallback adjacent chunk generation failed: {e}")
        
        return adjacent_chunks

def create_retrieval_engine(
    conversation_logger: Optional["ConversationLogger"] = None
) -> LayeredGraphRetrievalEngine:
    """
    Factory function to create a new LayeredGraphRetrievalEngine instance.
    
    Args:
        conversation_logger: Optional conversation logger for debugging
        
    Returns:
        Configured LayeredGraphRetrievalEngine instance
    """
    return LayeredGraphRetrievalEngine(conversation_logger=conversation_logger)

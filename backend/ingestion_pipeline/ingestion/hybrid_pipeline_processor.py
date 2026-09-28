"""
Hybrid Pipeline Processor for NLI Classification Service Update.

This module implements the 3-stage hybrid pipeline workflow:
1. Multi-Probe Retrieval: Execute 3 probe-specific vector searches
2. Aggregation & Deduplication: Merge and deduplicate by document ID
3. Unified NLI Classification: Single authoritative classification with ModernBERT

The hybrid pipeline maximizes retrieval coverage while ensuring classification accuracy
by separating the retrieval phase (cast a wide net) from the classification phase
(single authoritative judgment).
"""

from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
import time
import asyncio

from db.qdrant_client import search_paragraphs
from db.elasticsearch_client import get_paragraph
from external_clients.classification_client import ClassificationResponse
from ingestion.gating_router import GatingRouter
from utils.logging_utils import get_logger
from utils.errors import DatabaseError

logger = get_logger(__name__)


@dataclass
class TextPair:
    """Represents a text pair for classification."""
    id: str
    text_a: str  # Source text
    text_b: str  # Candidate text
    metadata: Dict[str, Any]  # Additional metadata (document_id, similarity, etc.)


@dataclass
class ProbeCandidate:
    """Represents a candidate from probe-specific vector search."""
    chunk_id: str
    document_id: str
    text: str
    similarity_score: float
    probe_type: str  # Which probe search found this candidate
    metadata: Dict[str, Any]


class HybridPipelineProcessor:
    """
    Implements the 3-stage hybrid pipeline workflow for probe graph creation.
    
    Stage 1: Multi-Probe Retrieval - Execute 4 probe-specific vector searches
    Stage 2: Aggregation & Deduplication - Merge and deduplicate by document ID
    Stage 3: Unified NLI Classification - Single authoritative classification
    """
    
    # Valid probe types for vector search (3 types, no unrelated search)
    PROBE_TYPES = ["dependency", "elaboration", "contradiction"]
    
    # Valid relationship types from classification (5 types including unrelated)
    RELATIONSHIP_TYPES = {"dependency", "expansion", "contradiction", "unrelated"}
    
    def __init__(self):
        """Initialize the hybrid pipeline processor."""
        self.classification_client = GatingRouter()
        
        # Load probe-specific vector search queries from environment
        self.probe_queries = self._load_probe_queries()
        
        # Text cache for reducing Redis calls and memory usage
        self._text_cache: Dict[str, str] = {}
        self._cache_max_size = 500  # Limit cache size to prevent memory bloat
        
        logger.info(
            f"Initialized HybridPipelineProcessor with {len(self.probe_queries)} probe types"
        )
    
    def _load_probe_queries(self) -> Dict[str, str]:
        """
        Load probe-specific vector search queries from prompts.yml.
        
        Returns:
            Dict mapping probe type to query template
        """
        from pathlib import Path
        import yaml

        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yml"
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        queries = data["probe_queries"]
        logger.debug(f"Loaded probe queries: {list(queries.keys())}")
        return queries
    
    async def process_source_paragraph(
        self,
        source_chunk_id: str,
        source_text: str,
        source_document_id: str,
        max_candidates_per_probe: Optional[int] = None,
        include_intra_document: bool = True,
        include_inter_document: bool = True
    ) -> ClassificationResponse:
        """
        Process a source paragraph through the complete 3-stage hybrid pipeline.
        
        Args:
            source_chunk_id: Unique identifier for the source chunk
            source_text: The source paragraph text
            source_document_id: Document ID of the source chunk
            max_candidates_per_probe: Maximum candidates to retrieve per probe type
            include_intra_document: Whether to include same-document connections
            include_inter_document: Whether to include cross-document connections
            
        Returns:
            ClassificationResponse with results from unified NLI classification
            
        Raises:
            DatabaseError: If pipeline fails
        """
        start_time = time.time()
        
        try:
            # Stage 1: Multi-Probe Retrieval
            logger.debug(f"Stage 1: Multi-probe retrieval for chunk {source_chunk_id}")
            candidates = await self.multi_probe_retrieval(
                source_text=source_text,
                max_candidates_per_probe=max_candidates_per_probe
            )
            
            if not candidates:
                logger.warning(f"No candidates found for chunk {source_chunk_id}")
                return ClassificationResponse(
                    results=[],
                    processing_time=time.time() - start_time,
                    model_info={}
                )
            
            # Stage 2: Aggregation & Deduplication
            logger.debug(f"Stage 2: Aggregation & deduplication for chunk {source_chunk_id}")
            unique_pairs = await self.aggregate_and_deduplicate(
                source_chunk_id=source_chunk_id,
                source_text=source_text,
                source_document_id=source_document_id,
                candidates=candidates,
                include_intra_document=include_intra_document,
                include_inter_document=include_inter_document
            )
            
            if not unique_pairs:
                logger.warning(f"No unique pairs after deduplication for chunk {source_chunk_id}")
                return ClassificationResponse(
                    results=[],
                    processing_time=time.time() - start_time,
                    model_info={}
                )
            
            # Stage 3: Unified NLI Classification
            logger.debug(f"Stage 3: Unified NLI classification for chunk {source_chunk_id}")
            classification_response = await self.classify_text_pairs(unique_pairs)
            
            total_time = time.time() - start_time
            logger.info(
                f"Hybrid pipeline completed for chunk {source_chunk_id}: "
                f"{len(candidates)} candidates -> {len(unique_pairs)} unique pairs -> "
                f"{len(classification_response.results)} classifications "
                f"(time: {total_time:.2f}s)"
            )
            
            return classification_response
            
        except Exception as e:
            logger.error(f"Hybrid pipeline failed for chunk {source_chunk_id}: {e}")
            raise DatabaseError(f"Hybrid pipeline processing failed: {e}") from e
    
    async def multi_probe_retrieval(
        self,
        source_text: str,
        max_candidates_per_probe: int = 50
    ) -> List[ProbeCandidate]:
        """
        Stage 1: Execute 3 probe-specific vector searches in parallel.
        
        This stage casts a wide net by executing multiple probe-specific searches.
        The probe type at this stage is just a hint - the final classification
        in Stage 3 will determine the actual relationship type.
        
        Args:
            source_text: The source paragraph text
            max_candidates_per_probe: Maximum candidates to retrieve per probe
            
        Returns:
            List of ProbeCandidate objects from all probe searches
        """
        all_candidates: List[ProbeCandidate] = []
        
        # Execute all 4 probe searches in parallel
        search_tasks = []
        for probe_type in self.PROBE_TYPES:
            query = f"{self.probe_queries[probe_type]} {source_text}"
            search_tasks.append(
                self._execute_probe_search(
                    query=query,
                    probe_type=probe_type,
                    top_k=max_candidates_per_probe
                )
            )
        
        # Wait for all searches to complete
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Collect candidates from all searches
        for probe_type, result in zip(self.PROBE_TYPES, search_results):
            if isinstance(result, Exception):
                logger.error(f"Probe search failed for {probe_type}: {result}")
                continue
            
            # Type narrowing: result is List[ProbeCandidate] here
            assert not isinstance(result, BaseException)
            candidates: List[ProbeCandidate] = result
            all_candidates.extend(candidates)
            logger.debug(f"Retrieved {len(candidates)} candidates for {probe_type} probe")
        
        logger.info(f"Stage 1 complete: Retrieved {len(all_candidates)} total candidates from {len(self.PROBE_TYPES)} probes")
        return all_candidates
    
    async def _execute_probe_search(
        self,
        query: str,
        probe_type: str,
        top_k: int
    ) -> List[ProbeCandidate]:
        """
        Execute a single probe-specific vector search.
        
        Args:
            query: The search query (probe query + source text)
            probe_type: The probe type for this search
            top_k: Maximum number of results
            
        Returns:
            List of ProbeCandidate objects
        """
        try:
            # Execute vector search with low threshold for initial filtering
            search_results = await search_paragraphs(
                query=query,
                top_k=top_k,
                score_threshold=0.3  # Low threshold - we'll filter later
            )
            
            # Convert search results to ProbeCandidate objects
            candidates = []
            for result in search_results:
                chunk_id = result.get("chunk_id")
                metadata = result.get("metadata", {})
                document_id = metadata.get("document_id", "")
                similarity_score = result.get("score", 0.0)
                
                # Get the full text for this chunk using cache
                chunk_text = self._get_candidate_text(chunk_id)
                if not chunk_text or not chunk_text.strip():
                    logger.debug(f"Skipping candidate {chunk_id} - no text content")
                    continue
                
                candidates.append(ProbeCandidate(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk_text,
                    similarity_score=similarity_score,
                    probe_type=probe_type,
                    metadata=metadata
                ))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error executing probe search for {probe_type}: {e}")
            return []
    
    def _get_candidate_text(self, chunk_id: str) -> Optional[str]:
        """
        Get candidate text with caching to reduce Redis calls and memory usage.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk text or None if not found
        """
        # Check cache first
        if chunk_id in self._text_cache:
            return self._text_cache[chunk_id]
        
        # Load from Redis
        text = get_paragraph(chunk_id)
        
        # Add to cache if not full
        if text and len(self._text_cache) < self._cache_max_size:
            self._text_cache[chunk_id] = text
        
        return text
    
    def _clear_text_cache(self):
        """Clear text cache to free memory."""
        cache_size = len(self._text_cache)
        self._text_cache.clear()
        logger.debug(f"Cleared text cache ({cache_size} entries)")
    
    def _get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._text_cache),
            "cache_max_size": self._cache_max_size
        }
    
    async def aggregate_and_deduplicate(
        self,
        source_chunk_id: str,
        source_text: str,
        source_document_id: str,
        candidates: List[ProbeCandidate],
        include_intra_document: bool = True,
        include_inter_document: bool = True
    ) -> List[TextPair]:
        """
        Stage 2: Merge and deduplicate candidates by document ID.
        
        This stage removes duplicates (same document may appear in multiple probe results)
        and applies document filtering rules. The probe type from Stage 1 is discarded -
        it was just a hint for retrieval, not ground truth.
        
        Args:
            source_chunk_id: Source chunk identifier
            source_text: Source paragraph text
            source_document_id: Source document ID
            candidates: List of candidates from Stage 1
            include_intra_document: Whether to include same-document connections
            include_inter_document: Whether to include cross-document connections
            
        Returns:
            List of unique TextPair objects ready for classification
        """
        seen_chunk_ids: Set[str] = set()
        unique_pairs: List[TextPair] = []
        
        # Track statistics
        stats = {
            "total_candidates": len(candidates),
            "self_matches": 0,
            "duplicates": 0,
            "intra_doc_filtered": 0,
            "inter_doc_filtered": 0,
            "unique_pairs": 0
        }
        
        for candidate in candidates:
            # Skip self-matches
            if candidate.chunk_id == source_chunk_id:
                stats["self_matches"] += 1
                continue
            
            # Skip duplicates (same chunk from different probe searches)
            if candidate.chunk_id in seen_chunk_ids:
                stats["duplicates"] += 1
                continue
            
            # Apply document filtering rules
            is_same_document = candidate.document_id == source_document_id
            
            if is_same_document and not include_intra_document:
                stats["intra_doc_filtered"] += 1
                continue
            
            if not is_same_document and not include_inter_document:
                stats["inter_doc_filtered"] += 1
                continue
            
            # Create unique text pair
            seen_chunk_ids.add(candidate.chunk_id)
            unique_pairs.append(TextPair(
                id=f"{source_chunk_id}_{candidate.chunk_id}",
                text_a=source_text,
                text_b=candidate.text,
                metadata={
                    "source_chunk_id": source_chunk_id,
                    "target_chunk_id": candidate.chunk_id,
                    "source_document_id": source_document_id,
                    "target_document_id": candidate.document_id,
                    "similarity_score": candidate.similarity_score,
                    "probe_hint": candidate.probe_type,  # Just a hint, not ground truth
                    "is_same_document": is_same_document
                }
            ))
            stats["unique_pairs"] += 1
        
        logger.info(
            f"Stage 2 complete: {stats['total_candidates']} candidates -> "
            f"{stats['unique_pairs']} unique pairs "
            f"(filtered: {stats['self_matches']} self, {stats['duplicates']} dup, "
            f"{stats['intra_doc_filtered']} intra, {stats['inter_doc_filtered']} inter)"
        )
        
        return unique_pairs
    
    async def classify_text_pairs(
        self,
        text_pairs: List[TextPair],
        batch_size: Optional[int] = None
    ) -> ClassificationResponse:
        """
        Stage 3: Unified NLI classification with competitive multi_label=False.
        
        This stage sends all unique pairs to ModernBERT for single authoritative
        classification. The model uses competitive classification (multi_label=False)
        to ensure each pair gets exactly one relationship type from the 5 categories.
        
        The "unrelated" category acts as a natural dump for unclear/confusing pairs.
        
        Args:
            text_pairs: List of TextPair objects to classify
            batch_size: Batch size for classification (uses settings.classification_batch_size if None)
            
        Returns:
            ClassificationResponse with results and metadata
            
        Raises:
            DatabaseError: If classification fails
        """
        if not text_pairs:
            return ClassificationResponse(
                results=[],
                processing_time=0.0,
                model_info={}
            )

        # Get batch size from settings if not provided
        if batch_size is None:
            batch_size = 30

        # Send ALL unique pairs to the classification model — no cap here.
        # Edge creation limits are applied downstream in ProbeGraphManager.
        
        # Prepare input for classification service
        classification_input = []
        for pair in text_pairs:
            classification_input.append({
                "id": pair.id,
                "text_a": pair.text_a,
                "text_b": pair.text_b
            })
        
        try:
            # Call classification service with new 4-category schema
            response = self.classification_client.classify_relationships(
                text_pairs=classification_input,
                batch_size=batch_size
            )
            
            # Enrich results with metadata from text pairs
            for i, result in enumerate(response.results):
                if i < len(text_pairs):
                    # Add metadata from the text pair
                    result.metadata = text_pairs[i].metadata
            
            logger.info(
                f"Stage 3 complete: Classified {len(response.results)} text pairs "
                f"(time: {response.processing_time:.2f}s)"
            )
            
            # Log relationship type distribution
            type_counts: Dict[str, int] = {}
            for result in response.results:
                type_counts[result.relation_type] = type_counts.get(result.relation_type, 0) + 1
            
            logger.debug(f"Relationship distribution: {type_counts}")
            
            return response
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise DatabaseError(f"Unified NLI classification failed: {e}") from e
    
    def process_source_paragraph_sync(
        self,
        source_chunk_id: str,
        source_text: str,
        source_document_id: str,
        max_candidates_per_probe: int = 50,
        include_intra_document: bool = True,
        include_inter_document: bool = True
    ) -> ClassificationResponse:
        """
        Synchronous wrapper for process_source_paragraph.
        
        Args:
            source_chunk_id: Unique identifier for the source chunk
            source_text: The source paragraph text
            source_document_id: Document ID of the source chunk
            max_candidates_per_probe: Maximum candidates to retrieve per probe type
            include_intra_document: Whether to include same-document connections
            include_inter_document: Whether to include cross-document connections
            
        Returns:
            ClassificationResponse with results from unified NLI classification
        """
        try:
            # Always create a new event loop for synchronous calls to avoid "Event loop is closed" errors
            # This is necessary because ThreadPoolExecutor may reuse threads with closed loops
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        self.process_source_paragraph(
                            source_chunk_id,
                            source_text,
                            source_document_id,
                            max_candidates_per_probe,
                            include_intra_document,
                            include_inter_document
                        )
                    )
                )
                return future.result()
        except Exception as e:
            logger.error(f"Synchronous pipeline processing failed: {e}")
            raise DatabaseError(f"Pipeline processing failed: {e}") from e


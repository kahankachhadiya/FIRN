"""
Probe Graph Manager for creating semantic relationship edges.

This module implements the ProbeGraphManager class that creates probe edges
(dependency, expansion, contradiction, unrelated) between chunks using the
new hybrid pipeline processor with 3-stage workflow:
1. Multi-Probe Retrieval: Execute 4 probe-specific vector searches
2. Aggregation & Deduplication: Merge and deduplicate by document ID
3. Unified NLI Classification: Single authoritative classification with ModernBERT
"""

from typing import Dict, Optional
import time

from db.falkor_client import ensure_probe_graph_exists, add_probe_edge
from db.probe_graph_client import should_create_graph_edge
from ingestion.hybrid_pipeline_processor import HybridPipelineProcessor
from ingestion.probe_config import probe_config_manager, ProbeConfig
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProbeGraphManager:
    """
    Manages probe edge creation for semantic relationships between chunks.
    
    Creates edges of type dependency, expansion, contradiction using the new
    hybrid pipeline processor. "unrelated" classifications skip edge creation.
    
    The hybrid pipeline implements a 3-stage workflow:
    1. Multi-Probe Retrieval: Execute 4 probe-specific vector searches
    2. Aggregation & Deduplication: Merge and deduplicate by document ID
    3. Unified NLI Classification: Single authoritative classification
    """
    
    def __init__(self):
        """
        Initialize the ProbeGraphManager with the hybrid pipeline processor.
        """
        self.hybrid_pipeline = HybridPipelineProcessor()
        self.probe_configs = probe_config_manager.get_all_configs()
        
        # Ensure probe graph exists
        ensure_probe_graph_exists()
        
        logger.info(
            "Initialized ProbeGraphManager with hybrid pipeline processor "
            "(4-category schema: dependency, expansion, contradiction, unrelated)"
        )
    
    def build_probe_edges(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        chunk_text: Optional[str] = None,
        include_intra_document: bool = True,
        include_inter_document: bool = True,
        max_candidates_per_probe: int = 50
    ) -> Dict[str, int]:
        """
        Create probe edges using the hybrid pipeline processor with 3-stage workflow.
        
        Stage 1: Multi-Probe Retrieval - Execute 4 probe-specific vector searches
        Stage 2: Aggregation & Deduplication - Merge and deduplicate by document ID
        Stage 3: Unified NLI Classification - Single authoritative classification
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk (used for vector search)
            document_id: Document ID to control intra/inter document connections
            chunk_text: Full chunk text (used for classification, falls back to summary)
            include_intra_document: Whether to include connections within the same document
            include_inter_document: Whether to include connections between different documents
            max_candidates_per_probe: Maximum candidates to retrieve per probe type
            
        Returns:
            Dict mapping relationship type to number of edges created
            (dependency, expansion, contradiction, unrelated)
        """
        if not summary or not summary.strip():
            logger.warning(f"Empty summary for chunk {chunk_id}, skipping probe edge creation")
            return {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}
        
        if not include_intra_document and not include_inter_document:
            logger.warning(f"Both intra and inter document connections disabled for {chunk_id}")
            return {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}
        
        # Use chunk_text if available, otherwise fall back to summary
        source_text = chunk_text if chunk_text and chunk_text.strip() else summary
        
        logger.debug(
            f"Creating probe edges for chunk {chunk_id} using hybrid pipeline "
            f"(intra_doc: {include_intra_document}, inter_doc: {include_inter_document})"
        )
        
        # Use the hybrid pipeline processor
        try:
            start_time = time.time()
            
            # Process through the 3-stage hybrid pipeline
            classification_response = self.hybrid_pipeline.process_source_paragraph_sync(
                source_chunk_id=chunk_id,
                source_text=source_text,
                source_document_id=document_id,
                max_candidates_per_probe=max_candidates_per_probe,
                include_intra_document=include_intra_document,
                include_inter_document=include_inter_document
            )
            
            # Create graph edges from classification results
            edge_counts = {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}

            # Per-relation-type top-N limits — select by highest NLI confidence score
            from config.settings import settings
            edge_limits = {
                "dependency": settings.graph_b_top_n_dependency_edges,
                "expansion":  settings.graph_b_top_n_expansion_edges,
                "contradiction": settings.graph_b_top_n_contradiction_edges,
            }
            edges_created_per_type: Dict[str, int] = {k: 0 for k in edge_limits}

            # Mapping from classification types to FalkorDB edge types
            classification_to_edge_type = {
                'dependency': 'DEPENDS_ON',
                'expansion': 'ELABORATES',
                'contradiction': 'CONTRADICTS'
            }

            # Sort by confidence descending so the cap selects the highest-scoring edges
            sorted_results = sorted(
                classification_response.results,
                key=lambda r: r.confidence_score,
                reverse=True
            )

            for result in sorted_results:
                relation_type = result.relation_type
                confidence = result.confidence_score

                # Track all relationship types (including unrelated)
                edge_counts[relation_type] = edge_counts.get(relation_type, 0) + 1

                # Only create graph edges for non-unrelated classifications
                if should_create_graph_edge(relation_type):
                    # Enforce per-type top-N limit
                    if edges_created_per_type.get(relation_type, 0) >= edge_limits.get(relation_type, 999):
                        logger.debug(
                            "Skipping %s edge for %s — limit of %d reached",
                            relation_type, chunk_id, edge_limits.get(relation_type)
                        )
                        continue

                    try:
                        target_chunk_id = result.metadata.get("target_chunk_id") if result.metadata else None

                        if not target_chunk_id:
                            logger.warning(f"Missing target_chunk_id in result metadata for {chunk_id}")
                            continue

                        edge_type = classification_to_edge_type.get(relation_type, relation_type.upper())

                        add_probe_edge(
                            source_chunk_id=chunk_id,
                            target_chunk_id=target_chunk_id,
                            probe_type=edge_type,
                            confidence=confidence
                        )

                        edges_created_per_type[relation_type] = edges_created_per_type.get(relation_type, 0) + 1

                        logger.debug(
                            f"Created {edge_type} edge: {chunk_id} -> {target_chunk_id} "
                            f"(confidence: {confidence:.3f})"
                        )

                    except Exception as e:
                        logger.warning(f"Failed to create {relation_type} edge for {chunk_id}: {e}")
                        edge_counts[relation_type] -= 1
            
            pipeline_time = time.time() - start_time
            total_classifications = sum(edge_counts.values())
            edges_created = sum(edges_created_per_type.values())

            logger.info(
                f"Hybrid pipeline completed for chunk {chunk_id}: "
                f"classified=({', '.join(f'{k}={v}' for k, v in edge_counts.items())}) "
                f"edges=({', '.join(f'{k}={v}' for k, v in edges_created_per_type.items())}) "
                f"(total classifications: {total_classifications}, edges created: {edges_created}, "
                f"time: {pipeline_time:.2f}s)"
            )

            return edges_created_per_type
            
        except Exception as e:
            logger.error(f"Error creating probe edges with hybrid pipeline for chunk {chunk_id}: {e}")
            # Return zero counts for all relationship types
            return {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}
    
    async def build_probe_edges_async(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        chunk_text: Optional[str] = None,
        include_intra_document: bool = True,
        include_inter_document: bool = True,
        max_candidates_per_probe: int = 50
    ) -> Dict[str, int]:
        """
        Async version of build_probe_edges for parallel processing.
        
        Uses the hybrid pipeline processor's native async support for better performance.
        
        Args:
            chunk_id: Unique chunk identifier
            summary: Summary text for the chunk
            document_id: Document ID to control intra/inter document connections
            chunk_text: Full chunk text for classification
            include_intra_document: Whether to include connections within the same document
            include_inter_document: Whether to include connections between different documents
            max_candidates_per_probe: Maximum candidates to retrieve per probe type
            
        Returns:
            Dict mapping relationship type to number of edges created
        """
        if not summary or not summary.strip():
            logger.warning(f"Empty summary for chunk {chunk_id}, skipping probe edge creation")
            return {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}
        
        if not include_intra_document and not include_inter_document:
            logger.warning(f"Both intra and inter document connections disabled for {chunk_id}")
            return {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}
        
        # Use chunk_text if available, otherwise fall back to summary
        source_text = chunk_text if chunk_text and chunk_text.strip() else summary
        
        try:
            start_time = time.time()
            
            # Use the hybrid pipeline's native async method
            classification_response = await self.hybrid_pipeline.process_source_paragraph(
                source_chunk_id=chunk_id,
                source_text=source_text,
                source_document_id=document_id,
                max_candidates_per_probe=max_candidates_per_probe,
                include_intra_document=include_intra_document,
                include_inter_document=include_inter_document
            )
            
            # Create graph edges from classification results
            edge_counts = {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}

            # Per-relation-type top-N limits — select by highest NLI confidence score
            from config.settings import settings
            edge_limits = {
                "dependency": settings.graph_b_top_n_dependency_edges,
                "expansion":  settings.graph_b_top_n_expansion_edges,
                "contradiction": settings.graph_b_top_n_contradiction_edges,
            }
            edges_created_per_type: Dict[str, int] = {k: 0 for k in edge_limits}

            # Mapping from classification types to FalkorDB edge types
            classification_to_edge_type = {
                'dependency': 'DEPENDS_ON',
                'expansion': 'ELABORATES',
                'contradiction': 'CONTRADICTS'
            }

            # Sort by confidence descending so the cap selects the highest-scoring edges
            sorted_results = sorted(
                classification_response.results,
                key=lambda r: r.confidence_score,
                reverse=True
            )

            for result in sorted_results:
                relation_type = result.relation_type
                confidence = result.confidence_score

                edge_counts[relation_type] = edge_counts.get(relation_type, 0) + 1

                if should_create_graph_edge(relation_type):
                    if edges_created_per_type.get(relation_type, 0) >= edge_limits.get(relation_type, 999):
                        logger.debug(
                            "Skipping %s edge for %s — limit of %d reached",
                            relation_type, chunk_id, edge_limits.get(relation_type)
                        )
                        continue

                    try:
                        target_chunk_id = result.metadata.get("target_chunk_id") if result.metadata else None

                        if not target_chunk_id:
                            logger.warning(f"Missing target_chunk_id in result metadata for {chunk_id}")
                            continue

                        edge_type = classification_to_edge_type.get(relation_type, relation_type.upper())

                        add_probe_edge(
                            source_chunk_id=chunk_id,
                            target_chunk_id=target_chunk_id,
                            probe_type=edge_type,
                            confidence=confidence
                        )

                        edges_created_per_type[relation_type] = edges_created_per_type.get(relation_type, 0) + 1

                        logger.debug(
                            f"Created {edge_type} edge: {chunk_id} -> {target_chunk_id} "
                            f"(confidence: {confidence:.3f})"
                        )

                    except Exception as e:
                        logger.warning(f"Failed to create {relation_type} edge for {chunk_id}: {e}")
                        edge_counts[relation_type] -= 1
            
            pipeline_time = time.time() - start_time
            total_classifications = sum(edge_counts.values())
            edges_created = sum(edges_created_per_type.values())

            logger.info(
                f"Async hybrid pipeline completed for chunk {chunk_id}: "
                f"classified=({', '.join(f'{k}={v}' for k, v in edge_counts.items())}) "
                f"edges=({', '.join(f'{k}={v}' for k, v in edges_created_per_type.items())}) "
                f"(total classifications: {total_classifications}, edges created: {edges_created}, "
                f"time: {pipeline_time:.2f}s)"
            )

            return edges_created_per_type

        except Exception as e:
            logger.error(f"Error in async probe edge creation for chunk {chunk_id}: {e}")
            return {"dependency": 0, "expansion": 0, "contradiction": 0, "unrelated": 0}
    
    def get_probe_query_template(self, probe_type: str) -> str:
        """
        Get the query template for a specific probe type.
        
        Args:
            probe_type: The probe type name
            
        Returns:
            Query suffix for the probe type
            
        Raises:
            ValueError: If probe_type is not valid
        """
        if not probe_config_manager.validate_probe_type(probe_type):
            valid_types = probe_config_manager.get_valid_probe_types()
            raise ValueError(f"Invalid probe type: {probe_type}. Must be one of {valid_types}")
        
        config = probe_config_manager.get_config(probe_type)
        return config.query_suffix
    
    def get_probe_config(self, probe_type: str) -> ProbeConfig:
        """
        Get the configuration for a specific probe type.
        
        Args:
            probe_type: The probe type name
            
        Returns:
            ProbeConfig for the specified type
            
        Raises:
            ValueError: If probe_type is not valid
        """
        return probe_config_manager.get_config(probe_type)
    
    def get_all_probe_configs(self) -> Dict[str, ProbeConfig]:
        """
        Get all probe configurations.
        
        Returns:
            Dictionary mapping probe type names to ProbeConfig objects
        """
        return self.probe_configs.copy()

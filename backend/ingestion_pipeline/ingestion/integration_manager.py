"""
Integration manager for layered graph RAG enhancements.

This module provides the main integration point for wiring together
all new components.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ingestion.parallel_edge_creator import ParallelEdgeCreator
from ingestion.probe_graph_manager import ProbeGraphManager
from db.falkor_client import ensure_graph_exists, ensure_probe_graph_exists
from utils.logging_utils import get_logger
from utils.errors import DatabaseError


logger = get_logger(__name__)


@dataclass
class IntegrationComponents:
    """Container for all integration components."""
    classification_client: Optional[Any] = None  # Zero-shot classification client
    parallel_edge_creator: Optional[ParallelEdgeCreator] = None
    probe_graph_manager: Optional[ProbeGraphManager] = None

    def is_fully_initialized(self) -> bool:
        """Check if all components are initialized."""
        return all([
            self.classification_client is not None,
            self.parallel_edge_creator is not None,
            self.probe_graph_manager is not None
        ])


class IntegrationManager:
    """Coordinates ingestion pipeline components for layered graph RAG."""

    def __init__(self):
        self.components = IntegrationComponents()
        self.initialization_errors: List[str] = []
        self.is_initialized = False
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all components."""
        logger.info("Initializing layered graph RAG integration components...")

        try:
            self._initialize_databases()
            self._initialize_classification_client()
            self._initialize_probe_graph_manager()
            self._initialize_parallel_edge_creator()

            self.is_initialized = True
            logger.info("Integration components initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize integration components: {e}"
            logger.error(error_msg, exc_info=True)
            self.initialization_errors.append(error_msg)

    def _initialize_databases(self) -> None:
        """Initialize database collections."""
        try:
            ensure_graph_exists()
            ensure_probe_graph_exists()

        except Exception as e:
            error_msg = f"Database initialization failed: {e}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)
            raise DatabaseError(error_msg) from e

    def _initialize_classification_client(self) -> None:
        """Initialize zero-shot classification client."""
        try:
            from external_clients.classification_client import get_classification_client
            self.components.classification_client = get_classification_client()
            logger.info("Zero-shot classification client initialized")

        except Exception as e:
            error_msg = f"Classification client initialization failed: {e}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)

    def _initialize_probe_graph_manager(self) -> None:
        """Initialize probe graph manager."""
        try:
            self.components.probe_graph_manager = ProbeGraphManager()
            logger.info("Probe graph manager initialized")

        except Exception as e:
            error_msg = f"Probe graph manager initialization failed: {e}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)

    def _initialize_parallel_edge_creator(self) -> None:
        """Initialize parallel edge creator."""
        try:
            self.components.parallel_edge_creator = ParallelEdgeCreator()
            logger.info("Parallel edge creator initialized")

        except Exception as e:
            error_msg = f"Parallel edge creator initialization failed: {e}"
            logger.error(error_msg)
            self.initialization_errors.append(error_msg)

    def process_ingestion(
        self,
        chunk_id: str,
        summary: str,
        document_id: str,
        metadata: Dict[str, Any],
        chunk_text: str
    ) -> Dict[str, Any]:
        """
        Process ingestion with enhanced edge creation.

        Args:
            chunk_id: Unique chunk identifier
            summary: Chunk summary for similarity search
            document_id: Document identifier
            metadata: Chunk metadata
            chunk_text: Full chunk text for probe analysis

        Returns:
            Dictionary with edge creation results
        """
        if not self.components.parallel_edge_creator:
            logger.warning("Parallel edge creator not initialized, skipping edge creation")
            return {
                "similar_edges": 0,
                "probe_edges": {},
                "total_edges": 0,
                "error": "Component not initialized"
            }

        try:
            return self.components.parallel_edge_creator.create_edges_parallel(
                chunk_id=chunk_id,
                summary=summary,
                document_id=document_id,
                metadata=metadata,
                chunk_text=chunk_text
            )

        except Exception as e:
            logger.error(f"Enhanced edge creation failed: {e}")
            return {
                "similar_edges": 0,
                "probe_edges": {},
                "total_edges": 0,
                "fallback_used": True,
                "error": f"Enhanced creation failed and legacy fallback removed: {e}"
            }


# Global integration manager instance
_integration_manager: Optional[IntegrationManager] = None


def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager singleton."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
    return _integration_manager

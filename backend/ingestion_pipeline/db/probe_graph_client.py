"""
Probe Graph Client for Graph B (typed semantic relationships).

This module provides utilities for the probe graph ingestion pipeline.
The retrieval engine queries Graph B directly via falkor_client.get_probe_neighbors.

Graph B Architecture (in Unified Knowledge Graph):
- Stores typed relationship edges: dependency, expansion, contradiction
- Edges are created using NLI (Natural Language Inference) classification
- "unrelated" classifications do not create graph edges by design
- Stored unidirectionally but queried bidirectionally

This is part of the unified Knowledge Graph architecture where Graph B provides typed
relationships and Graph A provides semantic relations via reranker scoring.
"""

from utils.logging_utils import get_logger

logger = get_logger(__name__)


# Mapping of probe types to whether they should create graph edges.
# "unrelated" classifications skip graph edge creation.
GRAPH_EDGE_TYPES = {
    'dependency': True,
    'expansion': True,
    'contradiction': True,
    'unrelated': False,
}


def should_create_graph_edge(probe_type: str) -> bool:
    """
    Determine if a probe type should create a graph edge.

    Returns False for "unrelated" classifications, True for all others.

    Example:
        >>> should_create_graph_edge('dependency')
        True
        >>> should_create_graph_edge('unrelated')
        False
    """
    return GRAPH_EDGE_TYPES.get(probe_type, False)

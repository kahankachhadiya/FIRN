"""
Probe configuration management for semantic relationship edges.

This module defines the configuration for the 3 active probe types used in the
Probe Graph database to create semantic relationships between chunks.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ProbeConfig:
    """Configuration for a probe type."""
    name: str
    query_suffix: str  # Text to append to anchor text for vector search


class ProbeConfigManager:
    """Manages probe configurations for all 3 active probe types."""

    def __init__(self):
        self._configs = {
            "CONTRADICTS": ProbeConfig(
                name="CONTRADICTS",
                query_suffix="contradicts opposes disputes challenges refutes argues against this claim",
            ),
            "ELABORATES": ProbeConfig(
                name="ELABORATES",
                query_suffix="provides detailed explanation technical details deeper analysis of this topic",
            ),
            "DEPENDS_ON": ProbeConfig(
                name="DEPENDS_ON",
                query_suffix="prerequisite requirement foundation necessary condition needed for this",
            ),
        }

    def get_config(self, probe_type: str) -> ProbeConfig:
        if probe_type not in self._configs:
            raise ValueError(f"Invalid probe type: {probe_type}. Must be one of {list(self._configs.keys())}")
        return self._configs[probe_type]

    def get_all_configs(self) -> Dict[str, ProbeConfig]:
        return self._configs.copy()

    def get_valid_probe_types(self) -> list[str]:
        return list(self._configs.keys())

    def validate_probe_type(self, probe_type: str) -> bool:
        return probe_type in self._configs


# Global probe config manager instance
probe_config_manager = ProbeConfigManager()

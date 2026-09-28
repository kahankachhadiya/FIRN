"""Configuration module using Pydantic BaseSettings for environment-based configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, field_validator

_ENV_FILE = str(Path(__file__).parent.parent / ".env")


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    This class manages all configuration for the multi-database RAG backend,
    including connection parameters for Redis, Qdrant, and Falkor DB, as well
    as embedding model configuration and system parameters.
    """
    
    # Elasticsearch Configuration
    elasticsearch_host: str = Field(..., description="Elasticsearch server host")
    elasticsearch_port: int = Field(..., description="Elasticsearch server port")
    elasticsearch_chunk_index: str = Field(..., description="Elasticsearch index for paragraph storage")
    elasticsearch_metadata_index: str = Field(..., description="Elasticsearch index for document metadata")
    
    # Qdrant Configuration
    qdrant_host: str = Field(..., description="Qdrant server host")
    qdrant_port: int = Field(..., description="Qdrant server port")
    qdrant_main_collection: str = Field(..., description="Qdrant collection for paragraph embeddings")
    
    # Falkor DB Configuration
    falkor_host: str = Field(..., description="Falkor DB server host")
    falkor_port: int = Field(..., description="Falkor DB server port")
    falkor_username: str = Field(..., description="Falkor DB username")
    falkor_password: str = Field(..., description="Falkor DB password")
    
    # Unified Knowledge Graph Configuration
    falkor_knowledge_graph_name: str = Field(..., description="Unified Knowledge Graph name in FalkorDB")
    falkor_vertex_collection: str = Field(..., description="Falkor DB vertex collection name")
    
    # Probe Graph Configuration (Graph B)
    
    # Graph A Configuration
    graph_a_top_n: int = Field(..., description="Number of top chunks to retrieve from Graph A")
    graph_a_reranker_threshold: float = Field(..., description="Minimum reranker score for Graph A edges")
    
    # Ingestion Pipeline Configuration
    vector_search_top_k_per_probe: int = Field(..., description="Maximum candidates to retrieve per probe from vector search")
    graph_a_reranker_candidates: int = Field(..., description="Number of candidates for Graph A reranker during ingestion")
    
    # Unified Knowledge Graph Service Configuration
    unified_kg_service_url: str = Field(..., description="Unified Knowledge Graph service endpoint URL (replaces separate embedding, classification, and reranker services)")
    
    # Service Communication Configuration
    service_retry_attempts: int = Field(..., description="Number of retry attempts for service requests")
    
    # Backward compatibility properties for existing code
    @property
    def embedding_service_url(self) -> str:
        """Backward compatibility: returns unified service URL"""
        return self.unified_kg_service_url
    
    @property
    def classification_service_url(self) -> str:
        """Backward compatibility: returns unified service URL"""
        return self.unified_kg_service_url
    
    @property
    def reranker_service_url(self) -> str:
        """Backward compatibility: returns unified service URL"""
        return self.unified_kg_service_url
    
    # LLM Configuration — Vision endpoint (image analysis, OCR, document figure understanding)
    llm_vision_base_url: str = Field(..., description="OpenAI-compatible base URL for the vision model, e.g. http://host:1234/v1")
    llm_vision_api_key: str = Field(default="", description="API key for vision model; empty string for local providers")
    llm_vision_model: str = Field(..., description="Vision model name")
    llm_vision_max_tokens: int = Field(..., description="Max tokens for vision completions")
    llm_vision_temperature: float = Field(..., description="Sampling temperature for vision")

    # LLM Configuration — Metadata extraction endpoint
    llm_metadata_base_url: str = Field(..., description="OpenAI-compatible base URL for the metadata extraction model")
    llm_metadata_api_key: str = Field(default="", description="API key for metadata model; empty string for local providers")
    llm_metadata_model: str = Field(..., description="Metadata extraction model name")
    llm_metadata_max_tokens: int = Field(default=1000, description="Max tokens for metadata completions")
    llm_metadata_temperature: float = Field(default=0.1, description="Sampling temperature for metadata extraction")

    # Classification Confidence Threshold
    # Pairs returned by the Fast Path below this score are dropped entirely.
    confidence_threshold: float = Field(..., description="Minimum confidence score for classification results (pairs below are dropped)")

    # Graph B — top-N edges per relation type, selected by highest NLI confidence score
    graph_b_top_n_dependency_edges: int = Field(..., description="Top N dependency edges per chunk by NLI confidence")
    graph_b_top_n_expansion_edges: int = Field(..., description="Top N expansion edges per chunk by NLI confidence")
    graph_b_top_n_contradiction_edges: int = Field(..., description="Top N contradiction edges per chunk by NLI confidence")

    # LM Studio Classification Model (Slow Path for GatingRouter)
    # Only used when CLASSIFICATION_SLOW_PATH_ENABLED=true AND CONFIDENCE_THRESHOLD > 0
    classification_slow_path_enabled: bool = Field(..., description="Master on/off switch for the GatingRouter Slow Path LLM fallback")
    llm_classification_base_url: str = Field(..., description="OpenAI-compatible base URL for the classification slow path model")
    llm_classification_api_key: str = Field(default="", description="API key for classification model; empty string for local providers")
    llm_classification_model: str = Field(..., description="Model name for classification slow path")
    llm_classification_max_tokens: int = Field(default=512, description="Max tokens for classification completions")
    llm_classification_temperature: float = Field(default=0.05, description="Sampling temperature for classification")

    model_config = ConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables not defined in this class
    )
    
    # Field Validators for New Parameters
    
    @field_validator('graph_a_top_n')
    @classmethod
    def validate_graph_a_top_n(cls, v: int) -> int:
        """Validate graph_a_top_n is positive."""
        if v <= 0:
            raise ValueError(f"graph_a_top_n must be positive, got {v}")
        if v > 100:
            raise ValueError(f"graph_a_top_n must be <= 100, got {v}")
        return v
    
    @field_validator('graph_a_reranker_threshold')
    @classmethod
    def validate_graph_a_reranker_threshold(cls, v: float) -> float:
        """Validate graph_a_reranker_threshold is in valid range [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"graph_a_reranker_threshold must be in range [0.0, 1.0], got {v}")
        return v
    
    @field_validator('graph_a_reranker_candidates')
    @classmethod
    def validate_graph_a_reranker_candidates(cls, v: int) -> int:
        """Validate graph_a_reranker_candidates is positive."""
        if v <= 0:
            raise ValueError(f"graph_a_reranker_candidates must be positive, got {v}")
        if v > 200:
            raise ValueError(f"graph_a_reranker_candidates must be <= 200, got {v}")
        return v
    
    @field_validator('vector_search_top_k_per_probe')
    @classmethod
    def validate_vector_search_top_k(cls, v: int) -> int:
        """Validate vector_search_top_k_per_probe is positive."""
        if v <= 0:
            raise ValueError(f"vector_search_top_k_per_probe must be positive, got {v}")
        if v > 200:
            raise ValueError(f"vector_search_top_k_per_probe must be <= 200, got {v}")
        return v
    
    @field_validator('unified_kg_service_url')
    @classmethod
    def validate_unified_kg_service_url(cls, v: str) -> str:
        """Validate unified KG service URL format."""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:'
            r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain with TLD
            r'[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?|'  # hostname without TLD
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # IP address
            r')'
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)?$', re.IGNORECASE)
        
        if not url_pattern.match(v):
            raise ValueError(f"unified_kg_service_url must be a valid URL, got {v}")
        return v
    
    @field_validator('elasticsearch_host')
    @classmethod
    def validate_elasticsearch_host(cls, v: str) -> str:
        """Validate Elasticsearch host is not empty."""
        if not v or not v.strip():
            raise ValueError("elasticsearch_host must not be empty")
        return v.strip()
    
    @field_validator('elasticsearch_port')
    @classmethod
    def validate_elasticsearch_port(cls, v: int) -> int:
        """Validate Elasticsearch port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"elasticsearch_port must be in range [1, 65535], got {v}")
        return v
    
    @field_validator('elasticsearch_chunk_index')
    @classmethod
    def validate_elasticsearch_chunk_index(cls, v: str) -> str:
        """Validate Elasticsearch chunk index name is not empty."""
        if not v or not v.strip():
            raise ValueError("elasticsearch_chunk_index must not be empty")
        return v.strip()
    
    @field_validator('elasticsearch_metadata_index')
    @classmethod
    def validate_elasticsearch_metadata_index(cls, v: str) -> str:
        """Validate Elasticsearch metadata index name is not empty."""
        if not v or not v.strip():
            raise ValueError("elasticsearch_metadata_index must not be empty")
        return v.strip()
    
    def validate_service_endpoints(self) -> bool:
        """
        Validate that service endpoints are properly configured and accessible.

        Returns:
            bool: True if all service endpoints are valid, False otherwise
        """
        import re

        # URL pattern validation (same as field validator)
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:'
            r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain with TLD
            r'[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?|'  # hostname without TLD
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # IP address
            r')'
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)?$', re.IGNORECASE)

        # Validate unified KG service URL
        if not url_pattern.match(self.unified_kg_service_url):
            return False

        return True

    
    def validate_all_configurations(self) -> tuple[bool, list[str]]:
        """
        Validate all configuration parameters.
        
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.validate_service_endpoints():
            errors.append("Invalid service endpoint URLs")
            
        return len(errors) == 0, errors


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings: The global settings instance
    """
    return settings

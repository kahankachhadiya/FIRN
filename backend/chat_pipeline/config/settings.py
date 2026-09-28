"""Configuration module using Pydantic BaseSettings for environment-based configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, field_validator

_ENV_FILE = str(Path(__file__).parent.parent / ".env")


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
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

    # Graph A Configuration
    graph_a_top_n: int = Field(..., description="Number of top chunks to retrieve from Graph A")

    # Probe Graph Configuration (Graph B) — per-probe top-N
    graph_b_contradicts_top_n: int = Field(..., description="Top-N CONTRADICTS edges to retrieve per anchor chunk")
    graph_b_elaborates_top_n: int = Field(..., description="Top-N ELABORATES edges to retrieve per anchor chunk")
    graph_b_depends_on_top_n: int = Field(..., description="Top-N DEPENDS_ON edges to retrieve per anchor chunk")

    # LLM Configuration — Chat endpoint
    llm_chat_base_url: str = Field(..., description="OpenAI-compatible base URL for the chat model")
    llm_chat_api_key: str = Field(default="", description="API key for chat model; empty string for local providers")
    llm_chat_model: str = Field(..., description="Chat model name")
    llm_chat_max_tokens: int = Field(..., description="Max tokens for chat completions")
    llm_chat_temperature: float = Field(..., description="Sampling temperature for chat")

    # Tool Calling Configuration
    max_retrieval_rounds: int = Field(..., description="Maximum number of retrieval rounds per query")
    tool_execution_timeout: int = Field(..., description="Tool execution timeout in seconds")

    # Chat History Configuration
    chat_history_dir: str = Field(..., description="Directory for chat history JSON files")

    # File Access Security Configuration
    allowed_file_extensions: list[str] = Field(..., description="List of allowed file extensions for file access")

    # API Configuration
    api_host: str = Field(..., description="API server host")
    api_port: int = Field(..., description="API server port")

    model_config = ConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator('graph_a_top_n')
    @classmethod
    def validate_graph_a_top_n(cls, v: int) -> int:
        if v <= 0 or v > 100:
            raise ValueError(f"graph_a_top_n must be in (0, 100], got {v}")
        return v
    @field_validator('elasticsearch_host')
    @classmethod
    def validate_elasticsearch_host(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("elasticsearch_host must not be empty")
        return v.strip()

    @field_validator('elasticsearch_port')
    @classmethod
    def validate_elasticsearch_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"elasticsearch_port must be in [1, 65535], got {v}")
        return v

    @field_validator('elasticsearch_chunk_index')
    @classmethod
    def validate_elasticsearch_chunk_index(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("elasticsearch_chunk_index must not be empty")
        return v.strip()

    @field_validator('elasticsearch_metadata_index')
    @classmethod
    def validate_elasticsearch_metadata_index(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("elasticsearch_metadata_index must not be empty")
        return v.strip()


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    return settings

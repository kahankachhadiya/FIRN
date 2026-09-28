"""
Production configuration module with environment-based configuration loading.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    _pipeline_env = Path(__file__).parent.parent / '.env'
    if _pipeline_env.exists():
        load_dotenv(_pipeline_env)
except ImportError:
    pass


@dataclass
class DatabaseConfig:
    """Minimal Qdrant connection configuration."""
    host: str
    port: int
    password: Optional[str] = None
    connection_timeout: int = 10
    prefer_grpc: bool = False
    https: bool = False
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    indexing_threshold: int = 10000


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the layered graph retrieval pipeline."""

    vector_search_top_k: int = 10
    keyword_search_top_k: int = 10
    vector_top_chunks: int = 5
    keyword_top_chunks: int = 5
    reranked_top_chunks: int = 8

    graph_a_top_n: int = 10

    # Per-probe top-N for Graph B
    graph_b_contradicts_top_n: int = 5
    graph_b_elaborates_top_n: int = 5
    graph_b_depends_on_top_n: int = 5

    context_expansion_enabled: bool = True
    context_adjacent_chunks: int = 1

    max_final_chunks: int = 20


@dataclass
class LLMConfig:
    """LLM configuration for query generation and response formatting."""

    chat_url: str = "http://127.0.0.1:1234"
    chat_model: str = "openai/gpt-oss-20b"
    chat_context_window: int = 20000
    chat_max_tokens: int = 4096
    chat_temperature: float = 0.1
    chat_top_p: float = 0.9
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0

    query_generation_timeout: int = 10
    response_generation_timeout: int = 30
    max_retries: int = 3

    query_generation_system_prompt: str = ""
    rag_response_system_prompt: str = ""


@dataclass
class ProductionConfig:
    """Production configuration loaded from environment variables."""

    qdrant_config: DatabaseConfig
    retrieval_config: RetrievalPipelineConfig
    llm_config: LLMConfig

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @staticmethod
    def _load_multiline_env_var(var_name: str) -> str:
        value = os.getenv(var_name, '')
        if value:
            value = value.replace('\\n', '\n')
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
        return value

    @staticmethod
    def _get_required_env(var_name: str, error_message: str) -> str:
        value = os.getenv(var_name)
        if value is None or value.strip() == '':
            raise EnvironmentError(
                f"Required environment variable '{var_name}' is not set. {error_message}"
            )
        return value.strip()

    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        logger = logging.getLogger(__name__)

        def _i(key, default): return int(os.getenv(key, str(default)))
        def _f(key, default): return float(os.getenv(key, str(default)))
        def _b(key, default): return os.getenv(key, str(default)).lower() == 'true'
        def _s(key, default=''): return os.getenv(key, default)

        qdrant_config = DatabaseConfig(
            host=_s('QDRANT_HOST', 'localhost'),
            port=_i('QDRANT_PORT', 6333),
            password=os.getenv('QDRANT_API_KEY') or None,
            connection_timeout=_i('QDRANT_TIMEOUT', 10),
            prefer_grpc=_b('QDRANT_PREFER_GRPC', False),
            https=_b('QDRANT_HTTPS', False),
            hnsw_m=_i('QDRANT_HNSW_M', 16),
            hnsw_ef_construct=_i('QDRANT_HNSW_EF_CONSTRUCT', 100),
            indexing_threshold=_i('QDRANT_INDEXING_THRESHOLD', 10000),
        )

        retrieval_config = RetrievalPipelineConfig(
            vector_search_top_k=_i('VECTOR_SEARCH_TOP_K', 12),
            keyword_search_top_k=_i('KEYWORD_SEARCH_TOP_K', 12),
            vector_top_chunks=_i('VECTOR_TOP_CHUNKS', 6),
            keyword_top_chunks=_i('KEYWORD_TOP_CHUNKS', 6),
            reranked_top_chunks=_i('RERANKED_TOP_CHUNKS', 10),
            graph_a_top_n=_i('GRAPH_A_TOP_N', 10),
            graph_b_contradicts_top_n=_i('GRAPH_B_CONTRADICTS_TOP_N', 5),
            graph_b_elaborates_top_n=_i('GRAPH_B_ELABORATES_TOP_N', 5),
            graph_b_depends_on_top_n=_i('GRAPH_B_DEPENDS_ON_TOP_N', 5),
            context_expansion_enabled=_b('CONTEXT_EXPANSION_ENABLED', True),
            context_adjacent_chunks=_i('CONTEXT_ADJACENT_CHUNKS', 2),
            max_final_chunks=_i('MAX_FINAL_CHUNKS', 25),
        )

        query_generation_prompt = cls._load_multiline_env_var('QUERY_GENERATION_SYSTEM_PROMPT')
        rag_response_prompt = cls._load_multiline_env_var('RAG_RESPONSE_SYSTEM_PROMPT')

        llm_config = LLMConfig(
            chat_url=_s('LLM_CHAT_BASE_URL', _s('LM_STUDIO_CHAT_URL', 'http://localhost:1234/v1')),
            chat_model=_s('LLM_CHAT_MODEL', _s('LM_STUDIO_CHAT_MODEL', 'local-model')),
            chat_context_window=_i('LM_STUDIO_CHAT_CONTEXT_WINDOW', 20000),
            chat_max_tokens=_i('LLM_CHAT_MAX_TOKENS', _i('LM_STUDIO_CHAT_MAX_TOKENS', 6144)),
            chat_temperature=_f('LLM_CHAT_TEMPERATURE', _f('LM_STUDIO_CHAT_TEMPERATURE', 0.05)),
            chat_top_p=_f('LM_STUDIO_CHAT_TOP_P', 0.9),
            chat_frequency_penalty=_f('LM_STUDIO_CHAT_FREQUENCY_PENALTY', 0.0),
            chat_presence_penalty=_f('LM_STUDIO_CHAT_PRESENCE_PENALTY', 0.0),
            query_generation_timeout=_i('LLM_QUERY_GENERATION_TIMEOUT', 999),
            response_generation_timeout=_i('LLM_RESPONSE_GENERATION_TIMEOUT', 999),
            max_retries=_i('LLM_MAX_RETRIES', 2),
            query_generation_system_prompt=query_generation_prompt,
            rag_response_system_prompt=rag_response_prompt,
        )

        config = cls(
            qdrant_config=qdrant_config,
            retrieval_config=retrieval_config,
            llm_config=llm_config,
            api_host=_s('API_HOST', 'localhost'),
            api_port=_i('API_PORT', 8000),
        )

        config.validate()
        logger.info("Production configuration loaded successfully")
        return config

    def validate(self) -> None:
        errors = []
        warnings = []
        logger = logging.getLogger(__name__)

        if self.retrieval_config.vector_search_top_k <= 0:
            errors.append(f"vector_search_top_k must be positive, got: {self.retrieval_config.vector_search_top_k}")
        if self.retrieval_config.keyword_search_top_k <= 0:
            errors.append(f"keyword_search_top_k must be positive, got: {self.retrieval_config.keyword_search_top_k}")
        if self.retrieval_config.vector_top_chunks > self.retrieval_config.vector_search_top_k:
            warnings.append(f"vector_top_chunks exceeds vector_search_top_k")
        if self.retrieval_config.keyword_top_chunks > self.retrieval_config.keyword_search_top_k:
            warnings.append(f"keyword_top_chunks exceeds keyword_search_top_k")
        if self.retrieval_config.max_final_chunks <= 0:
            errors.append(f"max_final_chunks must be positive, got: {self.retrieval_config.max_final_chunks}")
        if self.llm_config.chat_max_tokens <= 0:
            errors.append(f"LLM chat_max_tokens must be positive, got: {self.llm_config.chat_max_tokens}")
        if not (0.0 <= self.llm_config.chat_temperature <= 2.0):
            errors.append(f"LLM chat_temperature must be in [0,2], got: {self.llm_config.chat_temperature}")
        if not (1 <= self.api_port <= 65535):
            errors.append(f"API port must be in [1,65535], got: {self.api_port}")

        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")

        if errors:
            error_message = f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_message)
            raise ValueError(error_message)

        logger.info(f"Configuration validation passed with {len(warnings)} warnings")

    def get_pipeline_parameters(self) -> Dict[str, Any]:
        """Get all pipeline parameters for use by retrieval components."""
        return {
            'vector_search_top_k': self.retrieval_config.vector_search_top_k,
            'keyword_search_top_k': self.retrieval_config.keyword_search_top_k,
            'vector_top_chunks': self.retrieval_config.vector_top_chunks,
            'keyword_top_chunks': self.retrieval_config.keyword_top_chunks,
            'reranked_top_chunks': self.retrieval_config.reranked_top_chunks,
            'graph_a_top_n': self.retrieval_config.graph_a_top_n,
            'graph_b_contradicts_top_n': self.retrieval_config.graph_b_contradicts_top_n,
            'graph_b_elaborates_top_n': self.retrieval_config.graph_b_elaborates_top_n,
            'graph_b_depends_on_top_n': self.retrieval_config.graph_b_depends_on_top_n,
            'context_expansion_enabled': self.retrieval_config.context_expansion_enabled,
            'context_adjacent_chunks': self.retrieval_config.context_adjacent_chunks,
            'max_final_chunks': self.retrieval_config.max_final_chunks,
            'query_generation_system_prompt': self.llm_config.query_generation_system_prompt,
            'rag_response_system_prompt': self.llm_config.rag_response_system_prompt,
            'query_generation_timeout': self.llm_config.query_generation_timeout,
            'response_generation_timeout': self.llm_config.response_generation_timeout,
        }


# Global production configuration instance
production_config: Optional[ProductionConfig] = None


def get_production_config() -> ProductionConfig:
    global production_config
    if production_config is None:
        raise RuntimeError("Production configuration has not been initialized. Call initialize_production_config() first.")
    return production_config


def initialize_production_config() -> ProductionConfig:
    global production_config
    if production_config is None:
        production_config = ProductionConfig.from_env()
    return production_config


def reload_production_config() -> ProductionConfig:
    global production_config
    production_config = ProductionConfig.from_env()
    return production_config

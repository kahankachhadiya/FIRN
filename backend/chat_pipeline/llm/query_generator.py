"""
LLM Query Generator for structured RAG command generation.

This module provides the LLMQueryGenerator class that uses an LLM to convert
user queries into structured RAG commands containing:
- Optimized vector search queries
- Keyword extraction for exact match search
- Probe type selection from allow-list
- Metadata filter criteria

The generator uses system prompts loaded from ProductionConfig with environment variable support.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError, field_validator
from datetime import datetime
import uuid

from llm.openai_compatible_client import OpenAICompatibleClient
from config.settings import settings
from utils.errors import LLMCommandError
from utils.logging_utils import get_logger
from config.production import ProductionConfig, get_production_config

logger = get_logger(__name__)


@dataclass
class RAGCommand:
    """Structured command for RAG retrieval."""
    vector_query: str
    keyword_query: str
    metadata_filter: Optional[Dict[str, Any]]

    # Execution tracking fields
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    execution_duration: Optional[float] = None
    execution_status: str = "pending"
    execution_error: Optional[str] = None

    def __post_init__(self):
        self._validate_queries()

    def _validate_queries(self) -> None:
        if not self.vector_query or not self.vector_query.strip():
            raise ValueError("vector_query cannot be empty")
        if not self.keyword_query or not self.keyword_query.strip():
            raise ValueError("keyword_query cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "vector_query": self.vector_query,
            "keyword_query": self.keyword_query,
            "metadata_filter": self.metadata_filter,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_duration": self.execution_duration,
            "execution_status": self.execution_status,
            "execution_error": self.execution_error
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def start_execution(self) -> None:
        self.execution_status = "executing"
        self.executed_at = datetime.now()
        logger.info(f"RAGCommand {self.command_id} execution started")

    def complete_execution(self, duration: Optional[float] = None) -> None:
        self.execution_status = "completed"
        if duration is not None:
            self.execution_duration = duration
        elif self.executed_at:
            self.execution_duration = (datetime.now() - self.executed_at).total_seconds()
        logger.info(f"RAGCommand {self.command_id} execution completed in {self.execution_duration:.2f}s")

    def fail_execution(self, error: str, duration: Optional[float] = None) -> None:
        self.execution_status = "failed"
        self.execution_error = error
        if duration is not None:
            self.execution_duration = duration
        elif self.executed_at:
            self.execution_duration = (datetime.now() - self.executed_at).total_seconds()
        logger.error(f"RAGCommand {self.command_id} execution failed: {error}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGCommand':
        command = cls(
            vector_query=data.get("vector_query", ""),
            keyword_query=data.get("keyword_query", ""),
            metadata_filter=data.get("metadata_filter")
        )
        if "command_id" in data:
            command.command_id = data["command_id"]
        if "created_at" in data and data["created_at"]:
            command.created_at = datetime.fromisoformat(data["created_at"])
        if "executed_at" in data and data["executed_at"]:
            command.executed_at = datetime.fromisoformat(data["executed_at"])
        if "execution_duration" in data:
            command.execution_duration = data["execution_duration"]
        if "execution_status" in data:
            command.execution_status = data["execution_status"]
        if "execution_error" in data:
            command.execution_error = data["execution_error"]
        return command

    @classmethod
    def from_json(cls, json_str: str) -> 'RAGCommand':
        return cls.from_dict(json.loads(json_str))


class RAGCommandSchema(BaseModel):
    """Pydantic schema for RAG command validation."""
    vector_query: str = Field(..., description="Optimized query for semantic vector search")
    keyword_query: str = Field(..., description="Keywords for exact match search")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter criteria")

    @field_validator('metadata_filter')
    @classmethod
    def validate_metadata_filter(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        allowed = {"title", "author", "created_date", "source_type", "language"}
        unknown = set(v.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown metadata filter fields: {unknown}. Allowed: {allowed}")
        str_fields = {"title", "author", "source_type", "language"}
        for field in str_fields:
            if field in v and v[field] is not None and not isinstance(v[field], str):
                raise ValueError(f"metadata_filter.{field} must be a string")
        if "created_date" in v and v["created_date"] is not None:
            import re
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(v["created_date"])):
                raise ValueError("metadata_filter.created_date must be in YYYY-MM-DD format")
        return {k: val for k, val in v.items() if val is not None} or None


class LLMQueryGenerator:
    """Generates RAG commands using LLM with ProductionConfig-based system prompt."""

    def __init__(
        self, 
        lm_client: Optional[OpenAICompatibleClient] = None,
        production_config: Optional[ProductionConfig] = None
    ):
        """
        Initialize LLM Query Generator with ProductionConfig-based configuration.
        
        Args:
            lm_client: Optional OpenAI-compatible client instance. If None, creates new client.
            production_config: Optional ProductionConfig instance. If None, loads from global config.
            
        Raises:
            ValueError: If QUERY_GENERATION_SYSTEM_PROMPT is not set or is empty
        """
        self.lm_client = lm_client or OpenAICompatibleClient(
            base_url=settings.llm_chat_base_url,
            api_key=settings.llm_chat_api_key,
            model=settings.llm_chat_model,
            max_tokens=settings.llm_chat_max_tokens,
            temperature=settings.llm_chat_temperature,
        )
        
        # Load configuration from ProductionConfig
        if production_config is not None:
            self.production_config = production_config
        else:
            try:
                self.production_config = get_production_config()
            except RuntimeError:
                # Fallback: create config from environment if not initialized
                self.production_config = ProductionConfig.from_env()
        
        # Load query generation system prompt from config/prompts.yml — raises if missing
        from config.prompts import get_query_generation_prompt
        self.query_system_prompt = get_query_generation_prompt()
        
        logger.info("LLM Query Generator initialized with prompt from config/prompts.yml")
    
    def reload_config(self) -> None:
        """
        Reload configuration from ProductionConfig.
        
        This method allows hot-reloading of system prompts and other configuration
        without restarting the query generator.
        """
        try:
            # Try to get global config first
            try:
                self.production_config = get_production_config()
            except RuntimeError:
                # If global config not initialized, create from environment
                self.production_config = ProductionConfig.from_env()
            
            # Reload query generation system prompt from config/prompts.yml
            from config.prompts import _load_prompts, _prompts
            reloaded = _load_prompts()
            _prompts.update(reloaded)
            from config.prompts import get_query_generation_prompt
            self.query_system_prompt = get_query_generation_prompt()
            logger.info("Query generator prompt reloaded from config/prompts.yml")
        except Exception as e:
            logger.error(f"Failed to reload query generator configuration: {e}")
            raise

    async def generate_rag_command(
        self,
        user_query: str,
    ) -> RAGCommand:
        if not user_query or not user_query.strip():
            raise LLMCommandError("User query cannot be empty")

        logger.info(f"Generating RAG command for query: '{user_query[:100]}...'")

        messages = [
            {"role": "system", "content": self.query_system_prompt},
            {"role": "user", "content": f"Generate a RAG command for this query: {user_query}"}
        ]

        try:
            response = await self.lm_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )

            choices = response.get("choices", [])
            if not choices:
                raise LLMCommandError("No response choices returned from LLM")

            message = choices[0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                raise LLMCommandError("Empty response content from LLM")

            try:
                command_dict = json.loads(content.strip())
            except json.JSONDecodeError as e:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        command_dict = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        raise LLMCommandError(f"Failed to parse JSON from LLM response: {e}")
                else:
                    raise LLMCommandError(f"No valid JSON found in LLM response: {content}")

            try:
                validated_command = RAGCommandSchema(**command_dict)
            except ValidationError as e:
                raise LLMCommandError(f"Invalid RAG command structure: {e}")

            if not validated_command.vector_query.strip():
                raise LLMCommandError("Vector query cannot be empty")
            if not validated_command.keyword_query.strip():
                raise LLMCommandError("Keyword query cannot be empty")

            rag_command = RAGCommand(
                vector_query=validated_command.vector_query,
                keyword_query=validated_command.keyword_query,
                metadata_filter=validated_command.metadata_filter
            )

            logger.info(
                f"RAG command generated successfully: "
                f"vector_query='{rag_command.vector_query[:50]}...', "
                f"keyword_query='{rag_command.keyword_query[:30]}...', "
                f"has_metadata_filter={rag_command.metadata_filter is not None}"
            )

            return rag_command

        except Exception as e:
            if isinstance(e, LLMCommandError):
                raise
            logger.error(f"Failed to generate RAG command: {e}")
            raise LLMCommandError(f"RAG command generation failed: {e}") from e

    def generate_rag_command_sync(self, user_query: str) -> RAGCommand:
        return asyncio.run(self.generate_rag_command(user_query))

    async def close(self):
        if self.lm_client:
            await self.lm_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

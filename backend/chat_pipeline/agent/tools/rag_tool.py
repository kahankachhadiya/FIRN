"""
Production-Ready RAG Tool for Layered Graph RAG System.

This module provides the enhanced RAG tool that handles the complete layered graph
retrieval pipeline with comprehensive error handling, rate limiting, metrics collection,
conversation logging, and timeout management.

The tool integrates all production features:
- LLM-driven query generation
- 11-stage retrieval pipeline execution
- Connection pooling and resource management
- Rate limiting and request queuing
- Comprehensive metrics and monitoring
- Detailed conversation logging
- Graceful error handling and fallbacks
- Timeout management for all stages
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from llm.query_generator import LLMQueryGenerator, RAGCommand
from retrieval.layered_graph_retrieval_engine import LayeredGraphRetrievalEngine, EnhancedChunkOutput
from utils.conversation_logger import ConversationLogger
from utils.rate_limiter import RateLimiter
from utils.errors import (
    ToolExecutionError, 
    LLMCommandError, 
    DatabaseError, 
    TimeoutError,
    ToolValidationError
)

logger = logging.getLogger(__name__)


class ParameterValidator:
    """
    Comprehensive parameter validator for RAG tool arguments.
    
    Provides detailed validation for all RAG tool parameters with
    user-friendly error messages and recovery recommendations.
    """
    
    @staticmethod
    def validate_query(query: Any) -> str:
        """
        Validate the query parameter.
        
        Args:
            query: Query value to validate
            
        Returns:
            Validated query string
            
        Raises:
            ToolValidationError: If query is invalid
        """
        # Check if query exists
        if query is None:
            raise ToolValidationError(
                "Missing required parameter 'query'",
                details={
                    "parameter": "query",
                    "expected_type": "string",
                    "received_type": type(query).__name__,
                    "recovery_recommendations": [
                        "Provide a non-empty query string",
                        "Ensure the query parameter is included in the request"
                    ]
                }
            )
        
        # Check if query is a string
        if not isinstance(query, str):
            raise ToolValidationError(
                f"Parameter 'query' must be a string, got {type(query).__name__}",
                details={
                    "parameter": "query",
                    "expected_type": "string",
                    "received_type": type(query).__name__,
                    "received_value": str(query)[:100] + "..." if len(str(query)) > 100 else str(query),
                    "recovery_recommendations": [
                        "Convert the query to a string",
                        "Ensure the query is properly formatted as text"
                    ]
                }
            )
        
        # Check if query is not empty
        if not query.strip():
            raise ToolValidationError(
                "Parameter 'query' cannot be empty or whitespace-only",
                details={
                    "parameter": "query",
                    "expected": "non-empty string",
                    "received": "empty or whitespace-only string",
                    "recovery_recommendations": [
                        "Provide a meaningful query with actual content",
                        "Remove leading/trailing whitespace and add content"
                    ]
                }
            )
        
        # Check minimum length
        if len(query.strip()) < 1:
            raise ToolValidationError(
                "Parameter 'query' must be at least 1 character long",
                details={
                    "parameter": "query",
                    "minimum_length": 1,
                    "actual_length": len(query.strip()),
                    "recovery_recommendations": [
                        "Provide a query with at least 1 character",
                        "Ensure the query contains meaningful content"
                    ]
                }
            )
        
        # Check maximum length
        if len(query) > 10000:
            raise ToolValidationError(
                f"Parameter 'query' exceeds maximum length of 10000 characters (got {len(query)})",
                details={
                    "parameter": "query",
                    "maximum_length": 10000,
                    "actual_length": len(query),
                    "recovery_recommendations": [
                        "Shorten the query to under 10000 characters",
                        "Break long queries into smaller, focused questions",
                        "Remove unnecessary details or repetitive content"
                    ]
                }
            )
        
        return query.strip()
    
    @staticmethod
    def validate_client_id(client_id: Any) -> str:
        """
        Validate the client_id parameter.
        
        Args:
            client_id: Client ID value to validate
            
        Returns:
            Validated client ID string
            
        Raises:
            ToolValidationError: If client_id is invalid
        """
        # Use default if not provided
        if client_id is None:
            return "default"
        
        # Check if client_id is a string
        if not isinstance(client_id, str):
            raise ToolValidationError(
                f"Parameter 'client_id' must be a string, got {type(client_id).__name__}",
                details={
                    "parameter": "client_id",
                    "expected_type": "string",
                    "received_type": type(client_id).__name__,
                    "received_value": str(client_id),
                    "recovery_recommendations": [
                        "Convert the client_id to a string",
                        "Use 'default' if no specific client ID is needed"
                    ]
                }
            )
        
        # Check if client_id is not empty
        if not client_id.strip():
            raise ToolValidationError(
                "Parameter 'client_id' cannot be empty or whitespace-only",
                details={
                    "parameter": "client_id",
                    "expected": "non-empty string",
                    "received": "empty or whitespace-only string",
                    "recovery_recommendations": [
                        "Provide a valid client identifier",
                        "Use 'default' if no specific client ID is needed"
                    ]
                }
            )
        
        # Check minimum length
        if len(client_id.strip()) < 1:
            raise ToolValidationError(
                "Parameter 'client_id' must be at least 1 character long",
                details={
                    "parameter": "client_id",
                    "minimum_length": 1,
                    "actual_length": len(client_id.strip()),
                    "recovery_recommendations": [
                        "Provide a client ID with at least 1 character",
                        "Use 'default' if no specific client ID is needed"
                    ]
                }
            )
        
        # Check maximum length
        if len(client_id) > 100:
            raise ToolValidationError(
                f"Parameter 'client_id' exceeds maximum length of 100 characters (got {len(client_id)})",
                details={
                    "parameter": "client_id",
                    "maximum_length": 100,
                    "actual_length": len(client_id),
                    "recovery_recommendations": [
                        "Shorten the client_id to under 100 characters",
                        "Use a shorter, unique identifier"
                    ]
                }
            )
        
        # Check pattern (alphanumeric, underscore, hyphen only)
        import re
        trimmed_client_id = client_id.strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', trimmed_client_id):
            raise ToolValidationError(
                "Parameter 'client_id' contains invalid characters. Only alphanumeric characters, underscores, and hyphens are allowed",
                details={
                    "parameter": "client_id",
                    "allowed_pattern": "^[a-zA-Z0-9_-]+$",
                    "received_value": client_id,
                    "invalid_characters": [c for c in trimmed_client_id if not re.match(r'[a-zA-Z0-9_-]', c)],
                    "recovery_recommendations": [
                        "Remove or replace invalid characters",
                        "Use only letters, numbers, underscores, and hyphens",
                        "Example valid client IDs: 'user_123', 'api-client-1', 'default'"
                    ]
                }
            )
        
        return trimmed_client_id
    
    @staticmethod
    def validate_session_id(session_id: Any) -> Optional[str]:
        """
        Validate the session_id parameter.
        
        Args:
            session_id: Session ID value to validate
            
        Returns:
            Validated session ID string or None if not provided
            
        Raises:
            ToolValidationError: If session_id is invalid
        """
        # Allow None (will be auto-generated)
        if session_id is None:
            return None
        
        # Check if session_id is a string
        if not isinstance(session_id, str):
            raise ToolValidationError(
                f"Parameter 'session_id' must be a string, got {type(session_id).__name__}",
                details={
                    "parameter": "session_id",
                    "expected_type": "string",
                    "received_type": type(session_id).__name__,
                    "received_value": str(session_id),
                    "recovery_recommendations": [
                        "Convert the session_id to a string",
                        "Omit session_id to have it auto-generated"
                    ]
                }
            )
        
        # Check if session_id is not empty
        if not session_id.strip():
            raise ToolValidationError(
                "Parameter 'session_id' cannot be empty or whitespace-only",
                details={
                    "parameter": "session_id",
                    "expected": "non-empty string or null",
                    "received": "empty or whitespace-only string",
                    "recovery_recommendations": [
                        "Provide a valid session identifier",
                        "Omit session_id to have it auto-generated"
                    ]
                }
            )
        
        # Check minimum length
        if len(session_id.strip()) < 1:
            raise ToolValidationError(
                "Parameter 'session_id' must be at least 1 character long",
                details={
                    "parameter": "session_id",
                    "minimum_length": 1,
                    "actual_length": len(session_id.strip()),
                    "recovery_recommendations": [
                        "Provide a session ID with at least 1 character",
                        "Omit session_id to have it auto-generated"
                    ]
                }
            )
        
        # Check maximum length
        if len(session_id) > 100:
            raise ToolValidationError(
                f"Parameter 'session_id' exceeds maximum length of 100 characters (got {len(session_id)})",
                details={
                    "parameter": "session_id",
                    "maximum_length": 100,
                    "actual_length": len(session_id),
                    "recovery_recommendations": [
                        "Shorten the session_id to under 100 characters",
                        "Use a shorter, unique identifier"
                    ]
                }
            )
        
        # Check pattern (alphanumeric, underscore, hyphen only)
        import re
        trimmed_session_id = session_id.strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', trimmed_session_id):
            raise ToolValidationError(
                "Parameter 'session_id' contains invalid characters. Only alphanumeric characters, underscores, and hyphens are allowed",
                details={
                    "parameter": "session_id",
                    "allowed_pattern": "^[a-zA-Z0-9_-]+$",
                    "received_value": session_id,
                    "invalid_characters": [c for c in trimmed_session_id if not re.match(r'[a-zA-Z0-9_-]', c)],
                    "recovery_recommendations": [
                        "Remove or replace invalid characters",
                        "Use only letters, numbers, underscores, and hyphens",
                        "Example valid session IDs: 'session_abc123', 'conv-456', 'user_session_789'"
                    ]
                }
            )
        
        return trimmed_session_id
    
    @staticmethod
    def validate_max_chunks(max_chunks: Any) -> Optional[int]:
        """
        Validate the max_chunks parameter.
        
        Args:
            max_chunks: Max chunks value to validate
            
        Returns:
            Validated max chunks integer or None if not provided
            
        Raises:
            ToolValidationError: If max_chunks is invalid
        """
        # Allow None (will use default)
        if max_chunks is None:
            return None
        
        # Check if max_chunks is an integer or can be converted to one
        if isinstance(max_chunks, str):
            try:
                max_chunks = int(max_chunks)
            except ValueError:
                raise ToolValidationError(
                    f"Parameter 'max_chunks' must be an integer, got string '{max_chunks}' that cannot be converted to integer",
                    details={
                        "parameter": "max_chunks",
                        "expected_type": "integer",
                        "received_type": "string",
                        "received_value": max_chunks,
                        "recovery_recommendations": [
                            "Provide a valid integer value",
                            "Use a number between 1 and 50",
                            "Omit max_chunks to use the default value (20)"
                        ]
                    }
                )
        elif isinstance(max_chunks, float):
            if max_chunks.is_integer():
                max_chunks = int(max_chunks)
            else:
                raise ToolValidationError(
                    f"Parameter 'max_chunks' must be an integer, got float {max_chunks}",
                    details={
                        "parameter": "max_chunks",
                        "expected_type": "integer",
                        "received_type": "float",
                        "received_value": max_chunks,
                        "recovery_recommendations": [
                            "Provide an integer value instead of a float",
                            "Round the value to the nearest integer",
                            "Use a whole number between 1 and 50"
                        ]
                    }
                )
        elif not isinstance(max_chunks, int):
            raise ToolValidationError(
                f"Parameter 'max_chunks' must be an integer, got {type(max_chunks).__name__}",
                details={
                    "parameter": "max_chunks",
                    "expected_type": "integer",
                    "received_type": type(max_chunks).__name__,
                    "received_value": str(max_chunks),
                    "recovery_recommendations": [
                        "Provide an integer value",
                        "Use a number between 1 and 50",
                        "Omit max_chunks to use the default value (20)"
                    ]
                }
            )
        
        # Check minimum value
        if max_chunks < 1:
            raise ToolValidationError(
                f"Parameter 'max_chunks' must be at least 1, got {max_chunks}",
                details={
                    "parameter": "max_chunks",
                    "minimum_value": 1,
                    "received_value": max_chunks,
                    "recovery_recommendations": [
                        "Use a value of 1 or higher",
                        "Typical values are between 5 and 20 for most queries"
                    ]
                }
            )
        
        # Check maximum value
        if max_chunks > 50:
            raise ToolValidationError(
                f"Parameter 'max_chunks' cannot exceed 50, got {max_chunks}",
                details={
                    "parameter": "max_chunks",
                    "maximum_value": 50,
                    "received_value": max_chunks,
                    "recovery_recommendations": [
                        "Use a value of 50 or lower",
                        "Consider if you really need more than 50 chunks",
                        "Large numbers of chunks may impact performance"
                    ]
                }
            )
        
        return max_chunks
    
    @staticmethod
    def validate_all_parameters(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all RAG tool parameters comprehensively.
        
        Args:
            args: Dictionary of arguments to validate
            
        Returns:
            Dictionary of validated parameters
            
        Raises:
            ToolValidationError: If any parameter is invalid
        """
        validated_params = {}
        
        try:
            # Validate required parameters
            validated_params["query"] = ParameterValidator.validate_query(args.get("query"))
            
            # Validate optional parameters
            validated_params["client_id"] = ParameterValidator.validate_client_id(args.get("client_id"))
            validated_params["session_id"] = ParameterValidator.validate_session_id(args.get("session_id"))
            validated_params["max_chunks"] = ParameterValidator.validate_max_chunks(args.get("max_chunks"))
            
            # Check for unexpected parameters
            expected_params = {"query", "client_id", "session_id", "max_chunks"}
            unexpected_params = set(args.keys()) - expected_params
            
            if unexpected_params:
                raise ToolValidationError(
                    f"Unexpected parameters provided: {', '.join(unexpected_params)}",
                    details={
                        "unexpected_parameters": list(unexpected_params),
                        "expected_parameters": list(expected_params),
                        "recovery_recommendations": [
                            "Remove the unexpected parameters",
                            "Check the API documentation for valid parameters",
                            f"Valid parameters are: {', '.join(expected_params)}"
                        ]
                    }
                )
            
            return validated_params
            
        except ToolValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise ToolValidationError(
                f"Unexpected error during parameter validation: {str(e)}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "recovery_recommendations": [
                        "Check that all parameters are properly formatted",
                        "Ensure parameter values are of the correct type",
                        "Contact support if the error persists"
                    ]
                }
            ) from e


@dataclass
class RAGToolConfig:
    """Configuration for the RAG tool."""
    max_chunks: int = 20
    request_timeout: float = 30.0
    query_generation_timeout: float = 10.0
    pipeline_timeout: float = 25.0
    vector_search_timeout: float = 15.0
    keyword_search_timeout: float = 10.0
    relation_graph_timeout: float = 8.0
    probe_graph_timeout: float = 5.0
    context_expansion_timeout: float = 5.0
    reranking_timeout: float = 20.0
    merge_dedupe_timeout: float = 3.0
    final_assembly_timeout: float = 2.0
    database_query_timeout: float = 10.0
    enable_fallback: bool = False  # Disabled by default - throw errors instead
    enable_metrics: bool = True
    enable_conversation_logging: bool = True
    enable_rate_limiting: bool = True
    
    @classmethod
    def from_env(cls) -> 'RAGToolConfig':
        """Load configuration from environment variables."""
        import os
        
        return cls(
            max_chunks=int(os.getenv('MAX_FINAL_CHUNKS', '20')),
            request_timeout=float(os.getenv('RAG_TOOL_REQUEST_TIMEOUT', os.getenv('RAG_REQUEST_TIMEOUT', '30.0'))),
            query_generation_timeout=float(os.getenv('RAG_TOOL_QUERY_GENERATION_TIMEOUT', os.getenv('RAG_QUERY_GENERATION_TIMEOUT', os.getenv('LLM_QUERY_GENERATION_TIMEOUT', '999')))),
            pipeline_timeout=float(os.getenv('RAG_TOOL_PIPELINE_TIMEOUT', os.getenv('RAG_PIPELINE_TIMEOUT', '999'))),
            vector_search_timeout=float(os.getenv('VECTOR_SEARCH_TIMEOUT', '15.0')),
            keyword_search_timeout=float(os.getenv('KEYWORD_SEARCH_TIMEOUT', '10.0')),
            relation_graph_timeout=float(os.getenv('RELATION_GRAPH_TIMEOUT', '8.0')),
            probe_graph_timeout=float(os.getenv('PROBE_GRAPH_TIMEOUT', '5.0')),
            context_expansion_timeout=float(os.getenv('CONTEXT_EXPANSION_TIMEOUT', '5.0')),
            reranking_timeout=float(os.getenv('RERANKING_TIMEOUT', '20.0')),
            merge_dedupe_timeout=float(os.getenv('MERGE_DEDUPE_TIMEOUT', '3.0')),
            final_assembly_timeout=float(os.getenv('FINAL_ASSEMBLY_TIMEOUT', '2.0')),
            database_query_timeout=float(os.getenv('DATABASE_QUERY_TIMEOUT', '10.0')),
            enable_fallback=os.getenv('RAG_TOOL_ENABLE_FALLBACK', os.getenv('ENABLE_ERROR_RECOVERY', 'false')).lower() == 'true',
            enable_metrics=os.getenv('RAG_TOOL_ENABLE_METRICS', os.getenv('ENABLE_METRICS_COLLECTION', 'true')).lower() == 'true',
            enable_conversation_logging=os.getenv('RAG_TOOL_ENABLE_CONVERSATION_LOGGING', os.getenv('ENABLE_CONVERSATION_LOGGING', 'true')).lower() == 'true',
            enable_rate_limiting=os.getenv('RAG_TOOL_ENABLE_RATE_LIMITING', os.getenv('RATE_LIMIT_ENABLED', 'true')).lower() == 'true'
        )


class RateLimitExceededError(ToolExecutionError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class EnhancedRAGTool:
    """
    Production-ready RAG tool handling complete layered graph retrieval.
    
    This tool provides a comprehensive RAG retrieval system with:
    - LLM-driven query generation with structured RAG commands
    - 11-stage layered graph retrieval pipeline
    - Production-ready error handling and fallbacks
    - Rate limiting with per-client tracking
    - Comprehensive metrics collection and monitoring
    - Detailed conversation logging for debugging
    - Connection pooling for all database operations
    - Timeout management for all pipeline stages
    - Graceful degradation under load
    """
    
    def __init__(
        self,
        query_generator: Optional[LLMQueryGenerator] = None,
        retrieval_engine: Optional[LayeredGraphRetrievalEngine] = None,
        conversation_logger: Optional[ConversationLogger] = None,
        rate_limiter: Optional[RateLimiter] = None,
        metrics_collector: Optional[Any] = None,
        config: Optional[RAGToolConfig] = None
    ):
        """
        Initialize the enhanced RAG tool.
        
        Args:
            query_generator: LLM query generator for structured commands
            retrieval_engine: Layered graph retrieval engine
            conversation_logger: Logger for debugging conversations
            rate_limiter: Rate limiter for request throttling
            metrics_collector: Metrics collector for monitoring
            config: Tool configuration
        """
        self.config = config or RAGToolConfig()
        
        # Initialize components
        self.query_generator = query_generator or LLMQueryGenerator()
        self.retrieval_engine = retrieval_engine or LayeredGraphRetrievalEngine()
        self.conversation_logger = conversation_logger
        self.rate_limiter = rate_limiter
        self.metrics_collector = metrics_collector
        
        self.logger = logger
        
        self.logger.info(
            f"Enhanced RAG tool initialized with config: "
            f"max_chunks={self.config.max_chunks}, "
            f"request_timeout={self.config.request_timeout}s, "
            f"rate_limiting={self.config.enable_rate_limiting}, "
            f"metrics={self.config.enable_metrics}"
        )
    
    async def execute_retrieval(
        self,
        user_query: str,
        client_id: str = "default",
        session_id: Optional[str] = None,
        max_chunks: Optional[int] = None,
        request_weight: int = 1
    ) -> List[EnhancedChunkOutput]:
        """
        Execute complete layered graph RAG retrieval pipeline with production features.
        
        This method orchestrates the entire RAG pipeline:
        1. Rate limiting and request validation
        2. Initialize conversation logging with metrics
        3. LLM generates RAG command with metadata/probe filters
        4. Execute 11-stage retrieval pipeline with connection pooling
        5. Comprehensive error handling and logging
        6. Detailed logging and metrics collection
        
        Args:
            user_query: Original user query
            client_id: Client identifier for rate limiting
            session_id: Session identifier for logging (auto-generated if None)
            max_chunks: Maximum chunks to return (uses config default if None)
            request_weight: Weight for rate limiting (default: 1)
            
        Returns:
            List of enhanced chunks with complete metadata
            
        Raises:
            RateLimitExceededError: When rate limit is exceeded
            ToolExecutionError: When retrieval pipeline fails
            TimeoutError: When operation exceeds timeout
            DatabaseError: When database operations fail
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Use config default if max_chunks not specified
        if max_chunks is None:
            max_chunks = self.config.max_chunks
        
        start_time = time.time()
        log = None
        
        try:
            # Stage 1: Rate limiting
            if self.config.enable_rate_limiting and self.rate_limiter:
                await self._check_rate_limit(client_id, request_weight)
            
            # Stage 2: Start conversation logging
            if self.config.enable_conversation_logging and self.conversation_logger:
                log = self.conversation_logger.start_conversation(session_id, user_query)
            
            # Stage 3: Metrics tracking
            if self.config.enable_metrics and self.metrics_collector:
                self.metrics_collector.increment_counter('rag_requests_total')
                self.metrics_collector.increment_counter('rag_requests_active')
            
            # Stage 4: Execute retrieval with timeout
            results = await asyncio.wait_for(
                self._execute_retrieval_pipeline(user_query, log, max_chunks),
                timeout=self.config.request_timeout
            )
            
            # Stage 5: Success metrics and logging
            total_time = time.time() - start_time
            
            if self.config.enable_metrics and self.metrics_collector:
                self.metrics_collector.increment_counter('rag_requests_success')
                self.metrics_collector.record_histogram('rag_request_duration', total_time * 1000)
                self.metrics_collector.record_histogram('rag_chunks_returned', len(results))
                self.metrics_collector.increment_counter('rag_requests_active', -1)
            
            if log and self.conversation_logger:
                self.conversation_logger.log_final_response(log, results, "")
                self.conversation_logger.log_stage_timing(log, "total_pipeline", total_time)
            
            self.logger.info(
                f"RAG retrieval completed successfully: "
                f"session_id={session_id}, client_id={client_id}, "
                f"chunks={len(results)}, duration={total_time:.2f}s"
            )
            
            return results
            
        except asyncio.TimeoutError:
            # Handle timeout
            error_time = time.time() - start_time
            error_msg = f"RAG retrieval timed out after {self.config.request_timeout}s"
            
            await self._handle_error(
                error_msg, log, client_id, session_id, error_time, 'timeout',
                stage='request_timeout',
                context={'timeout_seconds': self.config.request_timeout},
                original_exception=None
            )
            raise TimeoutError(error_msg)
            
        except RateLimitExceededError:
            # Re-raise rate limit errors without modification
            raise
            
        except Exception as e:
            # Handle all other errors
            error_time = time.time() - start_time
            error_msg = f"RAG retrieval failed: {str(e)}"
            
            await self._handle_error(
                error_msg, log, client_id, session_id, error_time, type(e).__name__,
                stage='pipeline_execution',
                context={'user_query_length': len(user_query), 'max_chunks': max_chunks},
                original_exception=e
            )
            
            # Re-raise the original error instead of trying fallback
            if isinstance(e, DatabaseError):
                raise e
            else:
                raise ToolExecutionError(error_msg) from e
    
    async def _check_rate_limit(self, client_id: str, request_weight: int) -> None:
        """Check rate limit for client."""
        try:
            is_allowed = await self.rate_limiter.is_allowed(client_id, request_weight)
            if not is_allowed:
                reset_time = self.rate_limiter.get_reset_time(client_id)
                
                if self.metrics_collector:
                    self.metrics_collector.increment_counter('rag_rate_limit_exceeded')
                
                raise RateLimitExceededError(
                    f"Rate limit exceeded for client: {client_id}",
                    reset_time
                )
        except Exception as e:
            if isinstance(e, RateLimitExceededError):
                raise
            self.logger.error(f"Rate limit check failed for client {client_id}: {e}")
            # Continue without rate limiting on error (fail open)
    
    async def _execute_retrieval_pipeline(
        self,
        user_query: str,
        log: Optional[Any],
        max_chunks: int
    ) -> List[EnhancedChunkOutput]:
        """Execute the complete retrieval pipeline with timeout management."""
        
        # Stage 1: LLM Query Generation with timeout
        stage_start = time.time()
        try:
            rag_command = await asyncio.wait_for(
                self.query_generator.generate_rag_command(user_query),
                timeout=self.config.query_generation_timeout
            )
            
            query_gen_time = time.time() - stage_start
            
            if log and self.conversation_logger:
                self.conversation_logger.log_rag_command(log, rag_command)
                self.conversation_logger.log_stage_timing(log, "query_generation", query_gen_time)
            
            if self.metrics_collector:
                self.metrics_collector.record_histogram('rag_query_generation_time', query_gen_time * 1000)
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Query generation timed out after {self.config.query_generation_timeout}s")
        except Exception as e:
            # Enhanced fallback with detailed logging
            self.logger.warning(
                f"Query generation failed, using fallback: {e} "
                f"(type: {type(e).__name__}, query_length: {len(user_query)})"
            )
            
            # Log detailed error context if conversation logger is available
            if log and self.conversation_logger:
                fallback_context = {
                    'stage': 'query_generation_fallback',
                    'original_error': str(e),
                    'error_type': type(e).__name__,
                    'fallback_used': True,
                    'query_length': len(user_query)
                }
                self.conversation_logger.log_detailed_error(log, {
                    'message': f"Query generation failed, using fallback: {e}",
                    'context': fallback_context,
                    'recovery_recommendations': [
                        "Check LLM service availability and configuration",
                        "Validate query generation prompt format",
                        "Consider implementing query caching for common patterns"
                    ]
                })
            
            # Create fallback RAG command
            rag_command = RAGCommand(
                vector_query=user_query,
                keyword_query=user_query,
                metadata_filter=None
            )
        
        # Stage 2: Execute retrieval pipeline with timeout and stage-specific timeouts
        pipeline_start = time.time()
        try:
            # Pass timeout configuration to the retrieval engine
            timeout_config = {
                'vector_search_timeout': self.config.vector_search_timeout,
                'keyword_search_timeout': self.config.keyword_search_timeout,
                'relation_graph_timeout': self.config.relation_graph_timeout,
                'probe_graph_timeout': self.config.probe_graph_timeout,
                'context_expansion_timeout': self.config.context_expansion_timeout,
                'reranking_timeout': self.config.reranking_timeout,
                'merge_dedupe_timeout': self.config.merge_dedupe_timeout,
                'final_assembly_timeout': self.config.final_assembly_timeout,
                'database_query_timeout': self.config.database_query_timeout
            }
            
            results = await asyncio.wait_for(
                self.retrieval_engine.execute_pipeline(rag_command, log, timeout_config),
                timeout=self.config.pipeline_timeout
            )
            
            pipeline_time = time.time() - pipeline_start
            
            if log and self.conversation_logger:
                self.conversation_logger.log_stage_timing(log, "retrieval_pipeline", pipeline_time)
            
            if self.metrics_collector:
                self.metrics_collector.record_histogram('rag_pipeline_time', pipeline_time * 1000)
            
            # Limit results to max_chunks
            if len(results) > max_chunks:
                self.logger.info(f"Limiting results from {len(results)} to {max_chunks} chunks")
                results = results[:max_chunks]
            
            return results
            
        except asyncio.TimeoutError:
            pipeline_time = time.time() - pipeline_start
            timeout_context = {
                'stage': 'retrieval_pipeline',
                'timeout_seconds': self.config.pipeline_timeout,
                'actual_duration': pipeline_time,
                'timeout_config': timeout_config
            }
            
            # Log detailed timeout information
            if log and self.conversation_logger:
                self.conversation_logger.log_detailed_error(log, {
                    'message': f"Retrieval pipeline timed out after {self.config.pipeline_timeout}s",
                    'context': timeout_context,
                    'recovery_recommendations': [
                        f"Increase pipeline timeout from {self.config.pipeline_timeout}s",
                        "Optimize individual stage timeouts",
                        "Check database performance and connectivity",
                        "Consider reducing query complexity or result limits"
                    ]
                })
            
            raise TimeoutError(f"Retrieval pipeline timed out after {self.config.pipeline_timeout}s")
    
    async def _handle_error(
        self,
        error_msg: str,
        log: Optional[Any],
        client_id: str,
        session_id: str,
        error_time: float,
        error_type: str,
        stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ) -> None:
        """
        Handle errors with comprehensive logging, metrics, and detailed reporting.
        
        This enhanced error handler provides:
        - Detailed error context and stack traces
        - Stage-specific error tracking
        - Error categorization and severity levels
        - Recovery recommendations
        - Comprehensive metrics and alerting
        
        Args:
            error_msg: Human-readable error message
            log: Conversation log instance
            client_id: Client identifier
            session_id: Session identifier
            error_time: Time spent before error occurred
            error_type: Type of error (class name)
            stage: Pipeline stage where error occurred
            context: Additional error context
            original_exception: Original exception for stack trace
        """
        
        # Create detailed error context
        error_context = {
            'client_id': client_id,
            'session_id': session_id,
            'duration_seconds': round(error_time, 4),
            'error_type': error_type,
            'stage': stage or 'unknown',
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        # Add stack trace if available
        if original_exception:
            import traceback
            error_context['stack_trace'] = traceback.format_exception(
                type(original_exception), original_exception, original_exception.__traceback__
            )
            error_context['exception_details'] = {
                'type': type(original_exception).__name__,
                'message': str(original_exception),
                'module': getattr(original_exception, '__module__', 'unknown')
            }
        
        # Determine error severity
        severity = self._determine_error_severity(error_type, stage)
        error_context['severity'] = severity
        
        # Generate recovery recommendations
        recovery_recommendations = self._generate_recovery_recommendations(error_type, stage, context)
        error_context['recovery_recommendations'] = recovery_recommendations
        
        # Log error with full context
        log_message = (
            f"RAG retrieval error [{severity.upper()}]: {error_msg} "
            f"(client_id={client_id}, session_id={session_id}, "
            f"stage={stage or 'unknown'}, duration={error_time:.2f}s, type={error_type})"
        )
        
        if severity in ['critical', 'high']:
            self.logger.error(log_message, extra={'error_context': error_context}, exc_info=original_exception)
        elif severity == 'medium':
            self.logger.warning(log_message, extra={'error_context': error_context})
        else:
            self.logger.info(log_message, extra={'error_context': error_context})
        
        # Update conversation log with detailed error information
        if log and self.conversation_logger:
            detailed_error = {
                'message': error_msg,
                'context': error_context,
                'recovery_recommendations': recovery_recommendations
            }
            self.conversation_logger.log_detailed_error(log, detailed_error)
            self.conversation_logger.log_stage_timing(log, "total_pipeline", error_time)
            
            # Log stage-specific timing if stage is known
            if stage:
                self.conversation_logger.log_stage_timing(log, f"{stage}_error", error_time)
        
        # Update comprehensive metrics
        if self.metrics_collector:
            # Basic error metrics
            self.metrics_collector.increment_counter('rag_requests_error')
            self.metrics_collector.increment_counter('rag_error_by_type', tags={'error_type': error_type})
            self.metrics_collector.increment_counter('rag_error_by_severity', tags={'severity': severity})
            
            # Stage-specific error metrics
            if stage:
                self.metrics_collector.increment_counter('rag_error_by_stage', tags={'stage': stage})
            
            # Client-specific error metrics
            self.metrics_collector.increment_counter('rag_error_by_client', tags={'client_id': client_id})
            
            # Timing metrics
            self.metrics_collector.record_histogram('rag_request_duration', error_time * 1000)
            self.metrics_collector.record_histogram('rag_error_duration', error_time * 1000, tags={'error_type': error_type})
            
            # Active request tracking
            self.metrics_collector.increment_counter('rag_requests_active', -1)
            
            # Error rate tracking
            self.metrics_collector.set_gauge('rag_error_rate_last_minute', self._calculate_error_rate())
        
        # Trigger alerts for critical errors
        if severity == 'critical':
            await self._trigger_error_alert(error_context, error_msg)
    
    def _determine_error_severity(self, error_type: str, stage: Optional[str]) -> str:
        """
        Determine error severity based on error type and stage.
        
        Returns:
            Severity level: 'low', 'medium', 'high', 'critical'
        """
        # Critical errors that indicate system failure
        critical_errors = {
            'DatabaseError', 'ConnectionError', 'MemoryError', 
            'SystemError', 'OSError'
        }
        
        # High priority errors that affect functionality
        high_errors = {
            'TimeoutError', 'LLMCommandError', 'ToolExecutionError',
            'ValidationError', 'ConfigurationError'
        }
        
        # Medium priority errors that can be recovered from
        medium_errors = {
            'RateLimitExceededError', 'IngestionError', 'FileAccessError'
        }
        
        # Critical stages where any error is severe
        critical_stages = {'query_generation', 'final_assembly'}
        
        if error_type in critical_errors:
            return 'critical'
        elif error_type in high_errors or stage in critical_stages:
            return 'high'
        elif error_type in medium_errors:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recovery_recommendations(
        self, 
        error_type: str, 
        stage: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate specific recovery recommendations based on error context.
        
        Returns:
            List of actionable recovery recommendations
        """
        recommendations = []
        
        # Error type specific recommendations
        if error_type == 'TimeoutError':
            recommendations.extend([
                "Increase timeout configuration for the affected stage",
                "Check database connection health and network latency",
                "Consider reducing query complexity or chunk limits",
                "Monitor system resource usage (CPU, memory, disk I/O)"
            ])
        
        elif error_type == 'DatabaseError':
            recommendations.extend([
                "Verify database connectivity and health status",
                "Check database connection pool configuration",
                "Review database logs for underlying issues",
                "Consider implementing database failover mechanisms"
            ])
        
        elif error_type == 'RateLimitExceededError':
            recommendations.extend([
                "Implement exponential backoff retry logic",
                "Consider increasing rate limit thresholds",
                "Distribute load across multiple clients",
                "Implement request queuing for burst handling"
            ])
        
        elif error_type == 'LLMCommandError':
            recommendations.extend([
                "Validate LLM response format and structure",
                "Implement fallback query generation logic",
                "Check LLM service availability and configuration",
                "Consider using simpler query structures"
            ])
        
        elif error_type == 'MemoryError':
            recommendations.extend([
                "Reduce batch sizes and chunk limits",
                "Implement memory-efficient processing",
                "Monitor memory usage patterns",
                "Consider horizontal scaling"
            ])
        
        # Stage specific recommendations
        if stage == 'query_generation':
            recommendations.extend([
                "Implement query generation fallback to simple structure",
                "Validate LLM service health and response format",
                "Consider caching successful query patterns"
            ])
        
        elif stage == 'vector_search':
            recommendations.extend([
                "Check Qdrant service health and connectivity",
                "Verify vector index integrity",
                "Consider reducing search result limits"
            ])
        
        elif stage == 'keyword_search':
            recommendations.extend([
                "Check Redis service health and connectivity",
                "Verify search index configuration",
                "Consider simplifying search queries"
            ])
        
        elif stage == 'reranking':
            recommendations.extend([
                "Check cross-encoder model availability",
                "Reduce reranking batch size",
                "Implement reranking timeout handling"
            ])
        
        elif stage == 'graph_traversal':
            recommendations.extend([
                "Check FalkorDB service health and connectivity",
                "Verify graph data integrity",
                "Consider reducing graph traversal depth"
            ])
        
        # Context specific recommendations
        if context:
            if context.get('connection_pool_exhausted'):
                recommendations.append("Increase database connection pool size")
            
            if context.get('high_memory_usage'):
                recommendations.append("Implement memory cleanup and garbage collection")
            
            if context.get('slow_response_time'):
                recommendations.append("Optimize query performance and indexing")
        
        # General recommendations
        recommendations.extend([
            "Review system logs for additional error context",
            "Monitor system health metrics and alerts",
            "Consider implementing circuit breaker patterns",
            "Validate configuration and environment settings"
        ])
        
        return recommendations
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate for the last minute."""
        if not self.metrics_collector:
            return 0.0
        
        try:
            # Get metrics for the last minute
            perf_summary = self.metrics_collector.get_performance_summary()
            return perf_summary.error_rate
        except Exception:
            return 0.0
    
    async def _trigger_error_alert(self, error_context: Dict[str, Any], error_msg: str) -> None:
        """
        Trigger alerts for critical errors.
        
        This method can be extended to integrate with alerting systems
        like PagerDuty, Slack, email notifications, etc.
        """
        try:
            # Log critical alert
            alert_message = (
                f"CRITICAL RAG ERROR ALERT: {error_msg}\n"
                f"Session: {error_context.get('session_id')}\n"
                f"Client: {error_context.get('client_id')}\n"
                f"Stage: {error_context.get('stage')}\n"
                f"Error Type: {error_context.get('error_type')}\n"
                f"Duration: {error_context.get('duration_seconds')}s\n"
                f"Timestamp: {error_context.get('timestamp')}"
            )
            
            self.logger.critical(alert_message, extra={'alert': True, 'error_context': error_context})
            
            # TODO: Integrate with external alerting systems
            # Examples:
            # - Send to PagerDuty
            # - Post to Slack channel
            # - Send email notification
            # - Update monitoring dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to trigger error alert: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the RAG tool."""
        stats = {
            'config': {
                'max_chunks': self.config.max_chunks,
                'request_timeout': self.config.request_timeout,
                'query_generation_timeout': self.config.query_generation_timeout,
                'pipeline_timeout': self.config.pipeline_timeout,
                'enable_fallback': self.config.enable_fallback,
                'enable_metrics': self.config.enable_metrics,
                'enable_conversation_logging': self.config.enable_conversation_logging,
                'enable_rate_limiting': self.config.enable_rate_limiting
            }
        }
        
        # Add component stats if available
        if self.rate_limiter:
            stats['rate_limiter'] = self.rate_limiter.get_global_stats()
        
        if self.metrics_collector:
            stats['metrics'] = self.metrics_collector.export_metrics()
        
        if self.connection_manager:
            stats['connection_pools'] = self.connection_manager.get_comprehensive_resource_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all components."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Check query generator
            try:
                # Simple test query generation
                await asyncio.wait_for(
                    self.query_generator.generate_rag_command("test query"),
                    timeout=5.0
                )
                health['components']['query_generator'] = {
                    'status': 'healthy',
                    'response_time_ms': 0  # Would need to measure actual time
                }
            except Exception as e:
                health['components']['query_generator'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
            
            # Check retrieval engine
            try:
                # Check if retrieval engine is properly configured
                config = self.retrieval_engine._load_config()
                health['components']['retrieval_engine'] = {
                    'status': 'healthy',
                    'config_loaded': True,
                    'max_final_chunks': config.get('max_final_chunks', 0)
                }
            except Exception as e:
                health['components']['retrieval_engine'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
            
            # Check connection manager
            if self.connection_manager:
                try:
                    # Check overall health status
                    health_status = self.connection_manager.get_health_status()
                    
                    health['components']['connection_manager'] = {
                        'status': 'healthy' if health_status.get('overall_healthy', False) else 'degraded',
                        'pools': health_status.get('pools', {}),
                        'overall_healthy': health_status.get('overall_healthy', False)
                    }
                    
                    if not health_status.get('overall_healthy', False):
                        health['status'] = 'degraded'
                        
                except Exception as e:
                    health['components']['connection_manager'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health['status'] = 'degraded'
            
            # Check rate limiter
            if self.rate_limiter:
                try:
                    stats = self.rate_limiter.get_global_stats()
                    health['components']['rate_limiter'] = {
                        'status': 'healthy',
                        'active_clients': stats.get('active_clients', 0),
                        'total_requests': stats.get('total_requests', 0)
                    }
                except Exception as e:
                    health['components']['rate_limiter'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health['status'] = 'degraded'
            
            # Check metrics collector
            if self.metrics_collector:
                try:
                    perf_summary = self.metrics_collector.get_performance_summary()
                    health['components']['metrics_collector'] = {
                        'status': 'healthy',
                        'request_count': perf_summary.request_count,
                        'error_rate': perf_summary.error_rate
                    }
                except Exception as e:
                    health['components']['metrics_collector'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health['status'] = 'degraded'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health
    
    async def close(self) -> None:
        """Clean shutdown of all components."""
        self.logger.info("Starting RAG tool shutdown...")
        
        try:
            # Close query generator
            if hasattr(self.query_generator, 'close'):
                await self.query_generator.close()
            
            # Close rate limiter
            if self.rate_limiter and hasattr(self.rate_limiter, 'close'):
                await self.rate_limiter.close()
            
            # Stop metrics collector background tasks
            if self.metrics_collector and hasattr(self.metrics_collector, 'stop_background_tasks'):
                await self.metrics_collector.stop_background_tasks()
            
            # Close connection manager
            if hasattr(self, 'connection_manager') and self.connection_manager and hasattr(self.connection_manager, 'close'):
                await self.connection_manager.close()
            
            self.logger.info("RAG tool shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during RAG tool shutdown: {e}")


# Factory functions for easy instantiation

def create_rag_tool(
    config: Optional[RAGToolConfig] = None,
    enable_all_features: bool = True
) -> EnhancedRAGTool:
    """
    Factory function to create a fully configured RAG tool.
    Returns a cached singleton — the tool is expensive to initialize
    and should not be recreated on every request.
    """
    if config is None:
        config = RAGToolConfig.from_env()

    # Initialize components based on configuration
    components = {}

    if enable_all_features:
        try:
            from utils.rate_limiter import create_rate_limiter_from_env
            components['rate_limiter'] = create_rate_limiter_from_env()
        except Exception as e:
            logger.warning(f"Failed to initialize rate limiter: {e}")

        try:
            from utils.conversation_logger import ConversationLogger
            components['conversation_logger'] = ConversationLogger()
        except Exception as e:
            logger.warning(f"Failed to initialize conversation logger: {e}")

    return EnhancedRAGTool(config=config, **components)


# ── Module-level config cache ─────────────────────────────────────────────────
# Cache only the config (cheap, loop-independent).
# The EnhancedRAGTool itself contains an httpx.AsyncClient which is bound to
# the event loop it was created on — so it cannot be shared across threads.
# We recreate the tool per-call but reuse the config to avoid re-reading env vars.
_rag_tool_config: Optional['RAGToolConfig'] = None


def get_rag_tool() -> 'EnhancedRAGTool':
    """Create a RAG tool using the cached config.

    The tool is NOT a singleton because httpx.AsyncClient is bound to the
    event loop of the thread it was created on. Each ThreadPoolExecutor thread
    gets its own event loop, so we must create a fresh client per call.
    The config is cached to avoid re-reading env vars on every call.
    """
    global _rag_tool_config
    if _rag_tool_config is None:
        _rag_tool_config = RAGToolConfig.from_env()
    return create_rag_tool(config=_rag_tool_config)


def create_minimal_rag_tool(config: Optional[RAGToolConfig] = None) -> EnhancedRAGTool:
    """
    Factory function to create a minimal RAG tool without production features.
    
    Args:
        config: Optional tool configuration
        
    Returns:
        Minimal EnhancedRAGTool instance
    """
    if config is None:
        config = RAGToolConfig(
            enable_rate_limiting=False,
            enable_metrics=False,
            enable_conversation_logging=False
        )
    
    return EnhancedRAGTool(config=config)


# Tool handler function for integration with existing tool registry

async def handle_rag_retrieval(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle rag_retrieval tool call from the LLM.
    
    This function provides backward compatibility with the existing tool registry
    while using the new enhanced RAG tool implementation with comprehensive
    parameter validation.
    
    Args:
        args: Tool arguments containing the user query and optional parameters
              Expected structure:
              {
                  "query": str,  # Required: user query (1-10000 chars)
                  "client_id": str,  # Optional: client identifier (alphanumeric, _, -)
                  "session_id": str,  # Optional: session identifier (alphanumeric, _, -)
                  "max_chunks": int,  # Optional: maximum chunks to return (1-50)
              }
    
    Returns:
        Dict containing formatted response for LLM:
        {
            "success": True,
            "chunks": [
                {
                    "chunk_id": str,
                    "text": str,
                    "document_id": str,
                    "page_number": int,
                    "document_path": str,
                    "origin": str,
                    "confidence": float
                },
                ...
            ],
            "chunk_count": int,
            "message": str
        }
        
        Or on error:
        {
            "success": False,
            "error": str,
            "error_type": str,
            "message": str,
            "details": dict,  # Optional: detailed error information
            "recovery_recommendations": list  # Optional: recovery suggestions
        }
    """
    try:
        # Comprehensive parameter validation
        try:
            validated_params = ParameterValidator.validate_all_parameters(args)
        except ToolValidationError as e:
            # Return detailed validation error
            error_response = {
                "success": False,
                "error": str(e),
                "error_type": "parameter_validation_error",
                "message": f"RAG retrieval failed due to invalid parameters: {str(e)}"
            }
            
            # Add detailed error information if available
            if e.details:
                error_response["details"] = e.details
                if "recovery_recommendations" in e.details:
                    error_response["recovery_recommendations"] = e.details["recovery_recommendations"]
            
            logger.warning(f"Parameter validation failed: {e.get_detailed_message()}")
            return error_response
        
        # Extract validated parameters
        user_query = validated_params["query"]
        client_id = validated_params["client_id"]
        session_id = validated_params["session_id"]
        max_chunks = validated_params["max_chunks"]
        
        # Log successful validation
        logger.debug(
            f"Parameters validated successfully: "
            f"query_length={len(user_query)}, client_id={client_id}, "
            f"session_id={session_id}, max_chunks={max_chunks}"
        )
        
        # Use cached RAG tool singleton — avoids re-initializing LLM client on every call
        rag_tool = get_rag_tool()
        
        try:
            # Execute retrieval — always use config max_chunks, ignore LLM-provided value
            results = await rag_tool.execute_retrieval(
                user_query=user_query,
                client_id=client_id,
                session_id=session_id,
                max_chunks=None  # uses config default (RAG_TOOL_MAX_CHUNKS)
            )
            
            # Format results for LLM
            formatted_chunks = []
            for chunk in results:
                formatted_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "document_path": chunk.document_path,
                    "origin": chunk.origin,
                    "confidence": chunk.confidence
                })
            
            chunk_count = len(formatted_chunks)
            
            # Create success message
            if chunk_count == 0:
                message = "No chunks found matching your search criteria. Try broadening your search terms or removing filters."
            else:
                message = f"Retrieved {chunk_count} chunk(s) from layered graph RAG. IMPORTANT: Each chunk has a 'page_number' field - cite these page numbers in your response!"
            
            logger.info(
                f"RAG retrieval completed successfully: "
                f"query_length={len(user_query)}, client_id={client_id}, "
                f"chunks_returned={chunk_count}"
            )
            
            return {
                "success": True,
                "chunks": formatted_chunks,
                "chunk_count": chunk_count,
                "message": message
            }
            
        finally:
            # Close the per-call tool's httpx client to free the connection
            await rag_tool.close()
            
    except RateLimitExceededError as e:
        error_msg = f"Rate limit exceeded: {str(e)}"
        logger.warning(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "error_type": "rate_limit_exceeded",
            "message": f"RAG retrieval failed: {error_msg}",
            "retry_after": getattr(e, 'retry_after', 60),
            "recovery_recommendations": [
                "Wait for the specified retry_after period before making another request",
                "Implement exponential backoff in your client",
                "Consider distributing requests across multiple clients",
                "Contact support if rate limits are consistently exceeded"
            ]
        }
    
    except TimeoutError as e:
        error_msg = f"Request timeout: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "error_type": "timeout_error",
            "message": f"RAG retrieval failed: {error_msg}",
            "recovery_recommendations": [
                "Retry the request with a simpler query",
                "Reduce the max_chunks parameter to speed up processing",
                "Check system load and database connectivity",
                "Contact support if timeouts persist"
            ]
        }
    
    except ToolValidationError as e:
        # This should be caught above, but handle it here as a fallback
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": "parameter_validation_error",
            "message": f"RAG retrieval failed due to invalid parameters: {str(e)}"
        }
        
        if e.details:
            error_response["details"] = e.details
            if "recovery_recommendations" in e.details:
                error_response["recovery_recommendations"] = e.details["recovery_recommendations"]
        
        logger.error(f"Parameter validation error: {e.get_detailed_message()}")
        return error_response
    
    except LLMCommandError as e:
        error_msg = f"Invalid request: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "error_type": "validation_error",
            "message": f"RAG retrieval failed: {error_msg}",
            "recovery_recommendations": [
                "Verify that the query parameter is provided and not empty",
                "Check that optional parameters are in the correct format",
                "Ensure client_id and session_id are valid strings if provided",
                "Review the API documentation for correct parameter usage"
            ]
        }
    
    except DatabaseError as e:
        error_msg = f"Database operation failed: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "error_type": "database_error",
            "message": f"RAG retrieval failed: {error_msg}",
            "recovery_recommendations": [
                "Retry the request after a brief delay",
                "Check database service status and connectivity",
                "Verify that all required databases (Redis, Qdrant, FalkorDB) are running",
                "Contact support if database errors persist"
            ]
        }
    
    except Exception as e:
        error_msg = f"Unexpected error during RAG retrieval: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Determine if this is a recoverable error
        recoverable_errors = {
            'ConnectionError', 'ConnectTimeout', 'ReadTimeout', 
            'HTTPError', 'RequestException'
        }
        
        is_recoverable = type(e).__name__ in recoverable_errors
        
        recovery_recommendations = [
            "Contact support with the error details and session information",
            "Check system status and try again later"
        ]
        
        if is_recoverable:
            recovery_recommendations.insert(0, "Retry the request after a brief delay")
            recovery_recommendations.insert(1, "Check network connectivity and service availability")
        
        return {
            "success": False,
            "error": error_msg,
            "error_type": "internal_error",
            "message": f"RAG retrieval failed: {error_msg}",
            "is_recoverable": is_recoverable,
            "recovery_recommendations": recovery_recommendations
        }
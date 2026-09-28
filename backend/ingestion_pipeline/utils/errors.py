"""
Custom exception classes for the RAG system.

This module defines specific exception types for different failure modes
in the system, enabling precise error handling and debugging.
"""

from typing import Dict, Any, Optional


class IngestionError(Exception):
    """
    Raised when document ingestion fails.
    
    This exception should be raised for:
    - Invalid JSON structure or content
    - Missing required fields in document data
    - File reading or parsing errors during ingestion
    
    Example:
        raise IngestionError("Chunk paragraph_text exceeds character limit")
    """
    pass


class ValidationError(Exception):
    """
    Raised when Pydantic schema validation fails.
    
    This exception should be raised for:
    - Data that doesn't conform to Pydantic model schemas
    - Type mismatches in input data
    - Constraint violations (e.g., page_number < 1)
    
    Example:
        raise ValidationError("DocumentInput validation failed: page_number must be >= 1")
    """
    pass


class DatabaseError(Exception):
    """
    Raised when database operations fail.
    
    This exception should wrap underlying database errors from:
    - Redis connection or query failures
    - Qdrant connection or indexing failures
    - Falkor DB connection or graph operation failures
    
    The underlying exception should be wrapped to provide context.
    
    Example:
        try:
            redis_client.set(key, value)
        except RedisError as e:
            raise DatabaseError(f"Failed to write to Redis: {e}") from e
    """
    pass


class LLMCommandError(Exception):
    """
    Raised when LLM command validation or processing fails.
    
    This exception should be raised for:
    - Invalid command JSON structure
    - Unknown command mode
    - Missing required parameters in commands
    - Invalid parameter values in commands
    
    Example:
        raise LLMCommandError("Unknown command mode: 'invalid_mode'. Expected 'initial_retrieval' or 'context_expansion'")
    """
    pass


# Tool Calling Errors

class ToolError(Exception):
    """
    Base exception for tool-related errors.
    
    All tool-specific exceptions inherit from this class.
    """
    pass


class ToolNotFoundError(ToolError):
    """
    Raised when a tool call references a tool that doesn't exist in the registry.
    
    Example:
        raise ToolNotFoundError("Tool 'unknown_tool' not found in registry")
    """
    pass


class ToolValidationError(ToolError):
    """
    Raised when tool arguments don't match the tool's JSON schema.
    
    Supports detailed error information for better debugging and user experience.
    
    Attributes:
        message: Human-readable error message
        details: Optional dictionary with detailed error information
    
    Example:
        raise ToolValidationError(
            "Arguments for tool 'rag_retrieval' failed validation: missing required field 'mode'",
            details={
                "parameter": "mode",
                "expected_type": "string",
                "recovery_recommendations": ["Provide a valid mode parameter"]
            }
        )
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize ToolValidationError with optional details.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with detailed error information
        """
        super().__init__(message)
        self.details = details or {}
    
    def get_detailed_message(self) -> str:
        """
        Get a detailed error message including all available information.
        
        Returns:
            Formatted error message with details
        """
        if not self.details:
            return str(self)
        
        lines = [str(self)]
        
        if "parameter" in self.details:
            lines.append(f"Parameter: {self.details['parameter']}")
        
        if "expected_type" in self.details:
            lines.append(f"Expected type: {self.details['expected_type']}")
        
        if "received_type" in self.details:
            lines.append(f"Received type: {self.details['received_type']}")
        
        if "received_value" in self.details:
            lines.append(f"Received value: {self.details['received_value']}")
        
        if "recovery_recommendations" in self.details:
            lines.append("Recovery recommendations:")
            for rec in self.details["recovery_recommendations"]:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


class ToolExecutionError(ToolError):
    """
    Raised when a tool handler execution fails.
    
    This wraps underlying exceptions that occur during tool execution.
    
    Example:
        raise ToolExecutionError("Tool 'fetch_file' execution failed: file not found") from original_error
    """
    pass


# Session Errors

class SessionError(Exception):
    """
    Base exception for session-related errors.
    """
    pass


class SessionNotFoundError(SessionError):
    """
    Raised when a session ID doesn't exist.
    
    Example:
        raise SessionNotFoundError("Session 'abc123' not found")
    """
    pass


class SessionExpiredError(SessionError):
    """
    Raised when a session has expired.
    
    Example:
        raise SessionExpiredError("Session 'abc123' has expired")
    """
    pass


# File Access Errors

class FileAccessError(Exception):
    """
    Base exception for file access errors.
    """
    pass


class PathTraversalError(FileAccessError):
    """
    Raised when a file path contains path traversal attempts.
    
    Example:
        raise PathTraversalError("Path traversal detected in file path")
    """
    pass


class FileTypeNotAllowedError(FileAccessError):
    """
    Raised when a file extension is not in the allowed list.
    
    Example:
        raise FileTypeNotAllowedError("File extension '.exe' is not allowed")
    """
    pass


# LM Studio Errors

class LMStudioError(Exception):
    """
    Base exception for LM Studio API errors.
    """
    pass


class LMStudioConnectionError(LMStudioError):
    """
    Raised when connection to LM Studio API fails.
    
    Example:
        raise LMStudioConnectionError("Failed to connect to LM Studio at http://127.0.0.1:1234")
    """
    pass


class LMStudioTimeoutError(LMStudioError):
    """
    Raised when LM Studio API request times out.
    
    Example:
        raise LMStudioTimeoutError("LM Studio API request timed out after 120 seconds")
    """
    pass


class TimeoutError(Exception):
    """
    Raised when an operation times out.
    
    This exception should be raised for:
    - Database operations that exceed timeout limits
    - Network requests that take too long
    - Long-running computations that exceed time limits
    
    Example:
        raise TimeoutError("Operation timed out after 30 seconds")
    """
    pass


class RateLimitExceededError(Exception):
    """
    Raised when rate limit is exceeded.
    
    This exception should be raised when:
    - Client exceeds configured request rate limits
    - Token bucket is empty
    - Request queuing capacity is exceeded
    
    Attributes:
        retry_after: Seconds to wait before retrying
    
    Example:
        raise RateLimitExceededError("Rate limit exceeded", retry_after=60)
    """
    
    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = retry_after

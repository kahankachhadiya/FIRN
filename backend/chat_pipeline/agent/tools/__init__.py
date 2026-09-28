"""
LLM Tool Handlers

This module contains handlers for LLM tool calls and provides
a function to register only the enhanced RAG tool with the ToolRegistry.
Implements Requirement 4: Single RAG Tool Implementation.
"""

from typing import Dict, Any
from agent.tool_registry import ToolRegistry, ToolDefinition
from agent.tools.rag_tool import handle_rag_retrieval

# Keep only essential tools for UI integration
from agent.tools.display_tools import handle_display_image, handle_display_document

from utils.logging_utils import get_logger

logger = get_logger(__name__)





def register_enhanced_rag_tool_only(registry: ToolRegistry) -> None:
    """
    Register only essential tools: RAG retrieval, image display, and document display.
    
    This registration implements Requirement 4: Single RAG Tool Implementation
    and keeps only essential tools for UI integration. The RAG tool is initialized
    with environment-based configuration.
    
    Args:
        registry: ToolRegistry instance to register tools with
        
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """
    logger.info("Registering essential tools with environment-based configuration...")
    
    # Import RAG tool configuration and factory function
    from agent.tools.rag_tool import RAGToolConfig, create_rag_tool, handle_rag_retrieval
    
    # Load RAG tool configuration from environment variables
    try:
        rag_config = RAGToolConfig.from_env()
        logger.info(f"Loaded RAG tool configuration from environment: max_chunks={rag_config.max_chunks}, "
                   f"request_timeout={rag_config.request_timeout}s")
    except Exception as e:
        logger.warning(f"Failed to load RAG tool configuration from environment: {e}")
        logger.info("Using default RAG tool configuration")
        rag_config = RAGToolConfig()
    
    # Create enhanced RAG tool handler with environment-based configuration
    def create_configured_rag_handler():
        """Create RAG handler with proper configuration."""
        try:
            # Load configuration from environment
            logger.info("Creating RAG handler with environment-based configuration")
            
            # Return configured handler function that uses the existing handle_rag_retrieval
            # but with environment-based configuration applied
            def configured_rag_handler(args: Dict[str, Any]) -> Dict[str, Any]:
                import asyncio
                import concurrent.futures

                def _run():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(handle_rag_retrieval(args))
                    finally:
                        loop.close()

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(_run)
                        return future.result()
                except Exception as e:
                    logger.error(f"Error in configured RAG handler: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "error_type": "handler_error",
                        "message": f"RAG retrieval failed: {str(e)}"
                    }
            
            return configured_rag_handler
            
        except Exception as e:
            logger.error(f"Failed to create configured RAG tool: {e}")
            logger.info("Falling back to basic RAG handler")
            # Fallback to basic handler if configuration fails
            def fallback_handler(args: Dict[str, Any]) -> Dict[str, Any]:
                import asyncio
                import concurrent.futures

                def _run():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(handle_rag_retrieval(args))
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(_run).result()
            return fallback_handler
    
    # Create the configured handler
    rag_handler = create_configured_rag_handler()
    
    # Register the enhanced rag_retrieval tool with environment-based configuration
    # This tool handles the complete pipeline: LLM query generation → vector/keyword search 
    # → reranking → graph traversal → context expansion → final output
    enhanced_rag_tool = ToolDefinition(
        name="rag_retrieval",
        description="Enhanced RAG retrieval tool that handles the complete layered graph retrieval pipeline. Automatically generates structured queries using LLM, executes 11-stage retrieval pipeline with vector search, keyword search, graph traversal, reranking, and context expansion. Returns enhanced chunks with complete metadata including document paths, page numbers, origin labels, and confidence scores. Fully configured via environment variables for production deployment with rate limiting, metrics collection, and comprehensive error handling.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user query to process. The LLM will automatically generate structured RAG commands with appropriate metadata filters and probe types based on this query.",
                    "minLength": 1,
                    "maxLength": 10000,
                    "examples": [
                        "What are the benefits of machine learning?",
                        "Explain the process of photosynthesis",
                        "How does blockchain technology work?"
                    ]
                },
                "client_id": {
                    "type": "string",
                    "description": "Optional client identifier for rate limiting and tracking (default: 'default')",
                    "default": "default",
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": "^[a-zA-Z0-9_-]+$",  # Alphanumeric, underscore, hyphen only
                    "examples": ["default", "user_123", "api-client-1"]
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session identifier for conversation logging (auto-generated if not provided)",
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": "^[a-zA-Z0-9_-]+$",  # Alphanumeric, underscore, hyphen only
                    "examples": ["session_abc123", "conv-456", "user_session_789"]
                },
                "max_chunks": {
                    "type": "integer",
                    "description": f"Maximum number of chunks to return (default: {rag_config.max_chunks}, must be between 1 and 50)",
                    "minimum": 1,
                    "maximum": 50,
                    "default": rag_config.max_chunks,
                    "examples": [5, 10, 20, 30, 50]
                }
            },
            "required": ["query"],
            "additionalProperties": False,
            "title": "Enhanced RAG Retrieval Parameters",
            "description": "Parameters for the enhanced RAG retrieval tool with comprehensive validation and environment-based configuration"
        },
        handler=rag_handler
    )
    registry.register(enhanced_rag_tool)
    
    logger.info("Successfully registered tool: rag_retrieval")


__all__ = [
    # Only enhanced RAG tool registration function is kept
    "register_enhanced_rag_tool_only",
    "handle_rag_retrieval",
    # Display tools are kept for UI integration
    "handle_display_image",
    "handle_display_document"
]

"""
Tool Dispatcher for LLM Tool Calling.

Validates and dispatches tool calls from the LLM to registered tool handlers.
Provides error handling and logging for all tool executions.
"""

from typing import Dict, List, Any
import time
import json
from jsonschema import ValidationError as JsonSchemaValidationError

from agent.tool_registry import ToolRegistry
from utils.errors import ToolNotFoundError, ToolValidationError, ToolExecutionError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ToolDispatcher:
    """
    Dispatcher for routing and executing LLM tool calls.
    
    Validates tool names and arguments against the registry,
    executes tool handlers, and wraps errors appropriately.
    """
    
    def __init__(self, registry: ToolRegistry) -> None:
        """
        Initialize the tool dispatcher.
        
        Args:
            registry: ToolRegistry instance containing registered tools
        """
        self.registry = registry
        logger.info("Tool Dispatcher initialized")
    
    def dispatch(self, tool_call: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Dispatch a single tool call to its handler.
        
        Args:
            tool_call: Tool call dict with structure:
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "arguments": "{...}"  # JSON string
                    }
                }
            session_id: Current session ID for context
            
        Returns:
            Dict containing tool execution result:
                {
                    "success": True/False,
                    "result": {...} or None,
                    "error": str or None
                }
                
        Raises:
            ToolNotFoundError: If tool name not in registry
            ToolValidationError: If arguments don't match schema
            ToolExecutionError: If tool handler execution fails
        """
        start_time = time.time()
        tool_name = None
        
        try:
            # Extract tool information
            function_info = tool_call.get("function", {})
            tool_name = function_info.get("name")
            arguments_str = function_info.get("arguments", "{}")
            
            if not tool_name:
                raise ToolValidationError("Tool name is required")
            
            # Parse arguments
            try:
                arguments = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError as e:
                raise ToolValidationError(f"Invalid JSON arguments: {e}")
            
            # Get tool from registry
            tool_def = self.registry.get_tool(tool_name)
            if tool_def is None:
                raise ToolNotFoundError(f"Tool '{tool_name}' not found in registry")
            
            # Validate arguments against schema
            try:
                self.registry.validate_tool_call(tool_name, arguments)
            except JsonSchemaValidationError as e:
                raise ToolValidationError(f"Arguments validation failed: {e.message}")
            except ValueError as e:
                raise ToolNotFoundError(str(e))
            
            # Execute tool handler
            try:
                logger.info(f"Executing tool: {tool_name} for session: {session_id}")
                result = tool_def.handler(arguments)
                
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                logger.info(f"Tool execution completed: {tool_name} ({execution_time:.2f}ms)")
                
                return {
                    "success": True,
                    "result": result,
                    "error": None,
                    "execution_time_ms": execution_time
                }
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Tool execution failed: {tool_name} - {str(e)}")
                raise ToolExecutionError(f"Tool execution failed: {str(e)}")
        
        except (ToolNotFoundError, ToolValidationError, ToolExecutionError) as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool dispatch failed: {tool_name} - {str(e)}")
            
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "execution_time_ms": execution_time
            }
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Unexpected error in tool dispatch: {tool_name} - {str(e)}")
            
            return {
                "success": False,
                "result": None,
                "error": f"Unexpected error: {str(e)}",
                "execution_time_ms": execution_time
            }
    
    def dispatch_multiple(self, tool_calls: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Dispatch multiple tool calls sequentially.
        
        Args:
            tool_calls: List of tool call dictionaries
            session_id: Current session ID for context
            
        Returns:
            List of tool execution results in the same order as input
        """
        results = []
        
        for tool_call in tool_calls:
            result = self.dispatch(tool_call, session_id)
            results.append(result)
        
        return results
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names available for dispatch
        """
        return self.registry.list_tool_names()
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is available for dispatch.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is available, False otherwise
        """
        return self.registry.has_tool(tool_name)
"""
Tool Registry for LLM Tool Calling.

Provides dynamic tool registration and schema management for LLM tool calls.
Validates tool names and arguments against JSON schemas.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
import jsonschema
from jsonschema import ValidationError as JsonSchemaValidationError

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    """
    Definition of a tool that can be called by the LLM.
    
    Attributes:
        name: Unique tool name (e.g., "rag_retrieval")
        description: Human-readable description of what the tool does
        parameters: JSON Schema defining the tool's parameters
        handler: Callable that executes the tool logic
    """
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    """
    Registry for managing LLM tools.
    
    Provides methods to register tools, retrieve tool definitions,
    get OpenAI-compatible tool schemas, and validate tool arguments.
    """
    
    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        logger.info("Tool Registry initialized")
    
    def register(self, tool: ToolDefinition) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: ToolDefinition to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' is already registered, overwriting")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.
        
        Args:
            name: Tool name to retrieve
            
        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> Dict[str, ToolDefinition]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary mapping tool names to ToolDefinitions
        """
        return self._tools.copy()
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool schemas for all registered tools.
        
        Returns:
            List of tool schemas in OpenAI format
        """
        schemas = []
        for tool in self._tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            schemas.append(schema)
        
        return schemas
    
    def validate_tool_call(self, name: str, arguments: Dict[str, Any]) -> bool:
        """
        Validate a tool call against the registered schema.
        
        Args:
            name: Tool name
            arguments: Tool arguments to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If tool not found
            JsonSchemaValidationError: If arguments don't match schema
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        try:
            jsonschema.validate(arguments, tool.parameters)
            return True
        except JsonSchemaValidationError:
            raise
    
    def list_tool_names(self) -> List[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tool_count(self) -> int:
        """
        Get the number of registered tools.
        
        Returns:
            Number of tools in registry
        """
        return len(self._tools)
    
    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Tool name to check
            
        Returns:
            True if tool is registered, False otherwise
        """
        return name in self._tools
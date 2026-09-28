"""
JSON utility functions for loading and validating JSON files.
"""

import json
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from utils.errors import ValidationError, IngestionError


T = TypeVar('T', bound=BaseModel)


def load_json_file(path: str) -> dict:
    """
    Load a JSON file and return its contents as a dictionary.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary containing the parsed JSON data
        
    Raises:
        IngestionError: If the file cannot be read or parsed as JSON
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise IngestionError(f"JSON file not found: {path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict):
            raise IngestionError(f"JSON file must contain an object, got {type(data).__name__}")
            
        return data
        
    except json.JSONDecodeError as e:
        raise IngestionError(f"Invalid JSON in file {path}: {e}") from e
    except IOError as e:
        raise IngestionError(f"Failed to read file {path}: {e}") from e


def validate_json_schema(data: dict, schema: Type[T]) -> T:
    """
    Validate a dictionary against a Pydantic schema.
    
    Args:
        data: Dictionary to validate
        schema: Pydantic BaseModel class to validate against
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        ValidationError: If the data doesn't conform to the schema
    """
    try:
        return schema(**data)
    except PydanticValidationError as e:
        raise ValidationError(f"Schema validation failed for {schema.__name__}: {e}") from e

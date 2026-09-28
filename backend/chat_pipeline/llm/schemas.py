"""
Data schemas for the Layered Graph RAG system.

This module defines the data models used for the enhanced RAG retrieval pipeline,
including chunk output formats and command structures.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class EnhancedChunkOutput:
    """
    Enhanced chunk output with images embedded in text content.
    
    This dataclass represents the corrected chunk output format where images
    are embedded within the text content as references like [Image: /path/to/image.png]
    rather than as a separate image_urls field.
    
    Attributes:
        chunk_id: Unique identifier for the chunk (e.g., "DOC123_P5_C2")
        document_id: Identifier for the source document (e.g., "DOC123")
        page_number: Page number where the chunk appears (1-indexed)
        document_path: Full path to the source document (e.g., "/documents/research_paper.pdf")
        text: Complete text content with embedded image references in the format
              [Image: /path/to/image.png]. Images are NOT stored as a separate field.
        origin: Label indicating retrieval source. Valid values:
                - "anchor": Top chunk from reranking
                - "relation": From relation graph traversal
                - "probe_supports": From SUPPORTS probe graph
                - "probe_contradicts": From CONTRADICTS probe graph
                - "probe_example": From EXAMPLE_OF probe graph
                - "probe_elaborates": From ELABORATES probe graph
                - "probe_depends": From DEPENDS_ON probe graph
                - "context": From context expansion (±1 adjacent chunks)
        confidence: Confidence score for the chunk (0.0 to 1.0)
    
    Example:
        >>> chunk = EnhancedChunkOutput(
        ...     chunk_id="DOC123_P5_C2",
        ...     document_id="DOC123",
        ...     page_number=5,
        ...     document_path="/documents/research_paper.pdf",
        ...     text="The neural network architecture... [Image: /documents/figures/fig1.png] ...shows the layers.",
        ...     origin="anchor",
        ...     confidence=0.92
        ... )
    """
    chunk_id: str
    document_id: str
    page_number: int
    document_path: str
    text: str  # Contains embedded image references like [Image: /path/to/image.png]
    origin: str  # anchor, relation, probe_supports, probe_contradicts, probe_example, probe_elaborates, probe_depends, context
    confidence: float
    
    def __post_init__(self):
        """Validate chunk data after initialization."""
        # Validate required fields are not empty
        if not isinstance(self.chunk_id, str):
            raise ValueError("chunk_id must be a string")
        
        if not isinstance(self.document_id, str):
            raise ValueError("document_id must be a string")
        
        if not isinstance(self.page_number, int) or self.page_number < 0:
            raise ValueError("page_number must be a non-negative integer")
        
        if not isinstance(self.document_path, str):
            raise ValueError("document_path must be a string")
        
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        
        # Validate origin is one of the expected values
        valid_origins = {
            "anchor", "relation", 
            "probe_supports", "probe_contradicts", "probe_example", 
            "probe_elaborates", "probe_depends", 
            "context"
        }
        if self.origin not in valid_origins:
            raise ValueError(
                f"origin must be one of {valid_origins}, got '{self.origin}'"
            )
        
        # Validate confidence is between 0 and 1
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be a number between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the chunk to a dictionary representation.
        
        Returns:
            Dictionary containing all chunk fields
            
        Example:
            >>> chunk = EnhancedChunkOutput(...)
            >>> chunk_dict = chunk.to_dict()
            >>> print(chunk_dict['chunk_id'])
            'DOC123_P5_C2'
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert the chunk to a JSON string representation.
        
        Returns:
            JSON string containing all chunk fields
            
        Example:
            >>> chunk = EnhancedChunkOutput(...)
            >>> json_str = chunk.to_json()
            >>> print(json_str)
            '{"chunk_id": "DOC123_P5_C2", ...}'
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedChunkOutput':
        """
        Create an EnhancedChunkOutput instance from a dictionary.
        
        Args:
            data: Dictionary containing chunk fields
            
        Returns:
            EnhancedChunkOutput instance
            
        Raises:
            ValueError: If required fields are missing or invalid
            
        Example:
            >>> data = {
            ...     'chunk_id': 'DOC123_P5_C2',
            ...     'document_id': 'DOC123',
            ...     'page_number': 5,
            ...     'document_path': '/documents/research_paper.pdf',
            ...     'text': 'Content with [Image: /path/to/image.png]',
            ...     'origin': 'anchor',
            ...     'confidence': 0.92
            ... }
            >>> chunk = EnhancedChunkOutput.from_dict(data)
        """
        # Validate required fields are present
        required_fields = {
            'chunk_id', 'document_id', 'page_number', 
            'document_path', 'text', 'origin', 'confidence'
        }
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return cls(
            chunk_id=data['chunk_id'],
            document_id=data['document_id'],
            page_number=data['page_number'],
            document_path=data['document_path'],
            text=data['text'],
            origin=data['origin'],
            confidence=data['confidence']
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EnhancedChunkOutput':
        """
        Create an EnhancedChunkOutput instance from a JSON string.
        
        Args:
            json_str: JSON string containing chunk fields
            
        Returns:
            EnhancedChunkOutput instance
            
        Raises:
            ValueError: If JSON is invalid or required fields are missing
            json.JSONDecodeError: If JSON string is malformed
            
        Example:
            >>> json_str = '{"chunk_id": "DOC123_P5_C2", ...}'
            >>> chunk = EnhancedChunkOutput.from_json(json_str)
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
        
        if not isinstance(data, dict):
            raise ValueError("JSON must represent a dictionary")
        
        return cls.from_dict(data)

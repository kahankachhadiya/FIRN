"""JSON builder module for structured output generation.

This module constructs structured JSON output conforming to the unified schema,
including document metadata and chunks array.
"""

import json
from typing import List, Optional


class JSONBuilder:
    """
    Builder for constructing structured JSON output.
    
    This class combines metadata and chunks to create a complete JSON structure
    matching the unified pipeline output schema.
    """
    
    def __init__(self):
        """Initialize JSON builder."""
        pass
    
    def build_output(
        self,
        metadata: dict,
        chunks: List[dict],
        file_path: str,
        source_type: str,
        file_name: str,
        processing_device: Optional[str] = None
    ) -> dict:
        """
        Build complete JSON structure matching unified output schema.
        
        Args:
            metadata: Document metadata dict from Qwen extraction
            chunks: List of chunk dicts from Qwen chunking
            file_path: Absolute path to source document
            source_type: Type of source file (pdf, audio, docx, image, etc.)
            file_name: Name of the source file
            processing_device: Device used for processing (cuda or cpu), optional
            
        Returns:
            Dictionary ready for JSON serialization
        """
        # Build document metadata section
        document_metadata = self._build_document_metadata(
            metadata, file_path, source_type, file_name, processing_device, len(chunks)
        )
        
        # Build chunks array (may need to handle timestamp vs page_number)
        processed_chunks = self._process_chunks(chunks, source_type)
        
        # Construct final output
        output = {
            "document_metadata": document_metadata,
            "chunks": processed_chunks
        }
        
        return output
    
    def _build_document_metadata(
        self,
        metadata: dict,
        file_path: str,
        source_type: str,
        file_name: str,
        processing_device: Optional[str] = None,
        chunk_count: Optional[int] = None
    ) -> dict:
        """
        Build document_metadata section from Qwen extraction.
        
        Args:
            metadata: Metadata dict from Qwen (contains title, authors, created_date, etc.)
            file_path: Absolute path to source document
            source_type: Type of source file (pdf, audio, docx, image, etc.)
            file_name: Name of the source file
            processing_device: Device used for processing (cuda or cpu), optional
            chunk_count: Total number of chunks generated, optional
            
        Returns:
            Document metadata dictionary matching schema
        """
        # Extract from Qwen metadata or use defaults
        title = metadata.get("title", file_name)
        
        # Handle author field - Qwen returns "authors" array, schema wants "author" string
        authors = metadata.get("authors", [])
        author = ", ".join(authors) if authors else ""
        
        created_date = metadata.get("created_date", None)
        language = metadata.get("language", "en")
        tags = metadata.get("tags", [])
        
        # Get total_pages from metadata or extra_metadata
        total_pages = metadata.get("total_pages", None)
        if total_pages is None and "extra_metadata" in metadata:
            total_pages = metadata["extra_metadata"].get("total_pages", None)
        
        # Build extra_metadata from Qwen's extra_metadata
        extra_metadata = metadata.get("extra_metadata", {}).copy()
        
        # Add processing_device to extra_metadata if provided (Requirements 1.5, 6.3)
        if processing_device is not None:
            extra_metadata["processing_device"] = processing_device
        
        # Add chunk_count to extra_metadata if provided (Requirement 6.3)
        if chunk_count is not None:
            extra_metadata["chunk_count"] = chunk_count
        
        # Construct document_metadata matching schema
        doc_metadata = {
            "title": title,
            "path": file_path,
            "author": author,
            "created_date": created_date,
            "source_type": source_type,
            "language": language,
            "total_pages": total_pages,
            "file_name": file_name,
            "tags": tags,
            "extra_metadata": extra_metadata
        }
        
        return doc_metadata
    
    def _process_chunks(self, chunks: List[dict], source_type: str) -> List[dict]:
        """
        Process chunks array, handling audio timestamp vs page_number.
        
        Args:
            chunks: List of chunk dicts from Qwen chunking
            source_type: Type of source file (pdf, audio, docx, image, etc.)
            
        Returns:
            List of processed chunk dictionaries
        
        Raises:
            ValueError: If any chunk is missing required fields (Requirement 5.5, 6.2)
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Validate chunk completeness (Requirement 5.5, 6.2)
            if "paragraph_text" not in chunk or not chunk["paragraph_text"]:
                raise ValueError(f"Chunk {i} missing required field 'paragraph_text'")
            if "summary" not in chunk or not chunk["summary"]:
                raise ValueError(f"Chunk {i} missing required field 'summary'")
            
            # For audio files, use timestamp instead of page_number
            if source_type == "audio":
                # Ensure timestamp field exists, remove page_number if present
                processed_chunk = {
                    "timestamp": chunk.get("timestamp", chunk.get("page_number", "00:00:00")),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "paragraph_text": chunk.get("paragraph_text", ""),
                    "summary": chunk.get("summary", "")
                }
            else:
                # For non-audio files, use page_number
                processed_chunk = {
                    "page_number": chunk.get("page_number", 1),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "paragraph_text": chunk.get("paragraph_text", ""),
                    "summary": chunk.get("summary", "")
                }
            
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def save_to_file(self, output: dict, file_path: str):
        """
        Save structured output to JSON file.
        
        Args:
            output: Output dictionary
            file_path: Path to save JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

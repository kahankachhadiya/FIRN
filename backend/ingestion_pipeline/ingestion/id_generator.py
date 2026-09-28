"""
ID generation utilities for documents and chunks.

This module provides functions to generate stable, deterministic IDs
for documents and chunks based on their metadata and position.
"""

import hashlib
from ingestion.schemas import DocumentMetadata


def generate_document_id(metadata: DocumentMetadata) -> str:
    """
    Generate a deterministic Document ID from metadata.
    
    The Document ID is generated using SHA1 hash of the concatenation of:
    - file_name
    - created_date
    - title
    
    This ensures that the same document (identified by these three fields)
    always receives the same ID, enabling idempotent ingestion.
    
    Args:
        metadata: DocumentMetadata instance
        
    Returns:
        Document ID in format DOC_{SHA1_HASH[:16]}
        
    Example:
        >>> metadata = DocumentMetadata(
        ...     title="My Document",
        ...     author="John Doe",
        ...     created_date="2024-01-01",
        ...     source_type="pdf",
        ...     total_pages=10,
        ...     file_name="document.pdf"
        ... )
        >>> doc_id = generate_document_id(metadata)
        >>> doc_id.startswith("DOC_")
        True
    """
    # Concatenate the three identifying fields
    id_string = f"{metadata.file_name}{metadata.created_date}{metadata.title}"
    
    # Generate SHA1 hash
    hash_object = hashlib.sha1(id_string.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    # Return first 16 characters of hash with DOC_ prefix
    return f"DOC_{hash_hex[:16]}"


def generate_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    """
    Generate a Chunk ID from document ID and position.
    
    The Chunk ID format is: {DOCUMENT_ID}_P{page_number:02d}_C{chunk_index:02d}
    
    Args:
        document_id: The document's ID
        page_number: Page number (1-indexed)
        chunk_index: Chunk index within the page (1-indexed)
        
    Returns:
        Chunk ID in the specified format
        
    Example:
        >>> generate_chunk_id("DOC_abc123", 5, 3)
        'DOC_abc123_P05_C03'
    """
    return f"{document_id}_P{page_number:02d}_C{chunk_index:02d}"

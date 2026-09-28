"""
Metadata Client for chunk ordering and adjacent chunk retrieval.

This module provides a clean interface for retrieving adjacent chunks based on
document ordering (page number and chunk index). It wraps the underlying Elasticsearch
client functions with additional error handling and logging.

The metadata client is used during context expansion (Stage 9) to retrieve
adjacent chunks (±N) around an anchor chunk based on their position in the
original document.
"""

from typing import List, Dict, Any, Optional
from db.elasticsearch_client import get_paragraph
from utils.logging_utils import get_logger
from utils.retry import with_retry, ELASTICSEARCH_RETRY_CONFIG


logger = get_logger(__name__)


def parse_chunk_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Parse chunk ID to extract document ID, page number, and chunk index.
    
    Expected format: DOC{id}_P{page}_C{index}
    Example: DOC123_P5_C2 -> document_id=DOC123, page_number=5, chunk_index=2
    
    The document ID can contain underscores, so we parse from the end:
    - Last part must be C{index}
    - Second to last must be P{page}
    - Everything before that is the document ID
    
    Args:
        chunk_id: Chunk identifier in standard format
        
    Returns:
        Dictionary with document_id, page_number, and chunk_index if valid,
        None if parsing fails
    
    Example:
        >>> parse_chunk_id("DOC123_P5_C2")
        {'document_id': 'DOC123', 'page_number': 5, 'chunk_index': 2}
        >>> parse_chunk_id("DOC_ABC_123_P10_C0")
        {'document_id': 'DOC_ABC_123', 'page_number': 10, 'chunk_index': 0}
    """
    try:
        parts = chunk_id.split('_')
        if len(parts) < 3:
            logger.warning(f"Invalid chunk ID format: {chunk_id} (expected DOC*_P*_C*)")
            return None
        
        # Parse from the end to handle document IDs with underscores
        # Last part: C{index}
        if not parts[-1].startswith('C'):
            logger.warning(f"Invalid chunk format in chunk ID: {chunk_id}")
            return None
        chunk_index = int(parts[-1][1:])
        
        # Second to last part: P{page}
        if not parts[-2].startswith('P'):
            logger.warning(f"Invalid page format in chunk ID: {chunk_id}")
            return None
        page_number = int(parts[-2][1:])
        
        # Everything before that is the document ID
        document_id = '_'.join(parts[:-2])
        
        return {
            'document_id': document_id,
            'page_number': page_number,
            'chunk_index': chunk_index
        }
        
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse chunk ID {chunk_id}: {e}")
        return None


def construct_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    """
    Construct chunk ID from components.
    
    Args:
        document_id: Document identifier
        page_number: Page number
        chunk_index: Chunk index within the page
        
    Returns:
        Chunk ID in standard format: DOC{id}_P{page}_C{index}
    
    Example:
        >>> construct_chunk_id("DOC123", 5, 2)
        'DOC123_P5_C2'
    """
    return f"{document_id}_P{page_number}_C{chunk_index}"


@with_retry(
    max_retries=ELASTICSEARCH_RETRY_CONFIG.max_retries,
    base_delay=ELASTICSEARCH_RETRY_CONFIG.base_delay,
    max_delay=ELASTICSEARCH_RETRY_CONFIG.max_delay,
    retryable_exceptions=ELASTICSEARCH_RETRY_CONFIG.retryable_exceptions
)
def get_adjacent_chunks(
    chunk_id: str,
    document_id: Optional[str] = None,
    adjacent_count: int = 1
) -> List[Dict[str, Any]]:
    """
    Retrieve adjacent chunks (±N) around a given chunk based on document ordering.
    
    This function retrieves chunks that are adjacent to the anchor chunk in the
    original document. It uses the chunk index to determine ordering and retrieves
    chunks before and after the anchor.
    
    The function handles edge cases gracefully:
    - Negative indices: Skips chunks with negative indices (before document start)
    - Missing chunks: Skips chunks that don't exist in the database
    - Invalid chunk ID: Returns empty list
    
    Includes automatic retry logic with exponential backoff for failures.
    
    Args:
        chunk_id: Anchor chunk identifier to find adjacent chunks for
        document_id: Optional document ID (extracted from chunk_id if not provided)
        adjacent_count: Number of adjacent chunks to retrieve on each side (±N).
                       For example, adjacent_count=1 retrieves 1 chunk before and
                       1 chunk after (total of 2 chunks).
    
    Returns:
        List of adjacent chunk dictionaries, each containing:
        - chunk_id: Unique identifier of the adjacent chunk
        - document_id: Document identifier
        - page_number: Page number of the chunk
        - chunk_index: Chunk index within the page
        - text: Chunk text content
        - origin: Set to 'context' for all adjacent chunks
        
        Returns empty list if chunk_id is invalid or no adjacent chunks found.
    
    Example:
        >>> # Get ±1 adjacent chunks around DOC123_P5_C2
        >>> adjacent = get_adjacent_chunks("DOC123_P5_C2", adjacent_count=1)
        >>> # Returns chunks: DOC123_P5_C1 and DOC123_P5_C3 (if they exist)
        >>> 
        >>> # Get ±2 adjacent chunks
        >>> adjacent = get_adjacent_chunks("DOC123_P5_C2", adjacent_count=2)
        >>> # Returns chunks: DOC123_P5_C0, DOC123_P5_C1, DOC123_P5_C3, DOC123_P5_C4
    """
    adjacent_chunks: List[Dict[str, Any]] = []
    
    try:
        # Parse anchor chunk ID
        parsed = parse_chunk_id(chunk_id)
        if not parsed:
            logger.warning(f"Cannot retrieve adjacent chunks for invalid chunk ID: {chunk_id}")
            return adjacent_chunks
        
        # Use provided document_id or extracted one
        doc_id = document_id or parsed['document_id']
        page_num = parsed['page_number']
        anchor_index = parsed['chunk_index']
        
        # Generate adjacent chunk IDs (±N chunks)
        for offset in range(-adjacent_count, adjacent_count + 1):
            if offset == 0:
                continue  # Skip anchor chunk itself
            
            adjacent_index = anchor_index + offset
            
            # Skip negative indices (before document start)
            if adjacent_index < 0:
                logger.debug(f"Skipping negative chunk index: {adjacent_index}")
                continue
            
            # Construct adjacent chunk ID
            adjacent_chunk_id = construct_chunk_id(doc_id, page_num, adjacent_index)
            
            # Try to retrieve adjacent chunk text from Elasticsearch
            text = get_paragraph(adjacent_chunk_id)
            
            if text:
                adjacent_chunk = {
                    'chunk_id': adjacent_chunk_id,
                    'document_id': doc_id,
                    'page_number': page_num,
                    'chunk_index': adjacent_index,
                    'text': text,
                    'origin': 'context'
                }
                adjacent_chunks.append(adjacent_chunk)
            else:
                logger.debug(f"Adjacent chunk not found: {adjacent_chunk_id}")
        
        logger.debug(
            f"Retrieved {len(adjacent_chunks)} adjacent chunks for {chunk_id} "
            f"(±{adjacent_count})"
        )
        
        return adjacent_chunks
        
    except Exception as e:
        logger.error(f"Failed to retrieve adjacent chunks for {chunk_id}: {e}")
        return []


def get_adjacent_chunks_batch(
    chunk_ids: List[str],
    adjacent_count: int = 1
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve adjacent chunks for multiple anchor chunks in batch.
    
    This is a convenience function that retrieves adjacent chunks for multiple
    anchor chunks and returns a dictionary mapping each chunk_id to its
    adjacent chunks.
    
    Args:
        chunk_ids: List of anchor chunk identifiers
        adjacent_count: Number of adjacent chunks to retrieve on each side (±N)
    
    Returns:
        Dictionary mapping chunk_id to list of adjacent chunks.
        Chunks with no adjacent chunks will have empty lists.
    
    Example:
        >>> results = get_adjacent_chunks_batch(
        ...     ["DOC123_P5_C2", "DOC123_P6_C1"],
        ...     adjacent_count=1
        ... )
        >>> for chunk_id, adjacent in results.items():
        ...     print(f"{chunk_id}: {len(adjacent)} adjacent chunks")
    """
    results = {}
    
    for chunk_id in chunk_ids:
        adjacent = get_adjacent_chunks(
            chunk_id=chunk_id,
            adjacent_count=adjacent_count
        )
        results[chunk_id] = adjacent
    
    logger.debug(
        f"Batch retrieval completed for {len(chunk_ids)} chunks, "
        f"found adjacent chunks for {sum(1 for a in results.values() if a)} chunks"
    )
    
    return results


def get_chunk_ordering_info(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Get ordering information for a chunk.
    
    This function extracts the ordering information from a chunk ID, which
    can be used to determine the chunk's position in the document and find
    adjacent chunks.
    
    Args:
        chunk_id: Chunk identifier
        
    Returns:
        Dictionary with ordering information:
        - document_id: Document identifier
        - page_number: Page number
        - chunk_index: Chunk index within the page
        - chunk_id: Original chunk identifier
        
        Returns None if chunk_id is invalid.
    
    Example:
        >>> info = get_chunk_ordering_info("DOC123_P5_C2")
        >>> print(f"Chunk {info['chunk_id']} is at page {info['page_number']}, index {info['chunk_index']}")
    """
    parsed = parse_chunk_id(chunk_id)
    if not parsed:
        return None
    
    return {
        'chunk_id': chunk_id,
        'document_id': parsed['document_id'],
        'page_number': parsed['page_number'],
        'chunk_index': parsed['chunk_index']
    }


def validate_chunk_ordering(chunk_ids: List[str]) -> bool:
    """
    Validate that a list of chunk IDs are in correct document order.
    
    This function checks if chunks are ordered by page number and chunk index.
    Useful for validating context expansion results.
    
    Args:
        chunk_ids: List of chunk identifiers to validate
        
    Returns:
        True if chunks are in correct order, False otherwise
    
    Example:
        >>> chunks = ["DOC123_P5_C1", "DOC123_P5_C2", "DOC123_P5_C3"]
        >>> validate_chunk_ordering(chunks)
        True
        >>> 
        >>> chunks = ["DOC123_P5_C3", "DOC123_P5_C1"]
        >>> validate_chunk_ordering(chunks)
        False
    """
    if len(chunk_ids) < 2:
        return True
    
    try:
        # Parse all chunk IDs
        parsed_chunks = []
        for chunk_id in chunk_ids:
            parsed = parse_chunk_id(chunk_id)
            if not parsed:
                return False
            parsed_chunks.append(parsed)
        
        # Check ordering
        for i in range(1, len(parsed_chunks)):
            prev = parsed_chunks[i - 1]
            curr = parsed_chunks[i]
            
            # Check if from same document
            if prev['document_id'] != curr['document_id']:
                logger.warning("Chunks from different documents in ordering validation")
                return False
            
            # Check page ordering
            if curr['page_number'] < prev['page_number']:
                return False
            
            # If same page, check chunk index ordering
            if curr['page_number'] == prev['page_number']:
                if curr['chunk_index'] <= prev['chunk_index']:
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate chunk ordering: {e}")
        return False

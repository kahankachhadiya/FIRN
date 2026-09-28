"""
Metadata Client for chunk ordering, adjacent chunk retrieval, and metadata filtering.

This module provides:
- Resolving metadata filters to matching document_ids
- Retrieving adjacent chunks based on document ordering (context expansion)
"""

from typing import List, Dict, Any, Optional
from db.elasticsearch_client import get_paragraph, get_document_chunk_ids, get_paragraphs_batch, _get_client, _metadata_index
from utils.logging_utils import get_logger
from utils.retry import with_retry, ELASTICSEARCH_RETRY_CONFIG


logger = get_logger(__name__)


def resolve_metadata_filter(metadata_filter: Dict[str, Any]) -> Optional[List[str]]:
    """
    Resolve a metadata filter to a list of matching document_ids.

    Queries the documents metadata index and returns the IDs of documents
    that match ALL provided filter criteria (AND logic).

    Supported filter fields:
        title        (str) — case-insensitive substring match
        author       (str) — case-insensitive substring match
        created_date (str) — exact date match (YYYY-MM-DD)
        source_type  (str) — exact match e.g. "pdf"
        language     (str) — exact match e.g. "en"

    Returns:
        List of matching document_ids, None if filter is empty/invalid,
        or empty list if valid but no documents match.
    """
    if not metadata_filter:
        return None

    must_clauses = []

    title = metadata_filter.get("title")
    if title:
        must_clauses.append({"match": {"metadata.title": {"query": title, "operator": "and"}}})

    author = metadata_filter.get("author")
    if author:
        must_clauses.append({"match": {"metadata.author": {"query": author, "operator": "and"}}})

    created_date = metadata_filter.get("created_date")
    if created_date:
        must_clauses.append({"term": {"metadata.created_date": created_date}})

    source_type = metadata_filter.get("source_type")
    if source_type:
        must_clauses.append({"term": {"metadata.source_type.keyword": source_type}})

    language = metadata_filter.get("language")
    if language:
        must_clauses.append({"term": {"metadata.language.keyword": language}})

    if not must_clauses:
        return None

    try:
        client = _get_client()
        result = client.search(
            index=_metadata_index,
            body={"query": {"bool": {"must": must_clauses}}, "size": 1000, "_source": False}
        )
        doc_ids = []
        for hit in result.get("hits", {}).get("hits", []):
            raw_id = hit.get("_id", "")
            doc_id = raw_id[len("DOC:"):] if raw_id.startswith("DOC:") else raw_id
            doc_ids.append(doc_id)
        logger.info(f"Metadata filter resolved {len(doc_ids)} matching documents")
        return doc_ids
    except Exception as e:
        logger.warning(f"Metadata filter resolution failed, ignoring filter: {e}")
        return None


def _parse_chunk_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """Parse chunk ID into document_id, page_number, chunk_index."""
    try:
        parts = chunk_id.split('_')
        if len(parts) < 3:
            return None
        if not parts[-1].startswith('C') or not parts[-2].startswith('P'):
            return None
        return {
            'document_id': '_'.join(parts[:-2]),
            'page_number': int(parts[-2][1:]),
            'chunk_index': int(parts[-1][1:])
        }
    except (ValueError, IndexError):
        return None


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

    Fetches all chunk IDs for the document, sorts them by (page, chunk_index),
    then slices ±adjacent_count around the anchor — crossing page boundaries freely.

    Args:
        chunk_id: Anchor chunk identifier
        document_id: Optional document ID (extracted from chunk_id if not provided)
        adjacent_count: Number of adjacent chunks to retrieve on each side

    Returns:
        List of adjacent chunk dicts with chunk_id, document_id, page_number,
        chunk_index, text, origin='context'. Empty list if none found.
    """
    try:
        parsed = _parse_chunk_id(chunk_id)
        if not parsed:
            logger.warning(f"Cannot retrieve adjacent chunks for invalid chunk ID: {chunk_id}")
            return []

        doc_id = document_id or parsed['document_id']

        # Fetch all chunk IDs for this document, sorted by (page, chunk_index)
        all_chunk_ids = get_document_chunk_ids(doc_id)
        if not all_chunk_ids:
            logger.warning(f"No chunks found for document {doc_id}")
            return []

        # Find anchor position
        try:
            anchor_pos = all_chunk_ids.index(chunk_id)
        except ValueError:
            logger.warning(f"Anchor chunk {chunk_id} not found in document chunk list")
            return []

        # Slice ±adjacent_count, excluding the anchor itself
        start = max(0, anchor_pos - adjacent_count)
        end = min(len(all_chunk_ids), anchor_pos + adjacent_count + 1)
        neighbor_ids = all_chunk_ids[start:anchor_pos] + all_chunk_ids[anchor_pos + 1:end]

        # Batch fetch texts
        texts = get_paragraphs_batch(neighbor_ids)

        adjacent_chunks = []
        for cid in neighbor_ids:
            text = texts.get(cid)
            if not text:
                continue
            p = _parse_chunk_id(cid)
            adjacent_chunks.append({
                'chunk_id': cid,
                'document_id': doc_id,
                'page_number': p['page_number'] if p else 0,
                'chunk_index': p['chunk_index'] if p else 0,
                'text': text,
                'origin': 'context'
            })

        logger.debug(f"Retrieved {len(adjacent_chunks)} adjacent chunks for {chunk_id} (±{adjacent_count})")
        return adjacent_chunks

    except Exception as e:
        logger.error(f"Failed to retrieve adjacent chunks for {chunk_id}: {e}")
        return []

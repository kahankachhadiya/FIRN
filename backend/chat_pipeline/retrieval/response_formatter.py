"""
Response formatter for LLM responses using <link>URL</link> tag format.

This module provides functionality to parse <link> tags from LLM responses,
classify URLs as images or documents, and assemble the final-generation
system prompt with full chunk metadata.

Requirements satisfied:
- Requirement 8.4: Parse <link>URL</link> tags from response text
- Requirement 8.5: Classify image URLs (png, jpg, jpeg, gif, webp)
- Requirement 8.6: Classify document URLs (pdf, docx, doc, html, htm, txt, md)
- Requirement 9.1: Include original user query in assembled prompt
- Requirement 9.2: Include chunk metadata (document_path, page_number, origin, confidence)
- Requirement 9.3: Convert [Image: /path] references to full HTTP URLs
- Requirement 9.4: Include only chunks returned by RAG tool call
- Requirement 9.5: Reference user query field and structured chunk metadata format
"""

import re
import os
from typing import List, Tuple
from dataclasses import dataclass, field

from llm.schemas import EnhancedChunkOutput


# Module-level constants

LINK_PATTERN = re.compile(r'<link>(.*?)</link>|!\[.*?\]\((.*?)\)', re.DOTALL)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.html', '.htm', '.txt', '.md'}

# Pattern to match [Image: /path/to/image.ext] references in chunk text
_IMAGE_REF_PATTERN = re.compile(r'\[Image:\s*([^\]]+)\]')


@dataclass
class ParsedReferences:
    """
    Parsed URL references extracted from <link>URL</link> tags.

    Attributes:
        images: URLs ending in image extensions (.png, .jpg, .jpeg, .gif, .webp)
        documents: URLs ending in document extensions (.pdf, .docx, etc.)
    """
    images: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)


class ResponseFormatter:
    """
    Formats and parses LLM responses that use <link>URL</link> inline tags.

    Provides:
    - parse_link_tags(): extract and classify <link> URLs from response text
    - _image_path_to_url(): convert local image paths to full HTTP URLs
    - generate_llm_system_prompt(): build the final-generation system prompt
    """

    @staticmethod
    def parse_link_tags(response: str) -> Tuple[str, ParsedReferences]:
        """
        Parse all <link>URL</link> tags from response text.

        Tags are NOT stripped — they remain in the response for the frontend's
        custom markdown renderer to handle visually.

        Args:
            response: LLM response text that may contain <link>URL</link> tags.

        Returns:
            Tuple of (response_unchanged, ParsedReferences) where ParsedReferences
            contains two lists: images and documents, classified by URL extension.
        """
        if not response:
            return response, ParsedReferences()

        images: List[str] = []
        documents: List[str] = []

        for match in LINK_PATTERN.finditer(response):
            url = (match.group(1) or match.group(2) or "").strip()
            if not url:
                continue

            # Determine extension (case-insensitive)
            lower_url = url.lower()
            if any(lower_url.endswith(ext) for ext in IMAGE_EXTENSIONS):
                if url not in images:
                    images.append(url)
            elif any(lower_url.endswith(ext) for ext in DOCUMENT_EXTENSIONS):
                if url not in documents:
                    documents.append(url)
            # URLs with unrecognised extensions are silently ignored

        return response, ParsedReferences(images=images, documents=documents)

    @staticmethod
    def _image_path_to_url(path: str, base_url: str) -> str:
        """
        Convert a local image path to a full HTTP URL.

        Example:
            path     = "/Database/Processed_Docs/doc_pdf/figures/image_1.png"
            base_url = "http://10.0.0.1:8000"
            result   = "http://10.0.0.1:8000/api/files/Database/Processed_Docs/doc_pdf/figures/image_1.png"

        Args:
            path: Local filesystem path, typically starting with '/'.
            base_url: HTTP base URL of the backend (e.g. "http://host:port").

        Returns:
            Full HTTP URL string.
        """
        base_url = base_url.rstrip('/')
        # Strip leading slash from path so we don't get double slashes
        clean_path = path.lstrip('/')
        return f"{base_url}/api/files/{clean_path}"

    @staticmethod
    def generate_llm_system_prompt(
        chunks: List[EnhancedChunkOutput],
        original_query: str,
        base_url: str
    ) -> str:
        """
        Build the final-generation system prompt.

        Includes:
        - The RAG_RESPONSE_SYSTEM_PROMPT from environment / production config
        - The original user query
        - Each retrieved chunk with document_path, page_number, origin, confidence,
          and text content (with [Image: /path] references converted to full URLs)

        Args:
            chunks: Retrieved chunks to include in the prompt.
            original_query: The user's original query string.
            base_url: HTTP base URL used to convert image paths to full URLs.

        Returns:
            Assembled system prompt string.
        """
        # Load base system prompt from config/prompts.yml — raises if missing
        from config.prompts import get_rag_response_prompt
        base_prompt = get_rag_response_prompt()

        # Assemble prompt
        prompt_parts = [base_prompt, "", f"User Query: {original_query}", "", "Retrieved Context:"]

        for i, chunk in enumerate(chunks, 1):
            # ── Convert image references to full HTTP URLs ────────────────────
            # Handles both formats produced by the document processor:
            #   [Image: /path/to/image.png]          → legacy format
            #   ![Caption](path/to/image.png)        → markdown format

            def _normalize_path(raw: str) -> str:
                """Normalize Windows/relative paths to a clean path for URL construction."""
                # Replace Windows backslashes
                p = raw.replace("\\", "/")
                # Strip Windows drive letters (e.g. D:/Projects/RAG/RAG_MVP/)
                import re as _re
                p = _re.sub(r'^[A-Za-z]:/.*?RAG_MVP/', '', p)
                # Strip leading ./ (but preserve leading / for absolute paths)
                if p.startswith('./'):
                    p = p[2:]
                return p

            def _resolve_relative_path(img_path: str) -> str:
                if not img_path.startswith('/') and not img_path[1:3] == ':/':
                    if hasattr(chunk, 'document_path') and chunk.document_path:
                        import os
                        from pathlib import Path
                        doc_path = Path(chunk.document_path)
                        root_dir = None
                        current = doc_path.parent
                        while current.name != '' and current.name != '/':
                            if (current.parent / "Database" / "Processed_Docs").exists():
                                root_dir = current.parent
                                break
                            current = current.parent
                            
                        if root_dir:
                            doc_stem = doc_path.stem
                            processed_dir = root_dir / "Database" / "Processed_Docs" / f"{doc_stem}_pdf"
                            abs_img_path = processed_dir / img_path
                            if abs_img_path.exists():
                                return str(abs_img_path)
                return img_path

            def _replace_image_ref(m: re.Match) -> str:
                img_path = m.group(1).strip()
                img_path = _resolve_relative_path(img_path)
                img_path = _normalize_path(img_path)
                return f"![Image]({ResponseFormatter._image_path_to_url(img_path, base_url)})"

            def _replace_md_image(m: re.Match) -> str:
                caption = m.group(1).strip()
                img_path = m.group(2).strip()
                img_path = _resolve_relative_path(img_path)
                img_path = _normalize_path(img_path)
                url = ResponseFormatter._image_path_to_url(img_path, base_url)
                return f"![{caption}]({url})"

            chunk_text = _IMAGE_REF_PATTERN.sub(_replace_image_ref, chunk.text)
            # Also handle markdown image syntax: ![caption](path)
            chunk_text = re.sub(
                r'!\[([^\]]*)\]\(([^)]+)\)',
                _replace_md_image,
                chunk_text
            )
            # [Image Analysis] blobs are kept intact so the LLM can read them
            # and decide whether each image is relevant to the user's query.

            # ── Build document URL for Sources section ────────────────────────
            doc_path = chunk.document_path or ""
            if doc_path:
                doc_path = _normalize_path(doc_path)
                doc_url = f"{base_url.rstrip('/')}/api/files/{doc_path.lstrip('/')}"
            else:
                doc_url = ""

            prompt_parts.append(f"--- Chunk {i} ---")
            prompt_parts.append(f"Document: {doc_url or chunk.document_path or 'unknown'}")
            prompt_parts.append(f"Page: {chunk.page_number}")
            prompt_parts.append(f"Origin: {chunk.origin}")
            prompt_parts.append(f"Confidence: {chunk.confidence:.2f}")
            prompt_parts.append(f"Content: {chunk_text}")
            prompt_parts.append("")

        return "\n".join(prompt_parts)


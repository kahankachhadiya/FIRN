"""
Unified Chat API for LLM Tool Calling Extension.

Provides FastAPI endpoints for:
- Chat message processing with tool calling
- Session management
- File serving with security validation
"""

from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator
from pathlib import Path
import json
import os
import re
from uuid import uuid4
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from llm.session_manager import SessionManager
from llm.openai_compatible_client import OpenAICompatibleClient, StreamEvent
from agent.tool_dispatcher import ToolDispatcher
from agent.tool_registry import ToolRegistry
from retrieval.response_formatter import ResponseFormatter
from agent.tools import register_enhanced_rag_tool_only
from utils.errors import (
    ToolError,
    LMStudioError,
    PathTraversalError,
    FileTypeNotAllowedError
)
from utils.logging_utils import get_logger
from db.elasticsearch_client import (
    get_document_metadata,
    list_all_document_metadata,
    delete_document_metadata,
)

logger = get_logger(__name__)


def _clean_thinking_tags(message: str) -> Tuple[str, str]:
    """
    Separate thinking content from the main message.

    Handles two cases:
    1. Explicit <think>...</think> tags from reasoning models.
    2. Plain-text reasoning emitted before the first markdown heading (##),
       which some models output without any tags.

    Args:
        message: The full message with potential thinking content

    Returns:
        Tuple of (main_content, thinking_content)
    """
    if not message:
        return message, ""

    thinking = ""
    main_content = message

    # ── Case 1: explicit <think> tags ────────────────────────────────────────
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, message, re.DOTALL | re.IGNORECASE)
    if think_matches:
        thinking = "\n".join(match.strip() for match in think_matches)
        main_content = re.sub(think_pattern, '', message, flags=re.DOTALL | re.IGNORECASE)
        main_content = re.sub(r'\n{3,}', '\n\n', main_content).strip()

    # ── Case 2: plain-text reasoning before the first markdown heading ───────
    # If the message starts with prose (not a heading/list/table) and a ## heading
    # appears later, treat everything before that heading as leaked reasoning.
    if main_content and not main_content.lstrip().startswith('#'):
        heading_match = re.search(r'(^|\n)(#{1,3}\s+\S)', main_content)
        if heading_match and heading_match.start() > 0:
            pre_heading = main_content[:heading_match.start()].strip()
            # Only strip if the pre-heading block looks like internal monologue
            # (contains reasoning keywords or is a multi-sentence paragraph)
            reasoning_signals = [
                'let me', "let's", 'i need to', 'i should', 'i will',
                'proceed', 'now answer', 'provide', 'embed', 'the full url',
                'use that', 'craft', 'include', 'cite'
            ]
            lower_pre = pre_heading.lower()
            if any(sig in lower_pre for sig in reasoning_signals) or pre_heading.count('.') >= 2:
                thinking = (thinking + "\n" + pre_heading).strip() if thinking else pre_heading
                main_content = main_content[heading_match.start():].strip()

    return main_content, thinking


def _extract_balanced_json(text: str, start_pos: int = 0) -> tuple:
    """
    Extract a complete JSON object from text by finding balanced braces.
    
    Args:
        text: Text that contains a JSON object
        start_pos: Position to start searching from
        
    Returns:
        Tuple of (json_string, end_position) or (None, -1) if not found
    """
    # Find the first opening brace from start_pos
    start = text.find('{', start_pos)
    if start == -1:
        return None, -1
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1], i + 1
    
    return None, -1


def _clean_tool_call_artifacts(message: str) -> str:
    """
    Remove any remaining tool call artifacts from the assistant message.
    
    This handles cases where the LLM outputs tool calls as text that weren't
    fully cleaned during parsing. Handles:
    - <tool_call>...</tool_call> XML tags (complete and partial)
    - JSON objects with "name" and "arguments" fields matching tool call patterns
    - Nested JSON structures in tool calls
    - Multiple consecutive tool calls
    - Partial/malformed tool call syntax
    - Code blocks containing tool calls (```tool_call ... ```)
    
    Args:
        message: The assistant message to clean
        
    Returns:
        Cleaned message without tool call artifacts, or a fallback message
        if cleaning results in empty content
    """
    if not message:
        return "I've processed your request. Is there anything specific you'd like to know?"
    
    # First clean thinking tags
    cleaned, _ = _clean_thinking_tags(message)
    
    if not cleaned:
        cleaned = message
    
    # Known tool names for pattern matching
    tool_names = ['rag_retrieval', 'fetch_file']
    tool_names_pattern = '|'.join(tool_names)
    
    # Pattern 1: Remove <tool_call>...</tool_call> blocks (greedy to handle nested content)
    # Use a loop to handle multiple consecutive blocks
    prev_cleaned = None
    while prev_cleaned != cleaned:
        prev_cleaned = cleaned
        cleaned = re.sub(
            r'<tool_call>\s*.*?\s*</tool_call>',
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )
    
    # Pattern 2: Remove code blocks containing tool calls (```tool_call ... ```)
    cleaned = re.sub(
        r'```tool_call\s*.*?\s*```',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 3: Remove standalone/orphaned <tool_call> or </tool_call> tags
    cleaned = re.sub(r'<\s*/?\s*tool_call\s*>', '', cleaned, flags=re.IGNORECASE)
    
    # Pattern 4: Remove malformed tool_call tags (missing closing bracket, etc.)
    cleaned = re.sub(r'<\s*tool_call[^>]*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'^[^<]*tool_call\s*>', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Pattern 5: Remove JSON objects that look like tool calls with nested arguments
    # Use balanced brace extraction for proper nested JSON handling
    result_parts = []
    pos = 0
    while pos < len(cleaned):
        # Look for potential tool call JSON patterns
        match = re.search(
            rf'\{{\s*"name"\s*:\s*"({tool_names_pattern})"',
            cleaned[pos:],
            flags=re.IGNORECASE
        )
        
        if match:
            # Add text before the match
            result_parts.append(cleaned[pos:pos + match.start()])
            
            # Extract the full JSON object with balanced braces
            json_str, end_pos = _extract_balanced_json(cleaned, pos + match.start())
            
            if json_str:
                # Verify it's actually a tool call JSON (has "arguments" field)
                if '"arguments"' in json_str:
                    # Skip this JSON object (don't add to result)
                    pos = end_pos
                else:
                    # Not a tool call, keep it
                    result_parts.append(cleaned[pos + match.start():pos + match.start() + 1])
                    pos = pos + match.start() + 1
            else:
                # Couldn't extract balanced JSON, skip the match and continue
                result_parts.append(cleaned[pos + match.start():pos + match.start() + 1])
                pos = pos + match.start() + 1
        else:
            # No more matches, add remaining text
            result_parts.append(cleaned[pos:])
            break
    
    cleaned = ''.join(result_parts)
    
    # Pattern 6: Remove any remaining JSON-like structures with "name" and "arguments"
    # This catches any tool call patterns that might have been missed
    # Use non-greedy matching with proper nested brace handling
    prev_cleaned = None
    while prev_cleaned != cleaned:
        prev_cleaned = cleaned
        # Simple pattern for flat JSON structures
        cleaned = re.sub(
            r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}',
            '',
            cleaned,
            flags=re.DOTALL
        )
    
    # Pattern 7: Remove JSON with "name" field containing known tool names (even without arguments)
    for tool_name in tool_names:
        # Pattern for tool call JSON that might be malformed
        cleaned = re.sub(
            rf'\{{\s*"name"\s*:\s*"{tool_name}"[^}}]*\}}',
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )
    
    # Pattern 8: Remove any leftover fragments that look like tool call syntax
    # e.g., "arguments": {...} on its own line
    cleaned = re.sub(
        r'^\s*"arguments"\s*:\s*\{[^{}]*\}\s*$',
        '',
        cleaned,
        flags=re.MULTILINE
    )
    
    # Pattern 9: Remove lines that are just JSON fragments from tool calls
    cleaned = re.sub(
        r'^\s*"name"\s*:\s*"(' + tool_names_pattern + r')"\s*,?\s*$',
        '',
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE
    )
    
    # Pattern 10: Remove [fetch_file(...)] text patterns that LLM outputs as text
    # These should have been tool calls but LLM wrote them as text
    cleaned = re.sub(
        r'\[fetch_file\s*\([^\)]*\)\s*\]',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Pattern 11: Remove fetch_file(...) without brackets
    cleaned = re.sub(
        r'fetch_file\s*\(\s*file_path\s*=\s*["\'][^"\']*["\']\s*\)',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Clean up multiple newlines and whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)  # Remove trailing whitespace on lines
    cleaned = cleaned.strip()
    
    # Fallback message for empty cleaned responses (Requirement 2.4)
    # If the message is now empty or just whitespace, provide a meaningful default
    if not cleaned or cleaned.isspace():
        cleaned = "I've processed your request. Is there anything specific you'd like to know?"
    
    return cleaned


def format_source_attribution(chunks: List[Any]) -> List[str]:
    """
    Extract unique PDF filenames from chunks for source attribution.
    
    This function extracts document_ids from chunks, retrieves document metadata,
    and returns a deduplicated list of human-readable PDF filenames.
    
    Args:
        chunks: List of ChunkInfo objects or dicts with document_id field
        
    Returns:
        Deduplicated list of PDF filenames (ending in .pdf).
        Excludes chunks where filename cannot be determined.
        
    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    if not chunks:
        return []
    
    # Extract unique document_ids from chunks
    document_ids = set()
    for chunk in chunks:
        if hasattr(chunk, 'document_id'):
            doc_id = chunk.document_id
        elif isinstance(chunk, dict):
            doc_id = chunk.get('document_id')
        else:
            continue
        
        if doc_id:
            document_ids.add(doc_id)
    
    if not document_ids:
        logger.debug("No document_ids found in chunks for source attribution")
        return []
    
    # Retrieve file_name for each document_id
    pdf_filenames = set()
    for doc_id in document_ids:
        try:
            metadata = get_document_metadata(doc_id)
            if metadata:
                file_name = metadata.get('file_name')
                # Only include valid PDF filenames (Requirement 4.2, 4.4)
                if file_name and isinstance(file_name, str) and file_name.lower().endswith('.pdf'):
                    pdf_filenames.add(file_name)
                elif file_name:
                    logger.debug(f"Excluding non-PDF file_name: {file_name}")
            else:
                logger.debug(f"No metadata found for document_id: {doc_id}")
        except Exception as e:
            # Log and continue - missing metadata doesn't block response (Requirement 4.4)
            logger.warning(f"Failed to retrieve metadata for document_id '{doc_id}': {e}")
            continue
    
    # Return sorted list for consistent ordering (Requirement 4.3 - deduplication via set)
    result = sorted(pdf_filenames)
    logger.debug(f"Source attribution: {len(result)} unique PDF(s) from {len(document_ids)} document(s)")
    return result


# Pydantic Models

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: Optional[str] = Field(
        None,
        description="Session ID for continuing conversation. If None, creates new session."
    )
    message: str = Field(
        ...,
        description="User message content",
        min_length=1
    )
    settings: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional settings override (temperature, max_tokens, etc.)"
    )


class FileReference(BaseModel):
    """File reference in response."""
    file_path: str
    file_name: str
    file_type: str
    size_bytes: Optional[int] = None
    display_mode: str  # "new_tab" or "inline"
    url: str
    content_base64: Optional[str] = None


class ChunkInfo(BaseModel):
    """
    Retrieved chunk information matching EnhancedChunkOutput format.
    
    This model represents chunks returned by the layered graph RAG system
    with images embedded in text content and proper origin labeling.
    """
    chunk_id: str
    text: str  # Contains embedded image references like [Image: /path/to/image.png]
    document_id: str
    page_number: int
    document_path: str
    origin: str  # anchor, relation, probe_supports, probe_contradicts, probe_example, probe_elaborates, probe_depends, context
    confidence: float
    
    # Deprecated fields for backward compatibility
    source: Optional[str] = Field(default=None, description="Deprecated: use 'origin' instead")
    retrieval_mode: Optional[str] = Field(default=None, description="Deprecated: always 'enhanced_rag_retrieval'")
    
    def __init__(self, **data):
        # Handle backward compatibility
        if 'source' in data and 'origin' not in data:
            data['origin'] = data['source']
        if 'origin' in data and 'source' not in data:
            data['source'] = data['origin']
        if 'retrieval_mode' not in data:
            data['retrieval_mode'] = 'enhanced_rag_retrieval'
        super().__init__(**data)


class References(BaseModel):
    """Parsed URL references extracted from <link>URL</link> tags in the LLM response."""
    images: List[str] = Field(default_factory=list, description="Image URLs (png, jpg, jpeg, gif, webp)")
    documents: List[str] = Field(default_factory=list, description="Document URLs (pdf, docx, etc.)")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    session_id: str
    message: str
    thinking: Optional[str] = Field(default=None, description="Model's thinking process if available")
    chunks: List[ChunkInfo] = Field(default_factory=list)
    files: List[FileReference] = Field(default_factory=list)
    source_pdfs: List[str] = Field(default_factory=list, description="Human-readable PDF filenames used as sources")
    references: References = Field(default_factory=References, description="Parsed image and document references from <link> tags")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    """Session metadata for listing."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class SessionDetail(BaseModel):
    """Detailed session information."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response model."""
    error_type: str
    message: str
    recovery_suggestions: Optional[List[str]] = None


class MemoryItem(BaseModel):
    """A single memory entry scoped to a session."""
    id: str
    content: str
    createdAt: str


class MemoryCreateRequest(BaseModel):
    """Request body for creating a memory."""
    content: str


class MemoryUpdateRequest(BaseModel):
    """Request body for updating a memory."""
    content: str


class SourceResponse(BaseModel):
    """A processed document source returned by GET /api/sources."""
    id: str
    name: str
    size: int
    status: str  # always "ready" for indexed documents
    uploadedAt: str
    pages: Optional[int] = None


# Module-level in-memory stores (populated/mutated at runtime)
_memory_store: Dict[str, List[MemoryItem]] = {}  # session_id -> memories
_runtime_config: Dict[str, Any] = {}  # populated on startup from settings


# Initialize FastAPI app
app = FastAPI(
    title="LLM Tool Calling API",
    description="Unified API for multi-database RAG with LLM tool calling",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware with proper headers and error responses
from api.rate_limit_middleware import RateLimitMiddleware
app.add_middleware(
    RateLimitMiddleware,
    exempt_paths=[
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/metrics",
        "/api/config",  # Exempt config endpoints
    ]
)


# Mount admin router
from api.admin_api import admin_router
app.include_router(admin_router)


# Global state (initialized on startup)
session_manager: Optional[SessionManager] = None
llm_client: Optional[OpenAICompatibleClient] = None
tool_dispatcher: Optional[ToolDispatcher] = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    global session_manager, llm_client, tool_dispatcher, _runtime_config

    logger.info("Initializing LLM Tool Calling API...")

    # Initialize production configuration first — this satisfies all downstream
    # get_production_config() calls (qdrant_client, query_generator, etc.)
    try:
        from config.production import initialize_production_config
        initialize_production_config()
        logger.info("Production configuration initialized")
    except Exception as e:
        logger.warning(f"Production config init warning (non-fatal): {e}")
    
    # Populate runtime config from settings
    _runtime_config = {
        "llm_model": settings.llm_chat_model,
        "db_type": settings.qdrant_host,
        "backend_url": f"http://{settings.api_host}:{settings.api_port}",
    }
    logger.info("Runtime config populated from settings")
    
    # Initialize session manager
    session_manager = SessionManager(
        chat_history_dir=settings.chat_history_dir
    )
    logger.info("Session manager initialized")
    
    # Initialize tool registry and register enhanced RAG tool only
    tool_registry = ToolRegistry()
    register_enhanced_rag_tool_only(tool_registry)
    logger.info(f"Tool registry initialized with enhanced RAG tool only (total: {len(tool_registry._tools)} tools)")
    
    # Initialize tool dispatcher
    tool_dispatcher = ToolDispatcher(tool_registry)
    logger.info("Tool dispatcher initialized")
    
    # Initialize chat LLM client (tool calling, Q&A, RAG response generation)
    llm_client = OpenAICompatibleClient(
        base_url=settings.llm_chat_base_url,
        api_key=settings.llm_chat_api_key,
        model=settings.llm_chat_model,
        max_tokens=settings.llm_chat_max_tokens,
        temperature=settings.llm_chat_temperature,
    )
    logger.info("Chat LLM client initialized")
    
    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down LLM Tool Calling API...")
    
    if llm_client:
        await llm_client.close()
        logger.info("LLM client closed")
    
    logger.info("API shutdown complete")


async def status_callback(event_type: str, data: Dict[str, Any] = None) -> None:
    """Status callback for LLM processing."""
    logger.debug(f"LLM status: {event_type} {data or ''}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message with LLM tool calling.
    
    This endpoint handles the full conversation flow:
    1. Create or load session
    2. Clear memory for fresh query processing
    3. Add user message to history
    4. Build message context with system prompt
    5. Process with LLM and tool calling
    6. Extract memory reports and clear memory
    7. Save assistant response
    8. Return formatted response with chunks and files
    
    Args:
        request: ChatRequest with session_id, message, and optional settings
    
    Returns:
        ChatResponse with assistant message, retrieved chunks, files, and metadata
    
    Raises:
        HTTPException: On various errors (session, LLM, tool execution)
    """
    try:
        # Create or load session
        if request.session_id:
            session_id = request.session_id
            session = session_manager.get_session(session_id)
            if not session:
                # Session was deleted or expired — create a new one
                session_id = session_manager.create_session()
                logger.info(f"Session {request.session_id} not found, created new: {session_id}")
            else:
                logger.info(f"Loaded existing session: {session_id}")
        else:
            session_id = session_manager.create_session()
            logger.info(f"Created new session: {session_id}")
        
        logger.info(f"Starting query processing for session: {session_id}")
        
        # Add user message to history
        session_manager.add_message(
            session_id=session_id,
            role="user",
            content=request.message
        )
        
        # Build message context
        messages = []
        base_url = f"http://{settings.api_host}:{settings.api_port}"
        
        # Initial system prompt instructs the model to use the rag_retrieval tool.
        # The RAG_RESPONSE_SYSTEM_PROMPT (with retrieved chunks) is injected AFTER
        # retrieval completes — not here, to avoid telling the model "don't call tools"
        # before any retrieval has happened.
        from config.prompts import get_tool_calling_prompt
        messages.append({
            "role": "system",
            "content": get_tool_calling_prompt()
        })
        
        # Conversation history intentionally excluded — each query is stateless.
        # Sending prior turns caused hallucination from stale context.
        
        # Get tool schemas
        tool_schemas = tool_dispatcher.registry.get_all_schemas()
        
        # Process with LLM and tools
        logger.info(f"Processing message with LLM (session: {session_id})")
        
        result = await llm_client.process_with_tools(
            messages=messages,
            tools=tool_schemas,
            tool_dispatcher=tool_dispatcher,
            session_id=session_id,
            max_rounds=settings.max_retrieval_rounds,
            status_callback=status_callback
        )
        
        assistant_message = result["message"]
        
        # Separate thinking from message
        assistant_message, thinking_content = _clean_thinking_tags(assistant_message)
        
        # Clean up any remaining tool call artifacts from the message
        assistant_message = _clean_tool_call_artifacts(assistant_message)
        
        # Extract chunks and files from memory (if any retrieval was done)
        # This must be done BEFORE clearing memory (Requirement 1.3)
        chunks = []
        files = []
        
        # Extract chunks directly from RAG retrieval tool results
        retrieved_chunks_from_llm = result.get("retrieved_chunks", [])
        
        for chunk_data in retrieved_chunks_from_llm:
            # Handle both EnhancedChunkOutput format and legacy format
            if isinstance(chunk_data, dict):
                # Dictionary format from tool results
                chunk_id = chunk_data.get("chunk_id", "")
                text = chunk_data.get("text", "")
                document_id = chunk_data.get("document_id", "")
                page_number = chunk_data.get("page_number")
                document_path = chunk_data.get("document_path", "")
                origin = chunk_data.get("origin", "retrieval")
                confidence = chunk_data.get("confidence")
            else:
                # Assume it's an EnhancedChunkOutput object
                chunk_id = getattr(chunk_data, 'chunk_id', "")
                text = getattr(chunk_data, 'text', "")
                document_id = getattr(chunk_data, 'document_id', "")
                page_number = getattr(chunk_data, 'page_number', None)
                document_path = getattr(chunk_data, 'document_path', "")
                origin = getattr(chunk_data, 'origin', "retrieval")
                confidence = getattr(chunk_data, 'confidence', None)
            
            # Validate required fields
            if not chunk_id:
                logger.warning("Skipping chunk with missing chunk_id")
                continue
            
            # Ensure page_number is an integer
            if page_number is None:
                # Parse page_number from chunk_id if not provided
                # Format: {document_id}_P{page:02d}_C{chunk:02d}
                if chunk_id:
                    parts = chunk_id.split("_P")
                    if len(parts) > 1:
                        try:
                            page_str = parts[1].split("_C")[0]
                            page_number = int(page_str)
                            logger.debug(f"Parsed page_number {page_number} from chunk_id {chunk_id}")
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse page_number from chunk_id {chunk_id}: {e}")
                            page_number = 1  # Default to page 1
                    else:
                        page_number = 1  # Default to page 1
                else:
                    page_number = 1  # Default to page 1
            
            # Ensure confidence is a float
            if confidence is None:
                confidence = 0.0
            
            # Validate origin is one of the expected values
            valid_origins = {
                "anchor", "relation", 
                "probe_supports", "probe_contradicts", "probe_example", 
                "probe_elaborates", "probe_depends", 
                "context", "retrieval"  # Include legacy "retrieval" for backward compatibility
            }
            if origin not in valid_origins:
                logger.warning(f"Invalid origin '{origin}' for chunk {chunk_id}, using 'retrieval'")
                origin = "retrieval"
            
            # Ensure document_path is provided
            if not document_path and document_id:
                # Try to get document path from metadata
                try:
                    doc_metadata = get_document_metadata(document_id)
                    if doc_metadata:
                        document_path = doc_metadata.get("path") or doc_metadata.get("file_name") or f"/documents/{document_id}"
                    else:
                        document_path = f"/documents/{document_id}"
                except Exception as e:
                    logger.warning(f"Failed to get document metadata for {document_id}: {e}")
                    document_path = f"/documents/{document_id}"
            
            if chunk_id:
                chunks.append(ChunkInfo(
                    chunk_id=chunk_id,
                    text=text,
                    document_id=document_id,
                    page_number=page_number,
                    document_path=document_path,
                    origin=origin,
                    confidence=confidence
                ))
        
        if retrieved_chunks_from_llm:
            logger.info(f"Added {len(retrieved_chunks_from_llm)} chunks from RAG retrieval")
        
        # Memory functionality has been removed - chunks are now extracted directly from RAG tool results
        # The enhanced RAG tool returns chunks in the proper format via retrieved_chunks_from_llm
        logger.debug("Using enhanced RAG tool results - memory reports functionality removed")

        # Rebuild the system prompt with the actual retrieved chunks so the LLM response
        # is grounded in full chunk metadata (Requirements 9.1, 9.2, 9.3, 9.5).
        # We convert ChunkInfo → EnhancedChunkOutput for generate_llm_system_prompt.
        if chunks:
            from llm.schemas import EnhancedChunkOutput as _ECO
            enhanced_chunks = [
                _ECO(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    page_number=c.page_number,
                    document_path=c.document_path,
                    text=c.text,
                    origin=c.origin,
                    confidence=c.confidence,
                )
                for c in chunks
            ]
            enriched_system_prompt = ResponseFormatter.generate_llm_system_prompt(
                chunks=enhanced_chunks,
                original_query=request.message,
                base_url=base_url
            )
            logger.debug(
                f"Rebuilt system prompt with {len(enhanced_chunks)} chunk(s) "
                f"for session {session_id}"
            )
        else:
            enriched_system_prompt = get_tool_calling_prompt()
        
        # Parse <link>URL</link> tags from the assistant message (Requirement 8.4)
        _, parsed_references = ResponseFormatter.parse_link_tags(assistant_message)

        if parsed_references.images or parsed_references.documents:
            logger.info(
                f"Parsed link tags: {len(parsed_references.images)} image(s), "
                f"{len(parsed_references.documents)} document(s)"
            )

        # Build metadata with enhanced RAG tool information
        metadata = {
            "tool_calls_made": result["tool_calls_made"],
            "retrieval_rounds": result["rounds"],
            "total_tokens": result["total_tokens"],
            "finish_reason": result["finish_reason"],
            "enhanced_rag_used": True,  # Indicate new RAG system is being used
            "chunks_by_origin": {},  # Track chunk origins for debugging
        }

        # Analyze chunk origins for metadata
        if chunks:
            origin_counts = {}
            for chunk in chunks:
                origin = chunk.origin or "unknown"
                origin_counts[origin] = origin_counts.get(origin, 0) + 1
            metadata["chunks_by_origin"] = origin_counts
            logger.info(f"Chunk origins: {origin_counts}")
        
        # Store confidence from enhanced RAG tool results (if available)
        last_confidence = None
        if chunks:
            # Get confidence from the first chunk with confidence data
            for chunk in chunks:
                if chunk.confidence is not None and chunk.confidence > 0:
                    last_confidence = chunk.confidence
                    break
        
        logger.info(f"Response generation completed for session: {session_id}")
        
        # Save assistant response to history with enhanced metadata
        session_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
            metadata={
                "chunks_used": [c.chunk_id for c in chunks],
                "files_referenced": [f.file_path for f in files],
                "retrieval_rounds": result["rounds"],
                "confidence": last_confidence,
                "tool_calls": result["tool_calls_made"],
                "enhanced_rag_used": True,  # Indicate new system usage
                "chunk_origins": metadata.get("chunks_by_origin", {}),  # Track chunk sources
                "total_chunks": len(chunks)
            }
        )
        
        # Extract source PDF filenames for attribution (Requirements 4.1, 4.2, 4.3, 4.4)
        source_pdfs = format_source_attribution(chunks)
        if source_pdfs:
            logger.info(f"Source attribution: {len(source_pdfs)} PDF(s) for session: {session_id}")
        
        logger.info(
            f"Enhanced RAG chat response generated: session={session_id}, "
            f"chunks={len(chunks)}, files={len(files)}, "
            f"source_pdfs={len(source_pdfs)}, "
            f"tool_calls={result['tool_calls_made']}, "
            f"chunk_origins={metadata.get('chunks_by_origin', {})}"
        )
        
        return ChatResponse(
            session_id=session_id,
            message=assistant_message,
            thinking=thinking_content if thinking_content else None,
            chunks=chunks,
            files=files,
            source_pdfs=source_pdfs,
            references=References(
                images=parsed_references.images,
                documents=parsed_references.documents
            ),
            metadata=metadata
        )
        
    except LMStudioError as e:
        logger.error(f"LM Studio error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error_type": "LMStudioError",
                "message": str(e),
                "recovery_suggestions": [
                    "Check if LM Studio is running",
                    "Verify LM Studio API endpoint configuration",
                    "Try again in a few moments"
                ]
            }
        )
    
    except ToolError as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "ToolError",
                "message": str(e),
                "recovery_suggestions": [
                    "Check tool configuration",
                    "Verify database connections",
                    "Try rephrasing your query"
                ]
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "InternalError",
                "message": "An unexpected error occurred",
                "recovery_suggestions": [
                    "Try again",
                    "Contact support if the issue persists"
                ]
            }
        )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Process a chat message with LLM tool calling, streaming the response via SSE.

    Emits Server-Sent Events of the form:
      data: {"type": "token", "content": "..."}\\n\\n
      data: {"type": "citation", "citations": [...]}\\n\\n
      data: {"type": "done", "session_id": "...", "metadata": {...}}\\n\\n
      data: {"type": "error", "message": "..."}\\n\\n

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Create or load session — auto-create if the requested session no longer exists
            if request.session_id:
                session_id = request.session_id
                session = session_manager.get_session(session_id)
                if not session:
                    # Session was deleted or expired — recreate it with the same ID
                    session_id = session_manager.create_session()
                    logger.info(f"[stream] Session {request.session_id} not found, created new: {session_id}")
                else:
                    logger.info(f"[stream] Loaded existing session: {session_id}")
            else:
                session_id = session_manager.create_session()
                logger.info(f"[stream] Created new session: {session_id}")

            # Add user message to history
            session_manager.add_message(
                session_id=session_id,
                role="user",
                content=request.message
            )

            # Build message context
            base_url = f"http://{settings.api_host}:{settings.api_port}"
            # Tool-calling prompt — instructs the model to call rag_retrieval first.
            # The RAG response prompt is injected only after retrieval completes.
            from config.prompts import get_tool_calling_prompt
            messages = [{"role": "system", "content": get_tool_calling_prompt()}]

            # Add only the current user message — no history
            messages.append({"role": "user", "content": request.message})

            # Get tool schemas
            tool_schemas = tool_dispatcher.registry.get_all_schemas()

            # ── Prompt builder for the streaming path ─────────────────────────
            # This callback is called by process_with_tools_stream() AFTER the
            # tool-calling rounds complete but BEFORE the final answer is streamed.
            # It converts local `[Image: /path]` and document references in chunk
            # text to proper HTTP URLs, then builds the enriched RAG response
            # system prompt so the LLM generates its answer with correct URLs.
            from llm.schemas import EnhancedChunkOutput as _ECO

            # Capture the final LLM request for debug logging
            _debug_final_prompt: list = [None]   # [system_prompt_str]
            _debug_final_chunks: list = []        # eco_chunks list

            def _build_stream_prompt(raw_chunks: list, query: str) -> str:
                eco_chunks = []
                valid_origins = {
                    "anchor", "relation", "probe_supports", "probe_contradicts",
                    "probe_example", "probe_elaborates", "probe_depends", "context"
                }
                for c in raw_chunks:
                    if not isinstance(c, dict):
                        continue
                    try:
                        origin = c.get("origin", "context")
                        if origin not in valid_origins:
                            origin = "context"
                        eco_chunks.append(
                            _ECO(
                                chunk_id=c.get("chunk_id", "") or f"chunk_{len(eco_chunks)}",
                                document_id=c.get("document_id", "") or "unknown",
                                page_number=max(1, int(c.get("page_number") or 1)),
                                document_path=c.get("document_path", "") or "",
                                text=c.get("text", ""),
                                origin=origin,
                                confidence=float(c.get("confidence") or 0.0),
                            )
                        )
                    except Exception as _e:
                        logger.warning(f"[stream] Skipping malformed chunk: {_e}")
                if not eco_chunks:
                    from config.prompts import get_rag_response_prompt
                    prompt = get_rag_response_prompt()
                else:
                    prompt = ResponseFormatter.generate_llm_system_prompt(
                        chunks=eco_chunks,
                        original_query=query,
                        base_url=base_url,
                    )
                # Capture for debug log
                _debug_final_prompt[0] = prompt
                _debug_final_chunks.clear()
                _debug_final_chunks.extend(eco_chunks)
                return prompt

            # Stream via process_with_tools_stream
            full_content = ""
            done_event: Optional[StreamEvent] = None

            async for event in llm_client.process_with_tools_stream(
                messages=messages,
                tools=tool_schemas,
                tool_dispatcher=tool_dispatcher,
                session_id=session_id,
                original_query=request.message,
                max_rounds=settings.max_retrieval_rounds,
                prompt_builder=_build_stream_prompt,
            ):
                if event.type == "token":
                    token_content = event.content or ""
                    # Convert local absolute image paths to HTTP URLs in real-time
                    token_content = re.sub(
                        r'!\[([^\]]*)\]\((\/[^)]+\.(?:png|jpg|jpeg|gif|webp))\)',
                        lambda m: f'![{m.group(1)}]({base_url}/api/files{m.group(2)})',
                        token_content,
                        flags=re.IGNORECASE
                    )
                    full_content += token_content
                    yield f'data: {json.dumps({"type": "token", "content": token_content})}\n\n'

                elif event.type == "citation":
                    yield f'data: {json.dumps({"type": "citation", "citations": event.citations})}\n\n'

                elif event.type == "done":
                    done_event = event
                    # Strip thinking tags AND plain-text reasoning before saving
                    cleaned_message, _ = _clean_thinking_tags(full_content)
                    cleaned_message = _clean_tool_call_artifacts(cleaned_message)
                    # Convert any local absolute image paths the LLM may have output
                    cleaned_message = re.sub(
                        r'!\[([^\]]*)\]\((\/[^)]+\.(?:png|jpg|jpeg|gif|webp))\)',
                        lambda m: f'![{m.group(1)}]({base_url}/api/files{m.group(2)})',
                        cleaned_message,
                        flags=re.IGNORECASE
                    )

                    # ── Write full debug entry ────────────────────────────────
                    try:
                        from utils.conversation_logger import ConversationLogger
                        _cl = ConversationLogger()
                        _debug_chunks_serialized = []
                        for _c in _debug_final_chunks:
                            _debug_chunks_serialized.append({
                                "chunk_id": getattr(_c, "chunk_id", ""),
                                "document_path": getattr(_c, "document_path", ""),
                                "page_number": getattr(_c, "page_number", 0),
                                "origin": getattr(_c, "origin", ""),
                                "confidence": getattr(_c, "confidence", 0.0),
                                "text": getattr(_c, "text", ""),
                            })
                        import json as _json
                        _debug_entry = {
                            "type": "llm_final_call",
                            "timestamp": __import__('datetime').datetime.now().isoformat(),
                            "session_id": session_id,
                            "user_query": request.message,
                            "llm_final_request": {
                                "system_prompt": _debug_final_prompt[0] or "",
                                "user_query": request.message,
                                "chunks_count": len(_debug_chunks_serialized),
                                "chunks": _debug_chunks_serialized,
                            },
                            "llm_raw_output": full_content,
                            "llm_cleaned_output": cleaned_message,
                        }
                        import os as _os
                        _os.makedirs("logs", exist_ok=True)
                        with open("logs/conversation_debug.jsonl", "a", encoding="utf-8") as _f:
                            _f.write(_json.dumps(_debug_entry, default=str, ensure_ascii=False) + "\n")
                    except Exception as _de:
                        logger.warning(f"[stream] Debug log write failed: {_de}")
                    # ─────────────────────────────────────────────────────────
                    session_manager.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=cleaned_message,
                        metadata={
                            "streaming": True,
                            "tool_calls_made": (event.metadata or {}).get("tool_calls_made", 0),
                            "retrieval_rounds": (event.metadata or {}).get("rounds", 0),
                        }
                    )
                    # Parse <link>URL</link> tags from the LLM's response to extract
                    # only the images and documents the model actually used/cited.
                    # Images = <link> tags with image extensions embedded inline.
                    # Documents = <link> tags in the ## Sources section.
                    _, parsed_refs = ResponseFormatter.parse_link_tags(cleaned_message)
                    done_metadata = dict(event.metadata or {})
                    done_metadata["references"] = {
                        "images": parsed_refs.images,
                        "documents": parsed_refs.documents,
                    }
                    yield f'data: {json.dumps({"type": "done", "session_id": event.session_id, "metadata": done_metadata})}\n\n'
                    return


                elif event.type == "error":
                    yield f'data: {json.dumps({"type": "error", "message": event.message})}\n\n'
                    return

        except Exception as e:
            logger.error(f"Unexpected error in chat_stream: {e}", exc_info=True)
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/sessions", response_model=SessionMetadata)
async def create_session() -> SessionMetadata:
    """
    Create a new empty session and return its metadata.
    The frontend uses the returned session_id as its conversation ID,
    keeping frontend and backend IDs in sync from the start.
    """
    try:
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
        return SessionMetadata(
            session_id=session["session_id"],
            title=session["title"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            message_count=0,
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session")


@app.get("/api/sessions", response_model=List[SessionMetadata])
async def list_sessions() -> List[SessionMetadata]:
    """
    List all available chat sessions.
    
    Returns:
        List of session metadata sorted by updated_at (most recent first)
    """
    try:
        sessions = session_manager.list_sessions()
        
        return [
            SessionMetadata(
                session_id=s["session_id"],
                title=s["title"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                message_count=s["message_count"]
            )
            for s in sessions
        ]
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "InternalError",
                "message": "Failed to list sessions"
            }
        )


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    """
    Get detailed information for a specific session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        SessionDetail with full message history
    
    Raises:
        HTTPException: If session not found
    """
    try:
        session = session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return SessionDetail(
            session_id=session["session_id"],
            title=session["title"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            messages=session["messages"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "InternalError",
                "message": f"Failed to get session {session_id}"
            }
        )


@app.get("/api/files/{file_path:path}")
async def get_file(file_path: str) -> FileResponse:
    """
    Serve a file with security validation.
    
    Validates that:
    1. Path doesn't contain traversal attempts (..)
    2. Path is within allowed directories or is a valid absolute path
    3. File extension is allowed
    4. File exists
    
    Args:
        file_path: File path (can be relative or absolute)
    
    Returns:
        FileResponse with appropriate content type
    
    Raises:
        HTTPException: If file access is denied or file not found
    """
    try:
        # Re-add leading slash if the original path was absolute on Linux.
        # The {file_path:path} parameter strips the first slash in /api/files//home/...
        if not file_path.startswith('/'):
            file_path = '/' + file_path

        # Validate path security
        _validate_file_path(file_path)
        
        # Construct full path - handle both absolute and relative paths
        path = Path(file_path)
        if path.is_absolute():
            full_path = path.resolve()
        else:
            full_path = Path(file_path)
        
        # Check file exists
        if not full_path.exists():
            logger.warning(f"File not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        
        if not full_path.is_file():
            logger.warning(f"Path is not a file: {file_path}")
            raise HTTPException(
                status_code=400,
                detail="Path is not a file"
            )
        
        # Determine media type
        extension = full_path.suffix.lower().lstrip('.')
        media_type_map = {
            'pdf': 'application/pdf',
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        
        media_type = media_type_map.get(extension, 'application/octet-stream')
        
        logger.info(f"Serving file: {file_path} (type: {media_type})")
        
        return FileResponse(
            path=str(full_path),
            media_type=media_type,
            headers={"Content-Disposition": "inline"}
        )
        
    except PathTraversalError:
        logger.warning(f"Path traversal attempt blocked: {file_path}")
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    except FileTypeNotAllowedError:
        logger.warning(f"Disallowed file type requested: {file_path}")
        raise HTTPException(
            status_code=403,
            detail="File type not allowed"
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to serve file"
        )


def _validate_file_path(file_path: str) -> None:
    """
    Validate file path for security.

    Raises:
        PathTraversalError: If path contains traversal attempts
        FileTypeNotAllowedError: If file extension not allowed
    """
    if ".." in file_path:
        raise PathTraversalError(f"Path traversal detected: {file_path}")

    extension = Path(file_path).suffix.lower().lstrip('.')
    if extension not in settings.allowed_file_extensions:
        raise FileTypeNotAllowedError(f"File extension not allowed: {extension}")




@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "llm-tool-calling-api",
        "version": "1.0.0"
    }


# ---------------------------------------------------------------------------
# Config endpoints (Requirement 4.1, 4.2)
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """Return the current runtime configuration."""
    return _runtime_config


@app.put("/api/config")
async def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *updates* into the runtime config and return the full updated config."""
    global _runtime_config
    _runtime_config.update(updates)
    return _runtime_config


# ---------------------------------------------------------------------------
# Memory endpoints (Requirements 4.3 – 4.7)
# ---------------------------------------------------------------------------

@app.get("/api/memory", response_model=List[MemoryItem])
async def get_memories(session_id: str) -> List[MemoryItem]:
    """Return all memories for *session_id*, or 404 if the session is absent."""
    if session_id not in _memory_store:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _memory_store[session_id]


@app.post("/api/memory", response_model=MemoryItem, status_code=201)
async def add_memory(session_id: str, body: MemoryCreateRequest) -> MemoryItem:
    """Create a new memory for *session_id* and return the created item."""
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    item = MemoryItem(
        id=str(uuid4()),
        content=body.content,
        createdAt=datetime.utcnow().isoformat(),
    )
    _memory_store[session_id].append(item)
    return item


@app.put("/api/memory/{memory_id}", response_model=MemoryItem)
async def update_memory(memory_id: str, session_id: str, body: MemoryUpdateRequest) -> MemoryItem:
    """Update the content of an existing memory; 404 if session or id absent."""
    if session_id not in _memory_store:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    for item in _memory_store[session_id]:
        if item.id == memory_id:
            item.content = body.content
            return item
    raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")


@app.delete("/api/memory/{memory_id}", status_code=204)
async def delete_memory(memory_id: str, session_id: str) -> None:
    """Delete a memory; returns 204 on success, 404 if session or id absent."""
    if session_id not in _memory_store:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    memories = _memory_store[session_id]
    for i, item in enumerate(memories):
        if item.id == memory_id:
            memories.pop(i)
            return
    raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")


# ---------------------------------------------------------------------------
# Sources endpoints (Requirements 2.5, 2.7)
# ---------------------------------------------------------------------------

@app.get("/api/sources", response_model=List[SourceResponse])
async def get_sources() -> List[SourceResponse]:
    """
    List all processed/indexed document sources.

    Queries Elasticsearch for all document metadata records and maps them to
    the frontend ``Source`` type shape: {id, name, size, status, uploadedAt}.

    Requirements: 2.5
    """
    try:
        docs = list_all_document_metadata()
        sources: List[SourceResponse] = []
        for doc in docs:
            document_id = doc.get("document_id", "")
            file_name = doc.get("file_name") or doc.get("title") or document_id
            # size is not stored in metadata; default to 0
            size = int(doc.get("size", 0) or 0)
            uploaded_at = doc.get("ingestion_timestamp") or doc.get("created_date") or ""
            total_pages = doc.get("total_pages")
            sources.append(SourceResponse(
                id=document_id,
                name=file_name,
                size=size,
                status="ready",
                uploadedAt=uploaded_at,
                pages=int(total_pages) if total_pages is not None else None,
            ))
        return sources
    except Exception as e:
        logger.error(f"Error listing sources: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list sources")


@app.delete("/api/sources/{source_id}", status_code=204)
async def delete_source(source_id: str) -> None:
    """
    Delete a source record by its document ID.

    Returns HTTP 204 on success, 404 if the source does not exist.

    Requirements: 2.7
    """
    try:
        deleted = delete_document_metadata(source_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting source {source_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete source")

"""
Session Manager

This module manages conversation sessions and persistent chat history.
Sessions are stored as JSON files with unique session_ids, and the manager
handles message persistence, session lifecycle.
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manages conversation sessions and persistent chat history.
    
    This class handles:
    - Creating new sessions with unique IDs
    - Loading and saving chat history to JSON files
    - Adding messages to sessions
    - Listing available sessions
    """
    
    def __init__(
        self,
        chat_history_dir: str = "data/chat_history"
    ) -> None:
        """
        Initialize the SessionManager.
        
        Args:
            chat_history_dir: Directory path for storing chat history JSON files
        """
        self.chat_history_dir = Path(chat_history_dir)
        self._lock = Lock()
        
        # Create chat history directory if it doesn't exist
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"SessionManager initialized: chat_history_dir={self.chat_history_dir}"
        )
    
    def create_session(self) -> str:
        """
        Create a new conversation session.
        
        Generates a unique session_id and creates an empty chat history file.
        
        Returns:
            str: Unique session_id (UUID format)
        """
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "title": "New Conversation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": []
        }
        
        session_file = self.chat_history_dir / f"{session_id}.json"
        
        with self._lock:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session from its JSON file.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Dict containing session data with keys:
                - session_id: str
                - title: str
                - created_at: str (ISO format)
                - updated_at: str (ISO format)
                - messages: List[dict]
            
            Returns None if session file doesn't exist.
        """
        session_file = self.chat_history_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"Session file not found: {session_id}")
            return None
        
        try:
            with self._lock:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
            
            logger.debug(
                f"Loaded session {session_id}: "
                f"{len(session_data.get('messages', []))} message(s)"
            )
            
            return session_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session file {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}", exc_info=True)
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions with metadata.
        
        Returns:
            List of dicts containing session metadata:
            [
                {
                    "session_id": str,
                    "title": str,
                    "created_at": str,
                    "updated_at": str,
                    "message_count": int
                },
                ...
            ]
            
            Sorted by updated_at (most recent first).
        """
        sessions = []
        
        try:
            # Find all JSON files in chat history directory
            session_files = list(self.chat_history_dir.glob("*.json"))
            
            for session_file in session_files:
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # Extract metadata
                    sessions.append({
                        "session_id": session_data.get("session_id", session_file.stem),
                        "title": session_data.get("title", "Untitled"),
                        "created_at": session_data.get("created_at", ""),
                        "updated_at": session_data.get("updated_at", ""),
                        "message_count": len(session_data.get("messages", []))
                    })
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        f"Skipping invalid session file {session_file.name}: {e}"
                    )
                    continue
            
            # Sort by updated_at (most recent first)
            sessions.sort(
                key=lambda s: s.get("updated_at", ""),
                reverse=True
            )
            
            logger.debug(f"Listed {len(sessions)} session(s)")
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error listing sessions: {e}", exc_info=True)
            return []
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to a session's chat history.
        
        Appends the message to the session file and updates the updated_at timestamp.
        If the session doesn't exist, logs an error.
        
        Args:
            session_id: Unique session identifier
            role: Message role ("user", "assistant", or "system")
            content: Message content text
            metadata: Optional metadata dict containing:
                - chunks_used: List[str] - chunk IDs used
                - files_referenced: List[str] - file paths referenced
                - retrieval_rounds: int - number of retrieval rounds
                - confidence: float - confidence score
                - tool_calls: List[dict] - tool calls made
        """
        session_file = self.chat_history_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.error(f"Cannot add message: session {session_id} not found")
            return
        
        # Never persist system messages — they are rebuilt fresh on every request
        if role == "system":
            return
        
        try:
            # Load current session
            with self._lock:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Create message entry
                message = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "role": role,
                    "content": content
                }
                
                # Add metadata if provided
                if metadata:
                    message["metadata"] = metadata  # type: ignore
                
                # Append message
                session_data["messages"].append(message)
                
                # Update timestamp
                session_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                # Update title from first user message if still "New Conversation"
                if (session_data.get("title") == "New Conversation" and 
                    role == "user" and content):
                    # Use first 50 chars of first user message as title
                    session_data["title"] = content[:50] + ("..." if len(content) > 50 else "")
                
                # Save back to file
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"Added {role} message to session {session_id}: "
                f"content_length={len(content)}"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session file {session_id}: {e}")
        except Exception as e:
            logger.error(
                f"Error adding message to session {session_id}: {e}",
                exc_info=True
            )
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load message history for a session.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            List of message dicts with keys:
                - id: str (UUID)
                - timestamp: str (ISO format)
                - role: str ("user", "assistant", or "system")
                - content: str
                - metadata: dict (optional)
            
            Returns empty list if session doesn't exist.
        """
        session_data = self.get_session(session_id)
        
        if session_data is None:
            logger.warning(f"Cannot get messages: session {session_id} not found")
            return []
        
        messages = session_data.get("messages", [])
        
        logger.debug(f"Retrieved {len(messages)} message(s) from session {session_id}")
        
        return messages
    
    def end_session(self, session_id: str) -> None:
        """
        End a session.
        
        This method marks the session as ended but does NOT delete the chat 
        history file. The session remains available for future access.
        
        Args:
            session_id: Unique session identifier
        """
        logger.info(f"Session ended: {session_id}")

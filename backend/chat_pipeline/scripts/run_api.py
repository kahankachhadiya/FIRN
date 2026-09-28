"""
API Entry Point for LLM Tool Calling Extension.

This script starts the FastAPI application using uvicorn with configuration
from settings. It includes graceful shutdown handling and startup logging.

Usage:
    python scripts/run_api.py
"""

import sys
import signal
from pathlib import Path

# Insert pipeline root (backend/chat_pipeline/) as sys.path[0] before any application imports
pipeline_root = Path(__file__).parent.parent  # backend/chat_pipeline/scripts/../ → backend/chat_pipeline/
sys.path.insert(0, str(pipeline_root))

# Early import guards — raise ValidationError/EnvironmentError before any socket is opened
import config.settings
import config.production

import uvicorn
from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class GracefulShutdown:
    """Handler for graceful shutdown on SIGINT and SIGTERM."""
    
    def __init__(self):
        self.should_exit = False
        self.force_exit = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        if self.should_exit:
            # Second signal - force exit
            logger.warning("Received second shutdown signal, forcing exit...")
            self.force_exit = True
            sys.exit(1)
        else:
            # First signal - graceful shutdown
            logger.info(f"Received shutdown signal ({signal.Signals(signum).name}), shutting down gracefully...")
            self.should_exit = True


def preload_models():
    """
    Initialize service clients for Docker-based models.
    
    This ensures service clients are ready and validates connectivity
    to the Docker-based embedding and classification services.
    """
    logger.info("Initializing service clients for Docker-based models...")
    
    try:
        # Initialize embedding service client (Docker-based)
        from db.embedding_service_client import get_embedding_client
        embedding_client = get_embedding_client()
        health = embedding_client.health_check()
        # Service reports 'ok' (not 'healthy') — check worker_ready as the real signal
        if health.get('status') == 'ok' and health.get('worker_ready'):
            logger.info("Embedding service client initialized and healthy")
        else:
            logger.warning(f"Embedding service client initialized but not healthy: {health}")
    except Exception as e:
        logger.warning(f"Failed to initialize embedding service client: {e}")
    
    logger.info("Service client initialization complete")


def main():
    """
    Main entry point for the API server.
    
    Configures and starts uvicorn with settings from config.
    Includes graceful shutdown handling and comprehensive startup logging.
    """
    # Initialize graceful shutdown handler
    GracefulShutdown()
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("Starting LLM Tool Calling API Server")
    logger.info("=" * 80)
    logger.info(f"Host: {settings.api_host}")
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"LLM Chat URL: {settings.llm_chat_base_url}")
    logger.info(f"LLM Chat Model: {settings.llm_chat_model}")
    logger.info(f"Chat History Dir: {settings.chat_history_dir}")
    logger.info(f"Max Retrieval Rounds: {settings.max_retrieval_rounds}")
    logger.info(f"Allowed File Extensions: {', '.join(settings.allowed_file_extensions)}")
    logger.info("=" * 80)
    
    # Ensure chat history directory exists
    chat_history_path = Path(settings.chat_history_dir)
    if not chat_history_path.exists():
        logger.info(f"Creating chat history directory: {settings.chat_history_dir}")
        chat_history_path.mkdir(parents=True, exist_ok=True)
    
    # Preload ML models at startup (embedding + classification)
    preload_models()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app="api.chat_api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        access_log=False,  # Disable access log spam
        reload=False,  # Set to True for development
        workers=1,  # Single worker for now; increase for production
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info("API server starting...")
        logger.info("Press Ctrl+C to stop")
        
        # Run the server
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    
    except Exception as e:
        logger.error(f"Unexpected error during server execution: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        logger.info("API server stopped")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

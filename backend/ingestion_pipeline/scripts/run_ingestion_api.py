"""
API Entry Point for the Ingestion Pipeline.

This script starts the FastAPI application using uvicorn with configuration
from settings. It handles document upload, processing, and database ingestion.

Usage:
    python backend/ingestion_pipeline/scripts/run_ingestion_api.py
"""

import sys
from pathlib import Path

# Insert pipeline root (backend/ingestion_pipeline/) as sys.path[0] before any application imports
pipeline_root = Path(__file__).parent.parent  # backend/ingestion_pipeline/scripts/../ → backend/ingestion_pipeline/
sys.path.insert(0, str(pipeline_root))

import uvicorn
from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Port is hardcoded for the ingestion pipeline — always 8001
INGESTION_PORT = 8001


import logging

class _NoPollingFilter(logging.Filter):
    """Suppress noisy polling endpoints from uvicorn access logs."""
    _SKIP = {"/api/processing/jobs"}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SKIP)


def main():
    """
    Main entry point for the ingestion API server.

    Configures and starts uvicorn on port 8001. The port is hardcoded
    to prevent accidental conflicts with the chat pipeline (port 8000).
    """
    # Suppress polling spam before uvicorn starts
    logging.getLogger("uvicorn.access").addFilter(_NoPollingFilter())

    logger.info("=" * 80)
    logger.info("Starting Ingestion Pipeline API Server")
    logger.info("=" * 80)
    logger.info(f"Host: localhost")
    logger.info(f"Port: {INGESTION_PORT}")
    logger.info(f"Unified KG Service URL: {settings.unified_kg_service_url}")
    logger.info("=" * 80)

    uvicorn.run(
        "api.ingestion_api:app",
        host="localhost",
        port=INGESTION_PORT,
        log_level="info",
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()

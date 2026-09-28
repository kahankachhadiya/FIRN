"""
Ingestion Pipeline API

FastAPI application for document upload, processing, and ingestion.

Requirements:
- 5.3: Ingestion pipeline runs as independent FastAPI server
- 10.3: Health endpoint returns pipeline status
- 10.4: Metrics endpoint returns pipeline metrics
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.rate_limit_middleware import RateLimitMiddleware
from api.processing_api import router as processing_router

app = FastAPI(
    title="Ingestion Pipeline API",
    description="Document upload, processing, and ingestion",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    exempt_paths=["/health", "/metrics"]
)

# Mount processing router
app.include_router(processing_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "pipeline": "ingestion"}


@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    return {"pipeline": "ingestion", "status": "ok"}


@app.on_event("startup")
async def startup_event():
    """Start the processing service worker on application startup."""
    from config.production import initialize_production_config, get_production_config
    try:
        get_production_config()
    except RuntimeError:
        initialize_production_config()

    from api.processing_service import ProcessingService
    service = ProcessingService.get_instance()
    await service.start_worker()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the processing service worker on application shutdown."""
    from api.processing_service import ProcessingService
    service = ProcessingService.get_instance()
    await service.stop_worker()

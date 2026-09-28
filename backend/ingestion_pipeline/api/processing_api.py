"""
Processing API endpoints for document processing.

This module provides FastAPI endpoints for:
- File upload for processing
- Starting processing jobs
- Getting job status
- Listing all jobs

Requirements:
- 1.1: Display file upload interface supporting PDF, DOCX, PPTX, XLSX, MP3, WAV, JPG, PNG
- 1.2: Validate file extensions against supported formats
- 1.3: Initiate processing pipeline for uploaded files
- 1.4: Display error message and reject unsupported file types
"""

import os
import shutil
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from pydantic import BaseModel, Field

from api.processing_service import ProcessingService, JobInfo
from utils.logging_utils import get_logger

# Import PipelineWrapper for extension validation

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/processing", tags=["processing"])

# Supported file extensions (from requirements)
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.mp3', '.wav', '.jpg', '.png'}


# Pydantic Models

class UploadResponse(BaseModel):
    """Response model for file upload."""
    uploaded_files: List[str] = Field(description="List of successfully uploaded file paths")
    rejected_files: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of rejected files with reasons"
    )


class StartProcessingRequest(BaseModel):
    """Request model for starting processing."""
    file_paths: List[str] = Field(description="List of file paths to process")
    auto_inject: bool = Field(default=True, description="Whether to auto-inject to database")


class StartProcessingResponse(BaseModel):
    """Response model for starting processing."""
    job_id: str = Field(description="Job ID for tracking")
    message: str = Field(description="Status message")


class StageInfoResponse(BaseModel):
    """Stage information in response."""
    name: str
    display_name: str
    model: Optional[str]
    status: str
    progress: float
    duration_seconds: Optional[float]


class FileResultResponse(BaseModel):
    """File result in response."""
    filename: str
    success: bool
    json_path: Optional[str]
    initial_md_path: Optional[str]
    processed_md_path: Optional[str]
    processing_time: float
    error: Optional[str]
    stages_completed: List[str]


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    total_files: int
    processed_files: int
    current_file: Optional[str]
    current_stage: Optional[str]
    stages: List[StageInfoResponse]
    results: List[FileResultResponse]
    created_at: str
    updated_at: str
    auto_inject: bool
    error: Optional[str]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


# Helper functions

def validate_file_extension(filename: str) -> tuple[bool, str]:
    """
    Validate that a file has a supported extension.
    
    Implements Requirements 1.2, 1.4: Validate file extensions and reject unsupported types.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    ext = Path(filename).suffix.lower()
    
    if not ext:
        return False, f"File '{filename}' has no extension"
    
    if ext not in SUPPORTED_EXTENSIONS:
        supported_list = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"File type '{ext}' is not supported. Supported: {supported_list}"
    
    return True, ""


def _job_info_to_response(job: JobInfo) -> JobStatusResponse:
    """Convert JobInfo to JobStatusResponse."""
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value if hasattr(job.status, 'value') else str(job.status),
        total_files=job.total_files,
        processed_files=job.processed_files,
        current_file=job.current_file,
        current_stage=job.current_stage,
        stages=[
            StageInfoResponse(
                name=s.name,
                display_name=s.display_name,
                model=s.model,
                status=s.status.value if hasattr(s.status, 'value') else str(s.status),
                progress=s.progress,
                duration_seconds=s.duration_seconds
            )
            for s in job.stages
        ],
        results=[
            FileResultResponse(
                filename=r.filename,
                success=r.success,
                json_path=r.json_path,
                initial_md_path=r.initial_md_path,
                processed_md_path=r.processed_md_path,
                processing_time=r.processing_time,
                error=r.error,
                stages_completed=r.stages_completed
            )
            for r in job.results
        ],
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        auto_inject=job.auto_inject,
        error=job.error
    )


# Processing Endpoints

@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload files for processing.
    
    Implements Requirements:
    - 1.1: File upload interface supporting PDF, DOCX, PPTX, XLSX, MP3, WAV, JPG, PNG
    - 1.2: Validate file extensions against supported formats
    - 1.4: Display error message and reject unsupported file types
    
    Args:
        files: List of files to upload
        
    Returns:
        UploadResponse with uploaded and rejected files
    """
    uploaded_files = []
    rejected_files = []
    
    for file in files:
        # Validate extension
        is_valid, error_msg = validate_file_extension(file.filename)
        
        if not is_valid:
            rejected_files.append({
                "filename": file.filename,
                "reason": error_msg
            })
            logger.warning(f"Rejected file upload: {file.filename} - {error_msg}")
            continue
        
        # Generate unique filename to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{unique_id}_{file.filename}"
        tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        file_path = tmp_dir / safe_filename
        
        try:
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(str(file_path))
            logger.info(f"Uploaded file: {file.filename} -> {file_path}")
            
        except Exception as e:
            rejected_files.append({
                "filename": file.filename,
                "reason": f"Failed to save file: {str(e)}"
            })
            logger.error(f"Failed to save uploaded file {file.filename}: {e}")
    
    return UploadResponse(
        uploaded_files=uploaded_files,
        rejected_files=rejected_files
    )


@router.post("/start", response_model=StartProcessingResponse)
async def start_processing(request: StartProcessingRequest) -> StartProcessingResponse:
    """
    Start processing uploaded files.
    
    Implements Requirement 1.3: Initiate processing pipeline for uploaded files.
    
    Args:
        request: StartProcessingRequest with file paths
        
    Returns:
        StartProcessingResponse with job ID
    """
    if not request.file_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for processing"
        )
    
    # Validate all files exist
    missing_files = []
    for file_path in request.file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Files not found: {', '.join(missing_files)}"
        )
    
    # Get processing service instance
    service = ProcessingService.get_instance()
    
    # Submit job
    job_id = await service.submit_job(
        file_paths=request.file_paths,
        auto_inject=request.auto_inject
    )
    
    logger.info(f"Started processing job {job_id} with {len(request.file_paths)} files")
    
    return StartProcessingResponse(
        job_id=job_id,
        message=f"Processing started for {len(request.file_paths)} file(s)"
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get status of a processing job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        JobStatusResponse with current status
    """
    service = ProcessingService.get_instance()
    job = service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    return _job_info_to_response(job)


@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs() -> List[JobStatusResponse]:
    """
    List all processing jobs.
    
    Returns:
        List of JobStatusResponse objects
    """
    service = ProcessingService.get_instance()
    jobs = service.get_all_jobs()
    
    return [_job_info_to_response(job) for job in jobs if job]


# Export routers for mounting in main app
__all__ = ['router', 'validate_file_extension', 'SUPPORTED_EXTENSIONS']

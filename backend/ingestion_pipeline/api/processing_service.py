"""
Processing Service for managing document processing jobs.

This module provides a service for managing background document processing jobs,
including job queue management, status tracking, and integration with the
PipelineWrapper for actual document processing.

Requirements:
- 6.1: Execute pipeline in background task (non-blocking)
- 6.2: Keep Chat API responsive during processing
- 6.3: Return current progress without blocking
- 6.4: Process queued jobs sequentially without blocking API
- 4.4: Auto-inject to database on completion
"""

import asyncio
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum
import threading

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Status of a processing job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    """Status of a processing stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Processing stages with display names and model attribution
PROCESSING_STAGES = [
    ("file_classification", "File Classification", None),
    ("conversion", "Conversion", "Qwen"),
    ("image_transcription", "Image Transcription", "Qwen"),
    ("chunking", "Chunking", None),
    ("metadata_extraction", "Metadata Extraction", "Qwen"),
    ("json_generation", "JSON Generation", None),
    ("database_injection", "Database Injection", None),
]


@dataclass
class StageInfo:
    """Information about a processing stage."""
    name: str
    display_name: str
    model: Optional[str]
    status: StageStatus = StageStatus.PENDING
    progress: float = 0.0
    duration_seconds: Optional[float] = None
    start_time: Optional[float] = None


@dataclass
class FileResult:
    """Result of processing a single file."""
    filename: str
    success: bool
    json_path: Optional[str] = None
    initial_md_path: Optional[str] = None
    processed_md_path: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None
    stages_completed: List[str] = field(default_factory=list)


@dataclass
class JobInfo:
    """Information about a processing job."""
    job_id: str
    status: JobStatus
    total_files: int
    processed_files: int = 0
    current_file: Optional[str] = None
    current_stage: Optional[str] = None
    stages: List[StageInfo] = field(default_factory=list)
    results: List[FileResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    auto_inject: bool = True
    error: Optional[str] = None


class ProcessingService:
    """
    Service for managing document processing jobs.
    
    Provides non-blocking job submission, status tracking, and sequential
    job processing through a background worker.
    
    Requirements:
    - 6.1: Non-blocking job submission (returns job_id immediately)
    - 6.2: Background processing keeps API responsive
    - 6.3: Status retrieval is non-blocking
    - 6.4: Sequential job processing
    - 4.4: Auto-injection to database on completion
    """
    
    _instance: Optional['ProcessingService'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'ProcessingService':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, pipeline_wrapper=None):
        """
        Initialize the processing service.
        
        Args:
            pipeline_wrapper: Optional PipelineWrapper instance. If None,
                            will be lazily initialized when needed.
        """
        if self._initialized:
            return
        
        self._pipeline_wrapper = pipeline_wrapper
        self._job_queue: asyncio.Queue = None  # Initialized in start_worker
        self._jobs: Dict[str, JobInfo] = {}
        self._jobs_lock = threading.RLock()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._initialized = True
        
        logger.info("ProcessingService initialized")
    
    @classmethod
    def get_instance(cls, pipeline_wrapper=None) -> 'ProcessingService':
        """Get singleton instance."""
        return cls(pipeline_wrapper)
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing purposes)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._running = False
                if cls._instance._worker_task:
                    cls._instance._worker_task.cancel()
            cls._instance = None
    
    def set_pipeline_wrapper(self, pipeline_wrapper) -> None:
        """Set the pipeline wrapper instance."""
        self._pipeline_wrapper = pipeline_wrapper
    
    async def start_worker(self) -> None:
        """Start the background worker for processing jobs."""
        if self._running:
            return
        
        self._job_queue = asyncio.Queue()
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("ProcessingService worker started")
    
    async def stop_worker(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("ProcessingService worker stopped")
    
    async def _worker_loop(self) -> None:
        """
        Background worker loop for processing jobs sequentially.
        
        Implements Requirement 6.4: Process jobs sequentially without blocking API.
        """
        logger.info("Worker loop started")
        
        while self._running:
            try:
                # Wait for a job with timeout to allow checking _running flag
                try:
                    job_id = await asyncio.wait_for(
                        self._job_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the job
                await self._process_job(job_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
        
        logger.info("Worker loop stopped")

    
    async def submit_job(
        self,
        file_paths: List[str],
        auto_inject: bool = True
    ) -> str:
        """
        Submit a processing job to the queue.
        
        This method returns immediately with a job_id, implementing
        Requirement 6.1: Non-blocking job submission.
        
        Args:
            file_paths: Files to process
            auto_inject: Whether to auto-inject to database on completion
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        # Initialize stages for the job
        stages = [
            StageInfo(
                name=name,
                display_name=display_name,
                model=model
            )
            for name, display_name, model in PROCESSING_STAGES
        ]
        
        # Create job info
        job_info = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            total_files=len(file_paths),
            stages=stages,
            auto_inject=auto_inject
        )
        
        # Store job info
        with self._jobs_lock:
            self._jobs[job_id] = job_info
        
        # Store file paths for processing
        job_info._file_paths = file_paths
        
        # Add to queue
        if self._job_queue is None:
            # Auto-start worker if not running
            await self.start_worker()
        
        await self._job_queue.put(job_id)
        
        logger.info(f"Job {job_id} submitted with {len(file_paths)} files")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """
        Get current status of a processing job.
        
        This method is non-blocking, implementing Requirement 6.3.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobInfo if found, None otherwise
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                # Return a copy to prevent external modification
                return JobInfo(
                    job_id=job.job_id,
                    status=job.status,
                    total_files=job.total_files,
                    processed_files=job.processed_files,
                    current_file=job.current_file,
                    current_stage=job.current_stage,
                    stages=[
                        StageInfo(
                            name=s.name,
                            display_name=s.display_name,
                            model=s.model,
                            status=s.status,
                            progress=s.progress,
                            duration_seconds=s.duration_seconds
                        )
                        for s in job.stages
                    ],
                    results=[
                        FileResult(
                            filename=r.filename,
                            success=r.success,
                            json_path=r.json_path,
                            initial_md_path=r.initial_md_path,
                            processed_md_path=r.processed_md_path,
                            processing_time=r.processing_time,
                            error=r.error,
                            stages_completed=r.stages_completed.copy()
                        )
                        for r in job.results
                    ],
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    auto_inject=job.auto_inject,
                    error=job.error
                )
            return None
    
    def get_all_jobs(self) -> List[JobInfo]:
        """
        Get status of all jobs.
        
        Returns:
            List of JobInfo objects
        """
        with self._jobs_lock:
            return [self.get_job_status(job_id) for job_id in self._jobs.keys()]
    
    async def _process_job(self, job_id: str) -> None:
        """
        Process a single job.
        
        Args:
            job_id: Job identifier
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return
            
            job.status = JobStatus.PROCESSING
            job.updated_at = datetime.utcnow()
            file_paths = getattr(job, '_file_paths', [])
        
        logger.info(f"Processing job {job_id} with {len(file_paths)} files")
        
        try:
            # Ensure pipeline wrapper is available
            if self._pipeline_wrapper is None:
                # Lazy import to avoid circular dependencies
                from extractors.pipeline_wrapper import PipelineWrapper
                self._pipeline_wrapper = PipelineWrapper()
            
            # Process each file
            for file_path in file_paths:
                await self._process_file(job_id, file_path)
            
            # Update job status
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job:
                    # Check if all files succeeded
                    all_success = all(r.success for r in job.results)
                    job.status = JobStatus.COMPLETED if all_success else JobStatus.FAILED
                    job.updated_at = datetime.utcnow()
                    job.current_file = None
                    job.current_stage = None
            
            logger.info(f"Job {job_id} completed with status {job.status}")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.updated_at = datetime.utcnow()
    
    async def _process_file(self, job_id: str, file_path: str) -> None:
        """
        Process a single file within a job.
        
        Args:
            job_id: Job identifier
            file_path: Path to the file to process
        """
        from pathlib import Path
        filename = Path(file_path).name
        
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.current_file = filename
                job.updated_at = datetime.utcnow()
        
        logger.info(f"Starting file processing: {filename} (Job: {job_id})")
        logger.info(f"File path: {file_path}")
        
        # Create progress callback
        def progress_callback(stage: str, progress: float):
            self._update_stage_progress(job_id, stage, progress)
        
        try:
            # Reset stages for this file
            self._reset_stages(job_id)
            
            # Get auto_inject setting from job
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                auto_inject = job.auto_inject if job else True
            
            logger.info(f"Processing document: {filename} (auto_inject={auto_inject})")
            
            # Process the file using pipeline wrapper
            # PipelineWrapper now handles injection internally (Requirement 4.4)
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            processing_start = time.time()
            
            result = await loop.run_in_executor(
                None,
                lambda: self._pipeline_wrapper.process_file(
                    file_path, 
                    progress_callback,
                    auto_inject=auto_inject
                )
            )
            
            processing_time = time.time() - processing_start
            logger.info(f"Document processing completed for {filename} in {processing_time:.2f}s")
            logger.info(f"Processing result: success={result.success}")
            
            if result.success:
                logger.info(f"Γ£ô Document {filename} processed successfully")
                if result.json_path:
                    logger.info(f"  JSON output: {result.json_path}")
                if result.injection_success:
                    logger.info("  Γ£ô Database injection successful")
                elif result.injection_success is False:
                    logger.warning(f"  ΓÜá Database injection failed: {result.injection_error}")
            else:
                logger.error(f"Γ£ù Document processing failed for {filename}: {result.error}")
            
            # Update database_injection stage based on injection result
            if result.injection_success is not None:
                if result.injection_success:
                    self._update_stage_progress(job_id, "database_injection", 1.0)
                else:
                    # Mark injection stage as failed
                    with self._jobs_lock:
                        job = self._jobs.get(job_id)
                        if job:
                            for stage in job.stages:
                                if stage.name == "database_injection":
                                    stage.status = StageStatus.FAILED
                                    break
            
            # Store result
            file_result = FileResult(
                filename=result.filename,
                success=result.success,
                json_path=result.json_path,
                initial_md_path=result.initial_md_path,
                processed_md_path=result.processed_md_path,
                processing_time=result.processing_time,
                error=result.error,
                stages_completed=result.stages_completed
            )
            
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job:
                    job.results.append(file_result)
                    job.processed_files += 1
                    job.updated_at = datetime.utcnow()
            
            logger.info(f"File processing completed: {filename} (success={result.success})")
            
        except Exception as e:
            processing_time = time.time() - processing_start if 'processing_start' in locals() else 0
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            
            file_result = FileResult(
                filename=filename,
                success=False,
                processing_time=processing_time,
                error=str(e)
            )
            
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job:
                    job.results.append(file_result)
                    job.processed_files += 1
                    job.updated_at = datetime.utcnow()
    
    def _reset_stages(self, job_id: str) -> None:
        """Reset all stages to pending for a new file."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                for stage in job.stages:
                    stage.status = StageStatus.PENDING
                    stage.progress = 0.0
                    stage.duration_seconds = None
                    stage.start_time = None
    
    def _update_stage_progress(self, job_id: str, stage_name: str, progress: float) -> None:
        """
        Update progress for a specific stage.
        
        Args:
            job_id: Job identifier
            stage_name: Name of the stage
            progress: Progress value (0.0 to 1.0)
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            
            for stage in job.stages:
                if stage.name == stage_name:
                    if stage.status == StageStatus.PENDING:
                        stage.status = StageStatus.RUNNING
                        stage.start_time = time.time()
                    
                    stage.progress = progress
                    
                    if progress >= 1.0:
                        stage.status = StageStatus.COMPLETED
                        if stage.start_time:
                            stage.duration_seconds = time.time() - stage.start_time
                    
                    job.current_stage = stage.display_name
                    job.updated_at = datetime.utcnow()
                    break
    
    async def _inject_to_database(self, job_id: str, json_path: str) -> None:
        """
        Inject processed JSON to database.
        
        Implements Requirement 4.4: Auto-inject to database on completion.
        
        Args:
            job_id: Job identifier
            json_path: Path to the JSON file to inject
        """
        logger.info(f"Injecting {json_path} to database for job {job_id}")
        
        # Update stage progress
        self._update_stage_progress(job_id, "database_injection", 0.0)
        
        try:
            # Import injection function
            from ingestion.ingest_json import ingest_document_from_json
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: ingest_document_from_json(json_path)
            )
            
            self._update_stage_progress(job_id, "database_injection", 1.0)
            logger.info(f"Successfully injected {json_path} to database")
            
        except Exception as e:
            logger.error(f"Failed to inject {json_path}: {e}", exc_info=True)
            
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job:
                    for stage in job.stages:
                        if stage.name == "database_injection":
                            stage.status = StageStatus.FAILED
                            break
            
            # Don't fail the whole job, just log the error
            # The JSON is still saved and can be manually injected later

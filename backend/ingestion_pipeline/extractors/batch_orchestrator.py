"""Batch Pipeline Orchestrator for multi-modal document processing.

Orchestrates batch processing across all phases with GPU memory management.
Implements phase-based processing to maximize GPU efficiency by loading
each model once, processing all compatible inputs, then releasing GPU memory.

Processing Phases:
- Phase 1: Docling (PyTorch GPU) - Document conversion
- Phase 2: Whisper (PyTorch GPU) - Audio transcription
- Phase 3: Image Captioning (LM Studio) - Vision model
- Phase 4: Summarization (LM Studio) - Language model

Requirements: 1.1, 2.1-2.5, 6.5, 10.1
"""

import os
import logging
import atexit
import signal
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import torch
from extractors.config import BatchConfig
from extractors.progress_tracker import ProgressTracker, PhaseStatus
from extractors.file_classifier import FileClassifier, FileType
from extractors.docling_gpu_processor import (
    DoclingGPUProcessor,
    collect_image_paths_from_markdown,
)
from extractors.whisper_gpu_processor import (
    WhisperGPUProcessor,
    should_skip_phase as should_skip_whisper_phase,
)
from external_clients.openai_compatible_client import OpenAICompatibleClient as LMStudioClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline stubs replacing removed gpu_manager, batch_classifier, llm_model_manager
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    """Information about an image to be captioned."""
    path: str
    source: str = "extracted"
    source_file: str = ""
    markdown_file: str = ""
    page_number: int = 0


@dataclass
class ClassifiedBatch:
    """Files grouped by processing type."""
    documents: List[str] = field(default_factory=list)
    audio_files: List[str] = field(default_factory=list)
    standalone_images: List[ImageInfo] = field(default_factory=list)
    markdown_files: List[str] = field(default_factory=list)
    extracted_images: List[ImageInfo] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.documents) + len(self.audio_files) + len(self.standalone_images)

    @property
    def all_images(self) -> List[ImageInfo]:
        return list(self.standalone_images) + list(self.extracted_images)


class BatchClassifier:
    """Classifies files in a directory into processing groups."""

    DOCUMENT_EXTS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.html', '.htm'}
    AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.flac'}
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

    def scan_directory(self, input_dir: str) -> ClassifiedBatch:
        batch = ClassifiedBatch()
        for root, _, files in os.walk(input_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                ext = Path(fpath).suffix.lower()
                if ext in self.DOCUMENT_EXTS:
                    batch.documents.append(fpath)
                elif ext in self.AUDIO_EXTS:
                    batch.audio_files.append(fpath)
                elif ext in self.IMAGE_EXTS:
                    batch.standalone_images.append(ImageInfo(path=fpath, source="standalone"))
        return batch


class GPUMemoryManager:
    """Manages GPU memory for batch processing phases."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._peak_memory = 0.0

    @property
    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_peak_stats(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log_memory_status(self, label: str = "") -> None:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.debug(f"GPU [{label}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

    def cleanup(self) -> None:
        self.clear_cache()


@dataclass
class ModelLoadResult:
    success: bool
    message: str = ""
    load_time_seconds: float = 0.0


class LLMModelManager:
    """Manages LM Studio model loading/unloading via its REST API."""

    def __init__(self, endpoint: str = "http://127.0.0.1:1234", max_retries: int = 3):
        self.endpoint = endpoint.rstrip('/')
        self.max_retries = max_retries
        self._loaded_model: Optional[str] = None

    def load_model(self, model_id: str, verify_ready: bool = True) -> ModelLoadResult:
        import time
        import requests
        start = time.time()
        try:
            resp = requests.post(
                f"{self.endpoint}/api/v0/models/load",
                json={"identifier": model_id},
                timeout=120,
            )
            elapsed = time.time() - start
            if resp.status_code in (200, 201):
                self._loaded_model = model_id
                return ModelLoadResult(success=True, load_time_seconds=elapsed)
            return ModelLoadResult(success=False, message=resp.text, load_time_seconds=elapsed)
        except Exception as exc:
            return ModelLoadResult(success=False, message=str(exc), load_time_seconds=time.time() - start)

    def unload_model(self, wait_for_memory: bool = False) -> None:
        import requests
        if self._loaded_model:
            try:
                requests.post(
                    f"{self.endpoint}/api/v0/models/unload",
                    json={"identifier": self._loaded_model},
                    timeout=30,
                )
            except Exception as exc:
                logger.warning(f"LLMModelManager: unload failed: {exc}")
            finally:
                self._loaded_model = None


class JSONBuilder:
    """Builds and saves the final structured JSON output."""

    def build_output(
        self,
        metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        file_path: str,
        source_type: str,
        file_name: str,
        processing_device: str = "cpu",
    ) -> Dict[str, Any]:
        import uuid
        from datetime import datetime
        return {
            "document_id": str(uuid.uuid4()),
            "file_name": file_name,
            "file_path": file_path,
            "source_type": source_type,
            "processing_device": processing_device,
            "processed_at": datetime.utcnow().isoformat(),
            "document_metadata": metadata if isinstance(metadata, dict) else (
                metadata.__dict__ if hasattr(metadata, '__dict__') else {}
            ),
            "chunks": chunks,
        }

    def save_to_file(self, output: Dict[str, Any], json_path: str) -> None:
        import json as _json
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(output, f, indent=2, ensure_ascii=False)


@dataclass
class PhaseResult:
    """Result of a processing phase."""
    phase_name: str
    status: str  # "completed", "skipped", "failed"
    files_processed: int = 0
    files_failed: int = 0
    completed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    # Phase-specific outputs
    output_paths: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    extracted_images: List[ImageInfo] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str = ""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    phases: Dict[str, PhaseResult] = field(default_factory=dict)
    total_duration_seconds: float = 0.0
    output_directory: str = ""
    gpu_peak_memory_gb: float = 0.0


class BatchOrchestrator:
    """Orchestrates batch processing across all phases.
    
    Manages GPU memory efficiently by:
    - Loading each model once for all compatible inputs
    - Clearing CUDA cache between phases
    - Tracking progress for resume capability
    - Handling errors without stopping other files
    
    Example:
        config = BatchConfig()
        orchestrator = BatchOrchestrator(config)
        
        result = orchestrator.process_batch("input/")
        print(f"Processed {result.successful_files} files")
    
    Requirements: 1.1, 6.5, 10.1
    """
    
    # Default model names for LM Studio (will be overridden from config)
    VISION_MODEL = "google/gemma-3n-e4b"
    SUMMARIZATION_MODEL = "google/gemma-3n-e4b"
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize BatchOrchestrator.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.gpu_manager = GPUMemoryManager(self.config.cuda_device)
        self.progress_tracker = ProgressTracker(self.config.progress_file)
        self.classifier = BatchClassifier()
        self.lm_studio_manager = LLMModelManager(
            endpoint=self.config.lm_studio_endpoint,
            max_retries=self.config.max_retries,
        )
        
        # Track classified batch for phase coordination
        self._classified_batch: Optional[ClassifiedBatch] = None
        
        # Register cleanup handlers
        self._register_cleanup_handlers()
        
        logger.info(f"BatchOrchestrator initialized with device: {self.config.cuda_device}")

    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for graceful shutdown.
        
        Ensures GPU resources are released on exit or signal.
        
        Requirements: 6.5
        """
        atexit.register(self._cleanup_handler)
        
        # Register signal handlers (Windows-compatible)
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (ValueError, OSError):
            # Signal handling may not work in all contexts
            logger.warning("Could not register signal handlers")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, cleaning up...")
        self._cleanup_handler()
    
    def _cleanup_handler(self) -> None:
        """Cleanup handler for exit.
        
        Ensures GPU memory is released and progress is saved.
        
        Requirements: 6.5
        """
        try:
            # Clear GPU memory
            self.gpu_manager.cleanup()
            
            # Save progress
            if self.progress_tracker.is_initialized:
                self.progress_tracker.save()
            
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def classify_all_files(self, input_dir: str) -> ClassifiedBatch:
        """Classify all files in input directory.
        
        Args:
            input_dir: Path to input directory
            
        Returns:
            ClassifiedBatch with files grouped by processing type
            
        Requirements: 1.1
        """
        classified = self.classifier.scan_directory(input_dir)
        self._classified_batch = classified
        
        logger.info(
            f"Classified {classified.total_files} files: "
            f"{len(classified.documents)} documents, "
            f"{len(classified.audio_files)} audio, "
            f"{len(classified.standalone_images)} images"
        )
        
        return classified
    
    def _get_output_dir_for_file(self, file_path: str) -> str:
        """Get output directory for a file following naming convention.
        
        Creates directory structure: Database/Processed_Docs/{filename}_{ext}/
        
        Args:
            file_path: Path to input file
            
        Returns:
            Output directory path
            
        Requirements: 2.3
        """
        filename = Path(file_path).stem
        ext = Path(file_path).suffix.lower().lstrip('.')
        
        output_dir = os.path.join(
            self.config.output_base_dir,
            f"{filename}_{ext}"
        )
        
        return output_dir
    
    def execute_phase_1_docling(
        self,
        documents: List[str]
    ) -> PhaseResult:
        """Execute Phase 1: Docling document processing.
        
        Processes all documents with GPU-accelerated Docling:
        - Initialize DoclingGPUProcessor with CUDA
        - Process all documents sequentially with GPU
        - Collect image paths from generated .md files
        - Clear CUDA cache after completion
        - Update progress tracker
        
        Args:
            documents: List of document file paths to process
            
        Returns:
            PhaseResult with processing results
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        import time
        start_time = time.time()
        
        phase_name = ProgressTracker.PHASE_1_DOCLING
        
        # Handle empty input - skip phase
        if not documents:
            logger.info("Phase 1 (Docling): No documents to process, skipping")
            self.progress_tracker.mark_phase_skipped(phase_name, "No documents")
            return PhaseResult(
                phase_name=phase_name,
                status="skipped",
                files_processed=0,
                files_failed=0,
            )
        
        # Set phase status to in progress
        self.progress_tracker.set_phase_status(phase_name, PhaseStatus.IN_PROGRESS)
        self.progress_tracker.set_phase_files(phase_name, documents)
        
        # Get files to process with failed files first (Requirements 8.3, 10.5)
        files_to_process = self._get_files_with_failed_first(phase_name, documents)
        
        # Count failed files being retried
        failed_files = set(self.progress_tracker.get_failed_files(phase_name))
        retry_count = len([f for f in files_to_process if f in failed_files])
        skip_count = len(documents) - len(files_to_process) - len(failed_files)
        
        logger.info(
            f"Phase 1 (Docling): Processing {len(files_to_process)} documents "
            f"({skip_count} already completed, {retry_count} retrying failed)"
        )
        
        # Initialize result tracking
        result = PhaseResult(
            phase_name=phase_name,
            status="in_progress",
        )
        
        # Initialize Docling processor with GPU
        processor: Optional[DoclingGPUProcessor] = None
        
        try:
            # Log GPU memory before processing
            self.gpu_manager.reset_peak_stats()
            self.gpu_manager.log_memory_status("Phase 1 Start")
            
            # Initialize processor
            processor = DoclingGPUProcessor(device=self.config.cuda_device)
            processor.initialize()
            
            # Process each document sequentially
            for file_path in files_to_process:
                try:
                    # Get output directory
                    output_dir = self._get_output_dir_for_file(file_path)
                    
                    # Convert document
                    doc_result = processor.convert_document(file_path, output_dir)
                    
                    if doc_result.success:
                        # Mark as completed
                        self.progress_tracker.mark_file_completed(phase_name, file_path)
                        result.completed_files.append(file_path)
                        result.output_paths.append(doc_result.markdown_path)
                        
                        # Collect image paths
                        result.image_paths.extend(doc_result.image_paths)
                        
                        # Convert ExtractedImage to ImageInfo for phase coordination
                        for img in doc_result.images:
                            result.extracted_images.append(ImageInfo(
                                path=img.path,
                                source="extracted",
                                source_file=file_path,
                                markdown_file=doc_result.markdown_path,
                                page_number=img.page_number,
                            ))
                        
                        logger.info(f"Phase 1: Processed {file_path} -> {doc_result.markdown_path}")
                    else:
                        # Mark as failed
                        error_msg = doc_result.error or "Unknown error"
                        self.progress_tracker.mark_file_failed(phase_name, file_path, error_msg)
                        result.failed_files.append(file_path)
                        result.errors[file_path] = error_msg
                        
                        logger.error(f"Phase 1: Failed to process {file_path}: {error_msg}")
                        
                except Exception as e:
                    # Handle unexpected errors
                    error_msg = str(e)
                    self.progress_tracker.mark_file_failed(phase_name, file_path, error_msg)
                    result.failed_files.append(file_path)
                    result.errors[file_path] = error_msg
                    
                    logger.error(f"Phase 1: Error processing {file_path}: {e}")
                    
                    # Continue with next file (error isolation)
                    continue
            
            # Update result counts
            result.files_processed = len(result.completed_files)
            result.files_failed = len(result.failed_files)
            
            # Determine final status
            if result.files_failed == 0:
                result.status = "completed"
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            else:
                result.status = "completed"  # Completed with failures
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            
        except Exception as e:
            # Phase-level failure
            logger.error(f"Phase 1 (Docling): Critical error: {e}")
            result.status = "failed"
            result.errors["_phase"] = str(e)
            self.progress_tracker.set_phase_status(phase_name, PhaseStatus.FAILED)
            
        finally:
            # Cleanup processor
            if processor:
                processor.cleanup()
            
            # Clear CUDA cache (Requirements 2.5)
            if self.config.clear_cache_between_phases:
                self.gpu_manager.clear_cache()
                self.gpu_manager.log_memory_status("Phase 1 End")
        
        # Calculate duration
        result.duration_seconds = time.time() - start_time
        
        logger.info(
            f"Phase 1 (Docling): Completed in {result.duration_seconds:.2f}s. "
            f"Processed: {result.files_processed}, Failed: {result.files_failed}"
        )
        
        return result
    
    def _collect_images_from_markdown_files(
        self,
        markdown_paths: List[str]
    ) -> Tuple[List[str], List[ImageInfo]]:
        """Collect all image paths from generated markdown files.
        
        Scans markdown files for image references and returns paths.
        
        Args:
            markdown_paths: List of markdown file paths
            
        Returns:
            Tuple of (list of image paths, list of ImageInfo objects)
            
        Requirements: 2.4
        """
        all_image_paths = []
        all_image_infos = []
        
        for md_path in markdown_paths:
            if not os.path.exists(md_path):
                continue
            
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get relative image paths from markdown
                relative_paths = collect_image_paths_from_markdown(content)
                
                # Convert to absolute paths
                md_dir = os.path.dirname(md_path)
                for rel_path in relative_paths:
                    abs_path = os.path.normpath(os.path.join(md_dir, rel_path))
                    
                    if os.path.exists(abs_path):
                        all_image_paths.append(abs_path)
                        all_image_infos.append(ImageInfo(
                            path=abs_path,
                            source="extracted",
                            markdown_file=md_path,
                        ))
                        
            except Exception as e:
                logger.warning(f"Error reading markdown file {md_path}: {e}")
                continue
        
        return all_image_paths, all_image_infos
    
    def update_classified_batch_after_phase_1(
        self,
        phase_result: PhaseResult
    ) -> None:
        """Update classified batch with Phase 1 outputs.
        
        Adds markdown files and extracted images to the batch
        for subsequent phases.
        
        Args:
            phase_result: Result from Phase 1
        """
        if self._classified_batch is None:
            return
        
        # Add markdown files
        self._classified_batch.markdown_files.extend(phase_result.output_paths)
        
        # Add extracted images
        self._classified_batch.extracted_images.extend(phase_result.extracted_images)
        
        logger.info(
            f"Updated batch: {len(self._classified_batch.markdown_files)} markdown files, "
            f"{len(self._classified_batch.extracted_images)} extracted images"
        )
    
    def execute_phase_2_whisper(
        self,
        audio_files: List[str]
    ) -> PhaseResult:
        """Execute Phase 2: Whisper audio transcription.
        
        Processes all audio files with GPU-accelerated Whisper:
        - Skip phase if no audio files (Requirements 3.5)
        - Load Whisper model on GPU with CUDA 12.8 (Requirements 3.1)
        - Process all audio files sequentially (Requirements 3.2)
        - Generate markdown with [HH:MM:SS] Speaker N: format (Requirements 3.3)
        - Unload model and clear CUDA cache after completion (Requirements 3.4)
        - Update progress tracker
        
        Args:
            audio_files: List of audio file paths to process
            
        Returns:
            PhaseResult with processing results
            
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
        """
        import time
        start_time = time.time()
        
        phase_name = ProgressTracker.PHASE_2_WHISPER
        
        # Handle empty input - skip phase (Requirements 3.5)
        if should_skip_whisper_phase(audio_files):
            logger.info("Phase 2 (Whisper): No audio files to process, skipping")
            self.progress_tracker.mark_phase_skipped(phase_name, "No audio files")
            return PhaseResult(
                phase_name=phase_name,
                status="skipped",
                files_processed=0,
                files_failed=0,
            )
        
        # Set phase status to in progress
        self.progress_tracker.set_phase_status(phase_name, PhaseStatus.IN_PROGRESS)
        self.progress_tracker.set_phase_files(phase_name, audio_files)
        
        # Get files to process with failed files first (Requirements 8.3, 10.5)
        files_to_process = self._get_files_with_failed_first(phase_name, audio_files)
        
        # Count failed files being retried
        failed_files = set(self.progress_tracker.get_failed_files(phase_name))
        retry_count = len([f for f in files_to_process if f in failed_files])
        skip_count = len(audio_files) - len(files_to_process) - len(failed_files)
        
        logger.info(
            f"Phase 2 (Whisper): Processing {len(files_to_process)} audio files "
            f"({skip_count} already completed, {retry_count} retrying failed)"
        )
        
        # Initialize result tracking
        result = PhaseResult(
            phase_name=phase_name,
            status="in_progress",
        )
        
        # Initialize Whisper processor
        processor: Optional[WhisperGPUProcessor] = None
        
        try:
            # Log GPU memory before processing
            self.gpu_manager.reset_peak_stats()
            self.gpu_manager.log_memory_status("Phase 2 Start")
            
            # Initialize processor and load model (Requirements 3.1)
            processor = WhisperGPUProcessor(
                model_name="medium",
                device=self.config.cuda_device,
                enable_diarization=True
            )
            processor.load_model()
            
            # Process each audio file sequentially (Requirements 3.2)
            for file_path in files_to_process:
                try:
                    # Get output directory
                    output_dir = self._get_output_dir_for_file(file_path)
                    
                    # Transcribe audio (Requirements 3.2, 3.3)
                    processor.transcribe(file_path, output_dir)
                    
                    # Mark as completed
                    self.progress_tracker.mark_file_completed(phase_name, file_path)
                    result.completed_files.append(file_path)
                    
                    # Track output path
                    filename = Path(file_path).stem
                    md_path = os.path.join(output_dir, f"initial_{filename}.md")
                    result.output_paths.append(md_path)
                    
                    logger.info(f"Phase 2: Processed {file_path} -> {md_path}")
                    
                except Exception as e:
                    # Handle unexpected errors
                    error_msg = str(e)
                    self.progress_tracker.mark_file_failed(phase_name, file_path, error_msg)
                    result.failed_files.append(file_path)
                    result.errors[file_path] = error_msg
                    
                    logger.error(f"Phase 2: Error processing {file_path}: {e}")
                    
                    # Continue with next file (error isolation)
                    continue
            
            # Update result counts
            result.files_processed = len(result.completed_files)
            result.files_failed = len(result.failed_files)
            
            # Determine final status
            if result.files_failed == 0:
                result.status = "completed"
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            else:
                result.status = "completed"  # Completed with failures
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            
        except Exception as e:
            # Phase-level failure
            logger.error(f"Phase 2 (Whisper): Critical error: {e}")
            result.status = "failed"
            result.errors["_phase"] = str(e)
            self.progress_tracker.set_phase_status(phase_name, PhaseStatus.FAILED)
            
        finally:
            # Unload model and cleanup processor (Requirements 3.4)
            if processor:
                processor.unload_model()
            
            # Clear CUDA cache (Requirements 3.4)
            if self.config.clear_cache_between_phases:
                self.gpu_manager.clear_cache()
                self.gpu_manager.log_memory_status("Phase 2 End")
        
        # Calculate duration
        result.duration_seconds = time.time() - start_time
        
        logger.info(
            f"Phase 2 (Whisper): Completed in {result.duration_seconds:.2f}s. "
            f"Processed: {result.files_processed}, Failed: {result.files_failed}"
        )
        
        return result
    
    def update_classified_batch_after_phase_2(
        self,
        phase_result: PhaseResult
    ) -> None:
        """Update classified batch with Phase 2 outputs.
        
        Adds markdown files from audio transcriptions to the batch
        for subsequent phases.
        
        Args:
            phase_result: Result from Phase 2
        """
        if self._classified_batch is None:
            return
        
        # Add markdown files from audio transcriptions
        self._classified_batch.markdown_files.extend(phase_result.output_paths)
        
        logger.info(
            f"Updated batch after Phase 2: {len(self._classified_batch.markdown_files)} total markdown files"
        )
    
    def _get_forensic_visual_analyst_prompt(self) -> str:
        """Get the Forensic Visual Analyst prompt for image captioning.
        
        Returns:
            Prompt string for vision model
            
        Requirements: 4.3
        """
        return '''# Role
You are an expert Forensic Visual Analyst and Data Scientist. You perform "Deep Pixel Reasoning." You analyze why elements are present, detecting anomalies, specific states, and logical relationships.

# Task
Convert the visual input into a single, dense, narrative paragraph (max 250 words).
Combine Perfect OCR (reading text) with High-Level Inference (understanding context).

# Critical Reasoning Guidelines
1. Forensic Detail & State Analysis:
   - Identify object states and attributes (e.g., "weathered, red container labeled '12'")
   - Look for visual adjectives: highlighted, rusted, bolded, isolated

2. Contextual Inference (The "Why"):
   - If an element is unique, infer its significance
   - Example: "A unique red box labeled '12' is isolated, suggesting high-priority exception"

3. Complete Text Integration (OCR):
   - Transcribe every legible piece of text, label, and number
   - Weave into sentence structure. State exact values found

4. Logical Flow (For Diagrams/Charts):
   - Trace exact path of data: "Arrow moves from 'User' to 'Server', implying request flow"

# Output Schema
Return only this JSON object:
{"image_analysis_text": "<Your analysis paragraph>"}'''
    
    def _collect_all_images_for_captioning(
        self,
        classified_batch: ClassifiedBatch
    ) -> List[ImageInfo]:
        """Collect all images from all sources for Phase 3 captioning.
        
        Gathers images from:
        - Extracted figures from .md files (Phase 1 output)
        - Standalone input images (JPG, PNG, WEBP)
        - OCR-based PDF pages (scanned documents)
        
        Args:
            classified_batch: The classified batch with all file info
            
        Returns:
            List of ImageInfo objects for all images to caption
            
        Requirements: 4.2
        """
        return classified_batch.all_images
    
    def _caption_single_image(
        self,
        image_info: ImageInfo,
        lm_client: LMStudioClient,
        prompt: str
    ) -> Tuple[bool, str, str]:
        """Caption a single image using the vision model.
        
        Args:
            image_info: Information about the image to caption
            lm_client: LM Studio client for API calls
            prompt: Forensic Visual Analyst prompt
            
        Returns:
            Tuple of (success, caption_text, error_message)
            
        Requirements: 4.3
        """
        import json
        
        try:
            # Call vision API
            response = lm_client.vision_completion(
                image_path=image_info.path,
                prompt=prompt,
                max_tokens=512,
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                result = json.loads(response)
                caption_text = result.get('image_analysis_text', '')
                
                if not caption_text:
                    caption_text = response.strip()
            except json.JSONDecodeError:
                # If response is not JSON, use it directly
                caption_text = response.strip()
            
            if caption_text and not caption_text.startswith("Error:"):
                return True, caption_text, ""
            else:
                return False, "", caption_text or "Empty response from vision model"
                
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate-limited" in msg.lower():
                print(f"BatchOrchestrator: ✗ {image_info.path} — rate-limited, skipping after retries")
            return False, "", msg
    
    def _update_markdown_with_caption(
        self,
        markdown_path: str,
        image_path: str,
        caption: str
    ) -> bool:
        """Update a markdown file with an image caption.
        
        Finds the image reference in the markdown and adds the caption
        as a transcription below it.
        
        Args:
            markdown_path: Path to the markdown file
            image_path: Path to the image (to find in markdown)
            caption: Caption text to add
            
        Returns:
            True if update was successful
            
        Requirements: 4.4
        """
        if not markdown_path or not os.path.exists(markdown_path):
            return False
        
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get the relative path from markdown file to image
            md_dir = os.path.dirname(markdown_path)
            try:
                rel_image_path = os.path.relpath(image_path, md_dir)
            except ValueError:
                # On Windows, relpath can fail across drives
                rel_image_path = os.path.basename(image_path)
            
            # Also try with just the filename
            image_filename = os.path.basename(image_path)
            
            # Find image reference patterns
            # Pattern 1: ![caption](path)
            # Pattern 2: ![](path)
            import re
            
            # Try to find the image reference
            patterns = [
                # Match with relative path
                re.escape(rel_image_path),
                # Match with just filename
                re.escape(image_filename),
                # Match with figures/ prefix
                f"figures/{re.escape(image_filename)}",
            ]
            
            updated = False
            for pattern in patterns:
                # Look for markdown image syntax
                img_pattern = rf'(!\[[^\]]*\]\([^)]*{pattern}[^)]*\))'
                matches = list(re.finditer(img_pattern, content))
                
                if matches:
                    # Insert caption after the image reference
                    # Work backwards to preserve positions
                    for match in reversed(matches):
                        end_pos = match.end()
                        
                        # Check if there's already a caption/transcription
                        after_match = content[end_pos:end_pos + 200]
                        if '**Image Transcription:**' in after_match:
                            # Already has transcription, skip
                            continue
                        
                        # Add caption after the image
                        caption_block = f"\n\n**Image Transcription:** {caption}\n"
                        content = content[:end_pos] + caption_block + content[end_pos:]
                        updated = True
                    
                    if updated:
                        break
            
            if updated:
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            else:
                logger.warning(f"Could not find image reference for {image_path} in {markdown_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating markdown {markdown_path}: {e}")
            return False
    
    def execute_phase_3_captioning(
        self,
        images: List[ImageInfo]
    ) -> PhaseResult:
        """Execute Phase 3: Image captioning with Qwen3-VL.
        
        Processes all images with the vision model:
        - Load vision model in LM Studio (Requirements 4.1)
        - Collect images from all sources (Requirements 4.2)
        - Process all images with forensic visual analysis (Requirements 4.3)
        - Update .md files with captions (Requirements 4.4)
        - Unload model after completion (Requirements 4.5)
        - Update progress tracker
        
        Args:
            images: List of ImageInfo objects to process
            
        Returns:
            PhaseResult with processing results
            
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
        """
        import time
        start_time = time.time()
        
        phase_name = ProgressTracker.PHASE_3_CAPTIONING
        
        # Handle empty input - skip phase
        if not images:
            logger.info("Phase 3 (Captioning): No images to process, skipping")
            self.progress_tracker.mark_phase_skipped(phase_name, "No images")
            return PhaseResult(
                phase_name=phase_name,
                status="skipped",
                files_processed=0,
                files_failed=0,
            )
        
        # Set phase status to in progress
        self.progress_tracker.set_phase_status(phase_name, PhaseStatus.IN_PROGRESS)
        
        # Get image paths for progress tracking
        image_paths = [img.path for img in images]
        self.progress_tracker.set_phase_files(phase_name, image_paths)
        
        # Get files to process with failed files first (Requirements 8.3, 10.5)
        image_paths = [img.path for img in images]
        paths_to_process = self._get_files_with_failed_first(phase_name, image_paths)
        set(paths_to_process)
        
        # Filter images to match the ordered paths (failed first, then new)
        images_to_process = []
        path_to_image = {img.path: img for img in images}
        for path in paths_to_process:
            if path in path_to_image:
                images_to_process.append(path_to_image[path])
        
        # Count failed files being retried
        failed_files = set(self.progress_tracker.get_failed_files(phase_name))
        retry_count = len([f for f in paths_to_process if f in failed_files])
        skip_count = len(images) - len(images_to_process) - len(failed_files)
        
        logger.info(
            f"Phase 3 (Captioning): Processing {len(images_to_process)} images "
            f"({skip_count} already completed, {retry_count} retrying failed)"
        )
        
        # Initialize result tracking
        result = PhaseResult(
            phase_name=phase_name,
            status="in_progress",
        )
        
        # Initialize LM Studio client
        lm_client: Optional[LMStudioClient] = None
        model_loaded = False
        
        try:
            # Log GPU memory before processing
            self.gpu_manager.reset_peak_stats()
            self.gpu_manager.log_memory_status("Phase 3 Start")
            
            # Load vision model in LM Studio (Requirements 4.1)
            logger.info(f"Phase 3: Loading vision model {self.VISION_MODEL}")
            load_result = self.lm_studio_manager.load_model(
                self.VISION_MODEL,
                verify_ready=True
            )
            
            if not load_result.success:
                error_msg = f"Failed to load vision model: {load_result.message}"
                logger.error(f"Phase 3: {error_msg}")
                result.status = "failed"
                result.errors["_phase"] = error_msg
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.FAILED)
                return result
            
            model_loaded = True
            logger.info(f"Phase 3: Vision model loaded in {load_result.load_time_seconds:.1f}s")
            
            # Initialize LM Studio client for API calls
            lm_client = LMStudioClient(
                image_endpoint=self.config.lm_studio_endpoint,
            )
            
            # Get the forensic visual analyst prompt
            prompt = self._get_forensic_visual_analyst_prompt()
            
            # Process each image (Requirements 4.3)
            for image_info in images_to_process:
                try:
                    # Caption the image
                    success, caption, error = self._caption_single_image(
                        image_info, lm_client, prompt
                    )
                    
                    if success:
                        # Update markdown file with caption (Requirements 4.4)
                        if image_info.markdown_file:
                            md_updated = self._update_markdown_with_caption(
                                image_info.markdown_file,
                                image_info.path,
                                caption
                            )
                            if not md_updated:
                                logger.warning(
                                    f"Phase 3: Could not update markdown for {image_info.path}"
                                )
                        
                        # Mark as completed
                        self.progress_tracker.mark_file_completed(phase_name, image_info.path)
                        result.completed_files.append(image_info.path)
                        
                        logger.info(f"Phase 3: Captioned {image_info.path}")
                    else:
                        # Mark as failed
                        self.progress_tracker.mark_file_failed(phase_name, image_info.path, error)
                        result.failed_files.append(image_info.path)
                        result.errors[image_info.path] = error
                        
                        logger.error(f"Phase 3: Failed to caption {image_info.path}: {error}")
                        
                except Exception as e:
                    # Handle unexpected errors
                    error_msg = str(e)
                    self.progress_tracker.mark_file_failed(phase_name, image_info.path, error_msg)
                    result.failed_files.append(image_info.path)
                    result.errors[image_info.path] = error_msg
                    
                    logger.error(f"Phase 3: Error processing {image_info.path}: {e}")
                    
                    # Continue with next image (error isolation)
                    continue
            
            # Update result counts
            result.files_processed = len(result.completed_files)
            result.files_failed = len(result.failed_files)
            
            # Determine final status
            if result.files_failed == 0:
                result.status = "completed"
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            else:
                result.status = "completed"  # Completed with failures
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            
        except Exception as e:
            # Phase-level failure
            logger.error(f"Phase 3 (Captioning): Critical error: {e}")
            result.status = "failed"
            result.errors["_phase"] = str(e)
            self.progress_tracker.set_phase_status(phase_name, PhaseStatus.FAILED)
            
        finally:
            # Unload vision model (Requirements 4.5)
            if model_loaded:
                try:
                    logger.info(f"Phase 3: Unloading vision model {self.VISION_MODEL}")
                    self.lm_studio_manager.unload_model(wait_for_memory=True)
                except Exception as e:
                    logger.warning(f"Phase 3: Error unloading model: {e}")
            
            # Log GPU memory after processing
            self.gpu_manager.log_memory_status("Phase 3 End")
        
        # Calculate duration
        result.duration_seconds = time.time() - start_time
        
        logger.info(
            f"Phase 3 (Captioning): Completed in {result.duration_seconds:.2f}s. "
            f"Processed: {result.files_processed}, Failed: {result.files_failed}"
        )
        
        return result
    
    def execute_phase_4_summarization(
        self,
        markdown_files: List[str]
    ) -> PhaseResult:
        """Execute Phase 4: Summarization and metadata extraction.
        
        Processes all markdown files with the language model:
        - Load qwen/qwen3-vl-4b in LM Studio (Requirements 5.1)
        - Chunk all .md files with Chunklet (250 word limit) (Requirements 5.2)
        - Summarize all chunks (Requirements 5.3)
        - Extract metadata from each file (Requirements 5.4)
        - Generate final JSON (preserve existing structure) (Requirements 5.5)
        - Keep model loaded after completion (Requirements 5.5)
        - Update progress tracker
        
        Args:
            markdown_files: List of markdown file paths to process
            
        Returns:
            PhaseResult with processing results
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        import time
        start_time = time.time()
        
        phase_name = ProgressTracker.PHASE_4_SUMMARIZATION
        
        # Handle empty input - skip phase
        if not markdown_files:
            logger.info("Phase 4 (Summarization): No markdown files to process, skipping")
            self.progress_tracker.mark_phase_skipped(phase_name, "No markdown files")
            return PhaseResult(
                phase_name=phase_name,
                status="skipped",
                files_processed=0,
                files_failed=0,
            )
        
        # Set phase status to in progress
        self.progress_tracker.set_phase_status(phase_name, PhaseStatus.IN_PROGRESS)
        self.progress_tracker.set_phase_files(phase_name, markdown_files)
        
        # Get files to process with failed files first (Requirements 8.3, 10.5)
        files_to_process = self._get_files_with_failed_first(phase_name, markdown_files)
        
        # Count failed files being retried
        failed_files = set(self.progress_tracker.get_failed_files(phase_name))
        retry_count = len([f for f in files_to_process if f in failed_files])
        skip_count = len(markdown_files) - len(files_to_process) - len(failed_files)
        
        logger.info(
            f"Phase 4 (Summarization): Processing {len(files_to_process)} markdown files "
            f"({skip_count} already completed, {retry_count} retrying failed)"
        )
        
        # Initialize result tracking
        result = PhaseResult(
            phase_name=phase_name,
            status="in_progress",
        )
        
        # Initialize components
        lm_client: Optional[LMStudioClient] = None
        model_loaded = False
        
        try:
            # Log GPU memory before processing
            self.gpu_manager.reset_peak_stats()
            self.gpu_manager.log_memory_status("Phase 4 Start")
            
            # Load summarization model in LM Studio (Requirements 5.1)
            logger.info(f"Phase 4: Loading summarization model {self.SUMMARIZATION_MODEL}")
            load_result = self.lm_studio_manager.load_model(
                self.SUMMARIZATION_MODEL,
                verify_ready=True
            )
            
            if not load_result.success:
                error_msg = f"Failed to load summarization model: {load_result.message}"
                logger.error(f"Phase 4: {error_msg}")
                result.status = "failed"
                result.errors["_phase"] = error_msg
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.FAILED)
                return result
            
            model_loaded = True
            logger.info(f"Phase 4: Summarization model loaded in {load_result.load_time_seconds:.1f}s")
            
            # Initialize LM Studio client for API calls
            lm_client = LMStudioClient(
                image_endpoint=self.config.lm_studio_endpoint,
            )
            
            # Initialize Chunklet for chunking (Requirements 5.2)
            from extractors.chunklet import Chunklet, ChunkletConfig
            chunklet = Chunklet(ChunkletConfig(max_words=250))
            
            # Initialize metadata extractor (Requirements 5.4)
            from extractors.llm_metadata import LLMMetadataExtractor
            metadata_extractor = LLMMetadataExtractor(lm_client=lm_client)
            
            # Initialize JSON builder (Requirements 5.5)
            json_builder = JSONBuilder()
            
            # Process each markdown file
            for md_path in files_to_process:
                try:
                    # Process single markdown file
                    json_path = self._process_single_markdown_file(
                        md_path=md_path,
                        chunklet=chunklet,
                        lm_client=lm_client,
                        metadata_extractor=metadata_extractor,
                        json_builder=json_builder,
                    )
                    
                    if json_path:
                        # Mark as completed
                        self.progress_tracker.mark_file_completed(phase_name, md_path)
                        result.completed_files.append(md_path)
                        result.output_paths.append(json_path)
                        
                        logger.info(f"Phase 4: Processed {md_path} -> {json_path}")
                    else:
                        # Mark as failed
                        error_msg = "Failed to generate JSON output"
                        self.progress_tracker.mark_file_failed(phase_name, md_path, error_msg)
                        result.failed_files.append(md_path)
                        result.errors[md_path] = error_msg
                        
                        logger.error(f"Phase 4: Failed to process {md_path}")
                        
                except Exception as e:
                    # Handle unexpected errors
                    error_msg = str(e)
                    self.progress_tracker.mark_file_failed(phase_name, md_path, error_msg)
                    result.failed_files.append(md_path)
                    result.errors[md_path] = error_msg
                    
                    logger.error(f"Phase 4: Error processing {md_path}: {e}")
                    
                    # Continue with next file (error isolation)
                    continue
            
            # Update result counts
            result.files_processed = len(result.completed_files)
            result.files_failed = len(result.failed_files)
            
            # Determine final status
            if result.files_failed == 0:
                result.status = "completed"
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            else:
                result.status = "completed"  # Completed with failures
                self.progress_tracker.set_phase_status(phase_name, PhaseStatus.COMPLETED)
            
        except Exception as e:
            # Phase-level failure
            logger.error(f"Phase 4 (Summarization): Critical error: {e}")
            result.status = "failed"
            result.errors["_phase"] = str(e)
            self.progress_tracker.set_phase_status(phase_name, PhaseStatus.FAILED)
            
        finally:
            # NOTE: Keep model loaded after completion (Requirements 5.5)
            # Do NOT unload the summarization model - it stays loaded for queries
            if model_loaded:
                logger.info(f"Phase 4: Keeping {self.SUMMARIZATION_MODEL} loaded for queries")
            
            # Log GPU memory after processing
            self.gpu_manager.log_memory_status("Phase 4 End")
        
        # Calculate duration
        result.duration_seconds = time.time() - start_time
        
        logger.info(
            f"Phase 4 (Summarization): Completed in {result.duration_seconds:.2f}s. "
            f"Processed: {result.files_processed}, Failed: {result.files_failed}"
        )
        
        return result
    
    def _process_single_markdown_file(
        self,
        md_path: str,
        chunklet,
        lm_client: LMStudioClient,
        metadata_extractor,
        json_builder,
    ) -> Optional[str]:
        """Process a single markdown file through chunking, summarization, and JSON generation.
        
        Args:
            md_path: Path to markdown file
            chunklet: Chunklet instance for chunking
            lm_client: LM Studio client for API calls
            metadata_extractor: LLMMetadataExtractor instance
            json_builder: JSONBuilder instance
            
        Returns:
            Path to generated JSON file, or None on failure
            
        Requirements: 5.2, 5.3, 5.4, 5.5
        """
        
        # Read markdown content
        if not os.path.exists(md_path):
            logger.error(f"Markdown file not found: {md_path}")
            return None
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except Exception as e:
            logger.error(f"Error reading markdown file {md_path}: {e}")
            return None
        
        if not markdown_content.strip():
            logger.warning(f"Empty markdown file: {md_path}")
            return None
        
        # Step 1: Chunk the markdown (Requirements 5.2)
        chunks = chunklet.split_text(markdown_content)
        
        if not chunks:
            logger.warning(f"No chunks generated from {md_path}")
            # Create a single chunk with the entire content
            chunks = [{
                'page_number': 1,
                'chunk_index': 1,
                'paragraph_text': markdown_content[:1000],  # Limit to first 1000 chars
            }]
        
        logger.debug(f"Generated {len(chunks)} chunks from {md_path}")
        
        # Step 2: Summarize each chunk (Requirements 5.3)
        summarized_chunks = []
        for chunk in chunks:
            try:
                summary = self._summarize_chunk(
                    chunk['paragraph_text'],
                    lm_client
                )
                
                summarized_chunk = {
                    'page_number': chunk.get('page_number', 1),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'paragraph_text': chunk['paragraph_text'],
                    'summary': summary,
                }
                summarized_chunks.append(summarized_chunk)
                
            except Exception as e:
                logger.warning(f"Error summarizing chunk {chunk.get('chunk_index', 0)}: {e}")
                # Add chunk with fallback summary
                summarized_chunks.append({
                    'page_number': chunk.get('page_number', 1),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'paragraph_text': chunk['paragraph_text'],
                    'summary': "[Summary unavailable]",
                })
        
        # Step 3: Extract metadata (Requirements 5.4)
        # Determine source file info from markdown path
        md_dir = os.path.dirname(md_path)
        md_filename = os.path.basename(md_path)
        
        # Extract original filename from markdown filename (initial_{filename}.md)
        original_filename = md_filename
        if md_filename.startswith('initial_'):
            original_filename = md_filename[8:]  # Remove 'initial_' prefix
        if original_filename.endswith('.md'):
            original_filename = original_filename[:-3]  # Remove .md extension
        
        # Determine source type from directory name (e.g., a1_pdf -> pdf)
        source_type = "document"
        dir_name = os.path.basename(md_dir)
        if '_' in dir_name:
            source_type = dir_name.split('_')[-1]
        
        # Extract metadata
        metadata = metadata_extractor.extract_metadata(
            markdown=markdown_content,
            filename=original_filename,
            file_path=md_path,
            source_type=source_type,
        )
        
        # Step 4: Build final JSON (Requirements 5.5)
        # Determine original file path
        original_file_path = md_path  # Fallback
        
        output = json_builder.build_output(
            metadata=metadata,
            chunks=summarized_chunks,
            file_path=original_file_path,
            source_type=source_type,
            file_name=original_filename,
            processing_device="cuda" if self.gpu_manager.is_cuda_available else "cpu",
        )
        
        # Save JSON file
        json_filename = f"{original_filename}.json"
        json_path = os.path.join(md_dir, json_filename)
        
        try:
            json_builder.save_to_file(output, json_path)
            logger.info(f"Saved JSON output to {json_path}")
            return json_path
        except Exception as e:
            logger.error(f"Error saving JSON file {json_path}: {e}")
            return None
    
    def _summarize_chunk(
        self,
        paragraph_text: str,
        lm_client: LMStudioClient,
        min_words: int = 15,
        max_words: int = 30,
    ) -> str:
        """Summarize a single chunk using the LM Studio API.
        
        Args:
            paragraph_text: Text to summarize
            lm_client: LM Studio client
            min_words: Minimum words in summary
            max_words: Maximum words in summary
            
        Returns:
            Summary text
            
        Requirements: 5.3
        """
        if not paragraph_text or not paragraph_text.strip():
            return "[Empty content]"
        
        # Create summarization prompt
        prompt = (
            f"Please provide a concise summary of the following paragraph in {min_words}-{max_words} words. "
            f"Preserve key entities (proper nouns, numbers, scientific terms) and use action-oriented language "
            f"describing what the content states or finds.\n\n"
            f"Paragraph:\n{paragraph_text}\n\n"
            f"Summary:"
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise summaries of text."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = lm_client.chat_completion(
                messages=messages,
                max_tokens=100,
                temperature=0.1,
                json_mode=False,
            )
            
            if response:
                summary = response.strip()
                # Validate and truncate if needed
                words = summary.split()
                if len(words) > max_words:
                    summary = " ".join(words[:max_words])
                return summary
            else:
                return "[Summary unavailable]"
                
        except Exception as e:
            logger.warning(f"Error summarizing chunk: {e}")
            return "[Summary unavailable]"

    def _get_files_with_failed_first(
        self,
        phase: str,
        all_files: List[str]
    ) -> List[str]:
        """Get files to process with failed files prioritized first.
        
        Implements retry priority: failed files are retried before new files.
        
        Args:
            phase: Phase name
            all_files: Complete list of files for this phase
            
        Returns:
            List of files to process, with failed files first
            
        Requirements: 10.5
        """
        # Get completed files to skip
        completed = set(self.progress_tracker.get_completed_files(phase))
        
        # Get failed files for retry priority
        failed = self.progress_tracker.get_failed_files(phase)
        
        # Build ordered list: failed files first, then new files
        files_to_process = []
        
        # Add failed files first (retry priority)
        for f in failed:
            if f not in completed:
                files_to_process.append(f)
        
        # Add remaining files (not completed, not already in list)
        failed_set = set(failed)
        for f in all_files:
            if f not in completed and f not in failed_set:
                files_to_process.append(f)
        
        return files_to_process
    
    def check_for_resume(self) -> bool:
        """Check if there's an existing progress file to resume from.
        
        Returns:
            True if existing progress was loaded, False otherwise
            
        Requirements: 8.3
        """
        if os.path.exists(self.config.progress_file):
            loaded = self.progress_tracker.load_existing()
            if loaded:
                logger.info(
                    f"Found existing progress file. "
                    f"Batch ID: {self.progress_tracker.progress.batch_id}, "
                    f"Current phase: {self.progress_tracker.get_current_phase()}"
                )
                return True
        return False
    
    def process_batch(self, input_dir: str, resume: bool = True) -> BatchResult:
        """Process all files in batch mode.
        
        Main entry point for batch processing. Orchestrates all phases:
        1. Classify all files into processing groups
        2. Execute Phase 1: Docling document processing
        3. Execute Phase 2: Whisper audio transcription
        4. Execute Phase 3: Image captioning with Qwen3-VL
        5. Execute Phase 4: Summarization with Qwen3-VL
        
        Implements error isolation - failures in one file do not stop
        processing of other files.
        
        Supports resume capability - if a progress file exists and resume=True,
        completed files are skipped and failed files are retried first.
        
        Args:
            input_dir: Path to input directory containing files to process
            resume: If True, attempt to resume from existing progress file
            
        Returns:
            BatchResult with processing results and statistics
            
        Requirements: 1.1, 6.5, 8.3, 10.1, 10.5
        """
        import time
        import uuid
        
        start_time = time.time()
        
        # Check for resume capability (Requirements 8.3)
        is_resuming = False
        batch_id = None
        
        if resume and self.check_for_resume():
            is_resuming = True
            batch_id = self.progress_tracker.progress.batch_id
            logger.info(f"Resuming batch processing (ID: {batch_id}) for directory: {input_dir}")
        else:
            batch_id = str(uuid.uuid4())[:8]
            logger.info(f"Starting new batch processing (ID: {batch_id}) for directory: {input_dir}")
        
        # Initialize result
        result = BatchResult(
            batch_id=batch_id,
            output_directory=self.config.output_base_dir,
        )
        
        try:
            # Step 1: Classify all files (Requirements 1.1)
            logger.info("Step 1: Classifying files...")
            classified = self.classify_all_files(input_dir)
            result.total_files = classified.total_files
            
            if classified.total_files == 0:
                logger.warning(f"No files found in {input_dir}")
                result.total_duration_seconds = time.time() - start_time
                return result
            
            # Initialize or update progress tracking
            if not is_resuming:
                # New batch - initialize progress tracking
                self.progress_tracker.initialize_batch(
                    batch_id=batch_id,
                    classified_batch=classified,
                )
            else:
                # Resuming - update file lists for any new files
                logger.info("Resuming from existing progress - skipping completed files, retrying failed files first")
            
            # Step 2: Execute Phase 1 - Docling (Requirements 2.1-2.5)
            logger.info("Step 2: Executing Phase 1 (Docling)...")
            phase1_result = self.execute_phase_1_docling(classified.documents)
            result.phases[ProgressTracker.PHASE_1_DOCLING] = phase1_result
            
            # Update classified batch with Phase 1 outputs
            self.update_classified_batch_after_phase_1(phase1_result)
            
            # Clear GPU memory between phases (Requirements 6.5)
            if self.config.clear_cache_between_phases:
                self.gpu_manager.clear_cache()
                logger.info("GPU cache cleared after Phase 1")
            
            # Step 3: Execute Phase 2 - Whisper (Requirements 3.1-3.5)
            logger.info("Step 3: Executing Phase 2 (Whisper)...")
            phase2_result = self.execute_phase_2_whisper(classified.audio_files)
            result.phases[ProgressTracker.PHASE_2_WHISPER] = phase2_result
            
            # Update classified batch with Phase 2 outputs
            self.update_classified_batch_after_phase_2(phase2_result)
            
            # Clear GPU memory between phases (Requirements 6.5)
            if self.config.clear_cache_between_phases:
                self.gpu_manager.clear_cache()
                logger.info("GPU cache cleared after Phase 2")
            
            # Step 4: Execute Phase 3 - Image Captioning (Requirements 4.1-4.5)
            logger.info("Step 4: Executing Phase 3 (Image Captioning)...")
            all_images = self._collect_all_images_for_captioning(self._classified_batch)
            phase3_result = self.execute_phase_3_captioning(all_images)
            result.phases[ProgressTracker.PHASE_3_CAPTIONING] = phase3_result
            
            # Step 5: Execute Phase 4 - Summarization (Requirements 5.1-5.5)
            logger.info("Step 5: Executing Phase 4 (Summarization)...")
            phase4_result = self.execute_phase_4_summarization(
                self._classified_batch.markdown_files
            )
            result.phases[ProgressTracker.PHASE_4_SUMMARIZATION] = phase4_result
            
            # Calculate totals
            result.successful_files = sum(
                len(p.completed_files) for p in result.phases.values()
                if p.status != "skipped"
            )
            result.failed_files = sum(
                len(p.failed_files) for p in result.phases.values()
            )
            
            # Get GPU peak memory
            memory_info = self.gpu_manager.get_memory_info()
            if memory_info:
                # Handle both GPUMemoryInfo object and dict (for mocking)
                if hasattr(memory_info, 'max_allocated'):
                    result.gpu_peak_memory_gb = memory_info.max_allocated
                elif isinstance(memory_info, dict):
                    result.gpu_peak_memory_gb = memory_info.get('max_allocated', 0.0)
                else:
                    result.gpu_peak_memory_gb = 0.0
            else:
                result.gpu_peak_memory_gb = 0.0
            
            logger.info(
                f"Batch processing completed. "
                f"Successful: {result.successful_files}, Failed: {result.failed_files}"
            )
            
        except Exception as e:
            logger.error(f"Critical error during batch processing: {e}")
            # Ensure cleanup happens
            self._cleanup_handler()
            raise
        
        finally:
            # Calculate total duration
            result.total_duration_seconds = time.time() - start_time
            
            # Save final progress
            self.progress_tracker.save()
            
            # Generate failure report if there were failures
            if result.failed_files > 0:
                failure_report = self.generate_failure_report(result)
                logger.warning(f"Failure report:\n{failure_report}")
        
        return result

    def generate_failure_report(self, result: BatchResult) -> str:
        """Generate a failure report for batch processing.
        
        Creates a detailed report of all failed files with their
        error messages.
        
        Args:
            result: BatchResult from batch processing
            
        Returns:
            Formatted failure report string
            
        Requirements: 10.4
        """
        lines = [
            "=" * 60,
            "BATCH PROCESSING FAILURE REPORT",
            f"Batch ID: {result.batch_id}",
            f"Total Files: {result.total_files}",
            f"Successful: {result.successful_files}",
            f"Failed: {result.failed_files}",
            "=" * 60,
        ]
        
        for phase_name, phase_result in result.phases.items():
            if phase_result.failed_files:
                lines.append(f"\n{phase_name}:")
                lines.append("-" * 40)
                
                for failed_file in phase_result.failed_files:
                    error_msg = phase_result.errors.get(failed_file, "Unknown error")
                    lines.append(f"  File: {failed_file}")
                    lines.append(f"  Error: {error_msg}")
                    lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

    def get_failure_report_dict(self, result: BatchResult) -> Dict[str, Any]:
        """Get failure report as a dictionary.
        
        Returns structured failure information for programmatic access.
        
        Args:
            result: BatchResult from batch processing
            
        Returns:
            Dictionary with failure information
            
        Requirements: 10.4
        """
        failures = []
        
        for phase_name, phase_result in result.phases.items():
            for failed_file in phase_result.failed_files:
                error_msg = phase_result.errors.get(failed_file, "Unknown error")
                failures.append({
                    "phase": phase_name,
                    "file": failed_file,
                    "error": error_msg,
                })
        
        return {
            "batch_id": result.batch_id,
            "total_files": result.total_files,
            "successful_files": result.successful_files,
            "failed_files": result.failed_files,
            "failures": failures,
        }

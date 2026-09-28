"""Pipeline wrapper for programmatic access to the RAG document processing pipeline.

This module wraps the existing Pipeline class from document_processor/main.py to provide:
- Programmatic API access for document processing
- Progress callback support for stage updates
- ProcessingResult with json_path, md paths, and status
- Auto-injection to database after successful processing
- Comprehensive logging for all processing stages

Requirements:
- 4.1: Load RAG pipeline configuration and processors once on API initialization
- 4.2: Process files using initialized pipeline components
- 4.3: Return path to generated JSON file on completion
- 4.4: Pass JSON to injection pipeline automatically on completion
- 7.1: Save initial markdown to Database/Processed_Docs folder
- 7.2: Save processed markdown to Database/Processed_Docs folder
- 7.3: Save final JSON to Database/Processed_Docs folder
- 7.4: Only pass final JSON file path to injection pipeline
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path

from extractors.config import (
    load_config,
    DOCUMENT_EXTENSIONS,
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
)
from extractors.file_classifier import FileClassifier, FileType

# Import logging utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# Processing stages for progress tracking
PROCESSING_STAGES = [
    "file_classification",
    "conversion",
    "image_transcription",
    "chunking",
    "metadata_extraction",
    "json_generation",
    "database_injection",
]


@dataclass
class ProcessingResult:
    """Result of processing a single file.
    
    Attributes:
        filename: Name of the processed file (without extension)
        success: Whether processing completed successfully
        json_path: Path to the generated JSON file (None if failed)
        initial_md_path: Path to the initial markdown file (None if failed)
        processed_md_path: Path to the processed markdown file (None if failed)
        processing_time: Time taken to process in seconds
        error: Error message if processing failed
        stages_completed: List of completed processing stages
        injection_success: Whether database injection succeeded (None if not attempted)
        injection_error: Error message if injection failed
    """
    filename: str
    success: bool
    json_path: Optional[str] = None
    initial_md_path: Optional[str] = None
    processed_md_path: Optional[str] = None
    processing_time: float = 0.0
    error: Optional[str] = None
    stages_completed: List[str] = field(default_factory=list)
    injection_success: Optional[bool] = None
    injection_error: Optional[str] = None


class PipelineWrapper:
    """Wrapper for RAG pipeline providing programmatic access.
    
    This class wraps the existing Pipeline class to provide:
    - Single initialization of all pipeline components
    - Progress callback support for UI updates
    - Structured ProcessingResult returns
    - Auto-injection to database after successful processing
    
    Requirements:
    - 4.1: Load RAG pipeline configuration and processors once
    - 4.2: Process files using initialized pipeline components
    - 4.3: Return path to generated JSON file
    - 4.4: Pass JSON to injection pipeline automatically
    - 7.1, 7.2, 7.3: Save all output artifacts
    - 7.4: Only pass final JSON file path to injection
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with optional config overrides.
        
        Args:
            config_override: Optional dictionary of config values to override
        """
        logger.info("Initializing PipelineWrapper...")
        
        self._config = load_config()
        self._model_config, self._prompts_config, self._processing_config, self._directory_config = self._config
        
        # Apply config overrides if provided
        if config_override:
            logger.info(f"Applying config overrides: {list(config_override.keys())}")
            self._apply_config_overrides(config_override)
        
        # Initialize file classifier (lightweight, always available)
        self._file_classifier = FileClassifier()
        logger.info("File classifier initialized")
        
        # Lazy initialization of heavy components
        self._pipeline = None
        self._initialized = False
        
        logger.info("PipelineWrapper initialization complete")
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides.
        
        Args:
            overrides: Dictionary of config key-value pairs to override
        """
        for key, value in overrides.items():
            if hasattr(self._model_config, key):
                setattr(self._model_config, key, value)
            elif hasattr(self._processing_config, key):
                setattr(self._processing_config, key, value)
    
    def _ensure_initialized(self) -> None:
        """Ensure pipeline components are initialized (lazy loading).
        
        This implements Requirement 4.1: Load RAG pipeline configuration
        and processors once on API initialization.
        """
        if self._initialized:
            return
        
        logger.info("Initializing document processing pipeline components...")
        
        # Import Pipeline here to avoid circular imports and allow lazy loading
        from extractors.main import Pipeline
        
        start_time = time.time()
        self._pipeline = Pipeline(self._config)
        init_time = time.time() - start_time
        
        self._initialized = True
        logger.info(f"Document processing pipeline initialized successfully in {init_time:.2f}s")
    
    def process_file(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        auto_inject: bool = True
    ) -> ProcessingResult:
        """Process a single file through the pipeline.
        
        This implements Requirements:
        - 4.2: Process files using initialized pipeline components
        - 4.3: Return path to generated JSON file
        - 4.4: Pass JSON to injection pipeline automatically
        - 7.1, 7.2, 7.3: Save all output artifacts
        - 7.4: Only pass final JSON file path to injection
        
        Args:
            file_path: Path to the file to process
            progress_callback: Optional callback for progress updates.
                              Called with (stage_name, progress_fraction)
                              where progress_fraction is 0.0 to 1.0
            auto_inject: Whether to automatically inject to database after
                        successful processing (default: True)
        
        Returns:
            ProcessingResult with json_path and status
        """
        start_time = time.time()
        stages_completed = []
        filename = Path(file_path).name
        
        logger.info(f"Starting document processing for: {filename}")
        logger.info(f"File path: {file_path}")
        logger.info(f"Auto-inject enabled: {auto_inject}")
        
        def report_progress(stage: str, progress: float = 1.0):
            """Report progress for a stage."""
            if progress_callback:
                progress_callback(stage, progress)
            if progress >= 1.0:
                stages_completed.append(stage)
                logger.info(f"✓ Stage completed: {stage}")
        
        try:
            # Stage 1: File Classification
            logger.info("Stage 1/7: File Classification")
            report_progress("file_classification", 0.0)
            
            # Validate file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                return ProcessingResult(
                    filename=Path(file_path).stem,
                    success=False,
                    processing_time=time.time() - start_time,
                    error=error_msg,
                    stages_completed=stages_completed
                )
            
            # Get file size for logging
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            # Classify the file
            classified_file = self._file_classifier.classify(file_path)
            logger.info(f"File classified as: {classified_file.file_type.value}")
            
            # Validate file type is supported
            if classified_file.file_type == FileType.UNKNOWN:
                ext = Path(file_path).suffix
                error_msg = f"Unsupported file type: {ext}"
                logger.error(error_msg)
                return ProcessingResult(
                    filename=classified_file.filename,
                    success=False,
                    processing_time=time.time() - start_time,
                    error=error_msg,
                    stages_completed=stages_completed
                )
            
            report_progress("file_classification", 1.0)
            
            # Ensure pipeline is initialized
            logger.info("Ensuring pipeline components are initialized...")
            self._ensure_initialized()
            
            # Stage 2-6: Process through pipeline
            logger.info("Stage 2-6: Document Processing Pipeline")
            logger.info("Processing stages: Conversion → Image Transcription → Chunking → Metadata → JSON Generation")
            
            report_progress("conversion", 0.0)
            
            pipeline_start = time.time()
            result = self._pipeline.process_file(classified_file)
            pipeline_time = time.time() - pipeline_start
            
            logger.info(f"Pipeline processing completed in {pipeline_time:.2f}s")
            
            # Map pipeline stages to our progress stages
            if result.success:
                logger.info("✓ All pipeline stages completed successfully")
                report_progress("conversion", 1.0)
                report_progress("image_transcription", 1.0)
                report_progress("chunking", 1.0)
                report_progress("metadata_extraction", 1.0)
                report_progress("json_generation", 1.0)
                
                # Log output file paths
                if result.json_path:
                    logger.info(f"JSON output: {result.json_path}")
                if result.initial_md_path:
                    logger.info(f"Initial markdown: {result.initial_md_path}")
                if result.processed_md_path:
                    logger.info(f"Processed markdown: {result.processed_md_path}")
            else:
                logger.error(f"Pipeline processing failed: {result.error}")
            
            processing_time = time.time() - start_time
            
            # Validate JSON path exists (Requirement 4.3)
            json_path = result.json_path if result.success else None
            if json_path and not os.path.exists(json_path):
                error_msg = f"JSON file was not created: {json_path}"
                logger.error(error_msg)
                return ProcessingResult(
                    filename=result.filename,
                    success=False,
                    processing_time=processing_time,
                    error=error_msg,
                    stages_completed=stages_completed
                )
            
            # Auto-inject to database if enabled and processing succeeded
            # Implements Requirements 4.4 and 7.4
            injection_success = None
            injection_error = None
            
            if result.success and auto_inject and json_path:
                logger.info("Stage 7/7: Database Injection")
                report_progress("database_injection", 0.0)
                
                injection_start = time.time()
                injection_success, injection_error = self._inject_to_database(
                    json_path, 
                    result.filename
                )
                injection_time = time.time() - injection_start
                
                if injection_success:
                    logger.info(f"✓ Database injection completed in {injection_time:.2f}s")
                    report_progress("database_injection", 1.0)
                else:
                    logger.warning(f"Database injection failed: {injection_error}")
            
            total_time = time.time() - start_time
            logger.info(f"Document processing completed for {filename} in {total_time:.2f}s")
            logger.info(f"Success: {result.success}, Stages completed: {len(stages_completed)}/7")
            
            return ProcessingResult(
                filename=result.filename,
                success=result.success,
                json_path=json_path,
                initial_md_path=result.initial_md_path if result.success else None,
                processed_md_path=result.processed_md_path if result.success else None,
                processing_time=time.time() - start_time,
                error=result.error,
                stages_completed=stages_completed,
                injection_success=injection_success,
                injection_error=injection_error
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Pipeline error: {str(e)}"
            stack_trace = traceback.format_exc()
            
            logger.error(f"Document processing failed for {filename}: {error_msg}")
            logger.error(f"Stack trace:\n{stack_trace}")
            
            return ProcessingResult(
                filename=Path(file_path).stem,
                success=False,
                processing_time=time.time() - start_time,
                error=error_msg,
                stages_completed=stages_completed
            )
    
    def _inject_to_database(
        self,
        json_path: str,
        filename: str
    ) -> tuple:
        """
        Inject processed JSON to database.
        
        Implements Requirements:
        - 4.4: Pass JSON to injection pipeline automatically on completion
        - 7.4: Only pass final JSON file path to injection pipeline
        
        Args:
            json_path: Path to the JSON file to inject
            filename: Name of the file being processed (for logging)
            
        Returns:
            Tuple of (success: bool, error: Optional[str])
        """
        # Import here to avoid circular imports at module level
        import sys
        # Ensure parent directory is in path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from ingestion.ingest_json import ingest_document_from_json
        
        try:
            logger.info(f"Starting database injection for: {filename}")
            logger.info(f"JSON file: {json_path}")
            
            # Validate JSON file exists and get size
            if not os.path.exists(json_path):
                error_msg = f"JSON file not found: {json_path}"
                logger.error(error_msg)
                return (False, error_msg)
            
            json_size = os.path.getsize(json_path)
            logger.info(f"JSON file size: {json_size:,} bytes ({json_size / 1024:.2f} KB)")
            
            # Call injection pipeline with only the final JSON path (Requirement 7.4)
            logger.info("Calling ingestion pipeline...")
            injection_start = time.time()
            
            ingest_document_from_json(json_path)
            
            injection_time = time.time() - injection_start
            logger.info(f"✓ Database injection completed successfully in {injection_time:.2f}s")
            logger.info(f"Document {filename} is now available for RAG queries")
            
            return (True, None)
            
        except Exception as e:
            import traceback
            error_msg = f"Injection failed: {str(e)}"
            traceback.format_exc()
            
            logger.warning(f"Database injection failed for {filename}: {error_msg}")
            # Stack trace removed to reduce spam - database injection is optional
            
            return (False, error_msg)
    
    def process_files(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
        auto_inject: bool = True
    ) -> List[ProcessingResult]:
        """Process multiple files through the pipeline.
        
        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for progress updates.
                              Called with (filename, stage_name, progress_fraction)
            auto_inject: Whether to automatically inject to database after
                        successful processing (default: True)
        
        Returns:
            List of ProcessingResult objects, one per file
        """
        results = []
        
        for file_path in file_paths:
            filename = Path(file_path).name
            
            # Create a per-file callback that includes the filename
            def file_progress_callback(stage: str, progress: float):
                if progress_callback:
                    progress_callback(filename, stage, progress)
            
            result = self.process_file(file_path, file_progress_callback, auto_inject)
            results.append(result)
        
        return results
    
    @staticmethod
    def get_supported_extensions() -> set:
        """Get the set of supported file extensions.
        
        Returns:
            Set of supported file extensions (e.g., {'.pdf', '.docx', ...})
        """
        return DOCUMENT_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS
    
    @staticmethod
    def is_supported_extension(extension: str) -> bool:
        """Check if a file extension is supported.
        
        Args:
            extension: File extension to check (with or without leading dot)
        
        Returns:
            True if the extension is supported
        """
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        
        all_extensions = DOCUMENT_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS
        return ext in all_extensions
    
    def get_output_paths(self, filename: str, extension: str) -> Dict[str, str]:
        """Get expected output paths for a file.
        
        Args:
            filename: Base filename (without extension)
            extension: File extension (with leading dot)
        
        Returns:
            Dictionary with keys: 'output_dir', 'initial_md', 'processed_md', 'json'
        """
        output_dir = self._directory_config.get_file_output_dir(filename, extension)
        
        return {
            'output_dir': output_dir,
            'initial_md': os.path.join(output_dir, f"initial_{filename}.md"),
            'processed_md': os.path.join(output_dir, f"processed_{filename}.md"),
            'json': os.path.join(output_dir, f"{filename}.json"),
        }

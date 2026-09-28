"""Main pipeline orchestrator for multi-modal data ingestion and processing.

Unified Pipeline Flow:
1. ALL file types → Convert to initial markdown (.md)
2. Check if markdown contains image paths → If yes, transcribe images with Qwen
3. Chunk the markdown (250 word limit per chunk) with Qwen
4. Extract metadata with Qwen
5. Generate final structured JSON

Processing Modes:
- Individual Mode: Process files one at a time through all stages (default for single file)
- Batch Mode: Process files in phases by operation type (default for multiple files)

Mode Selection Logic (Requirements 9.1, 9.2):
- Single file → Individual mode (regardless of batch_mode flag)
- Multiple files + batch_mode=False → Individual mode
- Multiple files + batch_mode=True → Batch mode

Models:
- Qwen3 VL 30B A3B Instruct via LM Studio (configurable endpoint)
- Whisper on Intel NPU via OpenVINO for audio transcription
- Docling for document-to-markdown conversion
"""

import os
import time
import shutil
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from extractors.config import (
    load_config, print_config_summary,
    get_source_type, BatchConfig,
)
from extractors.file_classifier import FileClassifier, FileType, ClassifiedFile
from extractors.docling_gpu_processor import DoclingGPUProcessor as DoclingProcessor
from external_clients.openai_compatible_client import OpenAICompatibleClient as LMStudioClient
from extractors.unified_processor import UnifiedProcessor
from extractors.whisper_gpu_processor import WhisperGPUProcessor as WhisperNPU
from extractors.batch_orchestrator import BatchOrchestrator


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    filename: str
    success: bool
    initial_md_path: str
    processed_md_path: str
    json_path: str
    processing_time: float
    error: Optional[str] = None


@dataclass
class ProcessingSummary:
    """Summary of processing multiple files."""
    total_files: int
    successful: int
    failed: int
    total_time_seconds: float
    failed_files: List[str]


class Pipeline:
    """
    Main pipeline orchestrator for multi-modal data ingestion.
    
    Unified flow: File → Markdown → Check Images → Transcribe → Chunk → Metadata → JSON
    """
    
    def __init__(self, config: tuple):
        """
        Initialize all processors once at startup.
        
        Args:
            config: Tuple of (ModelConfig, PromptsConfig, ProcessingConfig, DirectoryConfig)
        """
        self.model_config, self.prompts_config, self.processing_config, self.directory_config = config
        
        print("\n" + "=" * 60)
        print("INITIALIZING UNIFIED MULTI-MODAL PIPELINE")
        print("=" * 60)
        
        # Initialize FileClassifier
        print("\n[1/5] Initializing File Classifier...")
        self.file_classifier = FileClassifier()
        print("FileClassifier: Initialized")
        
        # Initialize DoclingProcessor (for documents)
        print("\n[2/5] Initializing Docling Processor...")
        self.docling_processor = DoclingProcessor(device=self.model_config.whisper_device)
        self.docling_processor.initialize()
        
        # Initialize LM Studio Client (for vision/image analysis)
        print("\n[3/5] Initializing LM Studio Client...")
        self.lm_client = LMStudioClient(
            image_endpoint=self.model_config.lm_studio_image_url,
            api_key=self.model_config.lm_studio_api_key,
            image_model=self.model_config.lm_studio_image_model,
            max_tokens=self.model_config.lm_studio_max_tokens,
            temperature=self.model_config.lm_studio_temperature,
        )
        # Test connection
        self.lm_client.test_connection()

        # Initialize Metadata LLM Client (separate model for metadata extraction)
        print("\n[3b/5] Initializing Metadata LLM Client...")
        self.metadata_client = LMStudioClient(
            image_endpoint=self.model_config.metadata_base_url,
            api_key=self.model_config.metadata_api_key,
            image_model=self.model_config.metadata_model,
            max_tokens=self.model_config.metadata_max_tokens,
            temperature=self.model_config.metadata_temperature,
        )
        self.metadata_client.test_connection()
        
        # Initialize Whisper GPU (for audio)
        print("\n[4/5] Initializing Whisper GPU Processor...")
        self.whisper = WhisperNPU(
            model_name=self.model_config.whisper_model_name,
            device=self.model_config.whisper_device,
            enable_diarization=True
        )
        # Model loaded lazily on first transcription call
        
        # Initialize Unified Processor
        print("\n[5/5] Initializing Unified Processor...")
        self.unified_processor = UnifiedProcessor(
            lm_client=self.lm_client,
            metadata_client=self.metadata_client,
            prompts_config=self.prompts_config,
            directory_config=self.directory_config,
            model_config=self.model_config,
            docling_processor=self.docling_processor,
            whisper_processor=self.whisper,
        )
        
        print("\n" + "=" * 60)
        print("PIPELINE INITIALIZATION COMPLETE")
        print("=" * 60 + "\n")
        
        self.file_count = 0
    
    def process_all_files(self) -> ProcessingSummary:
        """Process all supported files in input directory."""
        input_dir = self.directory_config.abs_input
        classified_files = self.file_classifier.scan_directory(input_dir)
        
        if not classified_files:
            print(f"No supported files found in {input_dir}")
            return ProcessingSummary(
                total_files=0, successful=0, failed=0,
                total_time_seconds=0.0, failed_files=[]
            )
        
        # Group files by type
        grouped = self.file_classifier.group_by_type(classified_files)
        print(f"\nFound {len(classified_files)} file(s) to process:")
        print(f"  - Documents: {len(grouped[FileType.DOCUMENT])}")
        print(f"  - Audio: {len(grouped[FileType.AUDIO])}")
        print(f"  - Images: {len(grouped[FileType.IMAGE])}\n")
        
        start_time = time.time()
        results = []
        failed_files = []
        
        for i, classified_file in enumerate(classified_files, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing file {i}/{len(classified_files)}: {classified_file.filename}{classified_file.extension}")
            print(f"Type: {classified_file.file_type.value.upper()}")
            print(f"{'=' * 60}\n")
            
            result = self.process_file(classified_file)
            results.append(result)
            
            if not result.success:
                failed_files.append(result.filename)
            
            self._clear_gpu_cache_if_needed(i)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        
        summary = ProcessingSummary(
            total_files=len(classified_files),
            successful=successful,
            failed=len(results) - successful,
            total_time_seconds=total_time,
            failed_files=failed_files
        )
        
        self._print_processing_summary(summary)
        return summary
    
    def process_file(self, classified_file: ClassifiedFile) -> ProcessingResult:
        """
        Process a single file through the unified pipeline.
        
        Unified Flow:
        1. Classify file type
        2. Archive raw file to Database/Documents/{TYPE}/
        3. Convert to markdown (Docling/Whisper/Qwen based on type)
        4. Check for images in markdown
        5. Transcribe images with Qwen (if images exist)
        6. Chunk markdown with Qwen (250 word limit)
        7. Extract metadata with Qwen
        8. Generate final JSON
        9. Save outputs to Database/Processed_Docs/{filename}_{ext}/
        
        Implements Requirements:
        - 1.1: Classify and route to appropriate conversion
        - 1.2: Convert all file types to markdown first
        - 7.1: Archive raw files to Database/Documents/{TYPE}/
        - 7.2: Create output directory at Database/Processed_Docs/{filename}_{ext}/
        - 7.3: Save images to figures subdirectory
        - 7.4: Save {filename}.json in output directory
        - 7.5: Save initial_{filename}.md and processed_{filename}.md
        """
        start_time = time.time()
        
        # Step 1: Archive raw file to Database/Documents/{TYPE}/
        # Implements Requirement 7.1
        print("\n[Step 1/9] Archiving raw file...")
        self._archive_raw_file(classified_file)
        
        # Step 2: Get output directory Database/Processed_Docs/{filename}_{ext}/
        # Implements Requirement 7.2
        output_dir = self.directory_config.get_file_output_dir(
            classified_file.filename, classified_file.extension
        )
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        print(f"✓ Output directory: {output_dir}")
        
        try:
            # Step 3: Convert to markdown based on file type
            # Implements Requirements 1.1, 1.2, 2.3 (initial markdown saved by UnifiedProcessor)
            print("\n[Step 2/9] Converting to markdown...")
            initial_markdown, total_pages = self._convert_to_markdown(
                classified_file, output_dir
            )
            
            # Note: initial markdown is already saved by UnifiedProcessor
            initial_md_path = os.path.join(output_dir, f"initial_{classified_file.filename}.md")
            
            # Determine source type for proper chunk formatting
            # Audio files use timestamp instead of page_number (Requirement 6.3)
            source_type = get_source_type(classified_file.extension)
            
            # Step 4: Check for images in markdown
            # Implements Requirements 2.4, 2.5, 3.5
            print("\n[Step 3/9] Checking for images...")
            image_paths = self.unified_processor.check_for_images(initial_markdown, output_dir)
            
            # Step 5: Transcribe images (if any exist)
            # Implements Requirements 3.1-3.4
            if image_paths:
                print(f"\n[Step 4/9] Transcribing {len(image_paths)} image(s) with Qwen...")
                processed_markdown = self.unified_processor.transcribe_images(
                    initial_markdown, image_paths
                )
            else:
                print("\n[Step 4/9] No images found, skipping transcription...")
                processed_markdown = initial_markdown
            
            # Step 6: Chunk markdown with Chunklet (250 word limit)
            # Implements Requirements 4.1-4.5, 11.1
            print("\n[Step 5/9] Chunking markdown (250 word limit)...")
            chunks = self.unified_processor.chunk_markdown(processed_markdown, source_type=source_type)
            print(f"✓ Created {len(chunks)} chunk(s)")
            
            # Step 7: Extract metadata with Qwen
            # Implements Requirements 5.1-5.5, 11.3
            print("\n[Step 6/9] Extracting metadata...")
            file_info = {
                'filename': classified_file.filename,
                'path': os.path.join(
                    self.directory_config.get_archive_dir(classified_file.extension),
                    Path(classified_file.filepath).name
                ),
                'file_name': f"{classified_file.filename}{classified_file.extension}",
                'source_type': source_type,
                'total_pages': total_pages,
            }
            metadata = self.unified_processor.extract_metadata(initial_markdown, file_info)
            
            # Step 8: Generate final JSON
            # Implements Requirements 6.1-6.5
            print("\n[Step 7/9] Generating final JSON...")
            final_json = self.unified_processor.generate_final_json(metadata, chunks, file_info)
            
            # Step 9: Save outputs
            # Implements Requirements 7.4, 7.5
            print("\n[Step 8/9] Saving outputs...")
            
            # Save processed markdown
            processed_md_path = os.path.join(output_dir, f"processed_{classified_file.filename}.md")
            with open(processed_md_path, 'w', encoding='utf-8') as f:
                f.write(processed_markdown)
            print(f"✓ Saved processed markdown: {processed_md_path}")
            
            # Save JSON to individual folder
            json_path = os.path.join(output_dir, f"{classified_file.filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved JSON: {json_path}")
            
            processing_time = time.time() - start_time
            print(f"\n✓ Processing complete ({processing_time:.2f}s)")
            print(f"  - Initial MD: {initial_md_path}")
            print(f"  - Processed MD: {processed_md_path}")
            print(f"  - JSON: {json_path}")
            
            return ProcessingResult(
                filename=classified_file.filename,
                success=True,
                initial_md_path=initial_md_path,
                processed_md_path=processed_md_path,
                json_path=json_path,
                processing_time=processing_time,
            )
            
        except Exception as e:
            error_msg = f"Error processing {classified_file.filename}: {str(e)}"
            print(f"\n✗ {error_msg}")
            import traceback
            traceback.print_exc()
            
            return ProcessingResult(
                filename=classified_file.filename,
                success=False,
                initial_md_path="",
                processed_md_path="",
                json_path="",
                processing_time=time.time() - start_time,
                error=error_msg
            )
    
    def _convert_to_markdown(self, classified_file: ClassifiedFile, output_dir: str) -> tuple:
        """
        Convert file to markdown based on type using UnifiedProcessor.
        
        Implements Requirements:
        - 1.1: Route to appropriate conversion based on file type
        - 1.2: Convert all file types to markdown first
        - 2.1-2.3: Use Docling for documents (PDF, DOCX, PPTX, XLSX, HTML)
        - 2.4: Check for images before transcription
        - 2.5: Skip image transcription if no images
        - 8.1-8.4: Use Whisper NPU for audio (MP3, WAV, M4A, FLAC)
        - 9.1-9.3: Use Qwen for standalone images (JPG, PNG, JPEG, WEBP)
        
        Args:
            classified_file: The classified file to convert
            output_dir: Output directory for artifacts
            
        Returns:
            Tuple of (markdown_text, total_pages)
            
        Raises:
            ValueError: If file type is unsupported
        """
        # Use UnifiedProcessor to handle all file types consistently
        # This ensures proper routing and initial markdown generation
        print("  Using UnifiedProcessor for conversion...")
        markdown, file_info = self.unified_processor.process_to_markdown(
            classified_file.filepath, output_dir
        )
        
        total_pages = file_info.get('total_pages', 1)
        
        # Log conversion results based on file type
        if classified_file.file_type == FileType.DOCUMENT:
            print(f"  ✓ Converted {total_pages} page(s) to markdown via Docling")
        elif classified_file.file_type == FileType.AUDIO:
            duration = file_info.get('duration_seconds', 0)
            print(f"  ✓ Transcribed audio ({duration:.1f}s) via Whisper NPU")
        elif classified_file.file_type == FileType.IMAGE:
            print("  ✓ Analyzed image via Qwen Vision")
        
        return markdown, total_pages
    
    def _archive_raw_file(self, classified_file: ClassifiedFile):
        """
        Archive raw file to Database/Documents/{TYPE}/.
        
        Implements Requirement 7.1:
        WHEN the Pipeline processes a file THEN the Pipeline SHALL copy the raw file
        to Database/Documents/{TYPE}/ directory based on file type
        
        Args:
            classified_file: The classified file to archive
        """
        archive_dir = self.directory_config.get_archive_dir(classified_file.extension)
        dest_path = os.path.join(archive_dir, Path(classified_file.filepath).name)
        
        if not os.path.exists(dest_path):
            shutil.copy2(classified_file.filepath, dest_path)
            print(f"✓ Archived to: {archive_dir}/{Path(classified_file.filepath).name}")
        else:
            print(f"✓ Already archived: {archive_dir}/{Path(classified_file.filepath).name}")
    
    def _clear_gpu_cache_if_needed(self, file_count: int):
        """Clear GPU cache every N files."""
        if self.model_config.device == "cuda":
            if file_count % self.processing_config.gpu_cache_clear_interval == 0:
                torch.cuda.empty_cache()
                print(f"\n[GPU] Cache cleared after processing {file_count} files")
    
    def _print_processing_summary(self, summary: ProcessingSummary):
        """Print summary of processing run."""
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Files: {summary.total_files}")
        print(f"Successful: {summary.successful}")
        print(f"Failed: {summary.failed}")
        print(f"Total Time: {summary.total_time_seconds:.2f}s")
        
        if summary.failed_files:
            print("\nFailed Files:")
            for filename in summary.failed_files:
                print(f"  - {filename}")
        
        print("=" * 60 + "\n")
    
    def query_document(self, filename: str, query: str) -> str:
        """Query a processed document using Qwen."""
        # Find the JSON file
        json_path = None
        for ext in ['.pdf', '.docx', '.pptx', '.mp3', '.wav', '.jpg', '.png']:
            potential_path = self.directory_config.get_json_path(filename, ext)
            if os.path.exists(potential_path):
                json_path = potential_path
                break
        
        if not json_path:
            raise FileNotFoundError(f"No processed JSON found for: {filename}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Build context from chunks
        context = "\n\n".join([
            f"[Chunk {c['chunk_index']}]: {c['paragraph_text']}"
            for c in doc_data.get('chunks', [])
        ])
        
        # Query with Qwen
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided document context."},
            {"role": "user", "content": f"Document Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self.lm_client.chat_completion(messages, json_mode=False)
        return response


def determine_processing_mode(
    file_count: int,
    batch_mode_enabled: bool
) -> str:
    """Determine which processing mode to use.
    
    Mode Selection Logic (Requirements 9.1, 9.2):
    - Single file → Individual mode (regardless of batch_mode flag)
    - Multiple files + batch_mode=False → Individual mode
    - Multiple files + batch_mode=True → Batch mode
    
    Args:
        file_count: Number of files to process
        batch_mode_enabled: Whether batch mode is enabled in config
        
    Returns:
        "individual" or "batch"
    """
    # Single file always uses individual mode (Requirement 9.1)
    if file_count <= 1:
        return "individual"
    
    # Multiple files: check batch_mode flag (Requirement 9.2)
    if batch_mode_enabled:
        return "batch"
    else:
        return "individual"


def process_with_batch_mode(
    input_dir: str,
    model_config,
    processing_config,
    directory_config
) -> ProcessingSummary:
    """Process files using batch mode (phase-based processing).
    
    Batch mode processes files in phases by operation type:
    - Phase 1: All documents → Docling
    - Phase 2: All audio → Whisper
    - Phase 3: All images → Qwen Vision
    - Phase 4: All markdown → Summarization
    
    Args:
        input_dir: Input directory path
        model_config: Model configuration
        processing_config: Processing configuration
        directory_config: Directory configuration
        
    Returns:
        ProcessingSummary with results
    """
    print("\n" + "=" * 60)
    print("BATCH MODE: Processing files in phases by operation type")
    print("=" * 60 + "\n")
    
    # Create batch config from processing config
    batch_config = BatchConfig(
        enable_batch_mode=True,
        lm_studio_endpoint=model_config.lm_studio_endpoint,
        progress_file=processing_config.batch_progress_file,
        retry_failed_files=processing_config.batch_retry_failed,
        max_retries=processing_config.batch_max_retries,
        cuda_device=model_config.device if model_config.device.startswith("cuda") else "cuda:0",
        clear_cache_between_phases=True,
        output_base_dir=directory_config.abs_processed,
    )
    
    # Initialize batch orchestrator
    orchestrator = BatchOrchestrator(batch_config)
    
    # Process batch
    batch_result = orchestrator.process_batch(input_dir)
    
    # Convert BatchResult to ProcessingSummary for compatibility
    failed_files = []
    for phase_result in batch_result.phases.values():
        failed_files.extend(phase_result.failed_files)
    
    return ProcessingSummary(
        total_files=batch_result.total_files,
        successful=batch_result.successful_files,
        failed=batch_result.failed_files,
        total_time_seconds=batch_result.total_duration_seconds,
        failed_files=failed_files
    )


def main():
    """Main entry point for the pipeline."""
    import sys
    # Load configuration
    config = load_config()
    model_config, prompts_config, processing_config, directory_config = config
    print_config_summary(*config)
    
    # Accept input directory as CLI argument, default to Database/Documents
    input_dir = sys.argv[1] if len(sys.argv) > 1 else directory_config.abs_documents
    
    # Count files in input directory
    file_classifier = FileClassifier()
    classified_files = file_classifier.scan_directory(input_dir)
    file_count = len(classified_files)
    
    print(f"\nFound {file_count} file(s) in {input_dir}")
    
    if file_count == 0:
        print("No files to process. Exiting.")
        exit(0)
    
    # Determine processing mode
    mode = determine_processing_mode(
        file_count=file_count,
        batch_mode_enabled=processing_config.enable_batch_mode
    )
    
    print(f"Processing mode: {mode.upper()}")
    print(f"  - File count: {file_count}")
    print(f"  - Batch mode enabled: {processing_config.enable_batch_mode}")
    
    # Route to appropriate processing mode
    if mode == "batch":
        # Use batch mode (phase-based processing)
        summary = process_with_batch_mode(
            input_dir=input_dir,
            model_config=model_config,
            processing_config=processing_config,
            directory_config=directory_config
        )
    else:
        # Use individual mode (sequential per-file processing)
        print("\n" + "=" * 60)
        print("INDIVIDUAL MODE: Processing files sequentially through all stages")
        print("=" * 60 + "\n")
        
        # Initialize pipeline for individual processing
        pipeline = Pipeline(config)
        summary = pipeline.process_all_files()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Total Files: {summary.total_files}")
    print(f"Successful: {summary.successful}")
    print(f"Failed: {summary.failed}")
    print(f"Total Time: {summary.total_time_seconds:.2f}s")
    
    if summary.failed_files:
        print("\nFailed Files:")
        for filename in summary.failed_files:
            print(f"  - {filename}")
    
    print("=" * 60 + "\n")
    
    # Exit with appropriate code
    if summary.failed > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()

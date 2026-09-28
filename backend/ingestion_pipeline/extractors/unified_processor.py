"""Unified processor for multi-modal pipeline.

This module implements the unified pipeline flow:
1. Convert any file type to markdown
2. Check for image paths in markdown
3. Transcribe images with Qwen (if images exist)
4. Chunk markdown with Qwen (250 word limit)
5. Extract metadata with Qwen
6. Generate final JSON

Requirements:
- 1.1: Pipeline receives file and routes to appropriate conversion
- 1.2: All file types convert to markdown first
- 1.3: Generate single structured JSON with metadata and chunks
- 6.1-6.4: Structured JSON output with all required fields
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from extractors.config import (
    PromptsConfig, DirectoryConfig, ModelConfig,
    get_source_type, get_file_type
)
from external_clients.openai_compatible_client import OpenAICompatibleClient as LMStudioClient
from extractors.llm_metadata import LLMMetadataExtractor
from extractors.chunklet import Chunklet, ChunkletConfig

# Conditional imports to avoid circular dependencies
if TYPE_CHECKING:
    from extractors.docling_gpu_processor import DoclingGPUProcessor
    from extractors.whisper_gpu_processor import WhisperGPUProcessor


@dataclass
class ProcessedFile:
    """Result of processing a file through the unified pipeline."""
    filename: str
    extension: str
    source_type: str
    initial_md_path: str
    processed_md_path: str
    json_path: str
    success: bool
    error: Optional[str] = None


class UnifiedProcessor:
    """
    Unified processor for all file types.
    
    Implements the flow:
    File → Markdown → Check Images → Transcribe → Chunk → Metadata → JSON
    
    Implements Requirements:
    - 1.1: Classify file and route to appropriate conversion
    - 1.2: Convert all file types to markdown first
    - 1.3: Generate single structured JSON output
    - 2.4: Check if markdown contains image paths
    - 2.5: Skip image transcription if no images
    - 3.5: Handle gracefully when no images exist
    - 6.1-6.4: Generate structured JSON with metadata and chunks
    """
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        prompts_config: PromptsConfig,
        directory_config: DirectoryConfig,
        model_config: Optional[ModelConfig] = None,
        metadata_client: Optional[LMStudioClient] = None,
        docling_processor: Optional['DoclingGPUProcessor'] = None,
        whisper_processor: Optional['WhisperGPUProcessor'] = None,
    ):
        """
        Initialize unified processor.
        
        Args:
            lm_client: LM Studio client for vision/image analysis
            prompts_config: Prompts for image analysis, metadata
            directory_config: Directory configuration
            model_config: Model configuration (optional)
            metadata_client: Separate LLM client for metadata extraction (falls back to lm_client if None)
            docling_processor: Docling processor for documents (lazy init if None)
            whisper_processor: Whisper processor for audio (lazy init if None)
        """
        self.lm_client = lm_client
        self.prompts = prompts_config
        self.dirs = directory_config
        self.model_config = model_config or ModelConfig()
        
        # Lazy initialization of processors
        self._docling_processor = docling_processor
        self._whisper_processor = whisper_processor
        
        # Initialize Chunklet for fast code-based chunking
        self.chunklet = Chunklet(ChunkletConfig())
        
        # Initialize metadata extractor with its own dedicated client
        self.metadata_extractor = LLMMetadataExtractor(
            lm_client=metadata_client or lm_client,
            prompts_config=prompts_config
        )
        
        print("UnifiedProcessor: Initialized with Chunklet")
    
    @property
    def docling_processor(self) -> 'DoclingGPUProcessor':
        """Lazy initialization of Docling GPU processor."""
        if self._docling_processor is None:
            from extractors.docling_gpu_processor import DoclingGPUProcessor
            self._docling_processor = DoclingGPUProcessor(device=self.model_config.whisper_device)
            self._docling_processor.initialize()
        return self._docling_processor

    @property
    def whisper_processor(self) -> 'WhisperGPUProcessor':
        """Lazy initialization of Whisper GPU processor."""
        if self._whisper_processor is None:
            from extractors.whisper_gpu_processor import WhisperGPUProcessor
            self._whisper_processor = WhisperGPUProcessor(
                model_name=self.model_config.whisper_model_name,
                device=self.model_config.whisper_device,
                enable_diarization=True
            )
        return self._whisper_processor
    
    def process_to_markdown(self, file_path: str, output_dir: str) -> Tuple[str, Dict]:
        """
        Convert any file type to markdown.
        
        Implements Requirements:
        - 1.1: Route file to appropriate conversion based on type
        - 1.2: Convert all file types to markdown first
        - 2.1-2.3: Use Docling for documents
        - 8.1-8.4: Use Whisper for audio
        - 9.1-9.3: Use Qwen for standalone images
        
        Args:
            file_path: Path to input file
            output_dir: Directory for outputs
            
        Returns:
            Tuple of (markdown_content, file_info_dict)
            
        Raises:
            ValueError: If file type is unsupported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        path = Path(file_path)
        filename = path.stem
        extension = path.suffix.lower()
        file_type = get_file_type(file_path)
        source_type = get_source_type(extension)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        print(f"UnifiedProcessor: Processing {path.name} as {file_type}")
        
        # File info for metadata
        file_info = {
            'filename': filename,
            'file_name': path.name,
            'path': str(path.resolve()),
            'extension': extension,
            'source_type': source_type,
            'total_pages': 1,
        }
        
        # Route to appropriate converter
        if file_type == 'document':
            markdown, file_info = self._process_document(file_path, output_dir, file_info)
        elif file_type == 'audio':
            markdown, file_info = self._process_audio(file_path, output_dir, file_info)
        elif file_type == 'image':
            markdown, file_info = self._process_image(file_path, output_dir, file_info)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return markdown, file_info
    
    def _process_document(
        self, file_path: str, output_dir: str, file_info: Dict
    ) -> Tuple[str, Dict]:
        """
        Process document file (PDF, DOCX, PPTX, XLSX, HTML) to markdown.

        Uses DoclingGPUProcessor.convert_document() which returns a DoclingGPUResult.
        Raises on failure so the pipeline halts instead of silently continuing.
        """
        print("UnifiedProcessor: Converting document with Docling...")

        result = self.docling_processor.convert_document(file_path, output_dir)

        if not result.success:
            raise RuntimeError(
                f"Docling conversion failed for {file_path}: {result.error}"
            )

        markdown = result.markdown_text

        if not markdown or not markdown.strip():
            raise RuntimeError(
                f"Docling returned empty markdown for {file_path}"
            )

        # Count pages from the saved markdown path name (best-effort)
        # DoclingGPUResult doesn't expose page count directly; keep default of 1
        # unless we can infer it from the document.
        file_info['total_pages'] = file_info.get('total_pages', 1)

        # initial_{filename}.md is already written by convert_document — just
        # record the path so callers know where it is.
        initial_md_path = result.markdown_path
        print(f"UnifiedProcessor: Docling saved initial markdown → {initial_md_path}")

        return markdown, file_info
    
    def _process_audio(
        self, file_path: str, output_dir: str, file_info: Dict
    ) -> Tuple[str, Dict]:
        """
        Process audio file (MP3, WAV, M4A, FLAC) to markdown.
        
        Implements Requirements:
        - 8.1: Use Whisper via OpenVINO on Intel NPU
        - 8.2: Enable speaker diarization
        - 8.3: Format as [HH:MM:SS] Speaker N: text
        - 8.4: Use timestamp field instead of page_number
        
        Args:
            file_path: Path to audio file
            output_dir: Output directory
            file_info: File information dict
            
        Returns:
            Tuple of (markdown, updated_file_info)
            
        Raises:
            RuntimeError: If transcription fails or produces no output
        """
        print("UnifiedProcessor: Transcribing audio with Whisper...")
        print("UnifiedProcessor: No timeout - waiting for Whisper to complete...")
        
        # Transcribe with Whisper - no timeout, wait for completion
        # Let errors propagate - don't silently fail
        transcription = self.whisper_processor.transcribe(file_path, output_dir)
        
        # Convert to markdown
        markdown = transcription.to_markdown()
        
        # Validate we got actual content
        if not markdown or len(markdown.strip()) < 50:
            raise RuntimeError(f"Transcription produced insufficient content for {file_info['filename']}")
        
        # Update file info
        file_info['total_pages'] = 0  # Audio doesn't have pages
        file_info['duration_seconds'] = transcription.duration_seconds
        
        print(f"UnifiedProcessor: Audio transcription complete - {transcription.duration_seconds:.1f}s")
        
        return markdown, file_info
    
    def _process_image(
        self, file_path: str, output_dir: str, file_info: Dict
    ) -> Tuple[str, Dict]:
        """Process standalone image file to markdown. Raises on failure."""
        print("UnifiedProcessor: Analyzing image with vision model...")

        from extractors.image_pipeline import ImageHandler
        image_handler = ImageHandler(
            lm_client=self.lm_client,
            prompts_config=self.prompts
        )
        markdown = image_handler.process_image(file_path, output_dir)

        if not markdown or not markdown.strip():
            raise RuntimeError(f"Image analysis returned empty content for {file_path}")

        return markdown, file_info
    
    def check_for_images(self, markdown: str, output_dir: str = "") -> List[str]:
        """
        Detect image paths in markdown with safe detection.
        
        Implements Requirements:
        - 2.4: Check if markdown contains image paths
        - 2.5: Skip image transcription if no images
        - 3.5: Handle gracefully - never crash pipeline
        
        Scans for ![caption](path) patterns and validates paths exist.
        Returns empty list if no images found - pipeline continues without errors.
        
        Args:
            markdown: Markdown content to scan
            output_dir: Output directory for resolving relative paths
            
        Returns:
            List of valid image paths (empty if none found)
        """
        # Pattern for markdown images: ![caption](path)
        pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(pattern, markdown)
        
        if not matches:
            print("UnifiedProcessor: No image paths found in markdown")
            return []
        
        # Filter to only existing files
        valid_paths = []
        for path in matches:
            # Try absolute path first
            if os.path.exists(path):
                valid_paths.append(path)
            # Try relative to output_dir
            elif output_dir and os.path.exists(os.path.join(output_dir, path)):
                valid_paths.append(os.path.join(output_dir, path))
            else:
                print(f"UnifiedProcessor: Image path not found, skipping: {path}")
        
        if valid_paths:
            print(f"UnifiedProcessor: Found {len(valid_paths)} valid image(s)")
        else:
            print("UnifiedProcessor: No valid image paths found")
        
        return valid_paths
    
    def transcribe_images(self, markdown: str, image_paths: List[str]) -> str:
        """
        Replace image paths with LLM vision transcriptions, processed in parallel.

        All images are sent to the vision API concurrently using a thread pool
        (requests is blocking/sync, so threads give true parallelism here).
        Results are applied to the markdown in original order after all
        requests complete.

        Args:
            markdown: Original markdown with image paths
            image_paths: List of image paths to transcribe

        Returns:
            Markdown with image paths replaced by transcriptions
        """
        if not image_paths:
            return markdown

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Cap concurrency — most cloud providers handle 8-16 parallel requests
        # comfortably; local providers may need a lower value.
        MAX_WORKERS = min(len(image_paths), 8)

        print(f"UnifiedProcessor: Transcribing {len(image_paths)} image(s) "
              f"in parallel (workers={MAX_WORKERS})...")

        def _analyse_one(image_path: str):
            """Return (image_path, analysis_text) or (image_path, None) on error."""
            try:
                analysis = self.lm_client.analyze_image(
                    image_path,
                    self.prompts.image_analysis_prompt
                )
                return image_path, analysis
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "rate-limited" in msg.lower():
                    print(f"UnifiedProcessor: ✗ {Path(image_path).name} — rate-limited, skipping after retries")
                else:
                    print(f"UnifiedProcessor: ✗ {Path(image_path).name} — {exc}")
                return image_path, None

        # Submit all requests concurrently
        results: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_analyse_one, p): p for p in image_paths}
            for future in as_completed(futures):
                img_path, analysis = future.result()
                results[img_path] = analysis
                status = "✓" if analysis else "✗"
                print(f"UnifiedProcessor: {status} {Path(img_path).name}")

        # Apply results to markdown in original order
        result_markdown = markdown
        for image_path in image_paths:
            analysis = results.get(image_path)
            if not analysis:
                continue

            abs_path = image_path.replace("\\", "/")
            basename = os.path.basename(image_path)
            parent_basename = os.path.basename(os.path.dirname(image_path))

            path_variants = [
                abs_path,
                basename,
                f"{parent_basename}/{basename}",
                f"./{parent_basename}/{basename}",
            ]

            replaced = False
            for path_variant in path_variants:
                escaped_path = re.escape(path_variant)
                pattern = rf'(!\[.*?\]\({escaped_path}\))'
                if re.search(pattern, result_markdown):
                    replacement = rf'\1\n\n**[Image Analysis]:** {analysis}\n'
                    result_markdown = re.sub(pattern, replacement, result_markdown)
                    replaced = True
                    break

            if not replaced:
                print(f"UnifiedProcessor: Could not find image reference for {image_path}")

        return result_markdown
    
    def chunk_markdown(self, markdown: str, source_type: str = "document") -> List[Dict]:
        """
        Chunk markdown using Chunklet (fast code-based chunking).
        
        Implements Requirements:
        - 3.1: Use code-based chunking without LLM calls
        - 3.2: Remove noise patterns (page numbers, headers, footers)
        - 3.3: Header bonding - never split heading from content
        - 3.4: Table integrity - keep all table rows together
        - 3.5: Repeat heading for table chunk if split
        - 3.6: Keep images with preceding paragraph
        - 3.7: Split at sentence boundaries when exceeding 250 words
        - 3.8: Include most recent heading for context
        - 6.4: Use timestamp field instead of page_number for audio
        - 7.1: Complete without LLM API calls
        - 7.2: Complete in under 1 second for 10-page documents
        
        Args:
            markdown: Markdown content to chunk
            source_type: Type of source ("document", "audio", "image")
            
        Returns:
            List of chunk dictionaries with page_number/timestamp, chunk_index, paragraph_text
        """
        print("UnifiedProcessor: Chunking markdown with Chunklet (code-based)...")
        
        # Use Chunklet for fast code-based chunking (no LLM calls)
        chunks = self.chunklet.split_text(markdown)
        
        # Handle audio timestamp field (Requirement 6.4)
        is_audio = (source_type == "audio")
        if is_audio:
            # For audio, extract timestamps from markdown and replace page_number with timestamp
            chunks = self._convert_to_audio_timestamps(chunks, markdown)
        
        if not chunks:
            # Fallback: create single chunk from entire content
            print("UnifiedProcessor: Chunking failed, using fallback")
            chunks = [{
                "chunk_index": 1,
                "paragraph_text": markdown[:1000],  # First 1000 chars
            }]
            
            # Add page_number or timestamp based on source type
            if is_audio:
                chunks[0]["timestamp"] = "00:00:00"
            else:
                chunks[0]["page_number"] = 1
        
        print(f"UnifiedProcessor: Created {len(chunks)} chunk(s) using Chunklet")
        return chunks
    
    def _convert_to_audio_timestamps(self, chunks: List[Dict], markdown: str) -> List[Dict]:
        """
        Convert page_number field to timestamp field for audio chunks.
        
        Implements Requirement 6.4: Use timestamp instead of page_number for audio
        
        Args:
            chunks: List of chunks with page_number field
            markdown: Original markdown to extract timestamps from
            
        Returns:
            List of chunks with timestamp field instead of page_number
        """
        # Extract all timestamps from markdown (format: [HH:MM:SS])
        timestamp_pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]')
        timestamps = timestamp_pattern.findall(markdown)
        
        # Convert chunks to use timestamp field
        for i, chunk in enumerate(chunks):
            # Remove page_number field if it exists
            if 'page_number' in chunk:
                del chunk['page_number']
            
            # Add timestamp field
            # Try to match chunk to a timestamp, or use default
            if i < len(timestamps):
                chunk['timestamp'] = timestamps[i]
            else:
                # Default timestamp if we can't find one
                chunk['timestamp'] = "00:00:00"
        
        return chunks
    
    def extract_metadata(self, markdown: str, file_info: Dict) -> Dict:
        """
        Extract metadata using Qwen.
        
        Implements Requirements:
        - 5.1: Extract title from document header or first heading
        - 5.2: Clean author names (remove affiliation markers)
        - 5.3: Extract dates in YYYY-MM-DD format
        - 5.4: Extract language as ISO 639-1 code
        - 5.5: Generate 5-10 semantic keywords
        - 11.3: Use Bibliographic Metadata Extractor prompt
        
        Args:
            markdown: Markdown content
            file_info: File information (path, filename, extension)
            
        Returns:
            Metadata dictionary with all required fields
        """
        print("UnifiedProcessor: Extracting metadata...")
        
        # Use LLMMetadataExtractor for metadata extraction
        metadata = self.metadata_extractor.extract_metadata(
            markdown=markdown,
            filename=file_info.get('filename', ''),
            file_path=file_info.get('path', ''),
            source_type=file_info.get('source_type', 'document')
        )
        
        # Ensure all required fields for final JSON
        final_metadata = {
            "title": metadata.get('title', file_info.get('filename', 'Untitled')),
            "path": file_info.get('path', ''),
            "author": "",
            "created_date": metadata.get('created_date'),
            "source_type": file_info.get('source_type', 'unknown'),
            "language": metadata.get('language', 'en'),
            "total_pages": file_info.get('total_pages', 1),
            "file_name": file_info.get('file_name', ''),
            "tags": metadata.get('tags', []),
            "extra_metadata": metadata.get('extra_metadata', {})
        }
        
        # Handle authors field (convert list to string for author field)
        if 'authors' in metadata:
            if isinstance(metadata['authors'], list) and metadata['authors']:
                final_metadata['author'] = ', '.join(metadata['authors'])
            elif isinstance(metadata['authors'], str):
                final_metadata['author'] = metadata['authors']
        
        print(f"UnifiedProcessor: Extracted metadata for '{final_metadata.get('title', 'Unknown')}'")
        return final_metadata
    
    def generate_final_json(
        self,
        metadata: Dict,
        chunks: List[Dict],
        file_info: Dict
    ) -> Dict:
        """
        Generate final JSON structure with correct schema.
        
        Implements Requirements:
        - 6.1: Output contains document_metadata with all required fields
        - 6.2: Output contains chunks array with page_number/timestamp, chunk_index, paragraph_text, summary
        - 6.3: For audio files, use timestamp field instead of page_number
        - 6.4: Save as {filename}.json
        - 6.5: Include extra_metadata when available
        
        Args:
            metadata: Document metadata (already has all required fields)
            chunks: List of content chunks
            file_info: File information
            
        Returns:
            Final JSON structure matching the schema
        """
        # Metadata should already have all required fields from extract_metadata
        # Just ensure nothing is missing
        final_metadata = {
            "title": metadata.get('title', file_info.get('filename', 'Untitled')),
            "path": metadata.get('path', file_info.get('path', '')),
            "author": metadata.get('author', ''),
            "created_date": metadata.get('created_date'),
            "source_type": metadata.get('source_type', file_info.get('source_type', 'unknown')),
            "language": metadata.get('language', 'en'),
            "total_pages": metadata.get('total_pages', file_info.get('total_pages', len(chunks))),
            "file_name": metadata.get('file_name', file_info.get('file_name', '')),
            "tags": metadata.get('tags', []),
            "extra_metadata": metadata.get('extra_metadata', {})
        }
        
        # Ensure chunks have required fields
        validated_chunks = []
        for chunk in chunks:
            validated_chunk = {
                "chunk_index": chunk.get('chunk_index', len(validated_chunks) + 1),
                "paragraph_text": chunk.get('paragraph_text', ''),
                "summary": chunk.get('summary', '')
            }
            
            # Add page_number or timestamp based on what's in the chunk
            if 'timestamp' in chunk:
                validated_chunk['timestamp'] = chunk['timestamp']
            else:
                validated_chunk['page_number'] = chunk.get('page_number', 1)
            
            validated_chunks.append(validated_chunk)
        
        return {
            "document_metadata": final_metadata,
            "chunks": validated_chunks
        }
    
    def process_markdown_content(
        self,
        markdown: str,
        file_info: Dict,
        output_dir: str
    ) -> Tuple[Dict, str]:
        """
        Process markdown through the unified pipeline.
        
        Implements the complete unified flow:
        1. Check for images in markdown
        2. Transcribe images with Qwen (if images exist)
        3. Chunk markdown with Chunklet (fast code-based)
        4. Extract metadata with Qwen
        5. Generate final JSON
        
        Args:
            markdown: Initial markdown content
            file_info: File information
            output_dir: Output directory
            
        Returns:
            Tuple of (final_json, processed_markdown)
        """
        source_type = file_info.get('source_type', 'document')
        
        # Step 1: Check for images
        image_paths = self.check_for_images(markdown, output_dir)
        
        # Step 2: Transcribe images (if any exist)
        if image_paths:
            processed_markdown = self.transcribe_images(markdown, image_paths)
        else:
            processed_markdown = markdown
        
        # Step 3: Chunk markdown with Chunklet (fast code-based)
        chunks = self.chunk_markdown(processed_markdown, source_type=source_type)
        
        # Step 4: Extract metadata
        metadata = self.extract_metadata(markdown, file_info)
        
        # Step 5: Generate final JSON
        final_json = self.generate_final_json(metadata, chunks, file_info)
        
        return final_json, processed_markdown
    
    def save_outputs(
        self,
        initial_markdown: str,
        processed_markdown: str,
        final_json: Dict,
        output_dir: str,
        filename: str
    ) -> Tuple[str, str, str]:
        """
        Save all output files.
        
        Args:
            initial_markdown: Raw markdown from conversion
            processed_markdown: Markdown with image transcriptions
            final_json: Final JSON structure
            output_dir: Output directory
            filename: Base filename
            
        Returns:
            Tuple of (initial_md_path, processed_md_path, json_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save initial markdown
        initial_md_path = os.path.join(output_dir, f"initial_{filename}.md")
        with open(initial_md_path, 'w', encoding='utf-8') as f:
            f.write(initial_markdown)
        
        # Save processed markdown
        processed_md_path = os.path.join(output_dir, f"processed_{filename}.md")
        with open(processed_md_path, 'w', encoding='utf-8') as f:
            f.write(processed_markdown)
        
        # Save JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        
        print(f"UnifiedProcessor: Saved outputs to {output_dir}")
        
        return initial_md_path, processed_md_path, json_path

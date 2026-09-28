"""Docling GPU Processor for batch pipeline optimization.

This module provides GPU-accelerated document processing using Docling
with CUDA 12.8 support for batch processing workflows.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Import torch for GPU detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. GPU acceleration will not be available.")

# Import Docling components
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    logger.warning("Docling not installed. Document processing will not be available.")

# Try to import AcceleratorOptions for GPU configuration
try:
    from docling.datamodel.pipeline_options import AcceleratorOptions
    HAS_ACCELERATOR_OPTIONS = True
except ImportError:
    HAS_ACCELERATOR_OPTIONS = False
    logger.warning("AcceleratorOptions not available. GPU acceleration may be limited.")

# Try to import PyMuPDF for PDF image extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not installed. PDF image extraction will be limited.")

# Import PIL for image handling
try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not installed. Image processing will be limited.")


# Supported document formats for Docling
SUPPORTED_FORMATS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.html', '.htm'}


@dataclass
class ExtractedImage:
    """Image extracted from document with metadata."""
    index: int
    path: str
    caption: str
    page_number: int
    image_type: str  # "figure", "table", "diagram", etc.
    bbox: Optional[Dict[str, float]] = None
    source_file: Optional[str] = None  # Original document this image came from
    markdown_file: Optional[str] = None  # .md file to update with caption


@dataclass
class DoclingGPUResult:
    """Result of Docling GPU processing."""
    markdown_text: str
    markdown_path: str  # Path to saved initial_{filename}.md
    images: List[ExtractedImage] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)  # All image paths for Phase 3
    source_file: str = ""
    success: bool = True
    error: Optional[str] = None


class DoclingGPUProcessor:
    """GPU-accelerated document processor using Docling.
    
    Provides batch-optimized document processing with:
    - CUDA 12.8 GPU acceleration (Requirements 2.1)
    - Sequential processing with GPU (Requirements 2.2)
    - Output naming convention: initial_{filename}.md (Requirements 2.3)
    - Image extraction to figures/ directory (Requirements 2.4)
    - GPU memory cleanup after processing (Requirements 2.5)
    
    Example:
        processor = DoclingGPUProcessor(device="cuda:0")
        processor.initialize()
        
        result = processor.convert_document(
            "document.pdf",
            "Database/Processed_Docs/document_pdf"
        )
        
        processor.cleanup()
    """
    
    def __init__(self, device: str = "cuda:0"):
        """Initialize DoclingGPUProcessor.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
        """
        self.device = device
        self.converter: Optional[DocumentConverter] = None
        self._initialized = False
        
    def _verify_gpu(self) -> bool:
        """Verify GPU is available for Docling processing.
        
        Returns:
            True if GPU is available
            
        Raises:
            RuntimeError: If CUDA is not available
            
        Requirements: 2.1
        """
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed. Install with: "
                "pip3 install --pre torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/nightly/cu128"
            )
        
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Ensure you have:\n"
                "1. A CUDA-capable GPU\n"
                "2. PyTorch nightly with CUDA 12.8 installed:\n"
                "   pip3 install --pre torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/nightly/cu128"
            )
        
        # Extract device index
        device_idx = 0
        if ":" in self.device:
            try:
                device_idx = int(self.device.split(":")[1])
            except (ValueError, IndexError):
                device_idx = 0
        
        # Verify device is accessible
        if device_idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device_idx} not available. "
                f"Found {torch.cuda.device_count()} device(s)."
            )
        
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(device_idx)
        
        logger.info(f"DoclingGPUProcessor: CUDA Version: {cuda_version}")
        logger.info(f"DoclingGPUProcessor: GPU Device: {device_name}")
        
        return True
    
    def initialize(self) -> None:
        """Initialize Docling converter with GPU acceleration.
        
        Creates DocumentConverter with CUDA device configuration.
        
        Requirements: 2.1, 2.2
        """
        if self._initialized:
            logger.info("DoclingGPUProcessor already initialized")
            return
        
        if not HAS_DOCLING:
            raise RuntimeError("Docling is not installed")
        
        # Verify GPU availability
        self._verify_gpu()
        
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        
        # Configure accelerator device for GPU
        if HAS_ACCELERATOR_OPTIONS and HAS_TORCH:
            try:
                pipeline_options.accelerator_options = AcceleratorOptions(
                    num_threads=4,
                    device="cuda"
                )
                logger.info("DoclingGPUProcessor: AcceleratorOptions configured with CUDA")
            except Exception as e:
                logger.warning(f"Could not configure accelerator: {e}")
        
        # Create converter with PDF format options
        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("DoclingGPUProcessor: DocumentConverter initialized with GPU")
        except Exception as e:
            logger.warning(f"Could not create converter with custom options: {e}")
            logger.info("Falling back to default DocumentConverter")
            self.converter = DocumentConverter()
        
        self._initialized = True
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in SUPPORTED_FORMATS
    
    def convert_document(self, file_path: str, output_dir: str) -> DoclingGPUResult:
        """Convert a document to markdown using GPU acceleration.
        
        Processes the document and saves:
        - initial_{filename}.md in output_dir
        - Extracted images in output_dir/figures/
        
        Args:
            file_path: Path to the document file
            output_dir: Directory for output files (e.g., Database/Processed_Docs/doc_pdf)
            
        Returns:
            DoclingGPUResult with markdown text, paths, and extracted images
            
        Requirements: 2.2, 2.3, 2.4
        """
        if not self._initialized:
            self.initialize()
        
        if not os.path.exists(file_path):
            return DoclingGPUResult(
                markdown_text="",
                markdown_path="",
                source_file=file_path,
                success=False,
                error=f"File not found: {file_path}"
            )
        
        if not self.is_supported(file_path):
            return DoclingGPUResult(
                markdown_text="",
                markdown_path="",
                source_file=file_path,
                success=False,
                error=f"Unsupported format: {Path(file_path).suffix}"
            )
        
        try:
            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            figures_dir = os.path.join(output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Get filename for output naming
            filename = Path(file_path).stem
            
            # Convert document using Docling
            result = self.converter.convert(file_path)
            
            # Check conversion status
            if result.status.name == "FAILURE":
                error_msgs = []
                for page_no, page_result in result.pages.items():
                    if page_result.status.name == "FAILURE" and page_result.errors:
                        error_msgs.append(f"Page {page_no}: {'; '.join(page_result.errors)}")
                error_summary = "; ".join(error_msgs) if error_msgs else "Unknown error"
                
                return DoclingGPUResult(
                    markdown_text="",
                    markdown_path="",
                    source_file=file_path,
                    success=False,
                    error=f"Conversion failed: {error_summary}"
                )
            
            doc = result.document
            
            # Extract markdown
            markdown_text = doc.export_to_markdown()
            
            # Extract images if PDF
            images = []
            image_paths = []
            
            ext = Path(file_path).suffix.lower()
            if ext == '.pdf' and HAS_PYMUPDF and HAS_PIL:
                images, image_paths = self._extract_pdf_images(
                    doc, file_path, figures_dir, markdown_text, filename
                )
            
            # Process markdown to insert image paths
            final_markdown = self._process_markdown(markdown_text, images, output_dir)
            
            # Save initial_{filename}.md (Requirements 2.3)
            markdown_path = os.path.join(output_dir, f"initial_{filename}.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(final_markdown)
            
            logger.info(f"DoclingGPUProcessor: Converted {file_path} -> {markdown_path}")
            
            # Update image references with markdown file path
            for img in images:
                img.source_file = file_path
                img.markdown_file = markdown_path
            
            return DoclingGPUResult(
                markdown_text=final_markdown,
                markdown_path=markdown_path,
                images=images,
                image_paths=image_paths,
                source_file=file_path,
                success=True
            )
            
        except Exception as e:
            logger.error(f"DoclingGPUProcessor: Error processing {file_path}: {e}")
            return DoclingGPUResult(
                markdown_text="",
                markdown_path="",
                source_file=file_path,
                success=False,
                error=str(e)
            )

    
    def _extract_pdf_images(
        self,
        doc,
        pdf_path: str,
        figures_dir: str,
        markdown_text: str,
        filename: str
    ) -> Tuple[List[ExtractedImage], List[str]]:
        """Extract images from PDF using PyMuPDF.
        
        Args:
            doc: Docling document object
            pdf_path: Path to the PDF file
            figures_dir: Directory to save extracted images
            markdown_text: Markdown text to scan for figure captions
            filename: Base filename for output naming
            
        Returns:
            Tuple of (list of ExtractedImage, list of image paths)
            
        Requirements: 2.4
        """
        images = []
        image_paths = []
        
        if not hasattr(doc, 'pictures') or not doc.pictures:
            return images, image_paths
        
        # Open PDF with PyMuPDF
        try:
            pdf_document = fitz.open(pdf_path)
        except Exception as e:
            logger.warning(f"Could not open PDF with PyMuPDF: {e}")
            return images, image_paths
        
        try:
            # Build caption map from markdown
            caption_map = self._build_caption_map(markdown_text)
            
            # Extract each image
            for idx, picture_item in enumerate(doc.pictures):
                try:
                    # Get bounding box from provenance
                    if not hasattr(picture_item, 'prov') or not picture_item.prov:
                        continue
                    
                    prov = picture_item.prov[0]
                    page_no = prov.page_no - 1  # Convert to 0-based index
                    bbox = prov.bbox
                    
                    # Get the page
                    if page_no >= len(pdf_document):
                        continue
                    
                    page = pdf_document[page_no]
                    page_height = page.rect.height
                    
                    # Create rectangle for PyMuPDF
                    rect = fitz.Rect(
                        bbox.l,
                        page_height - bbox.t,
                        bbox.r,
                        page_height - bbox.b
                    )
                    
                    # Extract the image area as pixmap
                    mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Determine filename and caption
                    caption = ""
                    if idx in caption_map:
                        fig_label = caption_map[idx]
                        img_filename = fig_label.replace(" ", "_") + ".png"
                        caption = fig_label
                    else:
                        img_filename = f"image_{idx + 1}.png"
                        caption = f"Image {idx + 1}"
                    
                    # Save image
                    save_path = os.path.join(figures_dir, img_filename)
                    pil_image.save(save_path)
                    
                    # Determine image type from caption
                    image_type = self._determine_image_type(caption)
                    
                    # Create ExtractedImage object
                    extracted_image = ExtractedImage(
                        index=idx,
                        path=save_path,
                        caption=caption,
                        page_number=page_no + 1,
                        image_type=image_type,
                        bbox={
                            "left": bbox.l,
                            "top": bbox.t,
                            "right": bbox.r,
                            "bottom": bbox.b
                        }
                    )
                    images.append(extracted_image)
                    image_paths.append(save_path)
                    
                except Exception as e:
                    logger.warning(f"Could not extract image {idx}: {e}")
                    continue
                    
        finally:
            pdf_document.close()
        
        return images, image_paths
    
    def _build_caption_map(self, markdown_text: str) -> Dict[int, str]:
        """Build a map of image indices to figure captions from markdown.
        
        Args:
            markdown_text: Markdown text to scan
            
        Returns:
            Dictionary mapping image index to figure label
        """
        caption_map = {}
        figure_pattern = re.compile(r"^(Figure\s+\d+)[:\.\s]*(.*)", re.IGNORECASE)
        
        lines = markdown_text.split('\n')
        temp_image_idx = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if this is an image placeholder
            if line_stripped == '<!-- image -->':
                # Look ahead for a figure caption
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    match = figure_pattern.match(next_line)
                    if match:
                        fig_label = match.group(1)
                        caption_map[temp_image_idx] = fig_label
                        break
                temp_image_idx += 1
        
        return caption_map
    
    def _determine_image_type(self, caption: str) -> str:
        """Determine image type from caption text.
        
        Args:
            caption: Image caption text
            
        Returns:
            Image type string
        """
        caption_lower = caption.lower()
        
        if "table" in caption_lower:
            return "table"
        elif "diagram" in caption_lower:
            return "diagram"
        elif "chart" in caption_lower:
            return "chart"
        elif "graph" in caption_lower:
            return "graph"
        elif "figure" in caption_lower:
            return "figure"
        else:
            return "figure"
    
    def _process_markdown(
        self,
        markdown_text: str,
        images: List[ExtractedImage],
        output_dir: str
    ) -> str:
        """Process markdown to insert image paths.
        
        Args:
            markdown_text: Original markdown text
            images: List of extracted images
            output_dir: Output directory for relative path calculation
            
        Returns:
            Processed markdown with image paths inserted
        """
        figure_pattern = re.compile(r"^(Figure\s+\d+)[:\.\s]*(.*)", re.IGNORECASE)
        lines = markdown_text.split('\n')
        final_lines = []
        image_idx = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this is an image placeholder
            if line == '<!-- image -->':
                # Look ahead for a figure caption
                caption_found = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    match = figure_pattern.match(next_line)
                    if match:
                        fig_label = match.group(1)
                        caption_text = match.group(2).strip() if match.group(2) else ""
                        
                        # Insert image with markdown syntax using absolute path
                        if image_idx < len(images):
                            img = images[image_idx]
                            abs_path = os.path.abspath(img.path).replace("\\", "/")
                            caption = f"{fig_label}: {caption_text}" if caption_text else fig_label
                            final_lines.append(f"![{caption}]({abs_path})")
                            final_lines.append(f"*{caption}*")
                            image_idx += 1
                        
                        caption_found = True
                        i = j
                        break
                
                # If no caption found, just insert image
                if not caption_found and image_idx < len(images):
                    img = images[image_idx]
                    abs_path = os.path.abspath(img.path).replace("\\", "/")
                    final_lines.append(f"![{img.caption}]({abs_path})")
                    image_idx += 1
            
            elif line and line != '<!-- image -->':
                # Check if this line is a figure caption (standalone)
                match = figure_pattern.match(line)
                if not match:
                    final_lines.append(line)
            
            i += 1
        
        return "\n\n".join(final_lines)
    
    def cleanup(self) -> None:
        """Release GPU resources.
        
        Clears CUDA cache and releases converter.
        
        Requirements: 2.5
        """
        self.converter = None
        self._initialized = False
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("DoclingGPUProcessor: GPU resources released")
    
    def get_device(self) -> str:
        """Get the configured device.
        
        Returns:
            Device string (e.g., "cuda:0")
        """
        return self.device
    
    def is_initialized(self) -> bool:
        """Check if processor is initialized.
        
        Returns:
            True if initialized
        """
        return self._initialized


def generate_output_filename(input_filename: str, prefix: str = "initial_") -> str:
    """Generate output filename following naming convention.
    
    Args:
        input_filename: Original input filename (with or without extension)
        prefix: Prefix for output file (default: "initial_")
        
    Returns:
        Output filename following convention: {prefix}{filename}.md
        
    Requirements: 2.3, 3.3
    """
    # Get stem (filename without extension)
    stem = Path(input_filename).stem
    return f"{prefix}{stem}.md"


def collect_image_paths_from_markdown(markdown_content: str) -> List[str]:
    """Collect all image paths from markdown content.
    
    Extracts paths from markdown image syntax: ![caption](path)
    
    Args:
        markdown_content: Markdown text to scan
        
    Returns:
        List of image paths found in the markdown
        
    Requirements: 2.4, 4.2
    """
    # Pattern to match markdown image syntax: ![alt](path)
    pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
    
    matches = pattern.findall(markdown_content)
    
    # Filter out URLs (keep only local paths)
    local_paths = []
    for path in matches:
        # Skip URLs
        if path.startswith('http://') or path.startswith('https://'):
            continue
        # Skip data URIs
        if path.startswith('data:'):
            continue
        local_paths.append(path)
    
    return local_paths

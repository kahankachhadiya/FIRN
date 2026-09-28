"""Batch file classifier for multi-modal pipeline optimization.

This module classifies files into processing groups for batch processing,
enabling efficient GPU memory usage by loading each model once for all
compatible inputs.

Processing Groups:
- Documents: PDF, DOCX, PPTX, XLSX, HTML (text-based)
- Audio: MP3, WAV, M4A, FLAC
- Standalone Images: JPG, PNG, JPEG, WEBP
- OCR PDFs: PDF files that are primarily scanned images (future enhancement)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum


class ProcessingCategory(Enum):
    """Categories for batch processing."""
    DOCUMENT = "document"
    AUDIO = "audio"
    IMAGE = "image"


# Supported extensions by category
DOCUMENT_EXTENSIONS: Set[str] = {'.pdf', '.docx', '.pptx', '.xlsx', '.html'}
AUDIO_EXTENSIONS: Set[str] = {'.mp3', '.wav', '.m4a', '.flac'}
IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.png', '.jpeg', '.webp'}


@dataclass
class ImageInfo:
    """Information about an image to be captioned."""
    path: str
    source: str  # "extracted", "standalone", "ocr_pdf"
    source_file: Optional[str] = None  # Original file this image came from
    markdown_file: Optional[str] = None  # .md file to update with caption
    page_number: Optional[int] = None


@dataclass
class ClassifiedBatch:
    """All files classified into processing groups for batch processing."""
    documents: List[str] = field(default_factory=list)  # PDF, DOCX, PPTX, XLSX, HTML
    ocr_pdfs: List[str] = field(default_factory=list)  # PDF files requiring OCR
    audio_files: List[str] = field(default_factory=list)  # MP3, WAV, M4A, FLAC
    standalone_images: List[str] = field(default_factory=list)  # JPG, PNG, JPEG, WEBP
    
    # Populated after Phase 1 (Docling processing)
    extracted_images: List[ImageInfo] = field(default_factory=list)
    markdown_files: List[str] = field(default_factory=list)
    
    @property
    def all_images(self) -> List[ImageInfo]:
        """All images for Phase 3 captioning."""
        result = list(self.extracted_images)
        result.extend([
            ImageInfo(path=p, source="standalone") for p in self.standalone_images
        ])
        result.extend([
            ImageInfo(path=p, source="ocr_pdf") for p in self.ocr_pdfs
        ])
        return result
    
    @property
    def all_markdown_files(self) -> List[str]:
        """All markdown files for Phase 4 summarization."""
        return list(self.markdown_files)
    
    @property
    def total_files(self) -> int:
        """Total number of input files."""
        return (
            len(self.documents) +
            len(self.ocr_pdfs) +
            len(self.audio_files) +
            len(self.standalone_images)
        )
    
    def is_empty(self) -> bool:
        """Check if batch has no files."""
        return self.total_files == 0


def classify_file_extension(extension: str) -> Optional[ProcessingCategory]:
    """
    Classify a file extension into a processing category.
    
    Args:
        extension: File extension (with or without leading dot)
        
    Returns:
        ProcessingCategory or None if extension is not supported
    """
    # Normalize extension
    ext = extension.lower()
    if not ext.startswith('.'):
        ext = '.' + ext
    
    if ext in DOCUMENT_EXTENSIONS:
        return ProcessingCategory.DOCUMENT
    elif ext in AUDIO_EXTENSIONS:
        return ProcessingCategory.AUDIO
    elif ext in IMAGE_EXTENSIONS:
        return ProcessingCategory.IMAGE
    
    return None


def get_category_for_extension(extension: str) -> str:
    """
    Get the category name for a file extension.
    
    Args:
        extension: File extension (with or without leading dot)
        
    Returns:
        Category name string ('document', 'audio', 'image') or 'unknown'
    """
    category = classify_file_extension(extension)
    if category is None:
        return 'unknown'
    return category.value


class BatchClassifier:
    """Classifies files into processing groups for batch processing."""
    
    def __init__(self):
        """Initialize the batch classifier."""
        self._document_extensions = DOCUMENT_EXTENSIONS
        self._audio_extensions = AUDIO_EXTENSIONS
        self._image_extensions = IMAGE_EXTENSIONS
    
    def classify_file(self, filepath: str) -> Optional[ProcessingCategory]:
        """
        Classify a single file by its extension.
        
        Args:
            filepath: Path to the file
            
        Returns:
            ProcessingCategory or None if not supported
        """
        ext = Path(filepath).suffix.lower()
        return classify_file_extension(ext)
    
    def classify_all_files(self, input_paths: List[str]) -> ClassifiedBatch:
        """
        Classify all input files into processing groups.
        
        Args:
            input_paths: List of file paths to classify
            
        Returns:
            ClassifiedBatch with files grouped by processing type
        """
        batch = ClassifiedBatch()
        
        for filepath in input_paths:
            if not os.path.isfile(filepath):
                continue
            
            category = self.classify_file(filepath)
            
            if category == ProcessingCategory.DOCUMENT:
                batch.documents.append(filepath)
            elif category == ProcessingCategory.AUDIO:
                batch.audio_files.append(filepath)
            elif category == ProcessingCategory.IMAGE:
                batch.standalone_images.append(filepath)
        
        return batch
    
    def scan_directory(self, directory: str) -> ClassifiedBatch:
        """
        Scan a directory and classify all supported files.
        
        Args:
            directory: Path to directory to scan
            
        Returns:
            ClassifiedBatch with files grouped by processing type
        """
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return ClassifiedBatch()
        
        # Collect all files
        all_files = []
        for filepath in dir_path.iterdir():
            if filepath.is_file():
                all_files.append(str(filepath.resolve()))
        
        return self.classify_all_files(all_files)
    
    def get_supported_extensions(self) -> Set[str]:
        """Get all supported file extensions."""
        return (
            self._document_extensions |
            self._audio_extensions |
            self._image_extensions
        )
    
    def is_supported_extension(self, extension: str) -> bool:
        """Check if an extension is supported."""
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        return ext in self.get_supported_extensions()

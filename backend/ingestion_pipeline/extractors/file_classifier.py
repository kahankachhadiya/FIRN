"""File classifier and router for multi-modal pipeline.

This module handles file type detection and routing to appropriate
processing pipelines (Document, Audio, Image).
"""

import mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

from extractors.config import (
    DOCUMENT_EXTENSIONS,
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    EXTENSION_TO_DIR,
)


class FileType(Enum):
    """Enumeration of supported file types."""
    DOCUMENT = "document"
    AUDIO = "audio"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedFile:
    """Result of file classification."""
    filepath: str
    filename: str
    extension: str
    file_type: FileType
    mime_type: Optional[str]
    archive_dir_type: str  # e.g., 'PDF', 'WORD', 'AUDIO', 'IMAGES'
    
    @property
    def output_folder_name(self) -> str:
        """Get collision-safe output folder name: {filename}_{ext}."""
        ext_clean = self.extension.lower().lstrip('.')
        return f"{self.filename}_{ext_clean}"


class FileClassifier:
    """Classifies files by type and routes to appropriate pipeline."""
    
    def __init__(self):
        """Initialize the file classifier."""
        # Initialize mimetypes
        mimetypes.init()
        
        # MIME type to file type mapping
        self._mime_to_type = {
            # Documents
            'application/pdf': FileType.DOCUMENT,
            'application/msword': FileType.DOCUMENT,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCUMENT,
            'application/vnd.ms-powerpoint': FileType.DOCUMENT,
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': FileType.DOCUMENT,
            'application/vnd.ms-excel': FileType.DOCUMENT,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': FileType.DOCUMENT,
            'text/csv': FileType.DOCUMENT,
            'text/html': FileType.DOCUMENT,
            # Audio
            'audio/mpeg': FileType.AUDIO,
            'audio/mp3': FileType.AUDIO,
            'audio/wav': FileType.AUDIO,
            'audio/x-wav': FileType.AUDIO,
            'audio/mp4': FileType.AUDIO,
            'audio/x-m4a': FileType.AUDIO,
            'audio/flac': FileType.AUDIO,
            # Images
            'image/jpeg': FileType.IMAGE,
            'image/png': FileType.IMAGE,
            'image/webp': FileType.IMAGE,
        }
    
    def classify(self, filepath: str) -> ClassifiedFile:
        """
        Classify a file by its extension and MIME type.
        
        Args:
            filepath: Path to the file
            
        Returns:
            ClassifiedFile with type information
        """
        path = Path(filepath)
        filename = path.stem
        extension = path.suffix.lower()
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(filepath)
        
        # Determine file type (extension takes priority)
        file_type = self._get_file_type_by_extension(extension)
        
        # Fall back to MIME type if extension unknown
        if file_type == FileType.UNKNOWN and mime_type:
            file_type = self._mime_to_type.get(mime_type, FileType.UNKNOWN)
        
        # Get archive directory type
        archive_dir_type = EXTENSION_TO_DIR.get(extension, 'OTHER')
        
        return ClassifiedFile(
            filepath=str(path.resolve()),
            filename=filename,
            extension=extension,
            file_type=file_type,
            mime_type=mime_type,
            archive_dir_type=archive_dir_type,
        )
    
    def _get_file_type_by_extension(self, extension: str) -> FileType:
        """Get file type from extension."""
        ext = extension.lower()
        if ext in DOCUMENT_EXTENSIONS:
            return FileType.DOCUMENT
        elif ext in AUDIO_EXTENSIONS:
            return FileType.AUDIO
        elif ext in IMAGE_EXTENSIONS:
            return FileType.IMAGE
        return FileType.UNKNOWN
    
    def scan_directory(self, directory: str) -> List[ClassifiedFile]:
        """
        Scan a directory and classify all supported files.
        
        Args:
            directory: Path to directory to scan
            
        Returns:
            List of ClassifiedFile objects for supported files
        """
        classified_files = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return classified_files
        
        # Get all supported extensions
        all_extensions = DOCUMENT_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS
        
        for filepath in dir_path.iterdir():
            if filepath.is_file() and filepath.suffix.lower() in all_extensions:
                classified = self.classify(str(filepath))
                if classified.file_type != FileType.UNKNOWN:
                    classified_files.append(classified)
        
        return classified_files
    
    def group_by_type(self, files: List[ClassifiedFile]) -> dict:
        """
        Group classified files by their type.
        
        Args:
            files: List of ClassifiedFile objects
            
        Returns:
            Dictionary with FileType keys and lists of files
        """
        grouped = {
            FileType.DOCUMENT: [],
            FileType.AUDIO: [],
            FileType.IMAGE: [],
        }
        
        for f in files:
            if f.file_type in grouped:
                grouped[f.file_type].append(f)
        
        return grouped
    
    def get_routing_info(self, classified_file: ClassifiedFile) -> Tuple[str, str]:
        """
        Get routing information for a classified file.
        
        Args:
            classified_file: The classified file
            
        Returns:
            Tuple of (pipeline_name, archive_subdir)
        """
        pipeline_map = {
            FileType.DOCUMENT: "document_pipeline",
            FileType.AUDIO: "audio_pipeline",
            FileType.IMAGE: "image_pipeline",
        }
        
        pipeline = pipeline_map.get(classified_file.file_type, "unknown")
        return pipeline, classified_file.archive_dir_type

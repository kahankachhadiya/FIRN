"""Standalone image processing handler for multi-modal pipeline.

This module handles standalone image analysis (JPG, PNG, JPEG, WEBP) using Qwen
for OCR and visual description, then proceeds through the unified chunking/metadata flow.

Implements Requirements:
- 9.1: Send image to Qwen for OCR and visual analysis
- 9.2: Generate markdown with extracted text and description
- 9.3: Copy original to figures/figure_1.png
- 9.4: Proceed to unified chunking/metadata flow
- 9.5: Use forensic visual analysis prompt for detailed descriptions
"""

import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

from external_clients.openai_compatible_client import OpenAICompatibleClient as LMStudioClient
from extractors.config import PromptsConfig


@dataclass
class ImageMetadata:
    """Metadata for an image file."""
    filename: str
    width: int
    height: int
    format: str
    file_size_bytes: int
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


class ImageHandler:
    """
    Handles standalone image processing for the unified pipeline.
    
    Implements Requirements:
    - 9.1: Send image to Qwen for OCR and visual analysis
    - 9.2: Generate initial_{filename}.md with OCR + description
    - 9.3: Copy original to figures/figure_1.png
    - 9.4: Proceed to unified chunking/metadata flow
    - 9.5: Use forensic visual analysis prompt
    """
    
    def __init__(
        self,
        lm_client: LMStudioClient,
        prompts_config: PromptsConfig,
    ):
        """
        Initialize the image handler.
        
        Args:
            lm_client: LM Studio client for Qwen Vision API
            prompts_config: Prompts configuration (includes image_analysis_prompt)
        """
        self.lm_client = lm_client
        self.prompts = prompts_config
        
        print("ImageHandler: Initialized")
    
    def process_image(
        self,
        image_path: str,
        output_dir: str,
    ) -> str:
        """
        Process standalone image file to markdown.
        
        Implements Requirements:
        - 9.1: Send image to Qwen for OCR and visual analysis
        - 9.2: Generate markdown with extracted text and description
        - 9.3: Copy original to figures/figure_1.png
        
        Args:
            image_path: Path to image file (JPG, PNG, JPEG, WEBP)
            output_dir: Directory for outputs
            
        Returns:
            Markdown content with image analysis
        """
        path = Path(image_path)
        filename = path.stem
        
        # Create output directory and figures subdirectory
        os.makedirs(output_dir, exist_ok=True)
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        print(f"ImageHandler: Processing {path.name}...")
        
        # Get image metadata
        metadata = self._get_image_metadata(image_path)
        
        # Copy original image to figures/figure_1.png (Requirement 9.3)
        dest_ext = path.suffix  # Keep original extension
        figure_path = os.path.join(figures_dir, f"figure_1{dest_ext}")
        shutil.copy2(image_path, figure_path)
        print(f"ImageHandler: Copied image to {figure_path}")
        
        # Analyze image with Qwen (Requirements 9.1, 9.5)
        analysis = self._analyze_with_qwen(image_path)
        
        # Generate markdown (Requirement 9.2)
        markdown = self._generate_markdown(
            filename=filename,
            analysis=analysis,
            metadata=metadata,
            figure_rel_path=os.path.abspath(figure_path).replace("\\", "/")
        )
        
        # Save initial markdown
        initial_md_path = os.path.join(output_dir, f"initial_{filename}.md")
        with open(initial_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        print(f"ImageHandler: Saved initial markdown to {initial_md_path}")
        
        return markdown
    
    def _get_image_metadata(self, image_path: str) -> ImageMetadata:
        """
        Get image metadata using PIL.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ImageMetadata with dimensions and format info
        """
        path = Path(image_path)
        
        try:
            with Image.open(image_path) as img:
                return ImageMetadata(
                    filename=path.name,
                    width=img.width,
                    height=img.height,
                    format=img.format or path.suffix.upper().lstrip('.'),
                    file_size_bytes=os.path.getsize(image_path),
                )
        except Exception as e:
            print(f"ImageHandler: Could not read image metadata: {e}")
            # Return minimal metadata
            return ImageMetadata(
                filename=path.name,
                width=0,
                height=0,
                format=path.suffix.upper().lstrip('.'),
                file_size_bytes=os.path.getsize(image_path),
            )
    
    def _analyze_with_qwen(self, image_path: str) -> str:
        """
        Analyze image using Qwen with forensic visual analysis prompt.
        
        Implements Requirements:
        - 9.1: Send image to Qwen for OCR and visual analysis
        - 9.5: Use forensic visual analysis prompt
        
        Args:
            image_path: Path to image file
            
        Returns:
            Analysis text from Qwen (OCR + visual description)
        """
        try:
            # Use forensic visual analysis prompt from config (Requirement 9.5)
            analysis = self.lm_client.analyze_image(
                image_path,
                self.prompts.image_analysis_prompt
            )
            
            if analysis:
                print(f"ImageHandler: Received analysis from Qwen ({len(analysis)} chars)")
                return analysis
            else:
                raise RuntimeError(
                    f"ImageHandler: Empty analysis returned from Qwen for image: {image_path}. "
                    "Check LM Studio connectivity and model availability."
                )
                
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"ImageHandler: Error analyzing image with Qwen: {e}"
            ) from e
    
    def _generate_markdown(
        self,
        filename: str,
        analysis: str,
        metadata: ImageMetadata,
        figure_rel_path: str,
    ) -> str:
        """
        Generate markdown with image analysis.
        
        Implements Requirement 9.2: Generate markdown with extracted text and description
        
        Args:
            filename: Base filename
            analysis: Analysis text from Qwen
            metadata: Image metadata
            figure_rel_path: Relative path to figure
            
        Returns:
            Markdown content
        """
        lines = []
        lines.append(f"# Image Analysis: {filename}")
        lines.append("")
        
        # Add image reference with absolute path
        lines.append(f"![{filename}]({figure_rel_path})")
        lines.append("")
        
        # Add metadata section
        lines.append("## Image Information")
        lines.append("")
        if metadata.width > 0 and metadata.height > 0:
            lines.append(f"- **Resolution:** {metadata.width} x {metadata.height}")
        lines.append(f"- **Format:** {metadata.format}")
        lines.append(f"- **File Size:** {ImageMetadata._format_size(metadata.file_size_bytes)}")
        lines.append("")
        
        # Add visual analysis section
        lines.append("## Visual Analysis")
        lines.append("")
        lines.append(analysis)
        lines.append("")
        
        return "\n".join(lines)

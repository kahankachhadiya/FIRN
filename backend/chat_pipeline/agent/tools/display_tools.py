"""
Display tools for images and documents.

These tools handle displaying images and documents that were used in answer generation,
implementing selective asset display as per requirement 10.4.
"""

import os
from typing import Dict, Any
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def handle_display_image(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Display an image that was used in generating the answer.
    
    This tool only shows assets that were actually used in answer generation,
    not all assets from retrieved chunks.
    
    Args:
        args: Dictionary containing:
            - file_path: Path to the image file to display
            - caption: Optional caption for the image
    
    Returns:
        Dictionary with image display information
        
    **Validates: Requirements 10.2, 10.4**
    """
    try:
        file_path = args.get("file_path")
        caption = args.get("caption", "")
        
        if not file_path:
            return {
                "success": False,
                "error": "file_path is required"
            }
        
        # Validate file path exists and is an image
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"Image file not found: {file_path}"
            }
        
        # Check if it's an image file
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'}
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in image_extensions:
            return {
                "success": False,
                "error": f"File is not a supported image format: {file_ext}"
            }
        
        logger.info(f"Displaying image: {file_path}")
        
        return {
            "success": True,
            "type": "image_display",
            "file_path": file_path,
            "caption": caption,
            "message": f"Displaying image: {os.path.basename(file_path)}"
        }
        
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to display image: {str(e)}"
        }


def handle_display_document(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Display a document that was used in generating the answer.
    
    This tool only shows assets that were actually used in answer generation,
    not all assets from retrieved chunks.
    
    Args:
        args: Dictionary containing:
            - file_path: Path to the document file to display
            - page_number: Optional specific page number to display
            - highlight_text: Optional text to highlight in the document
    
    Returns:
        Dictionary with document display information
        
    **Validates: Requirements 10.3, 10.4**
    """
    try:
        file_path = args.get("file_path")
        page_number = args.get("page_number")
        highlight_text = args.get("highlight_text", "")
        
        if not file_path:
            return {
                "success": False,
                "error": "file_path is required"
            }
        
        # Validate file path exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"Document file not found: {file_path}"
            }
        
        # Check if it's a document file
        document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md', '.html', '.rtf'}
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in document_extensions:
            return {
                "success": False,
                "error": f"File is not a supported document format: {file_ext}"
            }
        
        logger.info(f"Displaying document: {file_path}")
        
        result = {
            "success": True,
            "type": "document_display",
            "file_path": file_path,
            "highlight_text": highlight_text,
            "message": f"Displaying document: {os.path.basename(file_path)}"
        }
        
        if page_number is not None:
            result["page_number"] = page_number
            result["message"] += f" (page {page_number})"
        
        return result
        
    except Exception as e:
        logger.error(f"Error displaying document: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to display document: {str(e)}"
        }
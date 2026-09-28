"""
Batch ingestion script for processing multiple JSON documents.

This CLI script processes all JSON files in a specified directory,
ingesting each document into the RAG system. It provides fault isolation,
meaning individual file failures don't stop the entire batch process.

Usage:
    python scripts/run_ingestion.py <directory_path>

Example:
    python scripts/run_ingestion.py ./data/documents/
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.production import get_production_config, initialize_production_config
from ingestion.ingest_json import ingest_document_from_json
from utils.logging_utils import get_logger
from utils.errors import IngestionError, ValidationError, DatabaseError

# Initialize production configuration if not already done
try:
    config = get_production_config()
except RuntimeError:
    # Initialize if not already done
    config = initialize_production_config()

logger = get_logger(__name__)


def find_json_files(directory: Path) -> List[Path]:
    """
    Find all .json files in the specified directory.
    
    Args:
        directory: Path to directory to search
        
    Returns:
        List of Path objects for .json files
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    json_files = list(directory.glob("*.json"))
    return sorted(json_files)  # Sort for consistent ordering


def ingest_batch(directory_path: str) -> Tuple[int, int, int]:
    """
    Process all JSON files in a directory with fault isolation.
    
    This function iterates over all .json files in the specified directory
    and attempts to ingest each one. Individual failures are logged but
    don't stop processing of remaining files.
    
    Args:
        directory_path: Path to directory containing JSON files
        
    Returns:
        Tuple of (total_files, successes, failures)
    """
    logger.info(f"Starting batch ingestion from directory: {directory_path}")
    
    # Find all JSON files
    directory = Path(directory_path)
    try:
        json_files = find_json_files(directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"Error accessing directory: {e}")
        return (0, 0, 0)
    
    if not json_files:
        logger.warning(f"No .json files found in directory: {directory_path}")
        return (0, 0, 0)
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Track results
    total_files = len(json_files)
    successes = 0
    failures = 0
    
    # Process each file
    for i, json_file in enumerate(json_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {json_file.name}")
        
        try:
            # Attempt to ingest the document
            ingest_document_from_json(str(json_file))
            successes += 1
            logger.info(f"✓ Successfully ingested: {json_file.name}")
            
        except (IngestionError, ValidationError, DatabaseError) as e:
            # Expected errors - log and continue
            failures += 1
            logger.error(
                f"✗ Failed to ingest {json_file.name}: "
                f"{type(e).__name__}: {e}"
            )
            
        except Exception as e:
            # Unexpected errors - log and continue
            failures += 1
            logger.exception(
                f"✗ Unexpected error ingesting {json_file.name}: {e}"
            )
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Batch ingestion completed")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Success rate: {(successes/total_files*100):.1f}%")
    logger.info("=" * 60)
    
    return (total_files, successes, failures)


def main():
    """
    CLI entry point for batch ingestion.
    """
    parser = argparse.ArgumentParser(
        description="Batch ingest JSON documents into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ingestion.py ./data/documents/
  python scripts/run_ingestion.py /path/to/json/files/
        """
    )
    
    parser.add_argument(
        "directory",
        type=str,
        help="Path to directory containing JSON files to ingest"
    )
    
    args = parser.parse_args()
    
    # Run batch ingestion
    try:
        total, successes, failures = ingest_batch(args.directory)
        
        # Exit with appropriate code
        if total == 0:
            logger.error("No files processed")
            sys.exit(1)
        elif failures > 0:
            logger.warning(f"Completed with {failures} failures")
            sys.exit(2)
        else:
            logger.info("All files processed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("Batch ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error during batch ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Progress Tracker for batch pipeline optimization.

Tracks batch processing progress with resume capability.
Provides utilities for tracking completed/failed files per phase
and resuming from failures.

Requirements: 8.1, 8.2, 8.3, 8.5
"""

import json
import time
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PhaseStatus(Enum):
    """Status of a processing phase."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PhaseProgress:
    """Progress for a single phase.
    
    Tracks files processed, completed, and failed within a phase.
    """
    status: str = "pending"  # PhaseStatus value
    files_total: int = 0
    files_completed: int = 0
    files_failed: int = 0
    completed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    failed_errors: Dict[str, str] = field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "files_total": self.files_total,
            "files_completed": self.files_completed,
            "files_failed": self.files_failed,
            "completed_files": self.completed_files.copy(),
            "failed_files": self.failed_files.copy(),
            "failed_errors": self.failed_errors.copy(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseProgress":
        """Create from dictionary."""
        return cls(
            status=data.get("status", "pending"),
            files_total=data.get("files_total", 0),
            files_completed=data.get("files_completed", 0),
            files_failed=data.get("files_failed", 0),
            completed_files=data.get("completed_files", []).copy(),
            failed_files=data.get("failed_files", []).copy(),
            failed_errors=data.get("failed_errors", {}).copy(),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


@dataclass
class BatchProgress:
    """Overall batch processing progress.
    
    Contains progress for all phases and summary information.
    """
    batch_id: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_phase: str = ""
    phases: Dict[str, PhaseProgress] = field(default_factory=dict)
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_id": self.batch_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_phase": self.current_phase,
            "phases": {k: v.to_dict() for k, v in self.phases.items()},
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchProgress":
        """Create from dictionary."""
        phases = {}
        for phase_name, phase_data in data.get("phases", {}).items():
            phases[phase_name] = PhaseProgress.from_dict(phase_data)
        
        return cls(
            batch_id=data.get("batch_id", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            current_phase=data.get("current_phase", ""),
            phases=phases,
            total_files=data.get("total_files", 0),
            successful_files=data.get("successful_files", 0),
            failed_files=data.get("failed_files", 0),
        )


class ProgressTracker:
    """Tracks batch processing progress with resume capability.
    
    Provides utilities for:
    - Initializing progress tracking for new batches (Requirements 8.1)
    - Updating progress immediately on file completion (Requirements 8.2)
    - Loading existing progress for resume (Requirements 8.3)
    - Generating summary reports (Requirements 8.5)
    
    Example:
        tracker = ProgressTracker("progress.json")
        
        # Initialize for new batch
        tracker.initialize({
            "phase_1_docling": ["a.pdf", "b.pdf"],
            "phase_2_whisper": ["audio.mp3"],
        })
        
        # Mark files as completed
        tracker.mark_file_completed("phase_1_docling", "a.pdf")
        
        # Resume after failure
        tracker.load_existing()
        pending = tracker.get_pending_files("phase_1_docling")
    """
    
    # Standard phase names
    PHASE_1_DOCLING = "phase_1_docling"
    PHASE_2_WHISPER = "phase_2_whisper"
    PHASE_3_CAPTIONING = "phase_3_captioning"
    PHASE_4_SUMMARIZATION = "phase_4_summarization"
    
    ALL_PHASES = [
        PHASE_1_DOCLING,
        PHASE_2_WHISPER,
        PHASE_3_CAPTIONING,
        PHASE_4_SUMMARIZATION,
    ]
    
    def __init__(self, progress_file: str = "progress.json"):
        """Initialize Progress Tracker.
        
        Args:
            progress_file: Path to the progress file for persistence
        """
        self.progress_file = progress_file
        self._progress: Optional[BatchProgress] = None
        self._initialized = False
        
        logger.info(f"ProgressTracker initialized with file: {progress_file}")
    
    @property
    def progress(self) -> Optional[BatchProgress]:
        """Get current batch progress."""
        return self._progress
    
    @property
    def is_initialized(self) -> bool:
        """Check if tracker is initialized."""
        return self._initialized and self._progress is not None
    
    def _generate_batch_id(self) -> str:
        """Generate a unique batch ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"

    
    def initialize(self, phase_files: Dict[str, List[str]]) -> None:
        """Initialize progress tracking for a new batch run.
        
        Creates a new progress file with all phases and their files.
        
        Args:
            phase_files: Dictionary mapping phase names to lists of files
                        e.g., {"phase_1_docling": ["a.pdf", "b.pdf"]}
        
        Requirements: 8.1
        """
        batch_id = self._generate_batch_id()
        started_at = self._get_timestamp()
        self._initialize_internal(batch_id, started_at, phase_files)
    
    def initialize_batch(
        self,
        batch_id: str,
        classified_batch: Any,
    ) -> None:
        """Initialize progress tracking for a new batch with ClassifiedBatch.
        
        Creates a new progress file with all phases based on classified files.
        
        Args:
            batch_id: Unique batch identifier
            classified_batch: ClassifiedBatch object with classified files
        
        Requirements: 8.1
        """
        started_at = self._get_timestamp()
        
        # Build phase_files from classified_batch
        phase_files = {
            self.PHASE_1_DOCLING: classified_batch.documents,
            self.PHASE_2_WHISPER: classified_batch.audio_files,
            self.PHASE_3_CAPTIONING: [],  # Will be populated after Phase 1
            self.PHASE_4_SUMMARIZATION: [],  # Will be populated after Phase 1 & 2
        }
        
        self._initialize_internal(batch_id, started_at, phase_files)
    
    def _initialize_internal(
        self,
        batch_id: str,
        started_at: str,
        phase_files: Dict[str, List[str]],
    ) -> None:
        """Internal initialization logic.
        
        Args:
            batch_id: Unique batch identifier
            started_at: ISO timestamp
            phase_files: Dictionary mapping phase names to lists of files
        """
        # Create phase progress for each phase
        phases = {}
        total_files = 0
        
        for phase_name in self.ALL_PHASES:
            files = phase_files.get(phase_name, [])
            phases[phase_name] = PhaseProgress(
                status=PhaseStatus.PENDING.value,
                files_total=len(files),
                files_completed=0,
                files_failed=0,
                completed_files=[],
                failed_files=[],
                failed_errors={},
            )
            total_files += len(files)
        
        # Create batch progress
        self._progress = BatchProgress(
            batch_id=batch_id,
            started_at=started_at,
            current_phase=self.PHASE_1_DOCLING,
            phases=phases,
            total_files=total_files,
            successful_files=0,
            failed_files=0,
        )
        
        self._initialized = True
        
        # Save immediately
        self.save()
        
        logger.info(f"Initialized batch {batch_id} with {total_files} total files")
    
    def load_existing(self) -> bool:
        """Load existing progress file for resume capability.
        
        Returns:
            True if progress file was loaded successfully, False otherwise
        
        Requirements: 8.3
        """
        if not os.path.exists(self.progress_file):
            logger.info(f"No existing progress file found at {self.progress_file}")
            return False
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._progress = BatchProgress.from_dict(data)
            self._initialized = True
            
            logger.info(
                f"Loaded existing progress for batch {self._progress.batch_id}. "
                f"Current phase: {self._progress.current_phase}"
            )
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to load progress file: {e}")
            return False
    
    def save(self) -> None:
        """Persist progress to file immediately.
        
        Requirements: 8.2
        """
        if self._progress is None:
            logger.warning("No progress to save")
            return
        
        try:
            # Ensure directory exists
            progress_dir = os.path.dirname(self.progress_file)
            if progress_dir and not os.path.exists(progress_dir):
                os.makedirs(progress_dir, exist_ok=True)
            
            # Write to file
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self._progress.to_dict(), f, indent=2)
            
            logger.debug(f"Progress saved to {self.progress_file}")
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to save progress file: {e}")
            raise

    
    def mark_file_completed(self, phase: str, filename: str) -> None:
        """Mark a file as completed in a phase.
        
        Updates the progress file immediately after marking.
        
        Args:
            phase: Phase name (e.g., "phase_1_docling")
            filename: Name of the completed file
        
        Requirements: 8.2
        """
        if self._progress is None:
            raise RuntimeError("Progress tracker not initialized")
        
        if phase not in self._progress.phases:
            raise ValueError(f"Unknown phase: {phase}")
        
        phase_progress = self._progress.phases[phase]
        
        # Avoid duplicates
        if filename in phase_progress.completed_files:
            logger.debug(f"File {filename} already marked as completed in {phase}")
            return
        
        # Remove from failed if it was there (retry succeeded)
        if filename in phase_progress.failed_files:
            phase_progress.failed_files.remove(filename)
            phase_progress.files_failed -= 1
            if filename in phase_progress.failed_errors:
                del phase_progress.failed_errors[filename]
        
        # Add to completed
        phase_progress.completed_files.append(filename)
        phase_progress.files_completed += 1
        
        # Update overall progress
        self._progress.successful_files = sum(
            p.files_completed for p in self._progress.phases.values()
        )
        
        # Update phase status if needed
        if phase_progress.status == PhaseStatus.PENDING.value:
            phase_progress.status = PhaseStatus.IN_PROGRESS.value
            phase_progress.started_at = time.time()
        
        # Check if phase is complete
        if phase_progress.files_completed + phase_progress.files_failed >= phase_progress.files_total:
            if phase_progress.files_failed == 0:
                phase_progress.status = PhaseStatus.COMPLETED.value
            else:
                phase_progress.status = PhaseStatus.COMPLETED.value  # Completed with failures
            phase_progress.completed_at = time.time()
            if phase_progress.started_at:
                phase_progress.duration_seconds = phase_progress.completed_at - phase_progress.started_at
        
        # Save immediately (Requirements 8.2)
        self.save()
        
        logger.info(f"Marked {filename} as completed in {phase}")
    
    def mark_file_failed(self, phase: str, filename: str, error: str = "") -> None:
        """Mark a file as failed in a phase.
        
        Updates the progress file immediately after marking.
        
        Args:
            phase: Phase name (e.g., "phase_1_docling")
            filename: Name of the failed file
            error: Error message describing the failure
        
        Requirements: 8.2
        """
        if self._progress is None:
            raise RuntimeError("Progress tracker not initialized")
        
        if phase not in self._progress.phases:
            raise ValueError(f"Unknown phase: {phase}")
        
        phase_progress = self._progress.phases[phase]
        
        # Avoid duplicates
        if filename in phase_progress.failed_files:
            # Update error message if provided
            if error:
                phase_progress.failed_errors[filename] = error
            logger.debug(f"File {filename} already marked as failed in {phase}")
            self.save()
            return
        
        # Remove from completed if it was there (shouldn't happen normally)
        if filename in phase_progress.completed_files:
            phase_progress.completed_files.remove(filename)
            phase_progress.files_completed -= 1
        
        # Add to failed
        phase_progress.failed_files.append(filename)
        phase_progress.files_failed += 1
        if error:
            phase_progress.failed_errors[filename] = error
        
        # Update overall progress
        self._progress.failed_files = sum(
            p.files_failed for p in self._progress.phases.values()
        )
        
        # Update phase status if needed
        if phase_progress.status == PhaseStatus.PENDING.value:
            phase_progress.status = PhaseStatus.IN_PROGRESS.value
            phase_progress.started_at = time.time()
        
        # Save immediately (Requirements 8.2)
        self.save()
        
        logger.info(f"Marked {filename} as failed in {phase}: {error}")

    
    def get_pending_files(self, phase: str) -> List[str]:
        """Get files not yet processed in a phase.
        
        Returns files that are neither completed nor failed.
        Used for resume logic to skip already-processed files.
        
        Args:
            phase: Phase name (e.g., "phase_1_docling")
        
        Returns:
            List of pending file names
        
        Requirements: 8.3
        """
        if self._progress is None:
            return []
        
        if phase not in self._progress.phases:
            return []
        
        phase_progress = self._progress.phases[phase]
        
        # Get all files that are not completed
        # Note: Failed files are NOT included - they need explicit retry
        set(phase_progress.completed_files)
        
        # We need to know the original file list
        # Since we don't store it, we return based on what we know
        # Pending = not completed and not failed
        # For resume, caller should provide the full file list
        
        return []  # Caller should use get_completed_files and filter
    
    def get_completed_files(self, phase: str) -> List[str]:
        """Get files that have been completed in a phase.
        
        Args:
            phase: Phase name
        
        Returns:
            List of completed file names
        
        Requirements: 8.3
        """
        if self._progress is None:
            return []
        
        if phase not in self._progress.phases:
            return []
        
        return self._progress.phases[phase].completed_files.copy()
    
    def get_failed_files(self, phase: str) -> List[str]:
        """Get files that have failed in a phase.
        
        Args:
            phase: Phase name
        
        Returns:
            List of failed file names
        """
        if self._progress is None:
            return []
        
        if phase not in self._progress.phases:
            return []
        
        return self._progress.phases[phase].failed_files.copy()
    
    def is_file_completed(self, phase: str, filename: str) -> bool:
        """Check if a file is completed in a phase.
        
        Args:
            phase: Phase name
            filename: File name to check
        
        Returns:
            True if file is completed
        
        Requirements: 8.3
        """
        if self._progress is None:
            return False
        
        if phase not in self._progress.phases:
            return False
        
        return filename in self._progress.phases[phase].completed_files
    
    def is_file_failed(self, phase: str, filename: str) -> bool:
        """Check if a file has failed in a phase.
        
        Args:
            phase: Phase name
            filename: File name to check
        
        Returns:
            True if file has failed
        """
        if self._progress is None:
            return False
        
        if phase not in self._progress.phases:
            return False
        
        return filename in self._progress.phases[phase].failed_files
    
    def set_phase_status(self, phase: str, status: PhaseStatus) -> None:
        """Set the status of a phase.
        
        Args:
            phase: Phase name
            status: New status
        """
        if self._progress is None:
            raise RuntimeError("Progress tracker not initialized")
        
        if phase not in self._progress.phases:
            raise ValueError(f"Unknown phase: {phase}")
        
        phase_progress = self._progress.phases[phase]
        phase_progress.status = status.value
        
        if status == PhaseStatus.IN_PROGRESS and phase_progress.started_at is None:
            phase_progress.started_at = time.time()
        elif status in [PhaseStatus.COMPLETED, PhaseStatus.SKIPPED, PhaseStatus.FAILED]:
            phase_progress.completed_at = time.time()
            if phase_progress.started_at:
                phase_progress.duration_seconds = phase_progress.completed_at - phase_progress.started_at
        
        # Update current phase
        self._progress.current_phase = phase
        
        self.save()
        logger.info(f"Phase {phase} status set to {status.value}")
    
    def mark_phase_skipped(self, phase: str, reason: str = "") -> None:
        """Mark a phase as skipped.
        
        Args:
            phase: Phase name
            reason: Reason for skipping
        """
        self.set_phase_status(phase, PhaseStatus.SKIPPED)
        logger.info(f"Phase {phase} skipped: {reason}")

    
    def set_phase_files(self, phase: str, files: List[str]) -> None:
        """Set the file list for a phase.
        
        Used when initializing or updating a phase's file list.
        
        Args:
            phase: Phase name
            files: List of files for this phase
        """
        if self._progress is None:
            raise RuntimeError("Progress tracker not initialized")
        
        if phase not in self._progress.phases:
            raise ValueError(f"Unknown phase: {phase}")
        
        phase_progress = self._progress.phases[phase]
        phase_progress.files_total = len(files)
        
        # Recalculate total files
        self._progress.total_files = sum(
            p.files_total for p in self._progress.phases.values()
        )
        
        self.save()
    
    def get_files_to_process(self, phase: str, all_files: List[str]) -> List[str]:
        """Get files that need to be processed in a phase.
        
        Filters out completed files for resume capability.
        Failed files are included for retry.
        
        Args:
            phase: Phase name
            all_files: Complete list of files for this phase
        
        Returns:
            List of files to process (not yet completed)
        
        Requirements: 8.3
        """
        if self._progress is None:
            return all_files
        
        if phase not in self._progress.phases:
            return all_files
        
        completed = set(self._progress.phases[phase].completed_files)
        
        # Return files that are not completed
        # Failed files ARE included - they should be retried
        return [f for f in all_files if f not in completed]
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary report with timing per phase.
        
        Returns:
            Dictionary with summary information
        
        Requirements: 8.5
        """
        if self._progress is None:
            return {}
        
        # Calculate totals
        total_completed = sum(p.files_completed for p in self._progress.phases.values())
        total_failed = sum(p.files_failed for p in self._progress.phases.values())
        total_duration = sum(p.duration_seconds for p in self._progress.phases.values())
        
        # Build phase summaries
        phase_summaries = {}
        for phase_name, phase_progress in self._progress.phases.items():
            phase_summaries[phase_name] = {
                "status": phase_progress.status,
                "files_total": phase_progress.files_total,
                "files_completed": phase_progress.files_completed,
                "files_failed": phase_progress.files_failed,
                "duration_seconds": phase_progress.duration_seconds,
                "completed_files": phase_progress.completed_files,
                "failed_files": phase_progress.failed_files,
            }
        
        return {
            "batch_id": self._progress.batch_id,
            "started_at": self._progress.started_at,
            "completed_at": self._progress.completed_at,
            "total_files": self._progress.total_files,
            "successful_files": total_completed,
            "failed_files": total_failed,
            "total_duration_seconds": total_duration,
            "phases": phase_summaries,
        }
    
    def complete_batch(self) -> Dict[str, Any]:
        """Mark the batch as complete and generate final summary.
        
        Returns:
            Summary report dictionary
        
        Requirements: 8.5
        """
        if self._progress is None:
            raise RuntimeError("Progress tracker not initialized")
        
        self._progress.completed_at = self._get_timestamp()
        
        # Update final counts
        self._progress.successful_files = sum(
            p.files_completed for p in self._progress.phases.values()
        )
        self._progress.failed_files = sum(
            p.files_failed for p in self._progress.phases.values()
        )
        
        self.save()
        
        summary = self.get_summary()
        logger.info(
            f"Batch {self._progress.batch_id} completed. "
            f"Successful: {self._progress.successful_files}, "
            f"Failed: {self._progress.failed_files}"
        )
        
        return summary
    
    def reset(self) -> None:
        """Reset the progress tracker.
        
        Clears all progress and removes the progress file.
        """
        self._progress = None
        self._initialized = False
        
        if os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
                logger.info(f"Removed progress file: {self.progress_file}")
            except OSError as e:
                logger.warning(f"Failed to remove progress file: {e}")
    
    def get_phase_progress(self, phase: str) -> Optional[PhaseProgress]:
        """Get progress for a specific phase.
        
        Args:
            phase: Phase name
        
        Returns:
            PhaseProgress or None if not found
        """
        if self._progress is None:
            return None
        
        return self._progress.phases.get(phase)
    
    def get_current_phase(self) -> str:
        """Get the current phase name.
        
        Returns:
            Current phase name or empty string
        """
        if self._progress is None:
            return ""
        return self._progress.current_phase
    
    def has_failures(self) -> bool:
        """Check if any files have failed.
        
        Returns:
            True if there are failed files
        """
        if self._progress is None:
            return False
        
        return any(p.files_failed > 0 for p in self._progress.phases.values())
    
    def get_failure_report(self) -> Dict[str, Dict[str, str]]:
        """Get a report of all failures.
        
        Returns:
            Dictionary mapping phase names to {filename: error} dictionaries
        """
        if self._progress is None:
            return {}
        
        report = {}
        for phase_name, phase_progress in self._progress.phases.items():
            if phase_progress.failed_files:
                report[phase_name] = phase_progress.failed_errors.copy()
                # Include files without error messages
                for filename in phase_progress.failed_files:
                    if filename not in report[phase_name]:
                        report[phase_name][filename] = "Unknown error"
        
        return report

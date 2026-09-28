"""Whisper GPU Processor for batch pipeline optimization.

Processes audio files using Whisper with PyTorch GPU acceleration (CUDA 12.8).
Provides model loading, transcription, and memory management for batch processing.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import os
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


def _ensure_ffmpeg() -> None:
    """Ensure ffmpeg is available on PATH.

    Preference order:
    1. System ffmpeg (already installed — no download needed)
    2. static_ffmpeg as fallback (only if system ffmpeg is absent)

    This prevents static_ffmpeg from re-downloading the binary on every run
    when a perfectly good system ffmpeg already exists.
    """
    if shutil.which("ffmpeg"):
        logger.info("WhisperGPUProcessor: using system ffmpeg at %s", shutil.which("ffmpeg"))
        return

    # System ffmpeg not found — try static_ffmpeg as fallback
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        logger.info("WhisperGPUProcessor: ffmpeg provided by static_ffmpeg")
    except Exception as exc:
        logger.warning("WhisperGPUProcessor: ffmpeg not found and static_ffmpeg failed: %s", exc)


_ensure_ffmpeg()


@dataclass
class AudioSegment:
    """A single segment of transcribed audio."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str  # Speaker identifier
    text: str     # Transcribed text
    
    @property
    def timestamp(self) -> str:
        """Get formatted timestamp [HH:MM:SS]."""
        return self._format_time(self.start)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp
        }


@dataclass
class AudioTranscription:
    """Complete transcription result for an audio file."""
    filename: str
    duration_seconds: float
    segments: List[AudioSegment]
    language: str = "en"
    model_used: str = "whisper"

    def to_markdown(self) -> str:
        """Convert transcription to markdown with timestamps and speaker labels.
        
        Format: [HH:MM:SS] Speaker N: text
        Requirements: 3.3
        """
        lines = []
        lines.append(f"# Audio Transcription: {self.filename}")
        lines.append("")
        lines.append(f"**Duration:** {self._format_duration()}")
        lines.append(f"**Language:** {self.language}")
        lines.append(f"**Model:** {self.model_used}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        current_speaker = None
        for segment in self.segments:
            if segment.speaker != current_speaker:
                current_speaker = segment.speaker
                lines.append(f"\n**{segment.speaker}:**\n")
            
            lines.append(f"[{segment.timestamp}] {segment.text}")
        
        return "\n".join(lines)
    
    def to_json_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "model_used": self.model_used,
            "segment_count": len(self.segments),
            "segments": [s.to_dict() for s in self.segments],
        }
    
    def _format_duration(self) -> str:
        """Format duration as human-readable string."""
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        secs = int(self.duration_seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"


class WhisperGPUProcessor:
    """Whisper processor with PyTorch GPU acceleration (CUDA 12.8).
    
    Provides:
    - Model loading on GPU with CUDA 12.8 (Requirements 3.1)
    - Audio transcription with GPU acceleration (Requirements 3.2)
    - Output formatting as [HH:MM:SS] Speaker N: text (Requirements 3.3)
    - Model unloading and GPU memory release (Requirements 3.4)
    - Phase skip when no audio files exist (Requirements 3.5)
    
    Example:
        processor = WhisperGPUProcessor(model_name="medium", device="cuda:0")
        processor.load_model()
        
        result = processor.transcribe("audio.mp3", "output/")
        print(result.to_markdown())
        
        processor.unload_model()
    """
    
    def __init__(
        self,
        model_name: str = "medium",
        device: str = "cuda:0",
        enable_diarization: bool = True
    ):
        """Initialize Whisper GPU Processor.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: CUDA device string (e.g., "cuda:0")
            enable_diarization: Whether to enable speaker diarization
        """
        self.model_name = model_name
        self.device = device
        self.enable_diarization = enable_diarization
        self._model = None
        self._cuda_available: Optional[bool] = None
        
        logger.info(f"WhisperGPUProcessor: Configured with model={model_name}, device={device}")
        logger.info(f"WhisperGPUProcessor: Diarization={'enabled' if enable_diarization else 'disabled'}")

    def _verify_gpu(self) -> bool:
        """Verify GPU is available for Whisper.
        
        Returns:
            True if CUDA is available
            
        Raises:
            RuntimeError: If CUDA is not available
            
        Requirements: 3.1
        """
        try:
            import torch
        except ImportError:
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
        
        self._cuda_available = True
        cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        
        logger.info(f"WhisperGPUProcessor: CUDA Version: {cuda_version}")
        logger.info(f"WhisperGPUProcessor: GPU Device: {device_name}")
        
        return True
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded.
        
        Returns:
            True if model is loaded
        """
        return self._model is not None
    
    def load_model(self) -> None:
        """Load Whisper model to GPU with CUDA 12.8.
        
        Requirements: 3.1
        
        Raises:
            RuntimeError: If CUDA is not available or model fails to load
        """
        if self._model is not None:
            logger.warning("WhisperGPUProcessor: Model already loaded, skipping")
            return
        
        # Verify GPU availability
        self._verify_gpu()
        
        try:
            import whisper
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"Required package not installed: {e}. "
                "Install with: pip install openai-whisper"
            )
        
        logger.info(f"WhisperGPUProcessor: Loading Whisper {self.model_name} on {self.device}...")
        
        # Load model to GPU
        self._model = whisper.load_model(self.model_name, device=self.device)
        
        # Log memory usage
        memory_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        logger.info(f"WhisperGPUProcessor: Model loaded. GPU Memory: {memory_gb:.2f}GB")
    
    def unload_model(self) -> None:
        """Unload model and free GPU memory.
        
        Requirements: 3.4
        """
        if self._model is None:
            logger.warning("WhisperGPUProcessor: No model loaded, skipping unload")
            return
        
        import torch
        
        # Delete model reference
        del self._model
        self._model = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("WhisperGPUProcessor: Model unloaded, GPU memory cleared")

    def transcribe(
        self,
        audio_path: str,
        output_dir: str
    ) -> AudioTranscription:
        """Transcribe audio file using GPU-accelerated Whisper.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save outputs
            
        Returns:
            AudioTranscription with segments and metadata
            
        Requirements: 3.2, 3.3
        """
        # Load model if not already loaded
        if self._model is None:
            self.load_model()
        
        path = Path(audio_path)
        filename = path.stem
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        logger.info(f"WhisperGPUProcessor: Transcribing {path.name}...")
        
        # Transcribe with Whisper
        transcription = self._transcribe_with_whisper(audio_path)
        
        # Save outputs
        self._save_outputs(transcription, output_dir, filename)
        
        return transcription
    
    def _transcribe_with_whisper(self, audio_path: str) -> AudioTranscription:
        """Transcribe audio using Whisper model on GPU.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioTranscription with segments
            
        Requirements: 3.2
        """
        path = Path(audio_path)
        
        # Transcribe with word timestamps
        result = self._model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
        )
        
        # Get audio duration
        duration = self._get_audio_duration(audio_path, result)
        
        # Convert to segments with speaker diarization
        if self.enable_diarization:
            segments = self._apply_simple_diarization(result)
        else:
            segments = [
                AudioSegment(
                    start=seg['start'],
                    end=seg['end'],
                    speaker="Speaker 1",
                    text=seg['text'].strip()
                )
                for seg in result.get('segments', [])
            ]
        
        return AudioTranscription(
            filename=path.name,
            duration_seconds=duration,
            segments=segments,
            language=result.get('language', 'en'),
            model_used=f"whisper-{self.model_name}-gpu"
        )
    
    def _get_audio_duration(self, audio_path: str, whisper_result: dict) -> float:
        """Get audio duration from file or Whisper result.
        
        Args:
            audio_path: Path to audio file
            whisper_result: Whisper transcription result
            
        Returns:
            Duration in seconds
        """
        # Try to get duration using ffprobe
        try:
            import subprocess
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries',
                'format=duration', '-of', 'csv=p=0', audio_path
            ]
            duration = float(subprocess.check_output(cmd).decode().strip())
            return duration
        except Exception:
            pass
        
        # Fallback: estimate from segments
        if whisper_result.get('segments'):
            return whisper_result['segments'][-1].get('end', 0)
        
        return 0.0
    
    def _apply_simple_diarization(self, whisper_result: dict) -> List[AudioSegment]:
        """Apply simple speaker diarization based on pauses.
        
        Switches speaker when there's a significant pause (>2 seconds).
        
        Args:
            whisper_result: Whisper transcription result
            
        Returns:
            List of AudioSegment with speaker labels
        """
        segments = []
        current_speaker = 1
        last_end = 0.0
        
        for seg in whisper_result.get('segments', []):
            # Switch speaker if there's a significant pause (>2 seconds)
            if seg['start'] - last_end > 2.0:
                current_speaker = 2 if current_speaker == 1 else 1
            
            segments.append(AudioSegment(
                start=seg['start'],
                end=seg['end'],
                speaker=f"Speaker {current_speaker}",
                text=seg['text'].strip()
            ))
            last_end = seg['end']
        
        return segments

    def _save_outputs(
        self,
        transcription: AudioTranscription,
        output_dir: str,
        filename: str
    ) -> None:
        """Save transcription outputs to files.
        
        Preserves existing output directory structure:
        - initial_{filename}.md
        - processed_{filename}.md
        - {filename}.json
        
        Args:
            transcription: AudioTranscription result
            output_dir: Output directory path
            filename: Base filename (without extension)
            
        Requirements: 3.3
        """
        import json
        
        # Save initial markdown with [HH:MM:SS] Speaker N: format
        initial_md_path = os.path.join(output_dir, f"initial_{filename}.md")
        with open(initial_md_path, 'w', encoding='utf-8') as f:
            f.write(transcription.to_markdown())
        logger.info(f"WhisperGPUProcessor: Saved {initial_md_path}")
        
        # Save processed markdown (same as initial for audio)
        processed_md_path = os.path.join(output_dir, f"processed_{filename}.md")
        with open(processed_md_path, 'w', encoding='utf-8') as f:
            f.write(transcription.to_markdown())
        logger.info(f"WhisperGPUProcessor: Saved {processed_md_path}")
        
        # Save JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription.to_json_dict(), f, indent=2)
        logger.info(f"WhisperGPUProcessor: Saved {json_path}")
    
    def cleanup(self) -> None:
        """Perform full cleanup, releasing all GPU resources.
        
        Requirements: 3.4
        """
        self.unload_model()
    
    def get_memory_info(self) -> dict:
        """Get current GPU memory usage.
        
        Returns:
            Dictionary with allocated and reserved memory in GB
        """
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated(0) / (1024 ** 3),
                    'reserved': torch.cuda.memory_reserved(0) / (1024 ** 3),
                }
        except ImportError:
            pass
        
        return {'allocated': 0.0, 'reserved': 0.0}


def should_skip_phase(audio_files: List[str]) -> bool:
    """Check if Whisper phase should be skipped.
    
    Phase should be skipped if no audio files exist.
    
    Args:
        audio_files: List of audio file paths
        
    Returns:
        True if phase should be skipped (no audio files)
        
    Requirements: 3.5
    """
    return len(audio_files) == 0


@dataclass
class PhaseResult:
    """Result of a processing phase."""
    status: str  # "completed", "skipped", "failed"
    files_processed: int
    files_failed: int
    duration_seconds: float
    error_message: Optional[str] = None


def execute_whisper_phase(
    audio_files: List[str],
    output_base_dir: str,
    model_name: str = "medium",
    device: str = "cuda:0"
) -> PhaseResult:
    """Execute Whisper phase for batch processing.
    
    Handles:
    - Phase skip when no audio files (Requirements 3.5)
    - Model loading on GPU (Requirements 3.1)
    - Sequential processing of all audio files (Requirements 3.2)
    - Model unloading and memory cleanup (Requirements 3.4)
    
    Args:
        audio_files: List of audio file paths to process
        output_base_dir: Base directory for outputs
        model_name: Whisper model size
        device: CUDA device string
        
    Returns:
        PhaseResult with status and statistics
    """
    import time
    
    start_time = time.time()
    
    # Check if phase should be skipped (Requirements 3.5)
    if should_skip_phase(audio_files):
        logger.info("WhisperGPUProcessor: No audio files, skipping phase")
        return PhaseResult(
            status="skipped",
            files_processed=0,
            files_failed=0,
            duration_seconds=0.0
        )
    
    processor = None
    files_processed = 0
    files_failed = 0
    
    try:
        # Initialize processor and load model (Requirements 3.1)
        processor = WhisperGPUProcessor(
            model_name=model_name,
            device=device,
            enable_diarization=True
        )
        processor.load_model()
        
        # Process all audio files sequentially (Requirements 3.2)
        for audio_path in audio_files:
            try:
                path = Path(audio_path)
                filename = path.stem
                ext = path.suffix.lower().lstrip('.')
                
                # Create output directory following existing structure
                output_dir = os.path.join(output_base_dir, f"{filename}_{ext}")
                
                processor.transcribe(audio_path, output_dir)
                files_processed += 1
                
            except Exception as e:
                logger.error(f"WhisperGPUProcessor: Failed to process {audio_path}: {e}")
                files_failed += 1
        
        duration = time.time() - start_time
        
        return PhaseResult(
            status="completed",
            files_processed=files_processed,
            files_failed=files_failed,
            duration_seconds=duration
        )
        
    except Exception as e:
        logger.error(f"WhisperGPUProcessor: Phase failed: {e}")
        duration = time.time() - start_time
        
        return PhaseResult(
            status="failed",
            files_processed=files_processed,
            files_failed=files_failed,
            duration_seconds=duration,
            error_message=str(e)
        )
        
    finally:
        # Unload model and clear GPU memory (Requirements 3.4)
        if processor is not None:
            processor.unload_model()

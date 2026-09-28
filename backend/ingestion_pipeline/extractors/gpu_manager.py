"""GPU Memory Manager for batch pipeline optimization.

Manages GPU memory for PyTorch models with CUDA 12.8 support.
Provides utilities for memory tracking, cache clearing, and resource management.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryInfo:
    """Information about GPU memory usage."""
    allocated: float  # GB
    reserved: float   # GB
    max_allocated: float  # GB
    free: float  # GB (estimated)
    total: float  # GB


class GPUMemoryManager:
    """Manages GPU memory for PyTorch models.
    
    Provides utilities for:
    - Verifying CUDA availability (Requirements 6.1)
    - Clearing GPU cache between phases (Requirements 6.2, 6.3)
    - Tracking memory usage (Requirements 6.4)
    - Ensuring cleanup on errors (Requirements 6.5)
    
    Example:
        gpu_manager = GPUMemoryManager()
        
        # Check memory before processing
        info = gpu_manager.get_memory_info()
        print(f"Allocated: {info.allocated:.2f} GB")
        
        # After processing phase, clear cache
        gpu_manager.clear_cache()
    """
    
    def __init__(self, device: str = "cuda:0"):
        """Initialize GPU Memory Manager.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
        """
        self.device = device
        self._cuda_available: Optional[bool] = None
        self._device_name: Optional[str] = None
        self._cuda_version: Optional[str] = None
        
    def _verify_cuda_available(self) -> bool:
        """Verify CUDA is available and accessible.
        
        Returns:
            True if CUDA is available
            
        Raises:
            RuntimeError: If CUDA is not available
            
        Requirements: 6.1
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
        
        # Cache device info
        self._cuda_available = True
        self._cuda_version = torch.version.cuda
        
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
        
        self._device_name = torch.cuda.get_device_name(device_idx)
        
        logger.info(f"CUDA Version: {self._cuda_version}")
        logger.info(f"GPU Device: {self._device_name}")
        logger.info(f"Device Count: {torch.cuda.device_count()}")
        
        return True
    
    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available (cached).
        
        Returns:
            True if CUDA is available
        """
        if self._cuda_available is None:
            try:
                self._verify_cuda_available()
            except RuntimeError:
                self._cuda_available = False
        return self._cuda_available
    
    @property
    def cuda_version(self) -> Optional[str]:
        """Get CUDA version string.
        
        Returns:
            CUDA version string or None if not available
        """
        if self._cuda_version is None and self.is_cuda_available:
            import torch
            self._cuda_version = torch.version.cuda
        return self._cuda_version
    
    @property
    def device_name(self) -> Optional[str]:
        """Get GPU device name.
        
        Returns:
            Device name string or None if not available
        """
        if self._device_name is None and self.is_cuda_available:
            import torch
            device_idx = self._get_device_index()
            self._device_name = torch.cuda.get_device_name(device_idx)
        return self._device_name
    
    def _get_device_index(self) -> int:
        """Extract device index from device string.
        
        Returns:
            Device index (0 if not specified)
        """
        if ":" in self.device:
            try:
                return int(self.device.split(":")[1])
            except (ValueError, IndexError):
                pass
        return 0
    
    def clear_cache(self) -> None:
        """Clear CUDA cache to free GPU memory.
        
        Calls torch.cuda.empty_cache() and synchronizes to ensure
        memory is actually freed before continuing.
        
        Requirements: 6.2, 6.3
        """
        if not self.is_cuda_available:
            logger.warning("CUDA not available, skipping cache clear")
            return
        
        import torch
        
        # Get memory before clearing for logging
        before_allocated = torch.cuda.memory_allocated(self._get_device_index())
        
        # Clear cache and synchronize
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get memory after clearing
        after_allocated = torch.cuda.memory_allocated(self._get_device_index())
        
        freed_mb = (before_allocated - after_allocated) / (1024 ** 2)
        logger.info(f"GPU cache cleared. Freed: {freed_mb:.2f} MB")
    
    def get_memory_info(self) -> GPUMemoryInfo:
        """Get current GPU memory usage information.
        
        Returns:
            GPUMemoryInfo with allocated, reserved, max_allocated, free, and total memory in GB
            
        Requirements: 6.4
        """
        if not self.is_cuda_available:
            return GPUMemoryInfo(
                allocated=0.0,
                reserved=0.0,
                max_allocated=0.0,
                free=0.0,
                total=0.0
            )
        
        import torch
        
        device_idx = self._get_device_index()
        
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 3)
        
        # Get total memory
        total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        total = total_bytes / (1024 ** 3)
        
        # Estimate free memory (total - reserved)
        free = total - reserved
        
        return GPUMemoryInfo(
            allocated=allocated,
            reserved=reserved,
            max_allocated=max_allocated,
            free=free,
            total=total
        )
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics for monitoring.
        
        Useful for tracking peak memory usage per phase.
        
        Requirements: 6.4
        """
        if not self.is_cuda_available:
            logger.warning("CUDA not available, skipping peak stats reset")
            return
        
        import torch
        
        torch.cuda.reset_peak_memory_stats(self._get_device_index())
        logger.info("GPU peak memory stats reset")
    
    def log_memory_status(self, phase_name: str = "") -> None:
        """Log current memory status for debugging.
        
        Args:
            phase_name: Optional name of current phase for context
        """
        info = self.get_memory_info()
        
        prefix = f"[{phase_name}] " if phase_name else ""
        logger.info(
            f"{prefix}GPU Memory - "
            f"Allocated: {info.allocated:.2f}GB, "
            f"Reserved: {info.reserved:.2f}GB, "
            f"Peak: {info.max_allocated:.2f}GB, "
            f"Free: {info.free:.2f}GB"
        )
    
    def ensure_memory_available(self, required_gb: float) -> bool:
        """Check if sufficient GPU memory is available.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if sufficient memory is available
            
        Requirements: 6.4
        """
        info = self.get_memory_info()
        
        if info.free < required_gb:
            logger.warning(
                f"Insufficient GPU memory. Required: {required_gb:.2f}GB, "
                f"Available: {info.free:.2f}GB"
            )
            return False
        
        return True
    
    def cleanup(self) -> None:
        """Perform full GPU cleanup.
        
        Clears cache and resets peak stats. Call this when
        exiting or on error to ensure resources are released.
        
        Requirements: 6.5
        """
        if not self.is_cuda_available:
            return
        
        import torch
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset stats
        torch.cuda.reset_peak_memory_stats(self._get_device_index())
        
        logger.info("GPU cleanup completed")

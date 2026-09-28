"""Configuration module for the multi-modal data ingestion pipeline.

Unified Pipeline Flow:
1. ALL file types → Convert to initial markdown (.md)
2. Check if markdown contains image paths → If yes, transcribe images with Qwen
3. Chunk the markdown (250 word limit per chunk) with Qwen
4. Extract metadata with Qwen
5. Generate final structured JSON

Model Configuration:
- Qwen3 VL 30B A3B Instruct (GGUF Q4_K_M) via LM Studio for all LLM tasks
- Whisper on Intel NPU via OpenVINO for audio transcription
"""

import os
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from the ingestion pipeline's .env file
_pipeline_env = Path(__file__).parent.parent / ".env"
load_dotenv(_pipeline_env, override=True)


# File type mappings for routing
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv', '.html', '.htm'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Directory type mappings
EXTENSION_TO_DIR = {
    '.pdf': 'PDF',
    '.docx': 'WORD', '.doc': 'WORD',
    '.pptx': 'PPT', '.ppt': 'PPT',
    '.xlsx': 'EXCEL', '.xls': 'EXCEL', '.csv': 'EXCEL',
    '.html': 'HTML', '.htm': 'HTML',
    '.mp3': 'AUDIO', '.wav': 'AUDIO', '.m4a': 'AUDIO', '.flac': 'AUDIO',
    '.jpg': 'IMAGES', '.jpeg': 'IMAGES', '.png': 'IMAGES', '.webp': 'IMAGES',
}

# Source type mappings for JSON output
EXTENSION_TO_SOURCE_TYPE = {
    '.pdf': 'pdf',
    '.docx': 'docx', '.doc': 'docx',
    '.pptx': 'pptx', '.ppt': 'pptx',
    '.xlsx': 'xlsx', '.xls': 'xlsx', '.csv': 'xlsx',
    '.html': 'web', '.htm': 'web',
    '.mp3': 'audio', '.wav': 'audio', '.m4a': 'audio', '.flac': 'audio',
    '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.webp': 'image',
}


@dataclass
class ModelConfig:
    """Configuration for AI models used in the pipeline.

    Reads vision model settings from the LLM_VISION_* environment variables.
    All four vision fields are REQUIRED — missing any one raises EnvironmentError
    at startup so the pipeline never silently falls back to None.
    """

    # Vision / image model — populated from LLM_VISION_* env vars
    lm_studio_image_url: str = field(default="")
    lm_studio_api_key: str = field(default="")
    lm_studio_image_model: str = field(default="")
    lm_studio_max_tokens: int = field(default=2048)
    lm_studio_temperature: float = field(default=0.1)

    # Metadata extraction model — populated from LLM_METADATA_* env vars
    metadata_base_url: str = field(default="")
    metadata_api_key: str = field(default="")
    metadata_model: str = field(default="")
    metadata_max_tokens: int = field(default=1000)
    metadata_temperature: float = field(default=0.1)

    # Whisper configuration
    whisper_model_name: str = field(default="large")
    whisper_device: str = field(default="cuda")

    # Legacy compatibility aliases (set in __post_init__)
    lm_studio_endpoint: str = field(default="", init=False)
    vision_model_name: str = field(default="", init=False)
    vision_api_base: str = field(default="", init=False)

    def __post_init__(self):
        """Read and validate LLM_VISION_* environment variables.

        Raises:
            EnvironmentError: If any required LLM_VISION_* variable is missing or empty.
        """
        required = {
            "LLM_VISION_BASE_URL": "lm_studio_image_url",
            "LLM_VISION_API_KEY": "lm_studio_api_key",
            "LLM_VISION_MODEL": "lm_studio_image_model",
        }
        missing = []
        for env_var, attr in required.items():
            value = os.environ.get(env_var, "").strip()
            if not value:
                missing.append(env_var)
            else:
                setattr(self, attr, value)

        if missing:
            raise EnvironmentError(
                f"Document processor requires the following environment variables "
                f"but they are missing or empty: {', '.join(missing)}. "
                f"Ensure LLM_VISION_BASE_URL, LLM_VISION_API_KEY, and LLM_VISION_MODEL "
                f"are set in your .env.production file."
            )

        # Optional numeric overrides
        max_tokens_str = os.environ.get("LLM_VISION_MAX_TOKENS", "").strip()
        if max_tokens_str:
            try:
                self.lm_studio_max_tokens = int(max_tokens_str)
            except ValueError:
                raise EnvironmentError(f"LLM_VISION_MAX_TOKENS must be an integer, got: {max_tokens_str!r}")

        temp_str = os.environ.get("LLM_VISION_TEMPERATURE", "").strip()
        if temp_str:
            try:
                self.lm_studio_temperature = float(temp_str)
            except ValueError:
                raise EnvironmentError(f"LLM_VISION_TEMPERATURE must be a float, got: {temp_str!r}")

        # Strip trailing /chat/completions from base URL if present — the client
        # appends the path itself.
        base = self.lm_studio_image_url
        for suffix in ("/chat/completions", "/v1/chat/completions"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        self.lm_studio_image_url = base.rstrip("/")

        # Legacy compatibility aliases
        self.lm_studio_endpoint = self.lm_studio_image_url
        self.vision_model_name = self.lm_studio_image_model
        self.vision_api_base = self.lm_studio_image_url

        print(f"Vision endpoint : {self.lm_studio_image_url}")
        print(f"Vision model    : {self.lm_studio_image_model}")
        print(f"Max tokens      : {self.lm_studio_max_tokens}")

        # ── Metadata extraction model (LLM_METADATA_*) ───────────────────────
        # Required: BASE_URL and MODEL. API_KEY is optional (empty = local provider).
        meta_required = {
            "LLM_METADATA_BASE_URL": "metadata_base_url",
            "LLM_METADATA_MODEL": "metadata_model",
        }
        meta_missing = []
        for env_var, attr in meta_required.items():
            value = os.environ.get(env_var, "").strip()
            if not value:
                meta_missing.append(env_var)
            else:
                setattr(self, attr, value)

        if meta_missing:
            raise EnvironmentError(
                f"Document processor requires the following environment variables "
                f"but they are missing or empty: {', '.join(meta_missing)}. "
                f"Ensure LLM_METADATA_BASE_URL and LLM_METADATA_MODEL are set in your .env file."
            )

        self.metadata_api_key = os.environ.get("LLM_METADATA_API_KEY", "").strip()

        meta_tokens_str = os.environ.get("LLM_METADATA_MAX_TOKENS", "").strip()
        if meta_tokens_str:
            try:
                self.metadata_max_tokens = int(meta_tokens_str)
            except ValueError:
                raise EnvironmentError(f"LLM_METADATA_MAX_TOKENS must be an integer, got: {meta_tokens_str!r}")

        meta_temp_str = os.environ.get("LLM_METADATA_TEMPERATURE", "").strip()
        if meta_temp_str:
            try:
                self.metadata_temperature = float(meta_temp_str)
            except ValueError:
                raise EnvironmentError(f"LLM_METADATA_TEMPERATURE must be a float, got: {meta_temp_str!r}")

        # Normalise metadata base URL
        meta_base = self.metadata_base_url
        for suffix in ("/chat/completions", "/v1/chat/completions"):
            if meta_base.endswith(suffix):
                meta_base = meta_base[: -len(suffix)]
                break
        self.metadata_base_url = meta_base.rstrip("/")

        print(f"Metadata endpoint: {self.metadata_base_url}")
        print(f"Metadata model   : {self.metadata_model}")
        print(f"Temperature     : {self.lm_studio_temperature}")


def _load_prompts_from_yaml() -> Dict[str, str]:
    """Load all prompts from the shared prompts.yml file.

    The file is resolved relative to this module's location:
    document_processor/src/ -> ../../config/prompts.yml

    Raises:
        FileNotFoundError: If prompts.yml does not exist.
        KeyError: If a required prompt key is missing from the file.
        yaml.YAMLError: If the file is not valid YAML.
    """
    prompts_path = Path(__file__).parent.parent / "config" / "prompts.yml"
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts configuration file not found: {prompts_path}. "
            "Ensure config/prompts.yml exists in the ingestion_pipeline directory."
        )
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"prompts.yml must be a YAML mapping, got {type(data).__name__}")
    required_keys = {"image_analysis_prompt", "metadata_prompt"}
    missing = required_keys - data.keys()
    if missing:
        raise KeyError(
            f"prompts.yml is missing required prompt keys: {sorted(missing)}. "
            "Add the missing keys to config/prompts.yml."
        )
    return data


@dataclass
class PromptsConfig:
    """Configuration for LLM prompts — loaded exclusively from config/prompts.yml.

    Raises:
        FileNotFoundError: If prompts.yml is missing.
        KeyError: If a required prompt key is absent from the file.
    """

    image_analysis_prompt: str = field(default="", init=False)
    metadata_prompt: str = field(default="", init=False)

    def __post_init__(self) -> None:
        data = _load_prompts_from_yaml()
        self.image_analysis_prompt = data["image_analysis_prompt"].strip()
        self.metadata_prompt = data["metadata_prompt"].strip()


@dataclass
class ProcessingConfig:
    """Configuration for document processing parameters."""
    
    chunk_max_words: int = 250
    gpu_cache_clear_interval: int = 10
    
    # Batch mode settings (Requirements 9.1, 9.2, 10.1)
    enable_batch_mode: bool = True
    batch_retry_failed: bool = True
    batch_max_retries: int = 3
    batch_progress_file: str = "progress.json"
    
    # Legacy compatibility
    chunk_size: int = 512
    overlap: int = 50
    summary_max_words: int = 30
    summary_min_words: int = 15
    vision_min_transcription_words: int = 50
    vision_batch_min: int = 4
    vision_batch_max: int = 8


@dataclass
class BatchConfig:
    """Configuration for batch processing pipeline.
    
    This configuration controls the batch processing mode which optimizes
    GPU memory usage by loading each model once, processing all compatible
    inputs, then releasing GPU memory before loading the next model.
    
    Processing Phases:
    - Phase 1: Docling (PyTorch GPU) - Document conversion
    - Phase 2: Whisper (PyTorch GPU) - Audio transcription  
    - Phase 3: Image Captioning (LM Studio) - Vision model
    - Phase 4: Text Processing (LM Studio) - Language model for chunking and metadata
    
    Requirements: 10.1
    """
    
    # Batch mode settings
    enable_batch_mode: bool = True
    retry_failed_files: bool = True
    max_retries: int = 3
    
    # Progress tracking
    progress_file: str = "progress.json"
    
    # GPU device configuration for CUDA 12.8
    # Install PyTorch nightly: pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    cuda_device: str = "cuda:0"
    clear_cache_between_phases: bool = True
    
    # LM Studio configuration
    lm_studio_endpoint: str = "http://127.0.0.1:1234"
    
    # LM Studio model name (will be overridden from config)
    captioning_model: str = "google/gemma-3n-e4b"  # Image captioning and text processing
    
    # Output directory settings
    output_base_dir: str = "Database/Processed_Docs"
    
    def __post_init__(self):
        """Validate and set up batch configuration."""
        # Allow environment variable overrides
        env_endpoint = os.environ.get("LM_STUDIO_ENDPOINT")
        if env_endpoint:
            self.lm_studio_endpoint = env_endpoint
        
        env_cuda_device = os.environ.get("CUDA_DEVICE")
        if env_cuda_device:
            self.cuda_device = env_cuda_device
        
        env_progress_file = os.environ.get("BATCH_PROGRESS_FILE")
        if env_progress_file:
            self.progress_file = env_progress_file
    
    def validate_cuda(self) -> bool:
        """Validate CUDA 12.8 availability.
        
        Returns:
            True if CUDA is available, False otherwise.
        """
        if not torch.cuda.is_available():
            return False
        return True
    
    def get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA device information.
        
        Returns:
            Dictionary with CUDA version, device name, and memory info.
        """
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.device_count() > 0 else 0,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3 if torch.cuda.device_count() > 0 else 0,
        }


@dataclass
class DirectoryConfig:
    """Configuration for input/output directories using Database hierarchy.
    
    Directory Structure:
    /Root/
    └── Database/
        ├── Documents/                  <- Raw file archive
        │   ├── PDF/
        │   ├── WORD/
        │   ├── PPT/
        │   ├── EXCEL/
        │   ├── HTML/
        │   ├── AUDIO/
        │   └── IMAGES/
        └── Processed_Docs/             <- Output artifacts
            └── {filename}_{ext}/       <- Unique folder per file
                ├── initial_{filename}.md
                ├── processed_{filename}.md
                ├── {filename}.json
                └── figures/
    """
    
    root_dir: str = "."
    database_dir: str = "Database"
    documents_subdir: str = "Documents"
    processed_subdir: str = "Processed_Docs"
    figures_subdir: str = "figures"
    
    # Absolute paths (set in __post_init__)
    abs_root: str = field(default="", init=False)
    abs_database: str = field(default="", init=False)
    abs_documents: str = field(default="", init=False)
    abs_processed: str = field(default="", init=False)
    output_dir: str = field(default="", init=False)
    
    def __post_init__(self):
        """Convert to absolute paths and create directory structure."""
        self.abs_root = str(Path(self.root_dir).resolve())
        self.abs_database = str(Path(self.abs_root, self.database_dir).resolve())
        self.abs_documents = str(Path(self.abs_database, self.documents_subdir).resolve())
        self.abs_processed = str(Path(self.abs_database, self.processed_subdir).resolve())
        self.output_dir = self.abs_processed
        
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the Database directory hierarchy."""
        for type_dir in ['PDF', 'WORD', 'PPT', 'EXCEL', 'HTML', 'AUDIO', 'IMAGES']:
            os.makedirs(os.path.join(self.abs_documents, type_dir), exist_ok=True)
        os.makedirs(self.abs_processed, exist_ok=True)
    
    def get_archive_dir(self, extension: str) -> str:
        """Get archive directory for a file type."""
        type_dir = EXTENSION_TO_DIR.get(extension.lower(), 'OTHER')
        return os.path.join(self.abs_documents, type_dir)
    
    def get_output_folder_name(self, filename: str, extension: str) -> str:
        """Get output folder name: {filename}_{ext}."""
        ext_clean = extension.lower().lstrip('.')
        return f"{filename}_{ext_clean}"
    
    def get_file_output_dir(self, filename: str, extension: str) -> str:
        """Get output directory for a specific file."""
        folder_name = self.get_output_folder_name(filename, extension)
        return os.path.join(self.abs_processed, folder_name)
    
    def get_figures_dir(self, filename: str, extension: str = "") -> str:
        """Get figures directory for a specific file."""
        if extension:
            output_dir = self.get_file_output_dir(filename, extension)
        else:
            output_dir = os.path.join(self.abs_processed, filename)
        return os.path.join(output_dir, self.figures_subdir)
    
    def get_initial_md_path(self, filename: str, extension: str) -> str:
        """Get path for initial markdown: initial_{filename}.md."""
        output_dir = self.get_file_output_dir(filename, extension)
        return os.path.join(output_dir, f"initial_{filename}.md")
    
    def get_processed_md_path(self, filename: str, extension: str) -> str:
        """Get path for processed markdown: processed_{filename}.md."""
        output_dir = self.get_file_output_dir(filename, extension)
        return os.path.join(output_dir, f"processed_{filename}.md")
    
    def get_json_path(self, filename: str, extension: str) -> str:
        """Get path for JSON output: {filename}.json."""
        output_dir = self.get_file_output_dir(filename, extension)
        return os.path.join(output_dir, f"{filename}.json")
    
    # Legacy compatibility
    def get_pdf_output_dir(self, pdf_name: str) -> str:
        return self.get_file_output_dir(pdf_name, '.pdf')
    
    def get_extracted_md_path(self, pdf_name: str) -> str:
        return self.get_initial_md_path(pdf_name, '.pdf')
    
    def get_transcribed_md_path(self, pdf_name: str) -> str:
        output_dir = self.get_file_output_dir(pdf_name, '.pdf')
        return os.path.join(output_dir, "transcribed_output.md")
    
    def get_enriched_md_path(self, pdf_name: str) -> str:
        return self.get_processed_md_path(pdf_name, '.pdf')
    
    def get_structured_json_path(self, pdf_name: str) -> str:
        return self.get_json_path(pdf_name, '.pdf')


def load_config() -> tuple:
    """Load and validate configuration for the pipeline.
    
    Returns:
        Tuple of (ModelConfig, PromptsConfig, ProcessingConfig, DirectoryConfig)
    """
    model_config = ModelConfig()
    prompts_config = PromptsConfig()
    processing_config = ProcessingConfig()
    directory_config = DirectoryConfig()
    
    return model_config, prompts_config, processing_config, directory_config


def load_batch_config() -> BatchConfig:
    """Load batch processing configuration.
    
    Returns:
        BatchConfig instance with default or environment-overridden values.
    """
    return BatchConfig()


def get_file_type(filepath: str) -> str:
    """Determine file type category from extension."""
    ext = Path(filepath).suffix.lower()
    if ext in DOCUMENT_EXTENSIONS:
        return 'document'
    elif ext in AUDIO_EXTENSIONS:
        return 'audio'
    elif ext in IMAGE_EXTENSIONS:
        return 'image'
    return 'unknown'


def get_source_type(extension: str) -> str:
    """Get source_type for JSON output."""
    return EXTENSION_TO_SOURCE_TYPE.get(extension.lower(), 'unknown')


def print_config_summary(
    model_config: ModelConfig,
    prompts_config: PromptsConfig,
    processing_config: ProcessingConfig,
    directory_config: DirectoryConfig,
    batch_config: Optional[BatchConfig] = None
) -> None:
    """Print a summary of the current configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print("\nModel Configuration:")
    print(f"  LM Studio Endpoint: {model_config.lm_studio_endpoint}")
    print(f"  Whisper Device: {model_config.whisper_device}")
    
    print("\nProcessing Configuration:")
    print(f"  Chunk Max Words: {processing_config.chunk_max_words}")
    print(f"  GPU Cache Clear Interval: {processing_config.gpu_cache_clear_interval}")
    
    if batch_config:
        print("\nBatch Processing Configuration:")
        print(f"  Batch Mode Enabled: {batch_config.enable_batch_mode}")
        print(f"  CUDA Device: {batch_config.cuda_device}")
        print(f"  Progress File: {batch_config.progress_file}")
        print(f"  Retry Failed Files: {batch_config.retry_failed_files}")
        print(f"  Max Retries: {batch_config.max_retries}")
        print(f"  Captioning Model: {batch_config.captioning_model}")
        
        # Print CUDA info if available
        cuda_info = batch_config.get_cuda_info()
        if cuda_info.get("available"):
            print(f"  CUDA Version: {cuda_info.get('cuda_version')}")
            print(f"  GPU Device: {cuda_info.get('device_name')}")
    
    print("\nDirectory Configuration (Absolute Paths):")
    print(f"  Root: {directory_config.abs_root}")
    print(f"  Documents Archive: {directory_config.abs_documents}")
    print(f"  Processed Output: {directory_config.abs_processed}")
    
    print("\nSupported File Types:")
    print(f"  Documents: {', '.join(sorted(DOCUMENT_EXTENSIONS))}")
    print(f"  Audio: {', '.join(sorted(AUDIO_EXTENSIONS))}")
    print(f"  Images: {', '.join(sorted(IMAGE_EXTENSIONS))}")
    
    print("=" * 60 + "\n")

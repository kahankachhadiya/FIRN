#!/usr/bin/env python3
"""
Document Processing Integration Script with GPU Model Preloading

This script integrates the document intelligence pipeline with the RAG system.
It processes various file types (PDF, audio, images, office docs) and converts
them to structured JSON format that can be ingested by the RAG system.

Features:
- Preloads all GPU models at startup (Whisper, Docling)
- Enforces GPU-only usage for maximum performance
- Integrates with RAG system for seamless workflow

Usage:
    python scripts/process_documents.py [input_directory] [output_directory]
    
Example:
    python scripts/process_documents.py raw_documents/ data/
"""

import sys
import shutil
import torch
from pathlib import Path

class GPUModelManager:
    """Manages GPU model preloading and validation."""
    
    def __init__(self):
        self.whisper_model = None
        self.docling_processor = None
        self.models_loaded = False
        
    def validate_gpu_availability(self):
        """Validate GPU availability and enforce GPU-only usage."""
        print("=" * 60)
        print("GPU VALIDATION")
        print("=" * 60)
        
        if not torch.cuda.is_available():
            print("✗ CUDA is not available!")
            print("\nThis script requires GPU acceleration. Please ensure:")
            print("1. NVIDIA GPU is installed and drivers are up to date")
            print("2. CUDA toolkit is installed")
            print("3. PyTorch with CUDA support is installed")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
            
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print("✓ CUDA is available!")
        print(f"✓ GPU Count: {gpu_count}")
        print(f"✓ GPU Name: {gpu_name}")
        print(f"✓ GPU Memory: {memory_total:.1f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        
        # Check minimum memory requirement (8GB recommended)
        if memory_total < 6.0:
            print(f"⚠ Warning: GPU has only {memory_total:.1f} GB memory.")
            print("  Minimum 6GB recommended for optimal performance.")
            print("  Processing may be slower or fail on large files.")
        
        return True
    
    def preload_models(self):
        """Preload all GPU models at startup."""
        print("\n" + "=" * 60)
        print("PRELOADING GPU MODELS")
        print("=" * 60)
        
        try:
            # Import required modules
            from src.config import ModelConfig
            from src.docling_processor import DoclingProcessor
            from src.whisper_npu import WhisperNPU
            
            # Create GPU-enforced configuration
            model_config = ModelConfig()
            model_config.device = "cuda"
            model_config.whisper_device = "cuda"
            
            print("\n[1/2] Loading Docling Processor (GPU)...")
            print("  - Initializing DocumentConverter with CUDA acceleration")
            print("  - This may take 30-60 seconds for first-time setup...")
            
            # Force GPU usage for Docling
            self.docling_processor = DoclingProcessor(device="cuda", force_gpu=True)
            print("  ✓ Docling Processor loaded successfully on GPU")
            
            print("\n[2/2] Loading Whisper Model (GPU)...")
            print("  - Loading Whisper Large model on CUDA")
            print("  - This may take 1-2 minutes for first-time download...")
            
            # Force GPU usage for Whisper
            self.whisper_model = WhisperNPU(
                model_name="large",
                device="cuda",
                enable_diarization=True
            )
            
            # Force model loading now (not lazy)
            self.whisper_model._ensure_model_loaded()
            print("  ✓ Whisper Model loaded successfully on GPU")
            
            # Clear any initialization cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.models_loaded = True
            
            print("\n" + "=" * 60)
            print("✓ ALL GPU MODELS PRELOADED SUCCESSFULLY")
            print("=" * 60)
            
            # Print GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
                print(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to preload GPU models: {e}")
            print("\nPlease ensure:")
            print("1. All dependencies are installed: pip install -r requirements.txt")
            print("2. CUDA-compatible PyTorch is installed")
            print("3. Sufficient GPU memory is available (6GB+ recommended)")
            return False
    
    def get_preloaded_processors(self):
        """Get preloaded processors for use in pipeline."""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call preload_models() first.")
        return self.docling_processor, self.whisper_model

def main():
    """Main entry point for document processing integration."""
    
    # Default directories
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_documents"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    print("=" * 60)
    print("DOCUMENT PROCESSING INTEGRATION (GPU-ACCELERATED)")
    print("=" * 60)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print()
    
    # Initialize GPU model manager
    gpu_manager = GPUModelManager()
    
    # Validate GPU availability
    if not gpu_manager.validate_gpu_availability():
        return 1
    
    # Preload all GPU models
    if not gpu_manager.preload_models():
        return 1
    
    # Check if document processor is available
    try:
        from main import Pipeline
        from src.config import load_config
        print("\n✓ Document processor modules loaded successfully")
    except ImportError as e:
        print(f"\n✗ Failed to load document processor: {e}")
        print("\nPlease ensure:")
        print("1. LM Studio is running with Qwen3 VL model")
        print("2. All dependencies are installed: pip install -r requirements.txt")
        print("3. Document processor is properly configured")
        return 1
    
    # Check input directory
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"\n✗ Input directory not found: {input_path}")
        print(f"Creating directory: {input_path}")
        input_path.mkdir(parents=True, exist_ok=True)
        print("Please place your documents in this directory and run again.")
        return 1
    
    # Check for files
    files = list(input_path.glob("*"))
    files = [f for f in files if f.is_file()]  # Only files, not directories
    if not files:
        print(f"\n✗ No files found in {input_path}")
        print("Please add documents to process.")
        return 1
    
    print(f"\n✓ Found {len(files)} file(s) to process:")
    for file in files:
        print(f"  - {file.name}")
    print()
    
    # Initialize document processor with preloaded models
    try:
        config = load_config()
        model_config, prompts_config, processing_config, directory_config = config
        
        # Force GPU usage in model config
        model_config.device = "cuda"
        model_config.whisper_device = "cuda"
        
        # Get preloaded processors
        docling_processor, whisper_processor = gpu_manager.get_preloaded_processors()
        
        # Create LM Studio client (this doesn't use GPU, it's API-based)
        from external_clients.openai_compatible_client import OpenAICompatibleClient as LMStudioClient
        lm_client = LMStudioClient(
            image_endpoint=model_config.lm_studio_image_url,
            api_key=model_config.lm_studio_api_key,
            image_model=model_config.lm_studio_image_model,
        )
        
        # Test LM Studio connection
        print("\n✓ Testing LM Studio connection...")
        lm_client.test_connection()
        print("✓ LM Studio connection successful")
        
        # Create unified processor with preloaded models
        from src.unified_processor import UnifiedProcessor
        unified_processor = UnifiedProcessor(
            lm_client=lm_client,
            prompts_config=prompts_config,
            directory_config=directory_config,
            model_config=model_config,
            docling_processor=docling_processor,
            whisper_processor=whisper_processor,
        )
        
        # Create a custom pipeline class that uses our preloaded models
        class PreloadedPipeline:
            def __init__(self, unified_processor, directory_config):
                self.unified_processor = unified_processor
                self.directory_config = directory_config
                
            def process_file(self, file_path):
                """Process a single file using preloaded models."""
                return self.unified_processor.process_file(file_path)
        
        pipeline = PreloadedPipeline(unified_processor, directory_config)
        
        print("✓ Document processor initialized with preloaded GPU models")
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Process files
    print("\n" + "=" * 60)
    print("PROCESSING FILES")
    print("=" * 60)
    
    processed_count = 0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(files, 1):
        try:
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            print("-" * 40)
            
            # Process with document processor (using preloaded GPU models)
            result = pipeline.process_file(str(file_path))
            
            if result.success:
                # Copy JSON output to RAG data directory
                json_file = Path(result.json_path)
                if json_file.exists():
                    output_json = output_path / json_file.name
                    shutil.copy2(json_file, output_json)
                    print(f"✓ Processed → {output_json}")
                    processed_count += 1
                    
                    # Print processing stats
                    print(f"  Processing Time: {result.processing_time:.2f}s")
                    if hasattr(result, 'chunks') and result.chunks:
                        print(f"  Chunks Generated: {len(result.chunks)}")
                else:
                    print(f"✗ JSON output not found: {json_file}")
            else:
                print(f"✗ Processing failed: {result.error}")
                
            # Clear GPU cache between files to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {processed_count}/{len(files)} files")
    
    if processed_count > 0:
        print(f"\n✓ JSON files ready for RAG ingestion in: {output_path}")
        print("\nNext steps:")
        print(f"1. Run: python scripts/run_ingestion.py {output_dir}")
        print("2. Start the RAG system: start.bat")
        print("3. Query your documents via the web interface")
        
        # Print final GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print("\nFinal GPU Memory Usage:")
            print(f"  Allocated: {memory_allocated:.2f} GB")
            print(f"  Reserved: {memory_reserved:.2f} GB")
    else:
        print("\n✗ No files were processed successfully.")
        print("Check the error messages above for troubleshooting.")
    
    return 0 if processed_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
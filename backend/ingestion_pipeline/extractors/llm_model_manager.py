"""LM Studio Model Manager for batch pipeline optimization.

Manages model loading/unloading in LM Studio with GPU memory management.
Ensures only one model is loaded at a time and provides retry logic.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import time
import logging
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """State of a model in LM Studio."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelLoadResult:
    """Result of a model load operation."""
    success: bool
    model_name: str
    message: str
    attempts: int = 1
    load_time_seconds: float = 0.0


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    state: ModelState
    loaded_at: Optional[float] = None


class LMStudioModelError(Exception):
    """Base exception for LM Studio model operations."""
    pass


class ModelLoadError(LMStudioModelError):
    """Model failed to load."""
    pass


class ModelUnloadError(LMStudioModelError):
    """Model failed to unload."""
    pass


class ModelTimeoutError(LMStudioModelError):
    """Model operation timed out."""
    pass


class LLMModelManager:
    """Manages model loading/unloading in LM Studio.
    
    Provides utilities for:
    - Loading models with previous model unload verification (Requirements 7.1)
    - Unloading models with GPU memory release wait (Requirements 7.2)
    - Retry logic with exponential backoff (Requirements 7.3)
    - Timeout handling (Requirements 7.4)
    - Keeping final model loaded (Requirements 7.5)
    
    Example:
        manager = LLMModelManager("http://127.0.0.1:1234")
        
        # Load vision model for captioning
        result = manager.load_model("google/gemma-3n-e4b")
        if result.success:
            # Process images...
            pass
        
        # Swap to summarization model
        result = manager.load_model("google/gemma-3n-e4b")
        # Previous model is automatically unloaded first
    """
    
    # Default model names for batch pipeline (will be overridden from config)
    VISION_MODEL = "google/gemma-3n-e4b"
    SUMMARIZATION_MODEL = "google/gemma-3n-e4b"
    
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:1234",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        load_timeout: int = 300,
        unload_timeout: int = 60,
        memory_release_wait: float = 5.0,
    ):
        """Initialize LM Studio Model Manager.
        
        Args:
            endpoint: LM Studio server endpoint
            max_retries: Maximum retry attempts for operations (Requirements 7.3)
            retry_delay: Initial delay between retries (exponential backoff)
            load_timeout: Timeout for model loading in seconds (Requirements 7.4)
            unload_timeout: Timeout for model unloading in seconds
            memory_release_wait: Time to wait for GPU memory release after unload (Requirements 7.2)
        """
        self.endpoint = endpoint.rstrip('/')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.load_timeout = load_timeout
        self.unload_timeout = unload_timeout
        self.memory_release_wait = memory_release_wait
        
        # Track current model state
        self._current_model: Optional[str] = None
        self._model_state: ModelState = ModelState.UNLOADED
        self._load_history: List[Dict[str, Any]] = []
        
        # API endpoints
        self._models_url = f"{self.endpoint}/v1/models"
        
        logger.info(f"LLMModelManager initialized with endpoint: {self.endpoint}")

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model name.
        
        Returns:
            Model name or None if no model is loaded
        """
        return self._current_model
    
    @property
    def model_state(self) -> ModelState:
        """Get the current model state.
        
        Returns:
            Current ModelState
        """
        return self._model_state
    
    @property
    def load_history(self) -> List[Dict[str, Any]]:
        """Get the history of model load operations.
        
        Returns:
            List of load operation records
        """
        return self._load_history.copy()
    
    def _make_request(
        self,
        method: str,
        url: str,
        timeout: Optional[int] = None,
        json_payload: Optional[Dict] = None,
    ) -> requests.Response:
        """Make HTTP request to LM Studio.
        
        Args:
            method: HTTP method ('GET', 'POST', 'DELETE')
            url: Request URL
            timeout: Request timeout in seconds
            json_payload: JSON payload for POST requests
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: On request failure
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        if method.upper() == 'GET':
            return requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == 'POST':
            return requests.post(url, headers=headers, json=json_payload, timeout=timeout)
        elif method.upper() == 'DELETE':
            return requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    def _retry_operation(
        self,
        operation: callable,
        operation_name: str,
        max_retries: Optional[int] = None,
    ) -> Any:
        """Execute operation with retry logic and exponential backoff.
        
        Args:
            operation: Callable to execute
            operation_name: Name for logging
            max_retries: Override default max_retries
            
        Returns:
            Operation result
            
        Raises:
            LMStudioModelError: If all retries fail
            
        Requirements: 7.3
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_exception = None
        
        for attempt in range(retries):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{retries}): {e}"
                )
                
                if attempt < retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying {operation_name} in {delay:.1f}s...")
                    time.sleep(delay)
        
        raise LMStudioModelError(
            f"{operation_name} failed after {retries} attempts. Last error: {last_exception}"
        )
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models from LM Studio.
        
        Returns:
            List of loaded model names
        """
        try:
            response = self._make_request('GET', self._models_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                return [m.get('id', '') for m in models if m.get('id')]
            else:
                logger.warning(f"Failed to get models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting loaded models: {e}")
            return []
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is currently loaded.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is loaded
        """
        loaded_models = self.get_loaded_models()
        return model_name in loaded_models
    
    def get_loaded_model(self) -> Optional[str]:
        """Get the currently loaded model name.
        
        Queries LM Studio to get the actual loaded model.
        
        Returns:
            Model name or None if no model is loaded
        """
        loaded_models = self.get_loaded_models()
        if loaded_models:
            # Return the first loaded model (LM Studio typically has one)
            return loaded_models[0]
        return None
    
    def verify_model_ready(self, model_name: str, timeout: int = 30) -> bool:
        """Verify a model is loaded and ready to respond.
        
        Args:
            model_name: Name of the model to verify
            timeout: Maximum time to wait for model to be ready
            
        Returns:
            True if model is ready
            
        Requirements: 7.4
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_model_loaded(model_name):
                # Try a simple completion to verify model is responding
                try:
                    response = self._make_request(
                        'POST',
                        f"{self.endpoint}/v1/chat/completions",
                        timeout=30,
                        json_payload={
                            "model": model_name,  # Required for JIT loading
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 1,
                        }
                    )
                    if response.status_code == 200:
                        logger.info(f"Model {model_name} is ready")
                        return True
                except Exception as e:
                    logger.debug(f"Model not ready yet: {e}")
            
            time.sleep(1)
        
        logger.warning(f"Model {model_name} not ready after {timeout}s")
        return False

    def unload_model(self, wait_for_memory: bool = True) -> bool:
        """Unload the current model and optionally wait for GPU memory release.
        
        Args:
            wait_for_memory: If True, wait for GPU memory to be released
            
        Returns:
            True if unload successful
            
        Raises:
            ModelUnloadError: If unload fails
            
        Requirements: 7.2
        """
        if self._current_model is None:
            logger.info("No model currently loaded, nothing to unload")
            return True
        
        model_to_unload = self._current_model
        logger.info(f"Unloading model: {model_to_unload}")
        
        self._model_state = ModelState.UNLOADING
        
        try:
            # LM Studio API: POST to /v1/models/unload or DELETE model
            # Try the unload endpoint first
            unload_url = f"{self.endpoint}/v1/models/unload"
            
            try:
                response = self._make_request(
                    'POST',
                    unload_url,
                    timeout=self.unload_timeout,
                    json_payload={"model": model_to_unload}
                )
                
                if response.status_code in [200, 204]:
                    logger.info(f"Model {model_to_unload} unload request sent")
                else:
                    # Try alternative: just wait for model to be replaced
                    logger.info(f"Unload endpoint returned {response.status_code}, will proceed with load")
                    
            except requests.exceptions.RequestException as e:
                # Unload endpoint might not exist, proceed anyway
                logger.debug(f"Unload endpoint not available: {e}")
            
            # Wait for GPU memory to be released
            if wait_for_memory:
                logger.info(f"Waiting {self.memory_release_wait}s for GPU memory release...")
                time.sleep(self.memory_release_wait)
            
            # Verify model is unloaded
            if not self.is_model_loaded(model_to_unload):
                self._current_model = None
                self._model_state = ModelState.UNLOADED
                logger.info(f"Model {model_to_unload} successfully unloaded")
                return True
            else:
                # Model might still be loaded, but we'll proceed
                # The load operation will replace it
                logger.warning(f"Model {model_to_unload} may still be loaded")
                self._current_model = None
                self._model_state = ModelState.UNLOADED
                return True
                
        except Exception as e:
            self._model_state = ModelState.ERROR
            raise ModelUnloadError(f"Failed to unload model {model_to_unload}: {e}")
    
    def load_model(
        self,
        model_name: str,
        timeout: Optional[int] = None,
        verify_ready: bool = True,
    ) -> ModelLoadResult:
        """Load a model in LM Studio, unloading current model first.
        
        This method ensures only one model is loaded at a time by:
        1. Checking if a model is currently loaded
        2. Unloading the current model if different
        3. Loading the new model with retry logic
        4. Verifying the model is ready
        
        Args:
            model_name: Name of the model to load
            timeout: Override default load timeout
            verify_ready: If True, verify model is responding after load
            
        Returns:
            ModelLoadResult with success status and details
            
        Requirements: 7.1, 7.3, 7.4
        """
        timeout = timeout or self.load_timeout
        start_time = time.time()
        attempts = 0
        
        logger.info(f"Loading model: {model_name}")
        
        # Check if model is already loaded
        if self._current_model == model_name and self.is_model_loaded(model_name):
            logger.info(f"Model {model_name} is already loaded")
            return ModelLoadResult(
                success=True,
                model_name=model_name,
                message="Model already loaded",
                attempts=0,
                load_time_seconds=0.0,
            )
        
        # Unload current model first (Requirements 7.1)
        if self._current_model is not None:
            logger.info(f"Unloading current model {self._current_model} before loading {model_name}")
            try:
                self.unload_model(wait_for_memory=True)
            except ModelUnloadError as e:
                logger.warning(f"Failed to unload current model: {e}")
                # Continue anyway, load might replace it
        
        self._model_state = ModelState.LOADING
        
        # Retry logic with exponential backoff (Requirements 7.3)
        last_error = None
        
        for attempt in range(self.max_retries):
            attempts = attempt + 1
            
            try:
                # LM Studio API: POST to load model
                # The exact endpoint depends on LM Studio version
                load_url = f"{self.endpoint}/v1/models/load"
                
                try:
                    response = self._make_request(
                        'POST',
                        load_url,
                        timeout=timeout,
                        json_payload={"model": model_name}
                    )
                    
                    if response.status_code in [200, 201, 202]:
                        logger.info(f"Model load request accepted for {model_name}")
                    elif response.status_code == 404:
                        # Load endpoint might not exist, model might auto-load on first request
                        logger.info("Load endpoint not available, model may auto-load on first request")
                    else:
                        logger.warning(f"Load request returned {response.status_code}: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Load endpoint request failed: {e}")
                
                # Wait for model to be ready
                if verify_ready:
                    if self.verify_model_ready(model_name, timeout=min(60, timeout)):
                        self._current_model = model_name
                        self._model_state = ModelState.LOADED
                        load_time = time.time() - start_time
                        
                        # Record in history
                        self._load_history.append({
                            "model": model_name,
                            "timestamp": time.time(),
                            "attempts": attempts,
                            "load_time": load_time,
                            "success": True,
                        })
                        
                        logger.info(f"Model {model_name} loaded successfully in {load_time:.1f}s")
                        
                        return ModelLoadResult(
                            success=True,
                            model_name=model_name,
                            message="Model loaded successfully",
                            attempts=attempts,
                            load_time_seconds=load_time,
                        )
                else:
                    # Don't verify, just assume success
                    self._current_model = model_name
                    self._model_state = ModelState.LOADED
                    load_time = time.time() - start_time
                    
                    return ModelLoadResult(
                        success=True,
                        model_name=model_name,
                        message="Model load requested (not verified)",
                        attempts=attempts,
                        load_time_seconds=load_time,
                    )
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Load attempt {attempts}/{self.max_retries} failed: {e}")
            
            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying model load in {delay:.1f}s...")
                time.sleep(delay)
        
        # All retries exhausted
        self._model_state = ModelState.ERROR
        load_time = time.time() - start_time
        
        # Record failure in history
        self._load_history.append({
            "model": model_name,
            "timestamp": time.time(),
            "attempts": attempts,
            "load_time": load_time,
            "success": False,
            "error": str(last_error),
        })
        
        error_msg = f"Failed to load model {model_name} after {attempts} attempts"
        logger.error(error_msg)
        
        return ModelLoadResult(
            success=False,
            model_name=model_name,
            message=f"{error_msg}. Last error: {last_error}",
            attempts=attempts,
            load_time_seconds=load_time,
        )
    
    def swap_model(
        self,
        new_model: str,
        wait_for_memory: bool = True,
    ) -> ModelLoadResult:
        """Swap from current model to a new model.
        
        Convenience method that ensures proper unload before load.
        
        Args:
            new_model: Name of the model to load
            wait_for_memory: Wait for GPU memory release after unload
            
        Returns:
            ModelLoadResult with success status
            
        Requirements: 7.1, 7.2
        """
        logger.info(f"Swapping model from {self._current_model} to {new_model}")
        
        # Unload current model with memory wait
        if self._current_model is not None:
            self.unload_model(wait_for_memory=wait_for_memory)
        
        # Load new model
        return self.load_model(new_model)
    
    def test_connection(self) -> bool:
        """Test connection to LM Studio server.
        
        Returns:
            True if connection successful
        """
        try:
            response = self._make_request('GET', self._models_url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current manager status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "endpoint": self.endpoint,
            "current_model": self._current_model,
            "model_state": self._model_state.value,
            "loaded_models": self.get_loaded_models(),
            "load_history_count": len(self._load_history),
            "max_retries": self.max_retries,
        }

import torch
import torch.nn as nn
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, Type, Union
from functools import lru_cache
from pathlib import Path
import platform
import gc

class LazyModelLoader:
    """Implements lazy loading for large models to optimize memory usage."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LazyModelLoader, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the model loader."""
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.models = {}
            self.model_load_times = {}
            self.last_access_times = {}
            self.auto_unload_interval = 300  # 5 minutes
            self.device = self._get_device()
            self.unload_thread = None
            self.stop_event = threading.Event()
            self.cache_dir = Path("models/cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            
            # Start background thread for automatic model unloading
            self._start_auto_unload_thread()
    
    def _get_device(self) -> torch.device:
        """Determine the appropriate device for model loading."""
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Apple Silicon (M1/M2)
            if torch.backends.mps.is_available():
                self.logger.info("Using MPS (Metal Performance Shaders) device")
                return torch.device("mps")
        elif torch.cuda.is_available():
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        
        self.logger.info("Using CPU device")
        return torch.device("cpu")
    
    def _start_auto_unload_thread(self):
        """Start a background thread to automatically unload unused models."""
        if self.unload_thread is not None and self.unload_thread.is_alive():
            return
            
        def auto_unload_task():
            while not self.stop_event.is_set():
                self._check_and_unload_unused_models()
                time.sleep(60)  # Check every minute
        
        self.unload_thread = threading.Thread(target=auto_unload_task, daemon=True)
        self.unload_thread.start()
        self.logger.info("Started automatic model unloading thread")
    
    def _check_and_unload_unused_models(self):
        """Check for and unload models that haven't been used recently."""
        current_time = time.time()
        models_to_unload = []
        
        for model_name, last_access in self.last_access_times.items():
            if current_time - last_access > self.auto_unload_interval:
                models_to_unload.append(model_name)
        
        for model_name in models_to_unload:
            self.unload_model(model_name)
            self.logger.info(f"Auto-unloaded unused model: {model_name} after {self.auto_unload_interval}s of inactivity")
    
    def get_model(self, model_name: str, model_loader: Callable, *args, **kwargs) -> Any:
        """Get a model, loading it if necessary.
        
        Args:
            model_name: Name of the model
            model_loader: Function to load the model if not already loaded
            *args: Arguments for the model loader
            **kwargs: Keyword arguments for the model loader
            
        Returns:
            The loaded model
        """
        # Update last access time
        self.last_access_times[model_name] = time.time()
        
        # Return if already loaded
        if model_name in self.models and self.models[model_name] is not None:
            self.logger.debug(f"Using already loaded model: {model_name}")
            return self.models[model_name]
        
        # Load the model
        self.logger.info(f"Lazy loading model: {model_name}")
        start_time = time.time()
        
        # Clear cache before loading
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
        
        # Load the model
        model = model_loader(*args, **kwargs)
        
        # Move model to device if it's a torch model
        if isinstance(model, nn.Module):
            model = model.to(self.device)
        
        # Store the model
        self.models[model_name] = model
        
        # Record load time
        load_time = time.time() - start_time
        self.model_load_times[model_name] = load_time
        self.logger.info(f"Loaded model {model_name} in {load_time:.2f} seconds")
        
        return model
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False otherwise
        """
        if model_name in self.models and self.models[model_name] is not None:
            model = self.models[model_name]
            
            # For PyTorch models, we can explicitly delete and clear cache
            if isinstance(model, nn.Module):
                del model
                self.models[model_name] = None
                
                # Clear CUDA cache if using GPU
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
            else:
                # For other types of models
                del model
                self.models[model_name] = None
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info(f"Unloaded model: {model_name}")
            return True
        
        return False
    
    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """List all currently loaded models with their metadata.
        
        Returns:
            Dictionary of model metadata
        """
        result = {}
        for model_name, model in self.models.items():
            if model is not None:
                result[model_name] = {
                    "load_time": self.model_load_times.get(model_name, 0),
                    "last_access": self.last_access_times.get(model_name, 0),
                    "type": type(model).__name__,
                    "device": self.device.type
                }
        return result
    
    def cleanup(self):
        """Clean up resources when shutting down."""
        self.stop_event.set()
        if self.unload_thread and self.unload_thread.is_alive():
            self.unload_thread.join(timeout=1)
        
        # Unload all models
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
        
        # Final cleanup
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache() 
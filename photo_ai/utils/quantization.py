"""Model quantization utilities for optimizing inference speed."""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import platform
import os
from functools import wraps
from enum import Enum, auto
import time

class QuantizationLevel(Enum):
    """Quantization precision levels."""
    NONE = auto()       # No quantization
    FP16 = auto()       # Half precision (16-bit)
    INT8 = auto()       # 8-bit integer precision 
    DYNAMIC = auto()    # Dynamic quantization

class ModelQuantizer:
    """Handles quantization of PyTorch models for faster inference."""
    
    def __init__(self):
        """Initialize the model quantizer."""
        self.logger = logging.getLogger(__name__)
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Determine the appropriate device for quantization."""
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
        
    def quantize_model(self, 
                       model: torch.nn.Module, 
                       level: QuantizationLevel = QuantizationLevel.FP16) -> torch.nn.Module:
        """Quantize a PyTorch model to reduce memory usage and increase inference speed.
        
        Args:
            model: The PyTorch model to quantize
            level: Quantization level
            
        Returns:
            Quantized model
        """
        if level == QuantizationLevel.NONE:
            return model
            
        # For Apple Silicon (MPS), only FP16 is properly supported
        if self.device.type == "mps" and level != QuantizationLevel.FP16:
            self.logger.warning(f"MPS device only supports FP16 quantization, ignoring {level}")
            level = QuantizationLevel.FP16
            
        # Make a copy of the model to avoid modifying the original
        model = model.to("cpu")
        
        try:
            if level == QuantizationLevel.FP16:
                # Half precision (16-bit floating point)
                self.logger.info("Applying FP16 quantization")
                model = model.half()
            elif level == QuantizationLevel.INT8:
                # 8-bit integer quantization (static)
                self.logger.info("Applying INT8 quantization")
                # Requires model to be in eval mode
                model.eval()
                # Apply static quantization if PyTorch supports it for this model
                if hasattr(torch.quantization, 'quantize_static'):
                    # Configure quantization
                    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                    # Prepare
                    model_prepared = torch.quantization.prepare(model)
                    # Calibrate with sample data (you'd need real data for accuracy)
                    # This part would need real calibration data for production
                    model_prepared(torch.randn(1, 3, 224, 224))
                    # Convert to quantized model
                    model = torch.quantization.convert(model_prepared)
            elif level == QuantizationLevel.DYNAMIC:
                # Dynamic quantization (quantizes weights, activations calculated at fp32)
                self.logger.info("Applying dynamic quantization")
                model.eval()
                model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.LSTM, torch.nn.Conv2d}, 
                    dtype=torch.qint8
                )
        except Exception as e:
            self.logger.error(f"Error during quantization: {e}")
            self.logger.warning("Falling back to original model")
            return model.to(self.device)
            
        # Move model to the appropriate device
        return model.to(self.device)
        
    def measure_quantization_impact(self, 
                                   model: torch.nn.Module, 
                                   input_tensor: torch.Tensor, 
                                   level: QuantizationLevel = QuantizationLevel.FP16,
                                   num_runs: int = 10) -> Dict[str, Any]:
        """Measure the performance impact of quantization.
        
        Args:
            model: Original PyTorch model
            input_tensor: Sample input tensor
            level: Quantization level
            num_runs: Number of inference runs for measurement
            
        Returns:
            Dictionary with performance metrics
        """
        # Original model metrics
        orig_model = model.to(self.device)
        orig_model.eval()
        
        # Warm-up run
        with torch.no_grad():
            orig_model(input_tensor.to(self.device))
            
        # Measure original model
        orig_start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                orig_model(input_tensor.to(self.device))
        orig_time = (time.time() - orig_start) / num_runs
        
        # Get original model size
        orig_size = self._get_model_size(orig_model)
        
        # Quantize the model
        quant_model = self.quantize_model(model, level)
        quant_model.eval()
        
        # Warm-up run for quantized model
        with torch.no_grad():
            quant_model(input_tensor.to(self.device))
            
        # Measure quantized model
        quant_start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                quant_model(input_tensor.to(self.device))
        quant_time = (time.time() - quant_start) / num_runs
        
        # Get quantized model size
        quant_size = self._get_model_size(quant_model)
        
        # Calculate improvements
        time_improvement = (orig_time - quant_time) / orig_time * 100
        size_improvement = (orig_size - quant_size) / orig_size * 100
        
        return {
            "original": {
                "inference_time": orig_time,
                "model_size_mb": orig_size
            },
            "quantized": {
                "inference_time": quant_time,
                "model_size_mb": quant_size,
                "quantization_level": level.name
            },
            "improvements": {
                "time_percent": time_improvement,
                "size_percent": size_improvement
            }
        }
    
    def _get_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
            
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training/inference if supported.
        
        This is different from model quantization - it uses the hardware's
        mixed precision capabilities while keeping the model in full precision.
        """
        if self.device.type == "cuda":
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                self.logger.info("Enabled TF32 mixed precision on CUDA device")
        elif self.device.type == "mps":
            # MPS already uses optimized precision
            self.logger.info("MPS device already uses optimized precision")
        else:
            self.logger.info("Mixed precision not available on this device")


def quantized(level: QuantizationLevel = QuantizationLevel.FP16):
    """Decorator for applying quantization to models used in a function.
    
    Args:
        level: Quantization level
        
    Returns:
        Decorated function
    """
    quantizer = ModelQuantizer()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Process all model arguments
            new_args = []
            for arg in args:
                if isinstance(arg, torch.nn.Module):
                    new_args.append(quantizer.quantize_model(arg, level))
                else:
                    new_args.append(arg)
            
            # Process kwargs that are models
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.nn.Module):
                    new_kwargs[key] = quantizer.quantize_model(value, level)
                else:
                    new_kwargs[key] = value
            
            # Call the function with quantized models
            return func(*new_args, **new_kwargs)
        
        return wrapper
    
    return decorator 
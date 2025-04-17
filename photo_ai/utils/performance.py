import concurrent.futures
from functools import lru_cache
from typing import Callable, List, Any, Dict, Optional
import numpy as np
import cv2
from pathlib import Path
import time
import psutil
import logging
from dataclasses import dataclass
from enum import Enum, auto
import threading
import torch
import torch.cuda as cuda
import platform
from photo_ai.core.image_processor import ImageProcessor

class ProcessingMode(Enum):
    """Processing modes for optimization."""
    CPU = auto()
    GPU = auto()
    HYBRID = auto()

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    processing_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None

class PerformanceOptimizer:
    """Handles performance optimizations for image processing."""

    def __init__(self, 
                 max_workers: Optional[int] = None,
                 cache_size: int = 100,
                 processing_mode: ProcessingMode = ProcessingMode.CPU):
        """Initialize the optimizer.

        Args:
            max_workers: Maximum number of parallel workers
            cache_size: Size of the LRU cache
            processing_mode: Processing mode to use
        """
        self.max_workers = max_workers or psutil.cpu_count(logical=False)
        self.cache_size = cache_size
        self.processing_mode = processing_mode
        self._setup_logging()
        self._setup_gpu()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PerformanceOptimizer')

    def _setup_gpu(self) -> None:
        """Setup GPU if available."""
        system = platform.system()
        machine = platform.machine()

        # Check for Apple Silicon
        if system == "Darwin" and machine == "arm64":
            self.gpu_available = torch.backends.mps.is_available()
            if self.gpu_available:
                self.device = torch.device("mps")
                self.logger.info("Apple Silicon GPU (MPS) available")
            else:
                self.device = torch.device("cpu")
                self.logger.warning("Apple Silicon GPU (MPS) not available, falling back to CPU")
        else:
            # For other systems, check CUDA
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.device = torch.device("cuda")
                self.logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                self.logger.warning("GPU not available, falling back to CPU")

    def to_gpu(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to GPU tensor.

        Args:
            image: Input image

        Returns:
            GPU tensor
        """
        if self.gpu_available and self.processing_mode != ProcessingMode.CPU:
            tensor = torch.from_numpy(image).to(self.device)
            if self.device.type == "mps":
                # Convert to float32 for MPS compatibility
                tensor = tensor.float()
            return tensor
        return torch.from_numpy(image)

    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to numpy array.

        Args:
            tensor: Input tensor

        Returns:
            Numpy array
        """
        if tensor.dtype == torch.float32:
            # Convert back to uint8 for image processing
            tensor = tensor.clamp(0, 255).to(torch.uint8)
        return tensor.cpu().numpy()

    def cached_image_load(self, image_path: str) -> np.ndarray:
        """Cache image loading operations.

        Args:
            image_path: Path to the image

        Returns:
            Loaded image
        """
        return ImageProcessor.cached_load_image(image_path)

    def parallel_process(self, 
                        func: Callable,
                        items: List[Any],
                        **kwargs) -> List[Any]:
        """Process items in parallel.

        Args:
            func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments for the function

        Returns:
            List of processed items
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(func, item, **kwargs): item 
                for item in items
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing item: {str(e)}")

        return results

    def measure_performance(self, 
                          func: Callable,
                          *args,
                          **kwargs) -> PerformanceMetrics:
        """Measure performance metrics for a function call.

        Args:
            func: Function to measure
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Performance metrics
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()

        # Get GPU memory if available
        if self.gpu_available:
            if self.device.type == "cuda":
                start_gpu_memory = cuda.memory_allocated() / 1024 / 1024
            else:  # MPS
                start_gpu_memory = None  # MPS doesn't provide memory stats
        else:
            start_gpu_memory = None

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()

        # Get GPU memory if available
        if self.gpu_available:
            if self.device.type == "cuda":
                end_gpu_memory = cuda.memory_allocated() / 1024 / 1024
                gpu_memory = end_gpu_memory - start_gpu_memory
            else:  # MPS
                gpu_memory = None
        else:
            gpu_memory = None

        return PerformanceMetrics(
            processing_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=end_cpu,
            gpu_memory=gpu_memory
        )

    def optimize_memory(self, image: np.ndarray) -> np.ndarray:
        """Optimize image memory usage.

        Args:
            image: Input image

        Returns:
            Memory-optimized image
        """
        # Convert to appropriate data type
        if image.dtype == np.float64:
            image = image.astype(np.float32)

        # Remove unnecessary channels
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        return image

    def batch_process_images(self,
                           image_paths: List[str],
                           process_func: Callable,
                           batch_size: int = 32,
                           **kwargs) -> List[Any]:
        """Process images in batches with memory optimization.

        Args:
            image_paths: List of image paths
            process_func: Function to process each image
            batch_size: Number of images to process at once
            **kwargs: Additional arguments for the processing function

        Returns:
            List of processed images
        """
        results = []
        total_images = len(image_paths)
        
        # Log the start of batch processing
        self.logger.info(f"Starting batch processing of {total_images} images with batch size {batch_size}")
        
        # Process in batches to conserve memory
        for i in range(0, total_images, batch_size):
            # Clear memory before processing each batch
            if self.gpu_available:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
            
            # Calculate progress
            batch_end = min(i + batch_size, total_images)
            progress = (i / total_images) * 100
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(total_images+batch_size-1)//batch_size} ({progress:.1f}%)")
            
            # Get current batch paths
            batch_paths = image_paths[i:batch_end]
            
            # Load images with optimized memory usage
            batch_images = []
            for path in batch_paths:
                # Use cached image loading to prevent duplicate loads
                image = self.cached_image_load(path)
                # Apply memory optimization to each image
                image = self.optimize_memory(image)
                batch_images.append(image)
            
            # Process batch with parallel processing if appropriate
            if len(batch_images) > 1 and self.max_workers > 1 and self.processing_mode != ProcessingMode.GPU:
                # For CPU or HYBRID mode, use parallel processing
                batch_results = self.parallel_process(process_func, batch_images, **kwargs)
            else:
                # For GPU mode or small batches, process sequentially
                batch_results = [process_func(img, **kwargs) for img in batch_images]
            
            # Append results
            results.extend(batch_results)
            
            # Force garbage collection after processing each batch
            import gc
            gc.collect()
        
        self.logger.info(f"Completed batch processing of {total_images} images")
        return results

    def get_performance_report(self) -> Dict[str, Any]:
        """Get current performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        report = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,
            "disk_usage": psutil.disk_usage('/').percent,
            "active_threads": threading.active_count()
        }

        if self.gpu_available:
            if self.device.type == "cuda":
                report.update({
                    "gpu_usage": cuda.utilization(),
                    "gpu_memory": cuda.memory_allocated() / 1024 / 1024,
                    "gpu_memory_total": cuda.get_device_properties(0).total_memory / 1024 / 1024
                })
            else:  # MPS
                report.update({
                    "gpu_type": "Apple Silicon (MPS)",
                    "gpu_available": True
                })

        return report 

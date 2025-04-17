from typing import Any, Dict, Optional, List, Tuple
import os
import shutil
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn.functional as F
import logging
import uuid
from ..utils.performance import PerformanceOptimizer, ProcessingMode
from .base_agent import BaseAgent
from ..core.image_processor import ImageProcessor
from ..utils.async_processor import AsyncProcessor, async_task, wait_for_task
from ..utils.progress_tracker import ProgressTracker, ProgressContext, with_progress

class BatchAgent(BaseAgent):
    """Agent for batch processing, photo ranking, and directory management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the batch agent.

        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        super().__init__(config)
        self._validate_config()  # Initialize defaults first
        self.performance_optimizer = PerformanceOptimizer(
            max_workers=self.config.get('max_workers'),
            cache_size=self.config.get('cache_size', 100),
            processing_mode=ProcessingMode(self.config.get('processing_mode', ProcessingMode.CPU))
        )
        self.rankings: List[Dict[str, Any]] = []
        self.processed_files: List[str] = []
        self.logger = logging.getLogger(__name__)
        self.async_processor = AsyncProcessor()
        self.progress_tracker = ProgressTracker()

    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        self.config.setdefault('min_rating', 1)
        self.config.setdefault('max_rating', 5)
        self.config.setdefault('supported_extensions', [
            # Standard formats
            '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif',
            # Modern formats
            '.webp', '.heif', '.heic',
            # Raw formats
            '.dng', '.arw', '.cr2', '.nef', '.orf', '.rw2'
        ])
        self.config.setdefault('quality_threshold', 3)
        self.config.setdefault('purge_low_quality', False)
        self.config.setdefault('processing_mode', ProcessingMode.CPU.value)
        self.config.setdefault('use_async', True)
        self.config.setdefault('batch_size', 32)
        self.config.setdefault('max_workers', 4)

    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         processing_order: List[BaseAgent],
                         purge_low_quality: bool = False) -> Dict[str, Any]:
        """Process all images in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            processing_order: List of agents to apply
            purge_low_quality: Whether to purge low quality photos

        Returns:
            Dictionary with processing results
        """
        # Generate a unique operation ID
        operation_id = f"batch_process_{uuid.uuid4().hex[:8]}"
        
        # Create a progress tracking operation
        self.progress_tracker.create_operation(
            operation_id,
            f"Processing images from {input_dir}",
            100  # Will update total steps once we know how many images there are
        )
        self.progress_tracker.start_operation(operation_id)
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get list of image paths
            image_paths = []
            for ext in self.config['supported_extensions']:
                image_paths.extend(list(Path(input_dir).glob(f"*{ext}")))
                image_paths.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
            
            image_paths = [str(p) for p in image_paths]
            total_images = len(image_paths)
            
            # Update progress tracking with total steps
            self.progress_tracker.update_progress(
                operation_id, 
                0, 
                f"Found {total_images} images to process"
            )
            
            # Update total steps now that we know how many images there are
            self.operations = self.progress_tracker.get_operation(operation_id)
            if self.operations:
                self.operations["total_steps"] = total_images
            
            # Log start of processing
            self.logger.info(f"Starting batch processing of {total_images} images from {input_dir} to {output_dir}")
            
            # Process images based on async configuration
            if self.config.get('use_async', True):
                result = self._process_directory_async(image_paths, output_dir, processing_order, purge_low_quality, operation_id)
            else:
                result = self._process_directory_sync(image_paths, output_dir, processing_order, purge_low_quality, operation_id)
            
            # Mark operation as completed
            self.progress_tracker.complete_operation(
                operation_id,
                f"Processed {result['processed_photos']} images, purged {result['purged_photos']} images, encountered {len(result['errors'])} errors"
            )
            
            return result
            
        except Exception as e:
            # Mark operation as failed
            self.progress_tracker.fail_operation(
                operation_id,
                f"Batch processing failed: {str(e)}"
            )
            raise

    def _process_directory_async(self,
                               image_paths: List[str],
                               output_dir: str,
                               processing_order: List[BaseAgent],
                               purge_low_quality: bool,
                               parent_operation_id: str) -> Dict[str, Any]:
        """Process images asynchronously.

        Args:
            image_paths: List of image paths
            output_dir: Output directory
            processing_order: List of agents to apply
            purge_low_quality: Whether to purge low quality photos
            parent_operation_id: ID of the parent progress tracking operation

        Returns:
            Dictionary with processing results
        """
        # Create sub-operations for each image
        image_operations = {}
        for i, image_path in enumerate(image_paths):
            image_name = Path(image_path).name
            op_id = f"{parent_operation_id}_image_{i}"
            
            self.progress_tracker.create_operation(
                op_id,
                f"Processing {image_name}",
                len(processing_order) + 2,  # Load + process agents + save
                parent_operation_id
            )
            
            image_operations[op_id] = image_path
        
        # Submit tasks for each image
        task_ids = []
        for op_id, image_path in image_operations.items():
            self.progress_tracker.start_operation(op_id)
            task_id = self.async_processor.submit_task(
                self._process_single_image,
                image_path,
                output_dir,
                processing_order,
                purge_low_quality,
                op_id
            )
            task_ids.append(task_id)
        
        # Create result tracking
        total_images = len(image_paths)
        processed_count = 0
        purged_count = 0
        errors = []
        
        # Wait for all tasks to complete and update progress
        for i, task_id in enumerate(task_ids):
            result = wait_for_task(task_id)
            
            # Update main progress
            self.progress_tracker.update_progress(
                parent_operation_id,
                i + 1,
                f"Processed {i+1}/{total_images} images"
            )
            
            if result:
                if result.get('success', False):
                    processed_count += 1
                elif result.get('purged', False):
                    purged_count += 1
                
                if result.get('error'):
                    errors.append(result['error'])
        
        # Cleanup completed tasks
        self.async_processor.clean_completed_tasks()
        
        return {
            "total_photos": total_images,
            "processed_photos": processed_count,
            "purged_photos": purged_count,
            "errors": errors
        }

    def _process_directory_sync(self,
                              image_paths: List[str],
                              output_dir: str,
                              processing_order: List[BaseAgent],
                              purge_low_quality: bool,
                              parent_operation_id: str) -> Dict[str, Any]:
        """Process images synchronously using batches.

        Args:
            image_paths: List of image paths
            output_dir: Output directory
            processing_order: List of agents to apply
            purge_low_quality: Whether to purge low quality photos
            parent_operation_id: ID of the parent progress tracking operation

        Returns:
            Dictionary with processing results
        """
        total_images = len(image_paths)
        
        # Process in batches using context manager for progress tracking
        results = []
        for batch_idx in range(0, total_images, self.config.get('batch_size', 32)):
            batch_end = min(batch_idx + self.config.get('batch_size', 32), total_images)
            batch = image_paths[batch_idx:batch_end]
            
            # Create a sub-operation for this batch
            batch_op_id = f"{parent_operation_id}_batch_{batch_idx//self.config.get('batch_size', 32)}"
            self.progress_tracker.create_operation(
                batch_op_id,
                f"Processing batch {batch_idx//self.config.get('batch_size', 32) + 1}/{(total_images+self.config.get('batch_size', 32)-1)//self.config.get('batch_size', 32)}",
                len(batch),
                parent_operation_id
            )
            self.progress_tracker.start_operation(batch_op_id)
            
            # Process the batch
            batch_results = []
            for i, image_path in enumerate(batch):
                # Create image sub-operation
                image_name = Path(image_path).name
                image_op_id = f"{batch_op_id}_image_{i}"
                
                self.progress_tracker.create_operation(
                    image_op_id,
                    f"Processing {image_name}",
                    len(processing_order) + 2,  # Load + process agents + save
                    batch_op_id
                )
                self.progress_tracker.start_operation(image_op_id)
                
                # Process image
                result = self._process_single_image(
                    image_path, 
                    output_dir, 
                    processing_order, 
                    purge_low_quality,
                    image_op_id
                )
                batch_results.append(result)
                
                # Update batch progress
                self.progress_tracker.update_progress(
                    batch_op_id,
                    i + 1,
                    f"Processed {i+1}/{len(batch)} images in batch"
                )
            
            # Complete batch operation
            self.progress_tracker.complete_operation(
                batch_op_id,
                f"Completed batch {batch_idx//self.config.get('batch_size', 32) + 1}"
            )
            
            # Update main progress operation
            batch_percent = batch_end / total_images * 100
            self.progress_tracker.update_progress(
                parent_operation_id,
                batch_end,
                f"Processed {batch_end}/{total_images} images ({batch_percent:.1f}%)"
            )
            
            results.extend(batch_results)
        
        # Count results
        processed_count = sum(1 for r in results if r.get('success', False))
        purged_count = sum(1 for r in results if r.get('purged', False))
        errors = [r['error'] for r in results if r.get('error')]
        
        return {
            "total_photos": total_images,
            "processed_photos": processed_count,
            "purged_photos": purged_count,
            "errors": errors
        }

    @async_task
    def process_image_async(self,
                          image_path: str,
                          output_dir: str,
                          processing_order: List[BaseAgent],
                          purge_low_quality: bool = False) -> Dict[str, Any]:
        """Process a single image asynchronously.

        Args:
            image_path: Path to the image
            output_dir: Output directory
            processing_order: List of agents to apply
            purge_low_quality: Whether to purge low quality photos

        Returns:
            Dictionary with processing results
        """
        # Create a progress operation
        operation_id = f"process_image_{uuid.uuid4().hex[:8]}"
        self.progress_tracker.create_operation(
            operation_id,
            f"Processing {Path(image_path).name}",
            len(processing_order) + 2  # Load + process agents + save
        )
        self.progress_tracker.start_operation(operation_id)
        
        return self._process_single_image(image_path, output_dir, processing_order, purge_low_quality, operation_id)

    def _process_single_image(self,
                            image_path: str,
                            output_dir: str,
                            processing_order: List[BaseAgent],
                            purge_low_quality: bool,
                            operation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image.

        Args:
            image_path: Path to the image
            output_dir: Output directory
            processing_order: List of agents to apply
            purge_low_quality: Whether to purge low quality photos
            operation_id: Optional progress tracking operation ID

        Returns:
            Dictionary with processing results
        """
        try:
            # Update progress if operation ID is provided
            if operation_id:
                self.progress_tracker.update_progress(
                    operation_id,
                    1,
                    f"Loading image {Path(image_path).name}"
                )
            
            # Load and optimize image
            image = self.performance_optimizer.cached_image_load(image_path)
            if image is None:
                if operation_id:
                    self.progress_tracker.fail_operation(
                        operation_id,
                        f"Failed to load image: {image_path}"
                    )
                return {
                    "success": False,
                    "purged": False,
                    "error": f"Failed to load image: {image_path}"
                }

            # Convert to GPU if available
            if self.performance_optimizer.gpu_available:
                image = self.performance_optimizer.to_gpu(image)

            # Assess quality
            quality_score = self._assess_photo_quality(image)
            rating = self._rank_photo(quality_score)

            # Check if image should be purged
            if purge_low_quality and rating < self.config['min_rating']:
                if operation_id:
                    self.progress_tracker.complete_operation(
                        operation_id,
                        f"Image purged due to low quality score: {quality_score:.2f}"
                    )
                return {
                    "success": False,
                    "purged": True,
                    "error": None
                }

            # Process image through agents
            processed = image.copy()
            for i, agent in enumerate(processing_order):
                if operation_id:
                    self.progress_tracker.update_progress(
                        operation_id,
                        i + 2,  # +2 because we already did step 1 (loading)
                        f"Applying {agent.__class__.__name__}"
                    )
                processed = agent.process(processed)

            # Convert back to CPU if necessary
            if isinstance(processed, torch.Tensor):
                processed = self.performance_optimizer.to_cpu(processed)

            # Save processed image
            if operation_id:
                self.progress_tracker.update_progress(
                    operation_id,
                    len(processing_order) + 2,
                    f"Saving processed image"
                )
                
            output_path = os.path.join(
                output_dir,
                os.path.basename(image_path)
            )
            
            try:
                # Use ImageProcessor to save
                ImageProcessor.save_image(processed, output_path)
                
                # Record successful processing
                self.processed_files.append(output_path)
                
                if operation_id:
                    self.progress_tracker.complete_operation(
                        operation_id,
                        f"Successfully processed and saved image with quality score: {quality_score:.2f}"
                    )
                
                return {
                    "success": True,
                    "purged": False,
                    "error": None,
                    "quality_score": quality_score,
                    "rating": rating,
                    "output_path": output_path
                }
            except Exception as e:
                if operation_id:
                    self.progress_tracker.fail_operation(
                        operation_id,
                        f"Failed to save image: {str(e)}"
                    )
                return {
                    "success": False,
                    "purged": False,
                    "error": f"Failed to save image: {str(e)}"
                }

        except Exception as e:
            if operation_id:
                self.progress_tracker.fail_operation(
                    operation_id,
                    f"Error processing {image_path}: {str(e)}"
                )
            return {
                "success": False,
                "purged": False,
                "error": f"Error processing {image_path}: {str(e)}"
            }

    def _assess_photo_quality(self, image: np.ndarray) -> float:
        """Assess the quality of a photo.

        Args:
            image: Input image

        Returns:
            Quality score (0.0 to 1.0)
        """
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = self.performance_optimizer.to_cpu(image)

        # Simple quality assessment metrics
        # 1. Check sharpness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Check contrast
        contrast = gray.std()
        
        # 3. Check brightness
        brightness = np.mean(gray)
        
        # 4. Check saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # Normalize and weight metrics
        norm_sharpness = min(1.0, sharpness / 1000)
        norm_contrast = min(1.0, contrast / 80)
        norm_brightness = 1.0 - abs((brightness - 128) / 128)
        norm_saturation = min(1.0, saturation / 128)
        
        # Combined score
        quality_score = (
            norm_sharpness * 0.4 + 
            norm_contrast * 0.3 + 
            norm_brightness * 0.2 + 
            norm_saturation * 0.1
        )
        
        return min(1.0, max(0.0, quality_score))

    def _rank_photo(self, quality_score: float) -> int:
        """Convert quality score to rating.

        Args:
            quality_score: Quality score (0.0 to 1.0)

        Returns:
            Rating (1 to 5)
        """
        min_rating = self.config['min_rating']
        max_rating = self.config['max_rating']
        rating_range = max_rating - min_rating
        
        # Convert to rating
        rating = min_rating + round(quality_score * rating_range)
        
        # Ensure within range
        return max(min_rating, min(max_rating, rating))

    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn from before/after image pairs."""
        # For batch processing, we don't need to learn from individual pairs
        # as we're focusing on quality assessment and ranking
        pass

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process a single image."""
        # For batch processing, we don't process individual images
        # as we're focusing on directory-level operations
        return image

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent.

        Returns:
            Dictionary containing agent status
        """
        performance_report = self.performance_optimizer.get_performance_report()
        return {
            "name": self.__class__.__name__,
            "config": self.config,
            "performance": performance_report
        }

    def get_processing_status(self) -> List[Dict[str, Any]]:
        """Get status of all current batch processing operations.
        
        Returns:
            List of operation statuses
        """
        return self.progress_tracker.get_all_operations(include_completed=False)
    
    def cancel_processing(self, operation_id: str) -> bool:
        """Cancel a processing operation.
        
        Args:
            operation_id: ID of the operation to cancel
            
        Returns:
            True if canceled successfully, False otherwise
        """
        # Cancel the operation in the progress tracker
        self.progress_tracker.cancel_operation(operation_id)
        return True 

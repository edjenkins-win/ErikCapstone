"""Progress tracking system for long-running operations."""

import time
import threading
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum, auto


class ProgressStatus(Enum):
    """Status of a progress operation."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()
    PAUSED = auto()


class ProgressTracker:
    """Tracks progress of long-running operations.
    
    This class provides a centralized way to track the progress of
    long-running operations like batch processing, model training, etc.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProgressTracker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the progress tracker."""
        if not self._initialized:
            self.operations = {}
            self.callbacks = {}
            self.logger = logging.getLogger(__name__)
            self._initialized = True
    
    def create_operation(self, 
                        operation_id: str, 
                        description: str, 
                        total_steps: int = 100,
                        parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new operation to track.
        
        Args:
            operation_id: Unique identifier for the operation
            description: Human-readable description of the operation
            total_steps: Total number of steps in the operation
            parent_id: Optional parent operation ID for hierarchical operations
            
        Returns:
            Operation data
        """
        operation = {
            "id": operation_id,
            "description": description,
            "parent_id": parent_id,
            "status": ProgressStatus.NOT_STARTED,
            "current_step": 0,
            "total_steps": total_steps,
            "progress": 0.0,  # Progress as percentage (0-100)
            "start_time": None,
            "end_time": None,
            "estimated_time_remaining": None,
            "messages": [],
            "sub_operations": [],
            "metadata": {}
        }
        
        self.operations[operation_id] = operation
        
        # Add to parent if specified
        if parent_id and parent_id in self.operations:
            if operation_id not in self.operations[parent_id]["sub_operations"]:
                self.operations[parent_id]["sub_operations"].append(operation_id)
        
        return operation
    
    def start_operation(self, operation_id: str) -> None:
        """Start an operation.
        
        Args:
            operation_id: ID of the operation to start
        """
        if operation_id in self.operations:
            self.operations[operation_id]["status"] = ProgressStatus.IN_PROGRESS
            self.operations[operation_id]["start_time"] = time.time()
            self._notify_update(operation_id)
            self.logger.info(f"Started operation: {operation_id}")
    
    def update_progress(self, 
                       operation_id: str, 
                       current_step: int,
                       message: Optional[str] = None) -> None:
        """Update the progress of an operation.
        
        Args:
            operation_id: ID of the operation to update
            current_step: Current step of the operation
            message: Optional status message to add
        """
        if operation_id not in self.operations:
            self.logger.warning(f"Attempted to update non-existent operation: {operation_id}")
            return
        
        operation = self.operations[operation_id]
        operation["current_step"] = current_step
        
        if operation["total_steps"] > 0:
            # Calculate progress percentage
            operation["progress"] = (current_step / operation["total_steps"]) * 100
            
            # Calculate estimated time remaining
            if (operation["start_time"] is not None 
                and current_step > 0 
                and operation["status"] == ProgressStatus.IN_PROGRESS):
                
                elapsed_time = time.time() - operation["start_time"]
                steps_remaining = operation["total_steps"] - current_step
                
                if elapsed_time > 0:
                    time_per_step = elapsed_time / current_step
                    operation["estimated_time_remaining"] = time_per_step * steps_remaining
        
        # Add message if provided
        if message:
            timestamp = time.time()
            operation["messages"].append({
                "timestamp": timestamp,
                "message": message,
                "step": current_step
            })
        
        # Notify callbacks
        self._notify_update(operation_id)
    
    def complete_operation(self, operation_id: str, message: Optional[str] = None) -> None:
        """Mark an operation as completed.
        
        Args:
            operation_id: ID of the operation to complete
            message: Optional completion message
        """
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            operation["status"] = ProgressStatus.COMPLETED
            operation["end_time"] = time.time()
            operation["current_step"] = operation["total_steps"]
            operation["progress"] = 100.0
            operation["estimated_time_remaining"] = 0
            
            if message:
                timestamp = time.time()
                operation["messages"].append({
                    "timestamp": timestamp,
                    "message": message,
                    "step": operation["current_step"]
                })
            
            self._notify_update(operation_id)
            self.logger.info(f"Completed operation: {operation_id}")
    
    def fail_operation(self, operation_id: str, error_message: str) -> None:
        """Mark an operation as failed.
        
        Args:
            operation_id: ID of the operation to fail
            error_message: Error message describing the failure
        """
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            operation["status"] = ProgressStatus.FAILED
            operation["end_time"] = time.time()
            
            timestamp = time.time()
            operation["messages"].append({
                "timestamp": timestamp,
                "message": f"ERROR: {error_message}",
                "step": operation["current_step"]
            })
            
            self._notify_update(operation_id)
            self.logger.error(f"Failed operation: {operation_id} - {error_message}")
    
    def cancel_operation(self, operation_id: str) -> None:
        """Cancel an operation.
        
        Args:
            operation_id: ID of the operation to cancel
        """
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            operation["status"] = ProgressStatus.CANCELED
            operation["end_time"] = time.time()
            
            timestamp = time.time()
            operation["messages"].append({
                "timestamp": timestamp,
                "message": "Operation canceled",
                "step": operation["current_step"]
            })
            
            # Also cancel any sub-operations
            for sub_id in operation["sub_operations"]:
                self.cancel_operation(sub_id)
            
            self._notify_update(operation_id)
            self.logger.info(f"Canceled operation: {operation_id}")
    
    def pause_operation(self, operation_id: str) -> None:
        """Pause an operation.
        
        Args:
            operation_id: ID of the operation to pause
        """
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            operation["status"] = ProgressStatus.PAUSED
            
            timestamp = time.time()
            operation["messages"].append({
                "timestamp": timestamp,
                "message": "Operation paused",
                "step": operation["current_step"]
            })
            
            self._notify_update(operation_id)
            self.logger.info(f"Paused operation: {operation_id}")
    
    def resume_operation(self, operation_id: str) -> None:
        """Resume a paused operation.
        
        Args:
            operation_id: ID of the operation to resume
        """
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            if operation["status"] == ProgressStatus.PAUSED:
                operation["status"] = ProgressStatus.IN_PROGRESS
                
                timestamp = time.time()
                operation["messages"].append({
                    "timestamp": timestamp,
                    "message": "Operation resumed",
                    "step": operation["current_step"]
                })
                
                self._notify_update(operation_id)
                self.logger.info(f"Resumed operation: {operation_id}")
    
    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of an operation.
        
        Args:
            operation_id: ID of the operation to get
            
        Returns:
            Operation data or None if not found
        """
        operation = self.operations.get(operation_id)
        if operation:
            # Convert enum to string for serialization
            operation_copy = operation.copy()
            operation_copy["status"] = operation["status"].name
            return operation_copy
        return None
    
    def get_all_operations(self, include_completed: bool = False) -> List[Dict[str, Any]]:
        """Get all operations.
        
        Args:
            include_completed: Whether to include completed operations
            
        Returns:
            List of operations
        """
        operations = []
        for op_id, operation in self.operations.items():
            if include_completed or operation["status"] not in (ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELED):
                # Convert enum to string for serialization
                operation_copy = operation.copy()
                operation_copy["status"] = operation["status"].name
                operations.append(operation_copy)
        return operations
    
    def remove_operation(self, operation_id: str) -> bool:
        """Remove an operation from tracking.
        
        Args:
            operation_id: ID of the operation to remove
            
        Returns:
            True if removed, False if not found
        """
        if operation_id in self.operations:
            # Remove from parent if it has one
            parent_id = self.operations[operation_id]["parent_id"]
            if parent_id and parent_id in self.operations:
                if operation_id in self.operations[parent_id]["sub_operations"]:
                    self.operations[parent_id]["sub_operations"].remove(operation_id)
            
            # Remove callbacks
            if operation_id in self.callbacks:
                del self.callbacks[operation_id]
            
            # Remove operation
            del self.operations[operation_id]
            return True
        
        return False
    
    def register_callback(self, operation_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for operation updates.
        
        Args:
            operation_id: ID of the operation to track
            callback: Function to call when the operation is updated
        """
        if operation_id not in self.callbacks:
            self.callbacks[operation_id] = []
        
        self.callbacks[operation_id].append(callback)
    
    def _notify_update(self, operation_id: str) -> None:
        """Notify callbacks of an operation update.
        
        Args:
            operation_id: ID of the updated operation
        """
        if operation_id in self.callbacks:
            operation = self.get_operation(operation_id)
            for callback in self.callbacks[operation_id]:
                try:
                    callback(operation)
                except Exception as e:
                    self.logger.error(f"Error in progress callback: {e}")
    
    def to_json(self) -> str:
        """Convert all operations to JSON string.
        
        Returns:
            JSON string of all operations
        """
        operations = self.get_all_operations(include_completed=True)
        return json.dumps(operations)
    
    def cleanup_completed(self, max_age: int = 3600) -> int:
        """Remove completed operations older than max_age.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            Number of operations removed
        """
        current_time = time.time()
        to_remove = []
        
        for op_id, operation in self.operations.items():
            if operation["status"] in (ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELED):
                if operation["end_time"] and (current_time - operation["end_time"]) > max_age:
                    to_remove.append(op_id)
        
        for op_id in to_remove:
            self.remove_operation(op_id)
        
        return len(to_remove)


class ProgressContext:
    """Context manager for tracking progress of a block of code.
    
    Example:
        with ProgressContext("process_images", "Processing images", 100) as progress:
            for i, image in enumerate(images):
                process_image(image)
                progress.update(i + 1)
    """
    
    def __init__(self, 
                 operation_id: str, 
                 description: str, 
                 total_steps: int = 100,
                 parent_id: Optional[str] = None):
        """Initialize the progress context.
        
        Args:
            operation_id: Unique identifier for the operation
            description: Human-readable description of the operation
            total_steps: Total number of steps in the operation
            parent_id: Optional parent operation ID
        """
        self.tracker = ProgressTracker()
        self.operation_id = operation_id
        self.description = description
        self.total_steps = total_steps
        self.parent_id = parent_id
    
    def __enter__(self) -> 'ProgressContext':
        """Enter the context manager."""
        self.tracker.create_operation(
            self.operation_id,
            self.description,
            self.total_steps,
            self.parent_id
        )
        self.tracker.start_operation(self.operation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        if exc_type is not None:
            # An exception occurred
            self.tracker.fail_operation(
                self.operation_id,
                f"{exc_type.__name__}: {str(exc_val)}"
            )
        else:
            # Normal exit
            self.tracker.complete_operation(self.operation_id)
    
    def update(self, current_step: int, message: Optional[str] = None) -> None:
        """Update the progress.
        
        Args:
            current_step: Current step
            message: Optional status message
        """
        self.tracker.update_progress(self.operation_id, current_step, message)


def with_progress(operation_id: str, description: str, total_steps: int = 100, parent_id: Optional[str] = None):
    """Decorator for tracking progress of a function.
    
    Example:
        @with_progress("process_batch", "Processing batch", 100)
        def process_batch(images):
            for i, image in enumerate(images):
                # Process image
                yield i + 1  # Update progress
    
    Args:
        operation_id: Unique identifier for the operation
        description: Human-readable description of the operation
        total_steps: Total number of steps in the operation
        parent_id: Optional parent operation ID
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get progress tracker
            tracker = ProgressTracker()
            
            # Create unique operation ID using function name if not provided
            nonlocal operation_id
            if operation_id is None:
                operation_id = f"{func.__name__}_{time.time()}"
            
            # Create operation
            tracker.create_operation(
                operation_id,
                description,
                total_steps,
                parent_id
            )
            tracker.start_operation(operation_id)
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Handle generators/iterators for step-by-step progress updates
                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                    processed_results = []
                    for i, item in enumerate(result):
                        # If the item is a tuple where the first element is an integer,
                        # assume it's a progress update
                        if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], int):
                            step = item[0]
                            message = item[1] if len(item) > 1 else None
                            tracker.update_progress(operation_id, step, message)
                            # Process the rest of the tuple (if any) as the result
                            if len(item) > 2:
                                processed_results.append(item[2:])
                            else:
                                processed_results.append(None)
                        else:
                            # Regular iterator, assume it's a step counter
                            tracker.update_progress(operation_id, i + 1)
                            processed_results.append(item)
                    
                    tracker.complete_operation(operation_id)
                    return processed_results
                else:
                    # Not an iterator, complete immediately
                    tracker.complete_operation(operation_id)
                    return result
                
            except Exception as e:
                tracker.fail_operation(operation_id, str(e))
                raise
                
        return wrapper
    
    return decorator 
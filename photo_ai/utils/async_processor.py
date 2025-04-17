"""Asynchronous processing utilities for non-blocking operations."""

import asyncio
import threading
import concurrent.futures
import queue
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple, Union, TypeVar
from functools import wraps
from enum import Enum, auto
import uuid

T = TypeVar('T')  # Return type
R = TypeVar('R')  # Result type

class TaskStatus(Enum):
    """Task status enum."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class ProcessingTask:
    """Represents an asynchronous processing task."""
    
    def __init__(self, 
                 task_id: str, 
                 func: Callable, 
                 args: Tuple = None, 
                 kwargs: Dict[str, Any] = None,
                 callback: Optional[Callable] = None):
        """Initialize a processing task.
        
        Args:
            task_id: Unique task identifier
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            callback: Optional callback function to call when task completes
        """
        self.task_id = task_id
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.callback = callback
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
    
    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
    
    def mark_completed(self, result: Any) -> None:
        """Mark task as completed with result.
        
        Args:
            result: Task result
        """
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
        
        # Call callback if provided
        if self.callback is not None:
            try:
                self.callback(result)
            except Exception as e:
                logging.error(f"Error in task callback: {e}")
    
    def mark_failed(self, error: Exception) -> None:
        """Mark task as failed with error.
        
        Args:
            error: Exception that caused the failure
        """
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()
        
        # Call callback with error if provided
        if self.callback is not None:
            try:
                self.callback(None, error)
            except Exception as e:
                logging.error(f"Error in task error callback: {e}")
    
    def mark_cancelled(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
    
    def get_execution_time(self) -> Optional[float]:
        """Get task execution time in seconds.
        
        Returns:
            Execution time or None if not completed
        """
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at
    
    def get_wait_time(self) -> Optional[float]:
        """Get task wait time in seconds.
        
        Returns:
            Wait time or None if not started
        """
        if self.started_at is None:
            return None
        return self.started_at - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary.
        
        Returns:
            Dictionary representation of task
        """
        return {
            "task_id": self.task_id,
            "status": self.status.name,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.get_execution_time(),
            "wait_time": self.get_wait_time(),
            "has_error": self.error is not None
        }

class AsyncProcessor:
    """Manages asynchronous processing of tasks."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern for AsyncProcessor."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AsyncProcessor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the async processor."""
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.tasks: Dict[str, ProcessingTask] = {}
            self.task_queue = queue.Queue()
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(4, (threading.active_count() + 4))
            )
            self.stop_event = threading.Event()
            self.max_concurrent_tasks = 4
            self.active_tasks = 0
            self.active_tasks_lock = threading.Lock()
            
            # Start worker threads
            self.worker_threads = []
            for _ in range(self.max_concurrent_tasks):
                thread = threading.Thread(target=self._worker_task, daemon=True)
                thread.start()
                self.worker_threads.append(thread)
            
            self._initialized = True
    
    def _worker_task(self) -> None:
        """Worker thread task."""
        while not self.stop_event.is_set():
            try:
                # Get a task from the queue
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if we should process this task
                with self.active_tasks_lock:
                    if self.active_tasks >= self.max_concurrent_tasks:
                        # Too many active tasks, put it back in the queue
                        self.task_queue.put(task)
                        continue
                    
                    # Mark task as running
                    task.mark_running()
                    self.active_tasks += 1
                
                # Execute the task
                try:
                    result = task.func(*task.args, **task.kwargs)
                    task.mark_completed(result)
                except Exception as e:
                    self.logger.error(f"Task {task.task_id} failed: {e}")
                    task.mark_failed(e)
                finally:
                    # Mark task as complete
                    with self.active_tasks_lock:
                        self.active_tasks -= 1
                    
                    # Mark the task as done in the queue
                    self.task_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in worker thread: {e}")
    
    def submit_task(self, 
                   func: Callable[..., R], 
                   *args, 
                   callback: Optional[Callable[[Optional[R], Optional[Exception]], None]] = None, 
                   **kwargs) -> str:
        """Submit a task for asynchronous execution.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            callback: Optional callback function to call when task completes
            **kwargs: Keyword arguments for the function
            
        Returns:
            Task ID
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task
        task = ProcessingTask(task_id, func, args, kwargs, callback)
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to queue
        self.task_queue.put(task)
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if not found
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        return task.to_dict()
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if not completed or found
        """
        task = self.tasks.get(task_id)
        if task is None or task.status != TaskStatus.COMPLETED:
            return None
        return task.result
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was cancelled, False if not found or already running/completed
        """
        task = self.tasks.get(task_id)
        if task is None or task.status != TaskStatus.PENDING:
            return False
        
        task.mark_cancelled()
        return True
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks.
        
        Returns:
            List of all tasks
        """
        return [task.to_dict() for task in self.tasks.values()]
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get active tasks.
        
        Returns:
            List of active tasks
        """
        return [
            task.to_dict() for task in self.tasks.values() 
            if task.status == TaskStatus.RUNNING
        ]
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending tasks.
        
        Returns:
            List of pending tasks
        """
        return [
            task.to_dict() for task in self.tasks.values() 
            if task.status == TaskStatus.PENDING
        ]
    
    def clean_completed_tasks(self, max_age: float = 3600.0) -> int:
        """Clean completed tasks older than max_age.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            Number of tasks cleaned
        """
        current_time = time.time()
        task_ids_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.completed_at and (current_time - task.completed_at) > max_age:
                    task_ids_to_remove.append(task_id)
        
        for task_id in task_ids_to_remove:
            del self.tasks[task_id]
        
        return len(task_ids_to_remove)
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self.stop_event.set()
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=False)


def async_task(func: Callable[..., T]) -> Callable[..., Union[T, str]]:
    """Decorator to make a function run asynchronously.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that submits a task to the AsyncProcessor
    """
    processor = AsyncProcessor()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if async=False is specified
        run_async = kwargs.pop('async', True)
        
        if not run_async:
            # Run synchronously
            return func(*args, **kwargs)
        
        # Get callback if provided
        callback = kwargs.pop('callback', None)
        
        # Submit task
        return processor.submit_task(func, *args, callback=callback, **kwargs)
    
    return wrapper


def wait_for_task(task_id: str, timeout: Optional[float] = None) -> Optional[Any]:
    """Wait for a task to complete.
    
    Args:
        task_id: Task ID
        timeout: Optional timeout in seconds
        
    Returns:
        Task result or None if timeout or task not found
    """
    processor = AsyncProcessor()
    start_time = time.time()
    
    while timeout is None or (time.time() - start_time) < timeout:
        status = processor.get_task_status(task_id)
        
        if status is None:
            return None
        
        if status['status'] == TaskStatus.COMPLETED.name:
            return processor.get_task_result(task_id)
        
        if status['status'] in (TaskStatus.FAILED.name, TaskStatus.CANCELLED.name):
            return None
        
        # Sleep briefly
        time.sleep(0.1)
    
    # Timeout
    return None 
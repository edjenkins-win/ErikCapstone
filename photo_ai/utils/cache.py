import os
import time
import hashlib
import pickle
import json
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, List, Tuple
from pathlib import Path
from functools import lru_cache, wraps
import threading
import numpy as np

T = TypeVar('T')  # Generic type for cached values

class CacheEntry(Generic[T]):
    """Entry in the cache."""
    
    def __init__(self, value: T, expiry: Optional[float] = None):
        """Initialize a cache entry.
        
        Args:
            value: Value to cache
            expiry: Optional expiration time (None means no expiration)
        """
        self.value = value
        self.expiry = expiry
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """Check if the entry has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expiry is None:
            return False
        return time.time() > self.expiry

    def access(self) -> None:
        """Record an access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1

class Cache(Generic[T]):
    """General-purpose caching system with memory and disk caching."""
    
    def __init__(self, 
                 name: str, 
                 max_memory_items: int = 100, 
                 max_disk_items: int = 1000,
                 cache_dir: str = "cache",
                 ttl: Optional[float] = None):
        """Initialize the cache.
        
        Args:
            name: Name of the cache
            max_memory_items: Maximum number of items to cache in memory
            max_disk_items: Maximum number of items to cache on disk
            cache_dir: Directory for disk cache
            ttl: Default time-to-live in seconds (None means no expiration)
        """
        self.name = name
        self.max_memory_items = max_memory_items
        self.max_disk_items = max_disk_items
        self.ttl = ttl
        
        # Setup disk cache
        self.cache_dir = Path(cache_dir) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup memory cache
        self.memory_cache: Dict[str, CacheEntry[T]] = {}
        self.memory_cache_lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger(f"Cache.{name}")
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True)
        self.stop_event = threading.Event()
        self.cleanup_thread.start()
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get an item from the cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        # Check memory cache first
        with self.memory_cache_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    entry.access()
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
        
        # Check disk cache
        disk_path = self._get_disk_path(key)
        if disk_path.exists():
            try:
                with open(disk_path, 'rb') as f:
                    entry_data = pickle.load(f)
                
                # Recreate cache entry
                entry = CacheEntry[T](
                    entry_data['value'],
                    entry_data.get('expiry')
                )
                entry.created_at = entry_data.get('created_at', time.time())
                entry.last_accessed = entry_data.get('last_accessed', time.time())
                entry.access_count = entry_data.get('access_count', 0) + 1
                
                if not entry.is_expired():
                    # Promote to memory cache
                    with self.memory_cache_lock:
                        self.memory_cache[key] = entry
                        self._evict_if_needed(memory=True)
                    
                    # Update disk entry
                    self._save_to_disk(key, entry)
                    
                    return entry.value
                else:
                    # Remove expired entry
                    disk_path.unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Error loading cache entry from disk: {e}")
        
        return default
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds (overrides instance default)
        """
        expiry = None
        if ttl is not None:
            expiry = time.time() + ttl
        elif self.ttl is not None:
            expiry = time.time() + self.ttl
        
        entry = CacheEntry[T](value, expiry)
        
        # Add to memory cache
        with self.memory_cache_lock:
            self.memory_cache[key] = entry
            self._evict_if_needed(memory=True)
        
        # Add to disk cache
        self._save_to_disk(key, entry)
    
    def _save_to_disk(self, key: str, entry: CacheEntry[T]) -> None:
        """Save a cache entry to disk.
        
        Args:
            key: Cache key
            entry: Cache entry
        """
        try:
            disk_path = self._get_disk_path(key)
            
            # Create parent directories if needed
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare serializable entry data
            entry_data = {
                'value': entry.value,
                'expiry': entry.expiry,
                'created_at': entry.created_at,
                'last_accessed': entry.last_accessed,
                'access_count': entry.access_count
            }
            
            # Save to disk
            with open(disk_path, 'wb') as f:
                pickle.dump(entry_data, f)
            
            # Check disk cache size after adding
            self._evict_if_needed(memory=False)
        except Exception as e:
            self.logger.warning(f"Error saving cache entry to disk: {e}")
    
    def _get_disk_path(self, key: str) -> Path:
        """Get the disk path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Hash the key to make it filesystem-safe
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        # Use the first two characters as a directory to avoid too many files in one directory
        return self.cache_dir / key_hash[:2] / key_hash[2:]
    
    def _evict_if_needed(self, memory: bool = True) -> None:
        """Evict items if cache is too large.
        
        Args:
            memory: True to evict from memory cache, False to evict from disk cache
        """
        if memory:
            # Evict from memory cache
            while len(self.memory_cache) > self.max_memory_items:
                # Find least recently used item
                lru_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k].last_accessed
                )
                del self.memory_cache[lru_key]
        else:
            # Evict from disk cache
            try:
                # Count files in cache directory
                file_count = sum(1 for _ in self.cache_dir.glob('**/*') if _.is_file())
                
                if file_count > self.max_disk_items:
                    # Get all cache files with their modification times
                    cache_files = [
                        (f, f.stat().st_mtime)
                        for f in self.cache_dir.glob('**/*')
                        if f.is_file()
                    ]
                    
                    # Sort by modification time (oldest first)
                    cache_files.sort(key=lambda x: x[1])
                    
                    # Delete oldest files
                    num_to_delete = file_count - self.max_disk_items
                    for i in range(min(num_to_delete, len(cache_files))):
                        cache_files[i][0].unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Error evicting from disk cache: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        deleted = False
        
        # Remove from memory cache
        with self.memory_cache_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
                deleted = True
        
        # Remove from disk cache
        disk_path = self._get_disk_path(key)
        if disk_path.exists():
            disk_path.unlink()
            deleted = True
        
        return deleted
    
    def clear(self) -> None:
        """Clear the entire cache."""
        # Clear memory cache
        with self.memory_cache_lock:
            self.memory_cache.clear()
        
        # Clear disk cache
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Error clearing disk cache: {e}")
    
    def _cleanup_task(self) -> None:
        """Background task for cleaning up expired entries."""
        while not self.stop_event.is_set():
            try:
                # Clean memory cache
                with self.memory_cache_lock:
                    expired_keys = [
                        key for key, entry in self.memory_cache.items()
                        if entry.is_expired()
                    ]
                    for key in expired_keys:
                        del self.memory_cache[key]
                
                # Clean disk cache (only check a subset each time)
                disk_files = list(self.cache_dir.glob('**/*'))
                sample_size = min(100, len(disk_files))
                if sample_size > 0:
                    import random
                    sample = random.sample(disk_files, sample_size)
                    
                    for file_path in sample:
                        if file_path.is_file():
                            try:
                                with open(file_path, 'rb') as f:
                                    entry_data = pickle.load(f)
                                
                                # Check if expired
                                expiry = entry_data.get('expiry')
                                if expiry is not None and time.time() > expiry:
                                    file_path.unlink(missing_ok=True)
                            except Exception:
                                # If we can't read it, consider it corrupt and delete
                                file_path.unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"Error in cache cleanup task: {e}")
            
            # Sleep for a while
            for _ in range(60):  # Check every minute
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def close(self) -> None:
        """Clean up resources."""
        self.stop_event.set()
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1)


def cached(cache_name: str, ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        cache_name: Name of the cache
        ttl: Optional time-to-live in seconds
        key_func: Optional function to generate cache key from function arguments
    
    Returns:
        Decorated function
    """
    # Create or get the cache
    cache = Cache(cache_name, ttl=ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func is not None:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__module__, func.__name__]
                
                # Add args and kwargs to key
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        key_parts.append(str(arg))
                    elif isinstance(arg, (np.ndarray, list, tuple)):
                        key_parts.append(str(hash(str(arg))))
                    else:
                        key_parts.append(str(id(arg)))
                
                for k, v in sorted(kwargs.items()):
                    if isinstance(v, (str, int, float, bool)):
                        key_parts.append(f"{k}={v}")
                    elif isinstance(v, (np.ndarray, list, tuple)):
                        key_parts.append(f"{k}={hash(str(v))}")
                    else:
                        key_parts.append(f"{k}={id(v)}")
                
                key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Not in cache, call the function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(key, result, ttl)
            
            return result
        
        # Add function to clear this specific function's cache
        wrapper.clear_cache = cache.clear
        
        return wrapper
    
    return decorator 
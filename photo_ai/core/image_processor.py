from typing import Tuple, List, Union, BinaryIO, Optional
import numpy as np
from PIL import Image
import cv2
import io
from functools import lru_cache
import os
import logging
from pathlib import Path
from ..utils.cache import Cache, cached

class ImageProcessor:
    """Core image processing functionality following SRP."""

    # List of supported image formats
    SUPPORTED_FORMATS = {
        # Standard formats
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.png': 'PNG',
        '.tiff': 'TIFF',
        '.tif': 'TIFF',
        '.bmp': 'BMP',
        '.gif': 'GIF',
        # Modern formats
        '.webp': 'WEBP',
        '.heif': 'HEIF',
        '.heic': 'HEIF',
        '.avif': 'AVIF',
        '.jxr': 'JXR',
        # Raw formats
        '.dng': 'DNG',
        '.arw': 'ARW',  # Sony
        '.cr2': 'CR2',  # Canon
        '.cr3': 'CR3',  # Canon (newer)
        '.nef': 'NEF',  # Nikon
        '.nrw': 'NRW',  # Nikon
        '.orf': 'ORF',  # Olympus
        '.rw2': 'RW2',  # Panasonic
        '.raf': 'RAF',  # Fujifilm
        '.pef': 'PEF',  # Pentax
        '.srw': 'SRW',  # Samsung
        '.x3f': 'X3F',  # Sigma
        # HDR formats
        '.hdr': 'HDR',
        '.exr': 'EXR',
    }
    
    # Format fallback mapping (if primary loader fails)
    FORMAT_FALLBACKS = {
        'HEIF': ['JPEG'],
        'AVIF': ['JPEG'],
        'DNG': ['TIFF'],
        'CR3': ['TIFF'],
        'HDR': ['PNG'],
        'EXR': ['TIFF'],
    }
    
    # Initialize the cache at class level
    _image_cache = Cache[np.ndarray](
        name="image_processor",
        max_memory_items=200,  # Store more images in memory
        max_disk_items=5000,   # Store many images on disk
        cache_dir="cache/images",
        ttl=3600 * 24  # Cache for 24 hours
    )
    
    _logger = logging.getLogger(__name__)

    @staticmethod
    def load_image(image_source: Union[str, BinaryIO]) -> np.ndarray:
        """Load an image from file path or file-like object.

        Args:
            image_source: Path to the image file or file-like object

        Returns:
            Loaded image as numpy array in RGB format

        Raises:
            RuntimeError: If image loading fails
        """
        try:
            if isinstance(image_source, str):
                # Try to load from cache first if it's a file path
                cached_image = ImageProcessor._image_cache.get(image_source)
                if cached_image is not None:
                    ImageProcessor._logger.debug(f"Cache hit for image: {image_source}")
                    return cached_image
                
                # Not in cache, load normally
                return ImageProcessor._load_from_path(image_source)
            else:
                # For file-like objects
                return ImageProcessor._load_from_file_object(image_source)
        except Exception as e:
            raise RuntimeError(f"Error loading image: {str(e)}")

    @staticmethod
    def _load_from_path(image_path: str) -> np.ndarray:
        """Load an image from a file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array in RGB format
            
        Raises:
            ValueError: If image could not be loaded
        """
        # Get file extension
        ext = '.' + image_path.split('.')[-1].lower()
        format_name = ImageProcessor.SUPPORTED_FORMATS.get(ext, 'JPEG')
        
        # Try PIL first
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_array = np.array(img)
                
                # Store in cache
                ImageProcessor._image_cache.set(image_path, image_array)
                return image_array
        except Exception as pil_error:
            ImageProcessor._logger.debug(f"PIL failed to load image {image_path}: {str(pil_error)}")
            
            # Try fallback formats if available
            fallbacks = ImageProcessor.FORMAT_FALLBACKS.get(format_name, [])
            for fallback_format in fallbacks:
                try:
                    ImageProcessor._logger.debug(f"Trying fallback format {fallback_format} for {image_path}")
                    with Image.open(image_path) as img:
                        # Force a specific format
                        img.format = fallback_format
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        image_array = np.array(img)
                        
                        # Store in cache
                        ImageProcessor._image_cache.set(image_path, image_array)
                        return image_array
                except Exception:
                    # Continue to next fallback
                    pass
            
            # Fall back to OpenCV for compatibility
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image at {image_path}: {str(pil_error)}")
                image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Store in cache
                ImageProcessor._image_cache.set(image_path, image_array)
                return image_array
            except Exception as cv_error:
                raise ValueError(f"Failed to load image with both PIL and OpenCV: {str(cv_error)}")

    @staticmethod
    def _load_from_file_object(file_obj: BinaryIO) -> np.ndarray:
        """Load an image from a file-like object.
        
        Args:
            file_obj: File-like object containing image data
            
        Returns:
            Loaded image as numpy array in RGB format
            
        Raises:
            ValueError: If image could not be loaded
        """
        # Save the current position
        pos = file_obj.tell()
        
        # Try PIL first
        try:
            # Reset to beginning
            file_obj.seek(0)

            with Image.open(file_obj) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_array = np.array(img)

            # Reset to original position
            file_obj.seek(pos)
            return image_array
        except Exception as pil_error:
            # Fall back to OpenCV
            try:
                # Reset to beginning
                file_obj.seek(0)
                file_bytes = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not decode image from file object: {str(pil_error)}")
                # Reset to original position
                file_obj.seek(pos)
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as cv_error:
                # Reset position before raising
                file_obj.seek(pos)
                raise ValueError(f"Failed to load image with both PIL and OpenCV: {str(cv_error)}")

    @staticmethod
    def cached_load_image(image_path: str) -> np.ndarray:
        """Load an image from file path with caching.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array in RGB format
        """
        # This now uses the improved Cache system instead of lru_cache
        return ImageProcessor.load_image(image_path)

    @staticmethod
    def clear_image_cache() -> None:
        """Clear the image cache to free memory."""
        ImageProcessor._image_cache.clear()
        ImageProcessor._logger.info("Image cache cleared")

    @staticmethod
    def get_cache_stats() -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        memory_cache_size = len(ImageProcessor._image_cache.memory_cache)
        
        # Count files in cache directory
        disk_cache_count = 0
        try:
            for root, _, files in os.walk(ImageProcessor._image_cache.cache_dir):
                disk_cache_count += len(files)
        except Exception:
            pass
            
        return {
            "memory_items": memory_cache_size,
            "disk_items": disk_cache_count
        }

    @staticmethod
    def get_format_from_extension(file_path: str) -> str:
        """Determine the image format from file extension.

        Args:
            file_path: Path to the image file

        Returns:
            PIL format name (e.g., 'JPEG', 'PNG')
        """
        ext = '.' + file_path.split('.')[-1].lower()
        return ImageProcessor.SUPPORTED_FORMATS.get(ext, 'JPEG')  # Default to JPEG if unknown

    @staticmethod
    def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> None:
        """Save an image to file path.

        Args:
            image: Image as numpy array in RGB format
            output_path: Path where the image will be saved
            quality: Quality for lossy formats (0-100)

        Raises:
            RuntimeError: If image saving fails
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')

            # Determine format from file extension
            format_name = ImageProcessor.get_format_from_extension(output_path)

            # Save with appropriate parameters
            save_args = {}
            if format_name == 'JPEG':
                save_args['quality'] = quality
                save_args['optimize'] = True
                save_args['subsampling'] = 0  # Better color accuracy
            elif format_name == 'PNG':
                save_args['optimize'] = True
                save_args['compress_level'] = 9  # Maximum compression
            elif format_name == 'WEBP':
                save_args['quality'] = quality
                save_args['method'] = 6  # Higher quality compression
            elif format_name == 'TIFF':
                save_args['compression'] = 'tiff_lzw'  # Lossless compression
            elif format_name == 'HEIF' or format_name == 'AVIF':
                save_args['quality'] = quality
                
            # Try to save with PIL
            try:
                pil_image.save(output_path, format=format_name, **save_args)
            except Exception as pil_error:
                # For formats PIL doesn't support well, fall back to OpenCV
                ImageProcessor._logger.warning(f"PIL failed to save {format_name} image: {str(pil_error)}")
                
                # OpenCV fallback
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Convert format name to OpenCV extension
                ext = os.path.splitext(output_path)[1].lower()
                
                # Set compression parameters for OpenCV
                if ext == '.jpg' or ext == '.jpeg':
                    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                elif ext == '.png':
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
                elif ext == '.webp':
                    params = [cv2.IMWRITE_WEBP_QUALITY, quality]
                else:
                    params = []
                    
                success = cv2.imwrite(output_path, bgr_image, params)
                
                if not success:
                    raise RuntimeError(f"OpenCV failed to save image as {format_name}")
                
        except Exception as e:
            raise RuntimeError(f"Error saving image to {output_path}: {str(e)}")

    @staticmethod
    def image_to_bytes(image: np.ndarray, format: str = '.jpg', quality: int = 95) -> bytes:
        """Convert an image to bytes for download or storage.

        Args:
            image: Image as numpy array in RGB format
            format: Image format extension (e.g., '.jpg', '.png', '.webp')
            quality: Quality for lossy formats (0-100)

        Returns:
            Image encoded as bytes

        Raises:
            RuntimeError: If image conversion fails
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')

            # Determine format name
            format_name = ImageProcessor.SUPPORTED_FORMATS.get(format.lower(), 'JPEG')

            # Save to bytes buffer
            buffer = io.BytesIO()

            # Save with appropriate parameters
            save_args = {}
            if format_name == 'JPEG':
                save_args['quality'] = quality
                save_args['optimize'] = True
                save_args['subsampling'] = 0  # Better color accuracy
            elif format_name == 'PNG':
                save_args['optimize'] = True
                save_args['compress_level'] = 9  # Maximum compression
            elif format_name == 'WEBP':
                save_args['quality'] = quality
                save_args['method'] = 6  # Higher quality compression
            elif format_name == 'TIFF':
                save_args['compression'] = 'tiff_lzw'  # Lossless compression
                
            # Try to save with PIL
            try:
                pil_image.save(buffer, format=format_name, **save_args)
                buffer.seek(0)
                return buffer.getvalue()
            except Exception as pil_error:
                # Fall back to OpenCV
                ImageProcessor._logger.warning(f"PIL failed to encode {format_name} image: {str(pil_error)}")
                
                # OpenCV fallback
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Set compression parameters for OpenCV
                if format.lower() in ['.jpg', '.jpeg']:
                    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                elif format.lower() == '.png':
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
                elif format.lower() == '.webp':
                    params = [cv2.IMWRITE_WEBP_QUALITY, quality]
                else:
                    params = []
                    
                success, encoded_image = cv2.imencode(format, bgr_image, params)
                
                if not success:
                    raise ValueError(f"Failed to encode image to {format} format")
                return encoded_image.tobytes()
        except Exception as e:
            raise RuntimeError(f"Error converting image to bytes: {str(e)}")

    @staticmethod
    def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
        """Get image dimensions (height, width)."""
        return image.shape[:2]

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image values to [0, 1] range."""
        return image.astype(np.float32) / 255.0

    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """Convert normalized image back to [0, 255] range."""
        return (image * 255).astype(np.uint8)

    @staticmethod
    def is_format_supported(file_path: str) -> bool:
        """Check if image format is supported.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            True if format is supported, False otherwise
        """
        ext = '.' + file_path.split('.')[-1].lower()
        return ext in ImageProcessor.SUPPORTED_FORMATS
        
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported image format extensions.
        
        Returns:
            List of supported extensions (with dots, e.g., '.jpg')
        """
        return list(ImageProcessor.SUPPORTED_FORMATS.keys())

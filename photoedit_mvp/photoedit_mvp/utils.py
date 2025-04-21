"""Utility functions for the Photo Editor application."""

import numpy as np
from typing import Tuple, List, Union, BinaryIO, Optional
from PIL import Image
import cv2
import io
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define supported image formats
SUPPORTED_FORMATS = {
    # Standard formats
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.png': 'PNG',
    '.tiff': 'TIFF',
    '.tif': 'TIFF',
    '.bmp': 'BMP',
    '.gif': 'GIF',
}

def load_image(source: Union[str, BinaryIO, np.ndarray]) -> np.ndarray:
    """Load an image from a file path, file object, or numpy array.
    
    Args:
        source: Source image as a file path, file object, or numpy array
        
    Returns:
        Loaded image as numpy array in RGB format
        
    Raises:
        ValueError: If image could not be loaded
    """
    # If source is already a numpy array, just return it
    if isinstance(source, np.ndarray):
        return source
        
    # If source is a file path, load from path
    if isinstance(source, str):
        try:
            with Image.open(source) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        except Exception as pil_error:
            # Fall back to OpenCV
            try:
                image = cv2.imread(source)
                if image is None:
                    raise ValueError(f"Could not load image at {source}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as cv_error:
                raise ValueError(f"Failed to load image with both PIL and OpenCV: {str(cv_error)}")
    
    # If source is a file-like object, load from file object
    try:
        # Save the current position
        pos = source.tell()
        
        # Reset to beginning
        source.seek(0)

        with Image.open(source) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_array = np.array(img)

        # Reset to original position
        source.seek(pos)
        return image_array
    except Exception as pil_error:
        # Fall back to OpenCV
        try:
            # Reset to beginning
            source.seek(0)
            file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image from file object")
            # Reset to original position
            source.seek(pos)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as cv_error:
            # Reset position before raising
            source.seek(pos)
            raise ValueError(f"Failed to load image with both PIL and OpenCV: {str(cv_error)}")

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
        ext = os.path.splitext(output_path)[1].lower()
        format_name = SUPPORTED_FORMATS.get(ext, 'JPEG')

        # Save with appropriate parameters
        save_args = {}
        if format_name == 'JPEG':
            save_args['quality'] = quality
            save_args['optimize'] = True
        elif format_name == 'PNG':
            save_args['optimize'] = True
            
        pil_image.save(output_path, format=format_name, **save_args)
    except Exception as e:
        raise RuntimeError(f"Error saving image to {output_path}: {str(e)}")

def image_to_bytes(image: np.ndarray, format: str = '.jpg', quality: int = 95) -> bytes:
    """Convert an image to bytes for storage.

    Args:
        image: Image as numpy array in RGB format
        format: Image format extension (e.g., '.jpg', '.png')
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
        format_name = SUPPORTED_FORMATS.get(format.lower(), 'JPEG')

        # Save to bytes buffer
        buffer = io.BytesIO()

        # Save with appropriate parameters
        save_args = {}
        if format_name == 'JPEG':
            save_args['quality'] = quality
            save_args['optimize'] = True
        elif format_name == 'PNG':
            save_args['optimize'] = True
            
        # Save with PIL
        pil_image.save(buffer, format=format_name, **save_args)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"Error converting image to bytes: {str(e)}")

def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """Get image dimensions (height, width)."""
    return image.shape[:2]

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image values to [0, 1] range."""
    return image.astype(np.float32) / 255.0

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Convert normalized image back to [0, 255] range."""
    return (image * 255).astype(np.uint8)

def is_supported_format(file_path: str) -> bool:
    """Check if image format is supported.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        True if format is supported, False otherwise
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_FORMATS

def get_supported_formats() -> List[str]:
    """Get list of supported image format extensions.
    
    Returns:
        List of supported extensions (with dots, e.g., '.jpg')
    """
    return list(SUPPORTED_FORMATS.keys())

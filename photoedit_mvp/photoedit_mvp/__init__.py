"""Photo Editor application.

An AI-driven tool that analyzes an uploaded image, generates a list of recommended adjustments,
lets the user pick a styling option, and applies those edits automatically.
"""

__version__ = "0.1.0"

from .analyzer import ImageAnalyzer, Adjustment
from .executor import ImageExecutor
from .styles import get_available_styles, get_style_preset
from .utils import load_image, save_image

# Import web app if Streamlit is installed
try:
    from .web import run_web_app
except ImportError:
    def run_web_app():
        """Placeholder function when Streamlit is not installed."""
        raise ImportError("Streamlit is required to run the web app. Please install it with 'pip install streamlit'")

def analyze_image(image_source):
    """Analyze an image and return recommended adjustments.
    
    Args:
        image_source: Path to the image or a numpy array
        
    Returns:
        List of recommended adjustments
    """
    analyzer = ImageAnalyzer()
    return analyzer.analyze(image_source)

def apply_adjustments(image_source, adjustments, style=None):
    """Apply adjustments and/or a style to an image.
    
    Args:
        image_source: Path to the image or a numpy array
        adjustments: List of adjustments to apply
        style: Optional style preset name
        
    Returns:
        Processed image as a numpy array
    """
    executor = ImageExecutor()
    return executor.apply(image_source, adjustments, style)

"""Image analysis module for the Photo Editor application.

This module is responsible for analyzing images and generating adjustment recommendations.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from .utils import load_image, normalize_image, denormalize_image

# Set up logging
logger = logging.getLogger(__name__)

class Adjustment:
    """Represents a single adjustment recommendation."""
    
    def __init__(self, parameter: str, suggested: float, unit: str, description: str = ""):
        """Initialize an adjustment.
        
        Args:
            parameter: The parameter name (e.g., "exposure", "contrast")
            suggested: The suggested adjustment value
            unit: The unit of the adjustment (e.g., "EV", "multiplier")
            description: A human-readable description of the adjustment
        """
        self.parameter = parameter
        self.suggested = suggested
        self.unit = unit
        self.description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the adjustment to a dictionary."""
        return {
            "parameter": self.parameter,
            "suggested": self.suggested,
            "unit": self.unit,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Adjustment':
        """Create an adjustment from a dictionary."""
        return cls(
            parameter=data["parameter"],
            suggested=data["suggested"],
            unit=data.get("unit", ""),
            description=data.get("description", "")
        )

class ImageAnalyzer:
    """Analyzes images to assess quality and suggest adjustments."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the image analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        defaults = {
            'detection_threshold': 0.5,
            'analyze_exposure': True,
            'analyze_contrast': True,
            'analyze_noise': True,
            'analyze_sharpness': True,
            'analyze_color': True,
            'exposure_target': 0.5,  # Target mean brightness (0-1)
            'contrast_target': 0.2,  # Target standard deviation (0-1)
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def analyze(self, image_source: Union[str, np.ndarray]) -> List[Adjustment]:
        """Analyze an image and generate adjustment recommendations.
        
        Args:
            image_source: Input image as file path or numpy array
            
        Returns:
            List of recommended adjustments
        """
        # Load the image if it's a file path
        image = load_image(image_source)
        
        # Initialize empty list of adjustments
        adjustments = []
        
        # Convert to various color spaces for analysis
        rgb = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Analyze exposure
        if self.config['analyze_exposure']:
            exposure_adjustment = self._analyze_exposure(gray)
            if exposure_adjustment is not None:
                adjustments.append(exposure_adjustment)
        
        # Analyze contrast
        if self.config['analyze_contrast']:
            contrast_adjustment = self._analyze_contrast(gray)
            if contrast_adjustment is not None:
                adjustments.append(contrast_adjustment)
        
        # Analyze noise
        if self.config['analyze_noise']:
            noise_adjustment = self._analyze_noise(gray)
            if noise_adjustment is not None:
                adjustments.append(noise_adjustment)
        
        # Analyze sharpness
        if self.config['analyze_sharpness']:
            sharpness_adjustment = self._analyze_sharpness(gray)
            if sharpness_adjustment is not None:
                adjustments.append(sharpness_adjustment)
        
        # Analyze color
        if self.config['analyze_color']:
            color_adjustments = self._analyze_color(rgb, hsv)
            adjustments.extend(color_adjustments)
        
        return adjustments
    
    def _analyze_exposure(self, gray: np.ndarray) -> Optional[Adjustment]:
        """Analyze exposure and recommend adjustment if needed.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Exposure adjustment or None if no adjustment needed
        """
        # Calculate mean brightness (0-1)
        mean_brightness = np.mean(gray) / 255.0
        target = self.config['exposure_target']
        
        # If brightness is significantly off target, recommend adjustment
        if abs(mean_brightness - target) > 0.05:
            # Calculate EV adjustment (logarithmic)
            ev_adjustment = np.log2(target / mean_brightness) if mean_brightness > 0 else 0
            
            # Clip to reasonable range (-2 to 2 EV)
            ev_adjustment = np.clip(ev_adjustment, -2.0, 2.0)
            
            # Round to nearest 0.1
            ev_adjustment = round(ev_adjustment * 10) / 10
            
            # Only recommend adjustment if it's significant
            if abs(ev_adjustment) >= 0.1:
                description = "Increase exposure" if ev_adjustment > 0 else "Decrease exposure"
                return Adjustment(
                    parameter="exposure",
                    suggested=ev_adjustment,
                    unit="EV",
                    description=description
                )
        
        return None
    
    def _analyze_contrast(self, gray: np.ndarray) -> Optional[Adjustment]:
        """Analyze contrast and recommend adjustment if needed.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Contrast adjustment or None if no adjustment needed
        """
        # Calculate standard deviation as measure of contrast
        std_dev = np.std(gray) / 255.0
        target = self.config['contrast_target']
        
        # If contrast is significantly off target, recommend adjustment
        if abs(std_dev - target) > 0.03:
            # Calculate multiplier
            multiplier = target / std_dev if std_dev > 0 else 1.0
            
            # Clip to reasonable range (0.5 to 2.0)
            multiplier = np.clip(multiplier, 0.5, 2.0)
            
            # Round to nearest 0.05
            multiplier = round(multiplier * 20) / 20
            
            # Only recommend adjustment if it's significant
            if abs(multiplier - 1.0) >= 0.05:
                description = "Increase contrast" if multiplier > 1.0 else "Decrease contrast"
                return Adjustment(
                    parameter="contrast",
                    suggested=multiplier,
                    unit="multiplier",
                    description=description
                )
        
        return None
    
    def _analyze_noise(self, gray: np.ndarray) -> Optional[Adjustment]:
        """Analyze noise and recommend noise reduction if needed.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Noise reduction adjustment or None if no adjustment needed
        """
        # Simple noise estimation
        # Apply Gaussian blur to remove details
        blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
        # Calculate difference between original and blurred
        diff = cv2.absdiff(gray, blurred)
        # Noise level is the mean of the difference
        noise_level = np.mean(diff) / 255.0
        
        # If noise level is significant, recommend noise reduction
        if noise_level > 0.03:
            # Scale from 0-1 based on noise level
            strength = min(noise_level * 5.0, 1.0)
            
            # Round to nearest 0.05
            strength = round(strength * 20) / 20
            
            return Adjustment(
                parameter="noise_reduction",
                suggested=strength,
                unit="strength",
                description="Apply noise reduction"
            )
        
        return None
    
    def _analyze_sharpness(self, gray: np.ndarray) -> Optional[Adjustment]:
        """Analyze sharpness and recommend sharpening if needed.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Sharpening adjustment or None if no adjustment needed
        """
        # Calculate Laplacian variance as measure of sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Normalize to 0-1 range (empirical values)
        normalized_sharpness = min(sharpness / 1000.0, 1.0)
        
        # If image isn't sharp enough, recommend sharpening
        if normalized_sharpness < 0.4:
            # Calculate strength based on current sharpness
            strength = 0.5 * (1.0 - normalized_sharpness)
            
            # Round to nearest 0.05
            strength = round(strength * 20) / 20
            
            # Only recommend if significant
            if strength >= 0.1:
                return Adjustment(
                    parameter="sharpening",
                    suggested=strength,
                    unit="strength",
                    description="Apply sharpening"
                )
        
        return None
    
    def _analyze_color(self, rgb: np.ndarray, hsv: np.ndarray) -> List[Adjustment]:
        """Analyze color balance and saturation.
        
        Args:
            rgb: RGB image
            hsv: HSV image
            
        Returns:
            List of color-related adjustments
        """
        adjustments = []
        
        # Analyze saturation
        mean_saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # If saturation is too low, recommend increase
        if mean_saturation < 0.3:
            strength = min((0.4 - mean_saturation) * 2.0, 0.5)
            strength = round(strength * 20) / 20
            
            if strength >= 0.1:
                adjustments.append(Adjustment(
                    parameter="saturation",
                    suggested=strength,
                    unit="increase",
                    description="Increase saturation"
                ))
        # If saturation is too high, recommend decrease
        elif mean_saturation > 0.7:
            strength = min((mean_saturation - 0.6) * 2.0, 0.5)
            strength = round(strength * 20) / 20
            
            if strength >= 0.1:
                adjustments.append(Adjustment(
                    parameter="saturation",
                    suggested=-strength,
                    unit="decrease",
                    description="Decrease saturation"
                ))
        
        # Analyze white balance (color temperature)
        mean_r = np.mean(rgb[:, :, 0])
        mean_g = np.mean(rgb[:, :, 1])
        mean_b = np.mean(rgb[:, :, 2])
        
        # Calculate ratios
        if mean_g > 0:
            rb_ratio = mean_r / mean_g
            bb_ratio = mean_b / mean_g
            
            # Determine if image is too warm or too cool
            if rb_ratio > 1.1 and bb_ratio < 0.9:
                # Too warm (reddish)
                adjustment = -0.2
                description = "Cool down image"
            elif rb_ratio < 0.9 and bb_ratio > 1.1:
                # Too cool (bluish)
                adjustment = 0.2
                description = "Warm up image"
            else:
                # Color temperature is acceptable
                adjustment = 0
                description = ""
            
            if abs(adjustment) > 0:
                adjustments.append(Adjustment(
                    parameter="temperature",
                    suggested=adjustment,
                    unit="shift",
                    description=description
                ))
        
        return adjustments

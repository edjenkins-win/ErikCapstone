"""Image executor module for the Photo Editor application.

This module applies adjustments to images based on the analyzer's recommendations.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional
from .analyzer import Adjustment
from .utils import load_image, save_image, normalize_image, denormalize_image

# Set up logging
logger = logging.getLogger(__name__)

class ImageExecutor:
    """Applies adjustments to images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the image executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        defaults = {
            'max_adjustment_intensity': 1.0,  # Scale factor for all adjustments (0-1)
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def apply(self, image_source: Union[str, np.ndarray], adjustments: List[Adjustment], style: str = None) -> np.ndarray:
        """Apply adjustments to an image.
        
        Args:
            image_source: Input image as file path or numpy array
            adjustments: List of adjustments to apply
            style: Optional style preset name
            
        Returns:
            Processed image as numpy array
        """
        # Load the image if it's a file path
        image = load_image(image_source)
        result = image.copy()
        
        # First apply the style preset if specified
        if style:
            result = self._apply_style(result, style)
        
        # Then apply individual adjustments
        for adjustment in adjustments:
            result = self._apply_adjustment(result, adjustment)
        
        return result
    
    def _apply_style(self, image: np.ndarray, style: str) -> np.ndarray:
        """Apply a style preset to an image.
        
        Args:
            image: Input image
            style: Style preset name
            
        Returns:
            Processed image
        """
        # Copy the image to avoid modifying the original
        result = image.copy()
        
        # Apply the appropriate style preset
        style_lower = style.lower().replace(' ', '-')
        
        if style_lower == "auto-enhance":
            result = self._style_auto_enhance(result)
        elif style_lower == "portrait":
            result = self._style_portrait(result)
        elif style_lower == "vintage":
            result = self._style_vintage(result)
        # Add new cinematic styles
        elif style_lower == "cinematic-teal-orange" or style_lower == "cinematic-teal-&-orange":
            result = self._style_cinematic_teal_orange(result)
        elif style_lower == "film-noir":
            result = self._style_film_noir(result)
        elif style_lower == "anamorphic":
            result = self._style_anamorphic(result)
        elif style_lower == "blockbuster":
            result = self._style_blockbuster(result)
        elif style_lower == "dreamy":
            result = self._style_dreamy(result)
        else:
            logger.warning(f"Unknown style preset: {style}")
        
        return result
    
    def _style_auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply the Auto-Enhance style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply slight sharpening
        enhanced = self._apply_sharpening(enhanced, 0.2)
        
        return enhanced
    
    def _style_portrait(self, image: np.ndarray) -> np.ndarray:
        """Apply the Portrait style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Start with auto-enhance
        result = self._style_auto_enhance(image)
        
        # Apply skin smoothing (bilateral filter)
        # Convert to float32 for better precision
        float_img = result.astype(np.float32) / 255.0
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(float_img, 9, 75, 75)
        # Convert back to uint8
        smoothed = (smoothed * 255).astype(np.uint8)
        
        # Warm up the image slightly
        result = self._apply_temperature(smoothed, 0.1)
        
        # Slightly increase saturation
        result = self._apply_saturation(result, 0.1)
        
        return result
    
    def _style_vintage(self, image: np.ndarray) -> np.ndarray:
        """Apply the Vintage style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Create a vintage look
        # Slightly reduce contrast
        result = self._apply_contrast(image, 0.8)
        
        # Apply a slight sepia tone
        sepia = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Convert to float for matrix multiplication
        float_img = result.astype(np.float32) / 255.0
        sepia_img = np.zeros_like(float_img)
        
        # Apply sepia matrix
        for i in range(3):
            sepia_img[:, :, i] = np.sum(float_img * sepia[i], axis=2)
        
        # Clip values to valid range
        sepia_img = np.clip(sepia_img, 0, 1.0)
        
        # Convert back to uint8
        result = (sepia_img * 255).astype(np.uint8)
        
        # Add slight vignette
        height, width = result.shape[:2]
        
        # Create vignette mask
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x**2 + y**2)
        
        # Create vignette
        vignette = np.clip(1.0 - radius * 0.5, 0, 1.0)
        vignette = np.dstack([vignette] * 3)
        
        # Apply vignette
        result = (result.astype(np.float32) * vignette).astype(np.uint8)
        
        return result
        
    def _style_cinematic_teal_orange(self, image: np.ndarray) -> np.ndarray:
        """Apply the Cinematic Teal & Orange style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Enhance contrast first
        result = self._apply_contrast(image, 1.3)
        
        # Convert to float for processing
        img_float = result.astype(np.float32) / 255.0
        
        # Split into channels
        b, g, r = cv2.split(img_float)
        
        # Manipulate shadows (blue/teal) and highlights (orange)
        # Boost blue in shadows
        shadows = 1.0 - ((r + g) / 2.0)  # Approximate shadow areas
        b_new = b + (shadows * 0.15)  # Add blue to shadows
        
        # Boost orange in highlights
        highlights = ((r + g) / 2.0)  # Approximate highlight areas
        r_new = r + (highlights * 0.1)  # Add red to highlights
        g_new = g + (highlights * 0.05)  # Add a bit of green for orange
        
        # Clip to valid range
        b_new = np.clip(b_new, 0, 1.0)
        g_new = np.clip(g_new, 0, 1.0)
        r_new = np.clip(r_new, 0, 1.0)
        
        # Merge channels
        result = cv2.merge([b_new, g_new, r_new])
        
        # Convert back to uint8
        result = (result * 255).astype(np.uint8)
        
        # Apply slight sharpening
        result = self._apply_sharpening(result, 0.25)
        
        # Add subtle vignette
        height, width = result.shape[:2]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x**2 + y**2)
        vignette = np.clip(1.0 - radius * 0.3, 0, 1.0)
        vignette = np.dstack([vignette] * 3)
        result = (result.astype(np.float32) * vignette).astype(np.uint8)
        
        return result
    
    def _style_film_noir(self, image: np.ndarray) -> np.ndarray:
        """Apply the Film Noir style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # First convert to black and white with high contrast
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE for better contrast distribution
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Enhance contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-10)
        
        # Convert back to RGB
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Add strong vignette for dramatic effect
        height, width = result.shape[:2]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x**2 + y**2)
        vignette = np.clip(1.0 - radius * 0.8, 0, 1.0)
        vignette = np.dstack([vignette] * 3)
        result = (result.astype(np.float32) * vignette).astype(np.uint8)
        
        # Add film grain
        noise = np.random.normal(0, 0.03, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) / 255.0 + noise, 0, 1) * 255
        result = result.astype(np.uint8)
        
        return result
    
    def _style_anamorphic(self, image: np.ndarray) -> np.ndarray:
        """Apply the Anamorphic style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Enhance contrast and color
        result = self._apply_contrast(image, 1.25)
        result = self._apply_saturation(result, 0.1)
        
        # Shift color balance slightly toward blue
        result = self._apply_temperature(result, -0.1)
        
        # Add horizontal lens flare effect (blue streak)
        height, width = result.shape[:2]
        
        # Create horizontal flare in random position
        flare_y = np.random.randint(height // 4, 3 * height // 4)
        flare = np.zeros_like(result, dtype=np.float32)
        
        # Create horizontal blue streak
        for y in range(max(0, flare_y - 5), min(height, flare_y + 6)):
            intensity = 1.0 - abs(y - flare_y) / 5.0
            for x in range(width):
                # Blue-tinted flare with falloff from center
                dist_from_center = abs(x - width // 2) / (width // 2)
                flare_intensity = intensity * (1.0 - dist_from_center**2) * 0.4
                flare[y, x, 0] = flare_intensity * 0.8  # Blue channel
                flare[y, x, 1] = flare_intensity * 0.3  # Green channel
                flare[y, x, 2] = flare_intensity * 0.2  # Red channel
        
        # Add flare to image
        result = np.clip(result.astype(np.float32) + flare * 255, 0, 255).astype(np.uint8)
        
        # Add letterbox effect to simulate widescreen
        letterbox_height = height // 6
        result[0:letterbox_height, :] = [0, 0, 0]
        result[height - letterbox_height:height, :] = [0, 0, 0]
        
        return result
    
    def _style_blockbuster(self, image: np.ndarray) -> np.ndarray:
        """Apply the Blockbuster style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Increase exposure slightly
        result = self._apply_exposure(image, 0.1)
        
        # Enhance contrast dramatically
        result = self._apply_contrast(result, 1.4)
        
        # Boost saturation for vivid colors
        result = self._apply_saturation(result, 0.25)
        
        # Enhance sharpness
        result = self._apply_sharpening(result, 0.3)
        
        # Add color tint based on dominant colors (typical blockbuster color grading)
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Shift colors slightly toward blue/cyan for shadows and orange/yellow for highlights
        # This creates the modern blockbuster look
        mask = v < 128  # Shadow areas
        h[mask] = np.clip(h[mask] + 10, 0, 179)  # Shift toward blue-cyan
        
        mask = v >= 128  # Highlight areas
        h[mask] = np.clip(h[mask] - 10, 0, 179)  # Shift toward orange-yellow
        
        # Merge channels
        hsv = cv2.merge([h, s, v])
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add subtle vignette
        height, width = result.shape[:2]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x**2 + y**2)
        vignette = np.clip(1.0 - radius * 0.2, 0, 1.0)
        vignette = np.dstack([vignette] * 3)
        result = (result.astype(np.float32) * vignette).astype(np.uint8)
        
        return result
    
    def _style_dreamy(self, image: np.ndarray) -> np.ndarray:
        """Apply the Dreamy style.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Reduce contrast
        result = self._apply_contrast(image, 0.85)
        
        # Add warmth
        result = self._apply_temperature(result, 0.15)
        
        # Apply soft glow effect
        # Create blurred version
        blur = cv2.GaussianBlur(result, (21, 21), 0)
        
        # Blend with original (soft glow effect)
        result = cv2.addWeighted(result, 0.7, blur, 0.3, 0)
        
        # Apply noise reduction
        result = self._apply_noise_reduction(result, 0.4)
        
        # Add dreamy haze/fog effect
        height, width = result.shape[:2]
        fog = np.ones_like(result) * 255  # White fog
        
        # Create fog gradient
        for y in range(height):
            for x in range(width):
                # Calculate distance from top left
                dist = np.sqrt((x / width)**2 + (y / height)**2)
                fog_alpha = 0.15 * (1.0 - dist)  # Stronger fog in top-left corner
                result[y, x] = cv2.addWeighted(result[y, x], 1 - fog_alpha, fog[y, x], fog_alpha, 0)
        
        # Add vignette for dreamy border effect
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)
        radius = np.sqrt(x**2 + y**2)
        vignette = np.clip(1.0 - radius * 0.4, 0, 1.0)
        vignette = np.dstack([vignette] * 3)
        result = (result.astype(np.float32) * vignette).astype(np.uint8)
        
        return result
    
    def _apply_adjustment(self, image: np.ndarray, adjustment: Adjustment) -> np.ndarray:
        """Apply a single adjustment to an image.
        
        Args:
            image: Input image
            adjustment: Adjustment to apply
            
        Returns:
            Processed image
        """
        # Scale the adjustment by the max intensity
        intensity = self.config['max_adjustment_intensity']
        
        # Apply the appropriate adjustment based on parameter
        if adjustment.parameter == "exposure":
            return self._apply_exposure(image, adjustment.suggested * intensity)
        elif adjustment.parameter == "contrast":
            return self._apply_contrast(image, adjustment.suggested)
        elif adjustment.parameter == "noise_reduction":
            return self._apply_noise_reduction(image, adjustment.suggested * intensity)
        elif adjustment.parameter == "sharpening":
            return self._apply_sharpening(image, adjustment.suggested * intensity)
        elif adjustment.parameter == "saturation":
            return self._apply_saturation(image, adjustment.suggested * intensity)
        elif adjustment.parameter == "temperature":
            return self._apply_temperature(image, adjustment.suggested * intensity)
        else:
            logger.warning(f"Unknown adjustment parameter: {adjustment.parameter}")
            return image
    
    def _apply_exposure(self, image: np.ndarray, ev: float) -> np.ndarray:
        """Apply exposure adjustment.
        
        Args:
            image: Input image
            ev: Exposure value adjustment in EV
            
        Returns:
            Processed image
        """
        # Convert to float
        float_img = image.astype(np.float32) / 255.0
        
        # Calculate multiplier from EV
        multiplier = 2 ** ev
        
        # Apply adjustment
        result = float_img * multiplier
        
        # Clip values
        result = np.clip(result, 0, 1.0)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)
    
    def _apply_contrast(self, image: np.ndarray, multiplier: float) -> np.ndarray:
        """Apply contrast adjustment.
        
        Args:
            image: Input image
            multiplier: Contrast multiplier
            
        Returns:
            Processed image
        """
        # Convert to float
        float_img = image.astype(np.float32) / 255.0
        
        # Calculate mean
        mean = np.mean(float_img, axis=(0, 1), keepdims=True)
        
        # Apply contrast adjustment (center around mean)
        result = (float_img - mean) * multiplier + mean
        
        # Clip values
        result = np.clip(result, 0, 1.0)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)
    
    def _apply_noise_reduction(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply noise reduction.
        
        Args:
            image: Input image
            strength: Strength of noise reduction (0-1)
            
        Returns:
            Processed image
        """
        # Apply bilateral filter
        # Scale strength to appropriate range for bilateral filter
        d = int(3 + strength * 7)  # Diameter of filter, 3-10
        sigma_color = 10 + strength * 90  # 10-100
        sigma_space = 10 + strength * 90  # 10-100
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def _apply_sharpening(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply sharpening.
        
        Args:
            image: Input image
            strength: Strength of sharpening (0-1)
            
        Returns:
            Processed image
        """
        # Apply unsharp mask
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(
            image, 1.0 + strength, 
            blurred, -strength, 
            0
        )
        
        return sharpened
    
    def _apply_saturation(self, image: np.ndarray, adjustment: float) -> np.ndarray:
        """Apply saturation adjustment.
        
        Args:
            image: Input image
            adjustment: Saturation adjustment (-1 to 1)
            
        Returns:
            Processed image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Calculate multiplier (1.0 means no change)
        if adjustment > 0:
            # Increase saturation
            multiplier = 1.0 + adjustment
        else:
            # Decrease saturation
            multiplier = 1.0 + adjustment
        
        # Apply multiplier to saturation channel
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * multiplier, 0, 255)
        
        # Convert back to BGR
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _apply_temperature(self, image: np.ndarray, adjustment: float) -> np.ndarray:
        """Apply color temperature adjustment.
        
        Args:
            image: Input image
            adjustment: Temperature adjustment (-1 to 1)
            
        Returns:
            Processed image
        """
        # Copy the image
        result = image.copy().astype(np.float32)
        
        # Apply temperature adjustment
        if adjustment > 0:
            # Warm up (increase red, decrease blue)
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 + adjustment * 0.2), 0, 255)  # Red
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 - adjustment * 0.1), 0, 255)  # Blue
        else:
            # Cool down (decrease red, increase blue)
            adjustment = abs(adjustment)
            result[:, :, 0] = np.clip(result[:, :, 0] * (1 - adjustment * 0.1), 0, 255)  # Red
            result[:, :, 2] = np.clip(result[:, :, 2] * (1 + adjustment * 0.2), 0, 255)  # Blue
        
        return result.astype(np.uint8)

from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from .base_agent import BaseAgent

class SkinAgent(BaseAgent):
    """Agent specialized in skin tone enhancement and retouching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.skin_profiles = []
        self.current_profile = None
    
    def _validate_config(self) -> None:
        """Validate skin agent configuration."""
        required_params = ['learning_rate', 'max_profiles', 'smoothness', 'skin_tone_range']
        defaults = {
            'learning_rate': 0.1,
            'max_profiles': 10,
            'smoothness': 0.5,  # Skin smoothing factor
            'skin_tone_range': ((0, 15, 0), (255, 240, 255))  # HSV range for skin tones
        }
        for param in required_params:
            if param not in self.config:
                self.config[param] = defaults[param]
    
    def _detect_skin(self, image: np.ndarray) -> np.ndarray:
        """Detect skin regions in the image."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define skin color range
        lower_skin = np.array(self.config['skin_tone_range'][0])
        upper_skin = np.array(self.config['skin_tone_range'][1])
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn skin enhancement from before/after image pair."""
        # Detect skin regions
        skin_mask = self._detect_skin(before_image)
        
        # Convert to LAB color space for better color analysis
        before_lab = cv2.cvtColor(before_image, cv2.COLOR_RGB2LAB)
        after_lab = cv2.cvtColor(after_image, cv2.COLOR_RGB2LAB)
        
        # Calculate color differences in skin regions
        skin_pixels = skin_mask > 0
        before_skin = before_lab[skin_pixels]
        after_skin = after_lab[skin_pixels]
        
        if len(before_skin) > 0:
            # Calculate average color adjustments
            color_diff = np.mean(after_skin - before_skin, axis=0)
            
            # Calculate smoothness factor
            before_blur = cv2.GaussianBlur(before_image, (0, 0), self.config['smoothness'])
            after_blur = cv2.GaussianBlur(after_image, (0, 0), self.config['smoothness'])
            smoothness_diff = np.mean(np.abs(after_blur - before_blur))
            
            # Store the learned profile
            profile = {
                'color_adjustment': color_diff,
                'smoothness': smoothness_diff
            }
            
            if len(self.skin_profiles) >= self.config['max_profiles']:
                self.skin_profiles.pop(0)
            self.skin_profiles.append(profile)
            
            # Update current profile
            self._update_current_profile()
    
    def _update_current_profile(self) -> None:
        """Update current skin profile based on learned profiles."""
        if not self.skin_profiles:
            return
        
        # Calculate weighted averages
        weights = np.linspace(0.1, 1.0, len(self.skin_profiles))
        weights = weights / weights.sum()
        
        color_adjustment = np.average(
            [p['color_adjustment'] for p in self.skin_profiles],
            weights=weights,
            axis=0
        )
        smoothness = np.average(
            [p['smoothness'] for p in self.skin_profiles],
            weights=weights
        )
        
        self.current_profile = {
            'color_adjustment': color_adjustment,
            'smoothness': smoothness
        }
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply learned skin enhancements to an image."""
        if self.current_profile is None:
            return image
        
        # Detect skin regions
        skin_mask = self._detect_skin(image)
        
        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply color adjustments to skin regions
        skin_pixels = skin_mask > 0
        lab_image[skin_pixels] = lab_image[skin_pixels] + self.current_profile['color_adjustment']
        
        # Apply smoothing to skin regions
        blurred = cv2.GaussianBlur(image, (0, 0), self.current_profile['smoothness'])
        image[skin_pixels] = blurred[skin_pixels]
        
        # Convert back to RGB
        return cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    def get_status(self) -> Dict[str, Any]:
        """Get skin agent status."""
        status = super().get_status()
        status.update({
            "profiles_learned": len(self.skin_profiles),
            "has_active_profile": self.current_profile is not None,
            "current_smoothness": self.current_profile['smoothness'] if self.current_profile else None
        })
        return status 
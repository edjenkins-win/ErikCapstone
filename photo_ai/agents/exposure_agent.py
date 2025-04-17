from typing import Any, Dict, Optional
import numpy as np
import cv2
from .base_agent import BaseAgent

class ExposureAgent(BaseAgent):
    """Agent specialized in exposure and contrast adjustments."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.exposure_profiles = []
        self.current_profile = None
    
    def _validate_config(self) -> None:
        """Validate exposure agent configuration."""
        required_params = ['learning_rate', 'max_profiles', 'contrast_range']
        defaults = {
            'learning_rate': 0.1,
            'max_profiles': 10,
            'contrast_range': (0.8, 1.2)  # Allowable contrast adjustment range
        }
        for param in required_params:
            if param not in self.config:
                self.config[param] = defaults[param]
    
    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn exposure adjustments from before/after image pair."""
        # Convert to LAB color space for better exposure analysis
        before_lab = cv2.cvtColor(before_image, cv2.COLOR_RGB2LAB)
        after_lab = cv2.cvtColor(after_image, cv2.COLOR_RGB2LAB)
        
        # Calculate brightness and contrast differences
        before_l = before_lab[:, :, 0].astype(np.float32)
        after_l = after_lab[:, :, 0].astype(np.float32)
        
        # Calculate brightness adjustment
        brightness_diff = np.mean(after_l - before_l)
        
        # Calculate contrast adjustment
        before_std = np.std(before_l)
        after_std = np.std(after_l)
        contrast_ratio = after_std / before_std if before_std > 0 else 1.0
        
        # Store the learned profile
        profile = {
            'brightness': brightness_diff,
            'contrast': contrast_ratio
        }
        
        if len(self.exposure_profiles) >= self.config['max_profiles']:
            self.exposure_profiles.pop(0)
        self.exposure_profiles.append(profile)
        
        # Update current profile
        self._update_current_profile()
    
    def _update_current_profile(self) -> None:
        """Update current exposure profile based on learned profiles."""
        if not self.exposure_profiles:
            return
        
        # Calculate weighted averages
        weights = np.linspace(0.1, 1.0, len(self.exposure_profiles))
        weights = weights / weights.sum()
        
        brightness = np.average(
            [p['brightness'] for p in self.exposure_profiles],
            weights=weights
        )
        contrast = np.average(
            [p['contrast'] for p in self.exposure_profiles],
            weights=weights
        )
        
        # Ensure contrast stays within allowed range
        min_contrast, max_contrast = self.config['contrast_range']
        contrast = np.clip(contrast, min_contrast, max_contrast)
        
        self.current_profile = {
            'brightness': brightness,
            'contrast': contrast
        }
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply learned exposure adjustments to an image."""
        if self.current_profile is None:
            return image
        
        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab_image[:, :, 0].astype(np.float32)
        
        # Apply brightness and contrast adjustments
        l_channel = (l_channel + self.current_profile['brightness']) * self.current_profile['contrast']
        
        # Clip values to valid range
        l_channel = np.clip(l_channel, 0, 255)
        
        # Update LAB image
        lab_image[:, :, 0] = l_channel
        
        # Convert back to RGB
        return cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    def get_status(self) -> Dict[str, Any]:
        """Get exposure agent status."""
        status = super().get_status()
        status.update({
            "profiles_learned": len(self.exposure_profiles),
            "has_active_profile": self.current_profile is not None,
            "current_brightness": self.current_profile['brightness'] if self.current_profile else None,
            "current_contrast": self.current_profile['contrast'] if self.current_profile else None
        })
        return status 
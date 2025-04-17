from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from .base_agent import BaseAgent

class NoiseAgent(BaseAgent):
    """Agent specialized in noise reduction and detail preservation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.noise_profiles = []
        self.current_profile = None
    
    def _validate_config(self) -> None:
        """Validate noise agent configuration."""
        required_params = ['learning_rate', 'max_profiles', 'denoise_strength_range', 'detail_preservation']
        defaults = {
            'learning_rate': 0.1,
            'max_profiles': 10,
            'denoise_strength_range': (1, 20),  # Allowable denoising strength
            'detail_preservation': 0.5  # Detail preservation factor (0-1)
        }
        for param in required_params:
            if param not in self.config:
                self.config[param] = defaults[param]
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local standard deviation
        kernel_size = 3
        local_std = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        local_std = np.abs(gray - local_std)
        
        # Estimate noise level
        noise_level = np.mean(local_std)
        return noise_level
    
    def _preserve_details(self, image: np.ndarray, denoised: np.ndarray) -> np.ndarray:
        """Preserve important details while reducing noise."""
        # Calculate detail mask
        detail_mask = cv2.Laplacian(image, cv2.CV_64F)
        detail_mask = np.abs(detail_mask)
        detail_mask = cv2.normalize(detail_mask, None, 0, 1, cv2.NORM_MINMAX)
        
        # Blend original and denoised images based on detail mask
        alpha = self.config['detail_preservation']
        result = image * (1 - alpha * detail_mask) + denoised * (alpha * detail_mask)
        return result.astype(np.uint8)
    
    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn noise reduction from before/after image pair."""
        # Estimate noise levels
        before_noise = self._estimate_noise(before_image)
        after_noise = self._estimate_noise(after_image)
        
        # Calculate denoising strength
        denoise_strength = before_noise - after_noise
        
        # Calculate detail preservation
        detail_mask = cv2.Laplacian(before_image, cv2.CV_64F)
        detail_mask = np.abs(detail_mask)
        detail_preservation = np.mean(detail_mask)
        
        # Store the learned profile
        profile = {
            'denoise_strength': denoise_strength,
            'detail_preservation': detail_preservation
        }
        
        if len(self.noise_profiles) >= self.config['max_profiles']:
            self.noise_profiles.pop(0)
        self.noise_profiles.append(profile)
        
        # Update current profile
        self._update_current_profile()
    
    def _update_current_profile(self) -> None:
        """Update current noise profile based on learned profiles."""
        if not self.noise_profiles:
            return
        
        # Calculate weighted averages
        weights = np.linspace(0.1, 1.0, len(self.noise_profiles))
        weights = weights / weights.sum()
        
        denoise_strength = np.average(
            [p['denoise_strength'] for p in self.noise_profiles],
            weights=weights
        )
        detail_preservation = np.average(
            [p['detail_preservation'] for p in self.noise_profiles],
            weights=weights
        )
        
        # Ensure values stay within allowed ranges
        min_strength, max_strength = self.config['denoise_strength_range']
        denoise_strength = np.clip(denoise_strength, min_strength, max_strength)
        
        self.current_profile = {
            'denoise_strength': denoise_strength,
            'detail_preservation': detail_preservation
        }
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply learned noise reduction to an image."""
        if self.current_profile is None:
            return image
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.current_profile['denoise_strength'],
            self.current_profile['denoise_strength'],
            7,  # template window size
            21  # search window size
        )
        
        # Preserve important details
        result = self._preserve_details(image, denoised)
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get noise agent status."""
        status = super().get_status()
        status.update({
            "profiles_learned": len(self.noise_profiles),
            "has_active_profile": self.current_profile is not None,
            "current_denoise_strength": self.current_profile['denoise_strength'] if self.current_profile else None,
            "current_detail_preservation": self.current_profile['detail_preservation'] if self.current_profile else None
        })
        return status 
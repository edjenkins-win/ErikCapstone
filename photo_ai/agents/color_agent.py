from typing import Any, Dict, Optional
import numpy as np
from .base_agent import BaseAgent

class ColorAgent(BaseAgent):
    """Agent specialized in color grading and adjustments."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.color_profiles = []
        self.current_profile = None
    
    def _validate_config(self) -> None:
        """Validate color agent configuration."""
        required_params = ['learning_rate', 'max_profiles']
        for param in required_params:
            if param not in self.config:
                self.config[param] = 0.1 if param == 'learning_rate' else 10
    
    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn color grading from before/after image pair."""
        # Convert to float for calculations
        before = before_image.astype(np.float32) / 255.0
        after = after_image.astype(np.float32) / 255.0
        
        # Calculate color transformation
        color_transform = after - before
        
        # Store the learned profile
        if len(self.color_profiles) >= self.config['max_profiles']:
            self.color_profiles.pop(0)
        self.color_profiles.append(color_transform)
        
        # Update current profile as weighted average
        self._update_current_profile()
    
    def _update_current_profile(self) -> None:
        """Update current color profile based on learned profiles."""
        if not self.color_profiles:
            return
        
        weights = np.linspace(0.1, 1.0, len(self.color_profiles))
        weights = weights / weights.sum()
        
        self.current_profile = np.average(
            self.color_profiles,
            weights=weights,
            axis=0
        )
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply learned color grading to an image."""
        if self.current_profile is None:
            return image
        
        # Convert to float for calculations
        processed = image.astype(np.float32) / 255.0
        
        # Apply color transformation
        processed = processed + self.current_profile
        
        # Clip values and convert back to uint8
        processed = np.clip(processed, 0, 1)
        return (processed * 255).astype(np.uint8)
    
    def get_status(self) -> Dict[str, Any]:
        """Get color agent status."""
        status = super().get_status()
        status.update({
            "profiles_learned": len(self.color_profiles),
            "has_active_profile": self.current_profile is not None
        })
        return status 
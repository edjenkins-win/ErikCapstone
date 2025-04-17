from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from .base_agent import BaseAgent

class CompositionAgent(BaseAgent):
    """Agent specialized in image composition and alignment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.composition_profiles = []
        self.current_profile = None
    
    def _validate_config(self) -> None:
        """Validate composition agent configuration."""
        required_params = ['learning_rate', 'max_profiles', 'aspect_ratios']
        defaults = {
            'learning_rate': 0.1,
            'max_profiles': 10,
            'aspect_ratios': ['1:1', '4:3', '16:9']  # Common aspect ratios
        }
        for param in required_params:
            if param not in self.config:
                self.config[param] = defaults[param]
    
    def _calculate_rule_of_thirds(self, image: np.ndarray) -> Tuple[float, float]:
        """Calculate how well the image follows the rule of thirds."""
        height, width = image.shape[:2]
        
        # Calculate grid lines
        third_width = width / 3
        third_height = height / 3
        
        # Calculate center of mass
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(gray)
        
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = width // 2, height // 2
        
        # Calculate distance from rule of thirds lines
        x_dist = min(abs(cx - third_width), abs(cx - 2 * third_width))
        y_dist = min(abs(cy - third_height), abs(cy - 2 * third_height))
        
        # Normalize distances
        x_score = 1 - (x_dist / (width / 3))
        y_score = 1 - (y_dist / (height / 3))
        
        return x_score, y_score
    
    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn composition rules from before/after image pair."""
        # Calculate rule of thirds scores
        before_x, before_y = self._calculate_rule_of_thirds(before_image)
        after_x, after_y = self._calculate_rule_of_thirds(after_image)
        
        # Calculate aspect ratio changes
        before_ratio = before_image.shape[1] / before_image.shape[0]
        after_ratio = after_image.shape[1] / after_image.shape[0]
        
        # Calculate crop adjustments
        height_diff = (after_image.shape[0] - before_image.shape[0]) / before_image.shape[0]
        width_diff = (after_image.shape[1] - before_image.shape[1]) / before_image.shape[1]
        
        # Store the learned profile
        profile = {
            'rule_of_thirds': {
                'x_improvement': after_x - before_x,
                'y_improvement': after_y - before_y
            },
            'aspect_ratio': after_ratio,
            'crop_adjustments': {
                'height': height_diff,
                'width': width_diff
            }
        }
        
        if len(self.composition_profiles) >= self.config['max_profiles']:
            self.composition_profiles.pop(0)
        self.composition_profiles.append(profile)
        
        # Update current profile
        self._update_current_profile()
    
    def _update_current_profile(self) -> None:
        """Update current composition profile based on learned profiles."""
        if not self.composition_profiles:
            return
        
        # Calculate weighted averages
        weights = np.linspace(0.1, 1.0, len(self.composition_profiles))
        weights = weights / weights.sum()
        
        rule_of_thirds_x = np.average(
            [p['rule_of_thirds']['x_improvement'] for p in self.composition_profiles],
            weights=weights
        )
        rule_of_thirds_y = np.average(
            [p['rule_of_thirds']['y_improvement'] for p in self.composition_profiles],
            weights=weights
        )
        
        aspect_ratio = np.average(
            [p['aspect_ratio'] for p in self.composition_profiles],
            weights=weights
        )
        
        crop_height = np.average(
            [p['crop_adjustments']['height'] for p in self.composition_profiles],
            weights=weights
        )
        crop_width = np.average(
            [p['crop_adjustments']['width'] for p in self.composition_profiles],
            weights=weights
        )
        
        self.current_profile = {
            'rule_of_thirds': {
                'x_improvement': rule_of_thirds_x,
                'y_improvement': rule_of_thirds_y
            },
            'aspect_ratio': aspect_ratio,
            'crop_adjustments': {
                'height': crop_height,
                'width': crop_width
            }
        }
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply learned composition improvements to an image."""
        if self.current_profile is None:
            return image
        
        height, width = image.shape[:2]
        
        # Apply crop adjustments
        new_height = int(height * (1 + self.current_profile['crop_adjustments']['height']))
        new_width = int(width * (1 + self.current_profile['crop_adjustments']['width']))
        
        # Calculate center crop
        start_y = (height - new_height) // 2
        start_x = (width - new_width) // 2
        
        # Ensure valid crop coordinates
        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = min(height, start_y + new_height)
        end_x = min(width, start_x + new_width)
        
        # Apply crop
        cropped = image[start_y:end_y, start_x:end_x]
        
        # Resize to maintain aspect ratio if needed
        if self.current_profile['aspect_ratio'] > 0:
            target_ratio = self.current_profile['aspect_ratio']
            current_ratio = cropped.shape[1] / cropped.shape[0]
            
            if abs(current_ratio - target_ratio) > 0.01:  # 1% tolerance
                if current_ratio > target_ratio:
                    new_width = int(cropped.shape[0] * target_ratio)
                    cropped = cropped[:, :new_width]
                else:
                    new_height = int(cropped.shape[1] / target_ratio)
                    cropped = cropped[:new_height, :]
        
        return cropped
    
    def get_status(self) -> Dict[str, Any]:
        """Get composition agent status."""
        status = super().get_status()
        status.update({
            "profiles_learned": len(self.composition_profiles),
            "has_active_profile": self.current_profile is not None,
            "current_aspect_ratio": self.current_profile['aspect_ratio'] if self.current_profile else None
        })
        return status 
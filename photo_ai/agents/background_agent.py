from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from .base_agent import BaseAgent

class BackgroundAgent(BaseAgent):
    """Agent specialized in background enhancement and subject-background separation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.background_profiles = []
        self.current_profile = None

    def _validate_config(self) -> None:
        """Validate background agent configuration."""
        required_params = ['learning_rate', 'max_profiles', 'blur_range', 'color_adjustment_range']
        defaults = {
            'learning_rate': 0.1,
            'max_profiles': 10,
            'blur_range': (0.0, 5.0),  # Allowable blur strength
            'color_adjustment_range': (-30, 30)  # Allowable color adjustment range
        }
        for param in required_params:
            if param not in self.config:
                self.config[param] = defaults[param]

    def _detect_background(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect background regions and create a mask."""
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Use k-means clustering to separate foreground and background
        pixel_values = lab.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 2
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Create background mask
        centers = np.uint8(centers)
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)

        # Create binary mask (0 for background, 1 for foreground)
        mask = np.zeros_like(image[:, :, 0])
        mask[labels.reshape(image.shape[:2]) == 0] = 255

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, segmented_image

    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn background enhancement from before/after image pair."""
        # Detect background regions
        before_mask, _ = self._detect_background(before_image)
        after_mask, _ = self._detect_background(after_image)

        # Calculate blur difference
        before_blur = cv2.GaussianBlur(before_image, (0, 0), 1.0)
        after_blur = cv2.GaussianBlur(after_image, (0, 0), 1.0)
        blur_diff = np.mean(np.abs(after_blur - before_blur))

        # Calculate color adjustments in background
        before_lab = cv2.cvtColor(before_image, cv2.COLOR_RGB2LAB)
        after_lab = cv2.cvtColor(after_image, cv2.COLOR_RGB2LAB)

        background_pixels = before_mask > 0
        color_diff = np.mean(after_lab[background_pixels] - before_lab[background_pixels], axis=0)

        # Store the learned profile
        profile = {
            'blur_strength': blur_diff,
            'color_adjustment': color_diff,
            'background_mask': before_mask
        }

        if len(self.background_profiles) >= self.config['max_profiles']:
            self.background_profiles.pop(0)
        self.background_profiles.append(profile)

        # Update current profile
        self._update_current_profile()

    def _update_current_profile(self) -> None:
        """Update current background profile based on learned profiles."""
        if not self.background_profiles:
            return

        # Calculate weighted averages
        weights = np.linspace(0.1, 1.0, len(self.background_profiles))
        weights = weights / weights.sum()

        blur_strength = np.average(
            [p['blur_strength'] for p in self.background_profiles],
            weights=weights
        )
        color_adjustment = np.average(
            [p['color_adjustment'] for p in self.background_profiles],
            weights=weights,
            axis=0
        )

        # Ensure values stay within allowed ranges
        min_blur, max_blur = self.config['blur_range']
        min_color, max_color = self.config['color_adjustment_range']

        blur_strength = np.clip(blur_strength, min_blur, max_blur)
        color_adjustment = np.clip(color_adjustment, min_color, max_color)

        self.current_profile = {
            'blur_strength': blur_strength,
            'color_adjustment': color_adjustment
        }

    def process(self, image: np.ndarray, **kwargs) -> tuple[np.ndarray, Dict[str, Any]]:
        """Apply learned background enhancements to an image.

        Args:
            image: Input image as numpy array
            **kwargs: Additional processing parameters

        Returns:
            Tuple containing:
                - Processed image as numpy array
                - Dictionary with processing metrics
        """
        if self.current_profile is None:
            return image, {"status": "no_profile_applied"}

        # Detect background
        background_mask, _ = self._detect_background(image)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply color adjustments to background
        background_pixels = background_mask > 0
        lab_image[background_pixels] = lab_image[background_pixels] + self.current_profile['color_adjustment']

        # Convert back to RGB
        processed = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # Apply blur to background
        blurred = cv2.GaussianBlur(processed, (0, 0), self.current_profile['blur_strength'])
        processed[background_pixels] = blurred[background_pixels]

        # Return processed image and metrics
        metrics = {
            "background_percentage": float(np.sum(background_pixels) / background_pixels.size * 100),
            "blur_strength": float(self.current_profile['blur_strength']),
            "color_adjustment": self.current_profile['color_adjustment'].tolist()
        }

        return processed, metrics

    def get_status(self) -> Dict[str, Any]:
        """Get background agent status."""
        status = super().get_status()
        status.update({
            "profiles_learned": len(self.background_profiles),
            "has_active_profile": self.current_profile is not None,
            "current_blur_strength": self.current_profile['blur_strength'] if self.current_profile else None
        })
        return status 

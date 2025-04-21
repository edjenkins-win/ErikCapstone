"""Tests for the executor module."""

import unittest
import numpy as np
import os
import tempfile
from pathlib import Path

from photoedit_mvp.executor import ImageExecutor
from photoedit_mvp.analyzer import Adjustment
from photoedit_mvp.utils import save_image

class TestExecutor(unittest.TestCase):
    """Test cases for the ImageExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test images
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Make the image dark (underexposed)
        self.test_image[:, :, :] = 50
        
        # Save the test image
        self.test_image_path = os.path.join(self.temp_dir.name, "test_image.jpg")
        save_image(self.test_image, self.test_image_path)
        
        # Create executor
        self.executor = ImageExecutor()

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_apply_no_adjustments(self):
        """Test that apply returns a copy of the image when no adjustments are provided."""
        # Apply no adjustments
        result = self.executor.apply(self.test_image, [])
        
        # Check that it returns an image with the same shape
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that it's a copy (not the same object)
        self.assertIsNot(result, self.test_image)
    
    def test_apply_exposure_adjustment(self):
        """Test that apply applies exposure adjustment correctly."""
        # Create an exposure adjustment
        adjustment = Adjustment(
            parameter="exposure",
            suggested=1.0,  # Increase by 1 EV (double the brightness)
            unit="EV",
            description="Increase exposure"
        )
        
        # Apply the adjustment
        result = self.executor.apply(self.test_image, [adjustment])
        
        # Check that the result is brighter
        self.assertTrue(np.mean(result) > np.mean(self.test_image))
    
    def test_apply_style(self):
        """Test that apply applies style preset correctly."""
        # Apply Auto-Enhance style
        result = self.executor.apply(self.test_image, [], style="Auto-Enhance")
        
        # Check that the result is different from the original
        self.assertFalse(np.array_equal(result, self.test_image))

if __name__ == "__main__":
    unittest.main()

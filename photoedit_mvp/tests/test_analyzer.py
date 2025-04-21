"""Tests for the analyzer module."""

import unittest
import numpy as np
import os
import tempfile
from pathlib import Path

from photoedit_mvp.analyzer import ImageAnalyzer, Adjustment
from photoedit_mvp.utils import save_image

class TestAnalyzer(unittest.TestCase):
    """Test cases for the ImageAnalyzer class."""

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
        
        # Create analyzer
        self.analyzer = ImageAnalyzer()

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_analyze_returns_adjustments(self):
        """Test that analyze returns a list of adjustments."""
        # Analyze the test image
        adjustments = self.analyzer.analyze(self.test_image_path)
        
        # Check that it returns a list of adjustments
        self.assertIsInstance(adjustments, list)
        for adj in adjustments:
            self.assertIsInstance(adj, Adjustment)
    
    def test_analyze_detects_exposure(self):
        """Test that analyze detects exposure issues."""
        # Analyze the dark test image
        adjustments = self.analyzer.analyze(self.test_image)
        
        # Find exposure adjustment
        exposure_adj = None
        for adj in adjustments:
            if adj.parameter == "exposure":
                exposure_adj = adj
                break
        
        # Check that exposure adjustment was found and is positive (increase)
        self.assertIsNotNone(exposure_adj)
        self.assertTrue(exposure_adj.suggested > 0)

if __name__ == "__main__":
    unittest.main()

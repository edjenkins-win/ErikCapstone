"""Styles module for the Photo Editor application.

This module defines preset styles for photo editing.
"""

import logging
from typing import Dict, List, Any

# Set up logging
logger = logging.getLogger(__name__)

class StylePreset:
    """Defines a preset style for photo editing."""
    
    def __init__(self, name: str, description: str, adjustments: List[Dict[str, Any]]):
        """Initialize a style preset.
        
        Args:
            name: The name of the style preset
            description: A description of the style preset
            adjustments: A list of adjustment dictionaries
        """
        self.name = name
        self.description = description
        self.adjustments = adjustments
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the style preset to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "adjustments": self.adjustments
        }

# Define the standard style presets
STYLE_PRESETS = {
    "auto-enhance": StylePreset(
        name="Auto-Enhance",
        description="Balanced automatic enhancement for most photos",
        adjustments=[
            {"parameter": "exposure", "suggested": 0.0, "unit": "EV"},
            {"parameter": "contrast", "suggested": 1.1, "unit": "multiplier"},
            {"parameter": "sharpening", "suggested": 0.2, "unit": "strength"},
        ]
    ),
    
    "portrait": StylePreset(
        name="Portrait",
        description="Optimized for portrait photography with skin tone enhancement",
        adjustments=[
            {"parameter": "exposure", "suggested": 0.1, "unit": "EV"},
            {"parameter": "contrast", "suggested": 1.05, "unit": "multiplier"},
            {"parameter": "noise_reduction", "suggested": 0.3, "unit": "strength"},
            {"parameter": "temperature", "suggested": 0.1, "unit": "shift"},
            {"parameter": "saturation", "suggested": 0.1, "unit": "increase"},
        ]
    ),
    
    "vintage": StylePreset(
        name="Vintage",
        description="Classic film-inspired look with faded colors",
        adjustments=[
            {"parameter": "contrast", "suggested": 0.8, "unit": "multiplier"},
            {"parameter": "saturation", "suggested": -0.15, "unit": "decrease"},
            {"parameter": "temperature", "suggested": 0.15, "unit": "shift"},
        ]
    ),
    
    # Cinematic style presets
    "cinematic-teal-orange": StylePreset(
        name="Cinematic Teal & Orange",
        description="Hollywood-style color grading with teal shadows and orange highlights",
        adjustments=[
            {"parameter": "contrast", "suggested": 1.3, "unit": "multiplier"},
            {"parameter": "temperature", "suggested": 0.2, "unit": "shift"},
            {"parameter": "saturation", "suggested": 0.15, "unit": "increase"},
            {"parameter": "sharpening", "suggested": 0.25, "unit": "strength"},
        ]
    ),
    
    "film-noir": StylePreset(
        name="Film Noir",
        description="High contrast black and white cinematic style with dramatic shadows",
        adjustments=[
            {"parameter": "contrast", "suggested": 1.5, "unit": "multiplier"},
            {"parameter": "saturation", "suggested": -1.0, "unit": "decrease"},
            {"parameter": "exposure", "suggested": -0.2, "unit": "EV"},
            {"parameter": "sharpening", "suggested": 0.3, "unit": "strength"},
        ]
    ),
    
    "anamorphic": StylePreset(
        name="Anamorphic",
        description="Widescreen cinematic look with enhanced contrast and blue lens flares",
        adjustments=[
            {"parameter": "contrast", "suggested": 1.25, "unit": "multiplier"},
            {"parameter": "saturation", "suggested": 0.1, "unit": "increase"},
            {"parameter": "temperature", "suggested": -0.1, "unit": "shift"},
            {"parameter": "sharpening", "suggested": 0.2, "unit": "strength"},
        ]
    ),
    
    "blockbuster": StylePreset(
        name="Blockbuster",
        description="Modern action film look with vibrant colors and high contrast",
        adjustments=[
            {"parameter": "exposure", "suggested": 0.1, "unit": "EV"},
            {"parameter": "contrast", "suggested": 1.4, "unit": "multiplier"},
            {"parameter": "saturation", "suggested": 0.25, "unit": "increase"},
            {"parameter": "sharpening", "suggested": 0.3, "unit": "strength"},
        ]
    ),
    
    "dreamy": StylePreset(
        name="Dreamy",
        description="Soft cinematic look with reduced contrast and warm tones",
        adjustments=[
            {"parameter": "contrast", "suggested": 0.85, "unit": "multiplier"},
            {"parameter": "temperature", "suggested": 0.15, "unit": "shift"},
            {"parameter": "noise_reduction", "suggested": 0.4, "unit": "strength"},
            {"parameter": "sharpening", "suggested": 0.1, "unit": "strength"},
        ]
    )
}

def get_style_preset(name: str) -> StylePreset:
    """Get a style preset by name.
    
    Args:
        name: The name of the style preset
        
    Returns:
        The style preset
        
    Raises:
        ValueError: If the style preset is not found
    """
    # Normalize the name (lowercase, remove spaces)
    normalized_name = name.lower().replace(' ', '-')
    
    if normalized_name in STYLE_PRESETS:
        return STYLE_PRESETS[normalized_name]
    else:
        raise ValueError(f"Style preset '{name}' not found")

def get_available_styles() -> List[str]:
    """Get a list of available style preset names.
    
    Returns:
        A list of style preset names
    """
    return [preset.name for preset in STYLE_PRESETS.values()]

def get_style_description(name: str) -> str:
    """Get the description of a style preset.
    
    Args:
        name: The name of the style preset
        
    Returns:
        The description of the style preset
        
    Raises:
        ValueError: If the style preset is not found
    """
    try:
        preset = get_style_preset(name)
        return preset.description
    except ValueError:
        raise

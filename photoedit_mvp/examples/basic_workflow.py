"""Basic workflow example for Photo Editor.

This example demonstrates how to analyze an image and apply adjustments and styles.
"""

import os
import argparse
import logging
from pathlib import Path

from photoedit_mvp import analyze_image, apply_adjustments, save_image, load_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the basic workflow example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Photo Editor basic workflow example")
    parser.add_argument("input", help="Input image file path")
    parser.add_argument("-o", "--output", help="Output directory (defaults to 'output')")
    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output or "output")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load the image
    logger.info(f"Loading image: {input_path}")
    image = load_image(str(input_path))

    # Step 2: Analyze the image
    logger.info("Analyzing image...")
    adjustments = analyze_image(image)
    
    # Log recommendations
    logger.info("Recommended adjustments:")
    for adj in adjustments:
        logger.info(f"- {adj.parameter}: {adj.suggested} {adj.unit} - {adj.description}")

    # Step 3: Apply the adjustments
    logger.info("Applying adjustments...")
    adjusted_image = apply_adjustments(image, adjustments)
    adjusted_path = output_dir / f"adjusted_{input_path.name}"
    save_image(adjusted_image, str(adjusted_path))
    logger.info(f"Saved adjusted image to: {adjusted_path}")

    # Step 4: Apply style presets
    for style in ["Auto-Enhance", "Portrait", "Vintage"]:
        logger.info(f"Applying '{style}' style...")
        styled_image = apply_adjustments(image, [], style=style)
        styled_path = output_dir / f"{style.lower()}_{input_path.name}"
        save_image(styled_image, str(styled_path))
        logger.info(f"Saved '{style}' styled image to: {styled_path}")

    return 0

if __name__ == "__main__":
    main()

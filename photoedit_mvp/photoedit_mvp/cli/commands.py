"""CLI commands for the Photo Editor application."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import numpy as np

from ..analyzer import ImageAnalyzer, Adjustment
from ..executor import ImageExecutor
from ..styles import get_available_styles, get_style_preset
from ..utils import load_image, save_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_command(args: argparse.Namespace) -> int:
    """Run the analyze command.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Check if input file exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 1
        
        # Create analyzer
        analyzer = ImageAnalyzer()
        
        # Load and analyze image
        logger.info(f"Analyzing image: {input_path}")
        adjustments = analyzer.analyze(str(input_path))
        
        # Convert adjustments to dictionaries
        adjustment_dicts = [adj.to_dict() for adj in adjustments]
        
        # Output adjustments
        if args.output:
            # Save to file
            output_path = Path(args.output)
            
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(adjustment_dicts, f, indent=2)
            logger.info(f"Adjustments saved to: {output_path}")
        else:
            # Print to stdout
            json_str = json.dumps(adjustment_dicts, indent=2)
            print(json_str)
        
        return 0
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return 1

def apply_command(args: argparse.Namespace) -> int:
    """Run the apply command.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Check if input file exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 1
        
        # Load adjustments if specified
        adjustments = []
        if args.adjustments:
            adjustments_path = Path(args.adjustments)
            if not adjustments_path.exists():
                logger.error(f"Adjustments file does not exist: {adjustments_path}")
                return 1
            
            with open(adjustments_path, 'r') as f:
                adjustment_dicts = json.load(f)
                
            # Convert dictionaries to Adjustment objects
            adjustments = [Adjustment.from_dict(adj) for adj in adjustment_dicts]
        
        # Create executor
        executor = ImageExecutor()
        
        # Load image
        logger.info(f"Loading image: {input_path}")
        image = load_image(str(input_path))
        
        # Apply style and/or adjustments
        logger.info("Processing image...")
        result = executor.apply(image, adjustments, args.style)
        
        # Save output
        output_path = args.output or f"processed_{input_path.name}"
        logger.info(f"Saving processed image to: {output_path}")
        save_image(result, output_path)
        
        return 0
    except Exception as e:
        logger.error(f"Error applying adjustments: {e}")
        return 1

def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Create top-level parser
    parser = argparse.ArgumentParser(
        description="AI-driven photo editing tool that analyzes and enhances photos."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create parser for the "analyze" command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze an image and generate adjustment recommendations"
    )
    analyze_parser.add_argument(
        "input",
        help="Input image file path"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output JSON file path for adjustment recommendations (defaults to stdout)"
    )
    
    # Create parser for the "apply" command
    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply adjustments and/or a style to an image"
    )
    apply_parser.add_argument(
        "input",
        help="Input image file path"
    )
    apply_parser.add_argument(
        "-a", "--adjustments",
        help="JSON file with adjustment recommendations"
    )
    apply_parser.add_argument(
        "-s", "--style",
        choices=get_available_styles(),
        help="Style preset to apply"
    )
    apply_parser.add_argument(
        "-o", "--output",
        help="Output image file path (defaults to 'processed_<input>')"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == "analyze":
        return analyze_command(args)
    elif args.command == "apply":
        return apply_command(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())

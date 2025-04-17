#!/usr/bin/env python
"""
Style Transfer Demo Script

This script demonstrates the enhanced style transfer capabilities with:
1. Progress tracking visualization
2. Multi-format output support
3. Enhanced image processing
"""

import argparse
import os
import sys
import time
import logging
from typing import List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from photo_ai.agents.style_agent import StyleAgent
from photo_ai.core.image_processor import ImageProcessor
from photo_ai.utils.progress_tracker import ProgressTracker

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Neural Style Transfer Demo')
    
    parser.add_argument('--content', '-c', type=str, required=True,
                      help='Path to content image')
    parser.add_argument('--style', '-s', type=str, required=True,
                      help='Path to style image')
    parser.add_argument('--output', '-o', type=str, default='output/stylized',
                      help='Base path for output image (without extension)')
    parser.add_argument('--steps', '-n', type=int, default=300,
                      help='Number of optimization steps')
    parser.add_argument('--formats', '-f', type=str, nargs='+',
                      default=['.jpg', '.png', '.webp'],
                      help='Output formats (e.g., .jpg .png .webp)')
    parser.add_argument('--quality', '-q', type=int, default=95,
                      help='Quality for lossy formats (0-100)')
    parser.add_argument('--style-weight', type=float, default=1e6,
                      help='Weight of style loss')
    parser.add_argument('--content-weight', type=float, default=1.0,
                      help='Weight of content loss')
    parser.add_argument('--image-size', type=int, default=512,
                      help='Size to resize images to (preserves aspect ratio)')
    
    return parser.parse_args()

def print_supported_formats():
    """Print the supported image formats."""
    formats = ImageProcessor.get_supported_formats()
    print("Supported image formats:")
    for i, fmt in enumerate(formats):
        print(f"  {fmt}", end=", " if (i + 1) % 5 != 0 else "\n")
    print("\n")

def show_operation_progress(operation_id: str):
    """Display the progress of an operation."""
    tracker = ProgressTracker()
    operation = tracker.get_operation(operation_id)
    
    if operation:
        progress = operation["progress"]
        description = operation["description"]
        current_step = operation["current_step"]
        total_steps = operation["total_steps"]
        
        # Get the most recent message
        messages = operation.get("messages", [])
        latest_message = messages[-1]["message"] if messages else ""
        
        # Print progress bar
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\r{description}: [{bar}] {progress:.1f}% ({current_step}/{total_steps}) {latest_message}", end='')
        sys.stdout.flush()
        
        if progress >= 100:
            print()  # Print newline when complete
            return False
            
        return True
    
    return False

def main():
    """Run the style transfer demo."""
    args = parse_arguments()
    
    # Print supported formats
    print_supported_formats()
    
    # Check if the input files exist
    if not os.path.exists(args.content):
        logger.error(f"Content image not found: {args.content}")
        return
        
    if not os.path.exists(args.style):
        logger.error(f"Style image not found: {args.style}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Load content and style images
    try:
        print(f"Loading content image: {args.content}")
        content_image = ImageProcessor.load_image(args.content)
        
        print(f"Loading style image: {args.style}")
        style_image = ImageProcessor.load_image(args.style)
    except Exception as e:
        logger.error(f"Error loading images: {str(e)}")
        return
    
    # Initialize style agent with custom configuration
    config = {
        'style_weight': args.style_weight,
        'content_weight': args.content_weight,
        'num_steps': args.steps,
        'image_size': args.image_size
    }
    
    style_agent = StyleAgent(config)
    print(f"Initialized style agent with image size: {args.image_size}")
    
    # Process and save the image
    operation_id = f"style_transfer_demo_{int(time.time())}"
    
    # Start processing in a separate thread
    import threading
    result = None
    
    def process_image():
        nonlocal result
        try:
            # Process the image and save it in multiple formats
            result = style_agent.process_and_save(
                content_image=content_image,
                style_image=style_image,
                output_path=args.output,
                formats=args.formats,
                num_steps=args.steps,
                quality=args.quality
            )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
    
    # Start processing thread
    process_thread = threading.Thread(target=process_image)
    process_thread.start()
    
    # Monitor progress
    print("Starting style transfer process...")
    
    while process_thread.is_alive():
        # Check for operations to display
        tracker = ProgressTracker()
        operations = tracker.get_all_operations()
        
        for op in operations:
            show_operation_progress(op["id"])
            
        time.sleep(0.5)
    
    # Wait for processing to complete
    process_thread.join()
    
    if result:
        print("\nStyle transfer completed!")
        print("\nProcessing metrics:")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value}")
            
        print("\nSaved files:")
        for fmt, path in result['saved_files'].items():
            print(f"  {fmt}: {path}")
    else:
        print("\nStyle transfer failed!")

if __name__ == "__main__":
    main() 
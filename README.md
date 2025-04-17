# AI-Powered Photo Editing System

An intelligent photo editing system that learns from before/after photo pairs to understand and apply consistent editing styles.

## Project Structure

```
photo_ai/
├── agents/                 # AI agents for different editing tasks
├── core/                   # Core functionality and utilities
├── models/                 # ML models and style transfer components
├── services/              # Business logic and service layer
├── api/                   # API endpoints and FastAPI application
└── tests/                 # Unit and integration tests
```

## Features

- Style learning from before/after photo pairs
- Automated batch processing
- Quality control and validation
- Consistent style application
- Specialized editing agents

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[Usage instructions will be added as the project develops]

## Development

This project follows clean code principles:
- Single Responsibility Principle (SRP)
- Don't Repeat Yourself (DRY)
- ACID principles for data consistency
- Modular and maintainable code structure

## License

[License information to be added]

# Enhanced Neural Style Transfer

This project implements an enhanced neural style transfer system with the following features:

- Detailed progress tracking with estimated time remaining
- Support for multiple image formats, including modern formats like HEIF, AVIF, and WebP
- Improved handling of raw camera formats from various manufacturers
- Multi-format output support (save stylized images in different formats simultaneously)
- Clean, modular code following SOLID principles

## Features

### 1. Progress Tracking

The style transfer process now provides detailed progress tracking with:
- Real-time step-by-step updates
- Estimated time remaining
- Detailed metrics during the style transfer process

### 2. Enhanced Image Format Support

The system can now process and save images in the following formats:

**Standard formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- GIF (.gif)

**Modern formats:**
- WebP (.webp)
- HEIF/HEIC (.heif, .heic)
- AVIF (.avif)
- JPEG XR (.jxr)

**Raw camera formats:**
- DNG (.dng)
- Sony ARW (.arw)
- Canon CR2/CR3 (.cr2, .cr3)
- Nikon NEF/NRW (.nef, .nrw)
- Olympus ORF (.orf)
- Panasonic RW2 (.rw2)
- Fujifilm RAF (.raf)
- Pentax PEF (.pef)
- Samsung SRW (.srw)
- Sigma X3F (.x3f)

**HDR formats:**
- HDR (.hdr)
- OpenEXR (.exr)

### 3. Multi-Format Output

You can now save stylized images in multiple formats simultaneously, making it easy to compare quality and file size or prepare images for different uses (web, print, archiving).

## Usage

### Running Style Transfer Demo

```bash
python examples/style_transfer_demo.py --content <content_image_path> --style <style_image_path> --output <output_path> --formats .jpg .png .webp --steps 300
```

#### Command-line Arguments

- `--content`, `-c`: Path to the content image (required)
- `--style`, `-s`: Path to the style image (required)
- `--output`, `-o`: Base path for output images (default: 'output/stylized')
- `--steps`, `-n`: Number of optimization steps (default: 300)
- `--formats`, `-f`: Output formats (default: ['.jpg', '.png', '.webp'])
- `--quality`, `-q`: Quality for lossy formats (default: 95)
- `--style-weight`: Weight for style loss (default: 1e6)
- `--content-weight`: Weight for content loss (default: 1.0)
- `--image-size`: Size to resize images to (default: 512)

## API Usage

```python
from photo_ai.agents.style_agent import StyleAgent
from photo_ai.core.image_processor import ImageProcessor

# Load images
content_image = ImageProcessor.load_image("path/to/content.jpg")
style_image = ImageProcessor.load_image("path/to/style.jpg")

# Create style agent with custom configuration
style_agent = StyleAgent({
    'style_weight': 1e6,
    'content_weight': 1.0,
    'num_steps': 300,
    'image_size': 512
})

# Process and save in multiple formats
result = style_agent.process_and_save(
    content_image=content_image,
    style_image=style_image,
    output_path="output/stylized",
    formats=['.jpg', '.png', '.webp'],
    quality=95
)

# Access results
stylized_image_paths = result['saved_files']
metrics = result['metrics']
```

## Implementation Details

- The style transfer implementation uses a pre-trained VGG19 model for feature extraction
- Image processing is handled by a robust `ImageProcessor` class with format-specific optimizations
- Progress is tracked using a `ProgressTracker` singleton that supports hierarchical operations
- Multiple format support is achieved through format-specific parameters and fallback mechanisms

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Pillow
- OpenCV
- NumPy 
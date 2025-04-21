# AI Photo Editor

An intelligent photo editing application that combines traditional image processing with advanced Gen AI capabilities.

## Features

### Traditional Photo Editing
- Auto adjustments based on image analysis
- Manual adjustment controls
- Style presets with cinematic looks

### Gen AI Capabilities

#### 1. Image Understanding
The application uses AI-powered image analysis to understand the content of your photos:
- Scene type detection (landscape, portrait, indoor, etc.)
- Lighting condition analysis
- Face detection
- Object recognition
- Color palette extraction
- Smart adjustment recommendations based on content

#### 2. Natural Language Editing
Edit your photos using plain English instructions:
- Describe desired changes in natural language
- AI interprets instructions and applies appropriate adjustments
- Function calling maps descriptions to specific editing operations
- Shows you exactly what changes were applied

#### 3. Retrieval Augmented Generation (RAG) for Style Recommendations
Get intelligent style recommendations:
- Knowledge base of cinematography techniques and styles
- Content-based recommendations tailored to your image
- Description-based matching using RAG architecture
- Detailed explanations of why styles were recommended

## Getting Started

### Installation

```bash
# Install the package
pip install -e .

# Run the web app
photo-web
```

### Usage

1. Upload an image using the file uploader
2. Try different editing modes:
   - Auto Adjustments: Apply recommended fixes
   - Manual Adjustments: Fine-tune your image
   - Style Presets: Apply cinematic looks
   - AI Assistant: Use Gen AI features
3. Download your edited image

## AI Assistant Tab

The AI Assistant tab provides access to all Gen AI capabilities:

### Image Understanding
- Click "Analyze Image Content" to get AI insights
- View scene type, detected objects, faces, and more
- Apply AI-recommended adjustments

### Natural Language Editing
- Type instructions like "Make the image warmer and increase contrast"
- The AI translates your words into specific editing operations
- View what operations were applied

### Style Recommendations
- Get style recommendations based on image content
- Or describe the style you want (e.g., "Like a dramatic movie scene")
- Apply recommended styles with one click

## Technical Details

The application uses:
- OpenCV and PIL for image processing
- Streamlit for the web interface
- Custom AI models for image analysis and understanding
- Function calling for natural language processing
- RAG architecture for style knowledge retrieval

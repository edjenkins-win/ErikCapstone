from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from typing import List, Dict, Any
import io
import cv2

from ..core.image_processor import ImageProcessor
from ..agents.color_agent import ColorAgent
from ..agents.exposure_agent import ExposureAgent
from ..agents.skin_agent import SkinAgent
from ..agents.composition_agent import CompositionAgent
from ..agents.background_agent import BackgroundAgent
from ..agents.noise_agent import NoiseAgent

app = FastAPI(
    title="AI Photo Editing System",
    description="API for AI-powered photo editing and style learning",
    version="0.1.0"
)

# Initialize components
image_processor = ImageProcessor()
color_agent = ColorAgent()
exposure_agent = ExposureAgent()
skin_agent = SkinAgent()
composition_agent = CompositionAgent()
background_agent = BackgroundAgent()
noise_agent = NoiseAgent()

# Agent processing order
PROCESSING_ORDER = [
    composition_agent,  # First, fix composition
    noise_agent,        # Reduce noise early to prevent artifacts
    exposure_agent,     # Then adjust exposure
    color_agent,        # Apply color grading
    skin_agent,         # Enhance skin
    background_agent    # Finally, enhance background
]

@app.post("/learn")
async def learn_from_pair(
    before: UploadFile = File(...),
    after: UploadFile = File(...)
):
    """Learn from a before/after image pair."""
    try:
        # Read and process images
        before_content = await before.read()
        after_content = await after.read()
        
        before_array = np.frombuffer(before_content, np.uint8)
        after_array = np.frombuffer(after_content, np.uint8)
        
        before_image = cv2.imdecode(before_array, cv2.IMREAD_COLOR)
        after_image = cv2.imdecode(after_array, cv2.IMREAD_COLOR)
        
        if before_image is None or after_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Convert to RGB
        before_image = cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB)
        after_image = cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB)
        
        # Learn from the pair with all agents
        for agent in PROCESSING_ORDER:
            agent.learn(before_image, after_image)
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Successfully learned from image pair",
                "agents_status": {
                    agent.__class__.__name__: agent.get_status()
                    for agent in PROCESSING_ORDER
                }
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_image(
    image: UploadFile = File(...),
    agent_config: Dict[str, Any] = None
):
    """Process an image using learned style."""
    try:
        # Read and process image
        content = await image.read()
        image_array = np.frombuffer(content, np.uint8)
        input_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if input_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Convert to RGB
        processed_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Apply processing in order
        for agent in PROCESSING_ORDER:
            processed_image = agent.process(processed_image, **(agent_config or {}))
        
        # Convert back to BGR for saving
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', processed_image)
        return JSONResponse(
            content={
                "status": "success",
                "message": "Image processed successfully",
                "image_size": processed_image.shape,
                "agents_status": {
                    agent.__class__.__name__: agent.get_status()
                    for agent in PROCESSING_ORDER
                }
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get current system status."""
    return JSONResponse(
        content={
            "status": "operational",
            "agents_status": {
                agent.__class__.__name__: agent.get_status()
                for agent in PROCESSING_ORDER
            }
        }
    ) 
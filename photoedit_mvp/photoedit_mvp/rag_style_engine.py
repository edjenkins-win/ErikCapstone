"""RAG Style Engine module for the Photo Editor application.

This module implements a Retrieval Augmented Generation (RAG) system for
suggesting and applying photo styles based on image content and user descriptions.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import cv2
from pathlib import Path
import random

# Import local modules
from .styles import StylePreset, get_style_preset
from .ai_analyzer import AIImageAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

class RAGStyleEngine:
    """Recommends and applies styles using RAG techniques."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RAG style engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._validate_config()
        
        # Initialize image analyzer
        self.image_analyzer = AIImageAnalyzer()
        
        # Initialize knowledge base
        self.knowledge_base = self._init_knowledge_base()
        
        # Initialize embedding database
        self.embedding_db = None
        self._init_embedding_database()
        
    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        defaults = {
            'knowledge_base_path': os.path.join(os.path.dirname(__file__), 'data', 'style_knowledge.json'),
            'embedding_db_path': os.path.join(os.path.dirname(__file__), 'data', 'style_embeddings.npz'),
            'custom_styles_path': os.path.join(os.path.dirname(__file__), 'data', 'custom_styles'),
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
                
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.config['knowledge_base_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['embedding_db_path']), exist_ok=True)
        os.makedirs(self.config['custom_styles_path'], exist_ok=True)
    
    def _init_knowledge_base(self) -> List[Dict[str, Any]]:
        """Initialize the style knowledge base.
        
        Returns:
            List of style knowledge entries
        """
        # Default knowledge base with cinematography styles and techniques
        default_knowledge = [
            {
                "style_name": "Cinematic Teal & Orange",
                "description": "Classic Hollywood color grading with teal shadows and orange highlights",
                "keywords": ["blockbuster", "cinematic", "movie", "film", "hollywood", "complementary"],
                "examples": ["Transformers", "Marvel movies", "Michael Bay", "action films"],
                "techniques": ["Shadow/highlight color split", "Blue shadows, orange highlights", "High contrast"]
            },
            {
                "style_name": "Film Noir",
                "description": "High contrast black and white with dramatic shadows and film grain",
                "keywords": ["noir", "detective", "dark", "mysterious", "contrast", "dramatic", "moody"],
                "examples": ["The Maltese Falcon", "Citizen Kane", "Double Indemnity"],
                "techniques": ["Hard shadows", "Low-key lighting", "Strong contrast", "Moody atmosphere"]
            },
            {
                "style_name": "Anamorphic",
                "description": "Widescreen cinematic look with lens flares and letterboxing",
                "keywords": ["widescreen", "anamorphic", "cinematic", "lens flare", "letterbox"],
                "examples": ["JJ Abrams films", "Star Trek", "Star Wars", "Sci-fi films"],
                "techniques": ["Horizontal lens flares", "Wide aspect ratio", "Anamorphic lens distortion"]
            },
            {
                "style_name": "Blockbuster",
                "description": "Vibrant colors and high contrast for modern action films",
                "keywords": ["action", "vibrant", "punchy", "bright", "dramatic", "dynamic"],
                "examples": ["Fast & Furious", "Mission Impossible", "Modern action films"],
                "techniques": ["Increased saturation", "High contrast", "Sharper details", "Vibrant colors"]
            },
            {
                "style_name": "Dreamy",
                "description": "Soft, ethereal look with warm tones and gentle glow",
                "keywords": ["soft", "ethereal", "dreamy", "romantic", "fantasy", "peaceful"],
                "examples": ["Romance films", "Fantasy sequences", "Music videos"],
                "techniques": ["Soft focus", "Glow effect", "Reduced contrast", "Warm tones"]
            },
            {
                "style_name": "Vintage",
                "description": "Classic film-inspired look with faded colors and subtle vignette",
                "keywords": ["retro", "vintage", "old", "classic", "film", "nostalgic"],
                "examples": ["Old photographs", "Analog film", "Instagram filters"],
                "techniques": ["Sepia tones", "Reduced contrast", "Faded blacks", "Vignette"]
            },
            {
                "style_name": "Portrait",
                "description": "Optimized for portrait photography with skin tone enhancement",
                "keywords": ["portrait", "person", "face", "skin", "beauty", "professional"],
                "examples": ["Professional portraits", "Headshots", "Fashion photography"],
                "techniques": ["Skin smoothing", "Warm tones", "Subtle contrast", "Detail preservation"]
            }
        ]
        
        # Try to load existing knowledge base
        try:
            if os.path.exists(self.config['knowledge_base_path']):
                with open(self.config['knowledge_base_path'], 'r') as f:
                    knowledge = json.load(f)
                    logger.info(f"Loaded {len(knowledge)} style knowledge entries")
                    return knowledge
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
        
        # If loading fails, use default and save it
        try:
            os.makedirs(os.path.dirname(self.config['knowledge_base_path']), exist_ok=True)
            with open(self.config['knowledge_base_path'], 'w') as f:
                json.dump(default_knowledge, f, indent=2)
                logger.info(f"Created default knowledge base with {len(default_knowledge)} entries")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
        
        return default_knowledge
    
    def _init_embedding_database(self) -> None:
        """Initialize the embedding database.
        
        In a real implementation, this would create embeddings for all style knowledge.
        For the MVP, we'll simulate embeddings.
        """
        # Attempt to load existing embeddings
        try:
            if os.path.exists(self.config['embedding_db_path']):
                # In a real implementation, load saved embeddings
                logger.info("Loaded style embeddings")
                self.embedding_db = True  # Placeholder, simulating successful loading
                return
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
        
        # If loading fails, generate simulated embeddings
        try:
            # In a real implementation, generate embeddings for all entries
            logger.info("Generating simulated style embeddings")
            self.embedding_db = True  # Placeholder, simulating successful generation
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
    
    def recommend_style(self, image: np.ndarray, description: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recommend styles based on image content and optional description.
        
        Args:
            image: Input image
            description: Optional user description of desired style
            
        Returns:
            List of recommended styles with reasoning
        """
        # Analyze image content
        _, analysis = self.image_analyzer.analyze(image)
        
        # Extract key features from analysis
        scene_type = analysis.get('scene_type', 'unknown')
        has_faces = analysis.get('has_faces', False)
        lighting = analysis.get('lighting_condition', 'normal')
        
        # Match against style knowledge
        if description:
            # For a real implementation, we would use embeddings to find similar styles
            # For now, we'll use keyword matching
            return self._match_by_description(description, scene_type, has_faces, lighting)
        else:
            # Content-based recommendation
            return self._match_by_content(scene_type, has_faces, lighting)
    
    def _match_by_description(self, description: str, scene_type: str, has_faces: bool, lighting: str) -> List[Dict[str, Any]]:
        """Match styles based on user description and image content.
        
        Args:
            description: User description of desired style
            scene_type: Detected scene type
            has_faces: Whether faces were detected
            lighting: Detected lighting condition
            
        Returns:
            List of recommended styles with reasoning
        """
        description_lower = description.lower()
        matches = []
        
        # Score each style entry
        for entry in self.knowledge_base:
            score = 0
            reasoning = []
            
            # Match based on style name
            if entry['style_name'].lower() in description_lower:
                score += 10
                reasoning.append(f"Explicitly mentioned {entry['style_name']}")
            
            # Match based on keywords
            for keyword in entry.get('keywords', []):
                if keyword.lower() in description_lower:
                    score += 3
                    reasoning.append(f"Mentioned '{keyword}'")
            
            # Match based on examples
            for example in entry.get('examples', []):
                if example.lower() in description_lower:
                    score += 5
                    reasoning.append(f"Referenced '{example}'")
            
            # Match based on techniques
            for technique in entry.get('techniques', []):
                tech_words = set(technique.lower().split())
                if any(word in description_lower for word in tech_words):
                    score += 2
                    reasoning.append(f"Described technique similar to '{technique}'")
            
            # Content-based boosting
            if scene_type == "landscape" and entry['style_name'] in ["Cinematic Teal & Orange", "Anamorphic"]:
                score += 2
                reasoning.append("Good match for landscape content")
            
            if has_faces and entry['style_name'] in ["Portrait", "Film Noir"]:
                score += 2
                reasoning.append("Good match for portrait content")
            
            if lighting == "low_light" and entry['style_name'] in ["Film Noir", "Anamorphic"]:
                score += 2
                reasoning.append("Good match for low-light conditions")
            
            # Add to matches if above threshold
            if score > 0:
                matches.append({
                    'style': entry['style_name'],
                    'description': entry['description'],
                    'score': score,
                    'reasoning': reasoning
                })
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:3]  # Return top 3 matches
    
    def _match_by_content(self, scene_type: str, has_faces: bool, lighting: str) -> List[Dict[str, Any]]:
        """Match styles based on image content.
        
        Args:
            scene_type: Detected scene type
            has_faces: Whether faces were detected
            lighting: Detected lighting condition
            
        Returns:
            List of recommended styles with reasoning
        """
        matches = []
        
        # Content-specific recommendations
        if has_faces:
            if lighting == "low_light":
                # Portrait in low light
                matches.append({
                    'style': "Film Noir",
                    'description': "High contrast black and white with dramatic shadows",
                    'score': 9,
                    'reasoning': ["Detected faces in low light", "Dramatic lighting suits noir style"]
                })
            else:
                # Standard portrait
                matches.append({
                    'style': "Portrait",
                    'description': "Optimized for portrait photography with skin tone enhancement",
                    'score': 9,
                    'reasoning': ["Detected faces", "Standard portrait enhancement"]
                })
        
        if scene_type in ["landscape", "nature"]:
            # Landscape scene
            matches.append({
                'style': "Cinematic Teal & Orange",
                'description': "Hollywood-style color grading with teal shadows and orange highlights",
                'score': 8,
                'reasoning': ["Detected landscape/nature scene", "Popular cinematic look for landscapes"]
            })
            
            matches.append({
                'style': "Anamorphic",
                'description': "Widescreen cinematic look with enhanced contrast and blue lens flares",
                'score': 7,
                'reasoning': ["Detected landscape/nature scene", "Wide cinematic style suits landscape views"]
            })
        
        if lighting == "low_light":
            # Low light scene
            if not has_faces and scene_type not in ["landscape", "nature"]:
                matches.append({
                    'style': "Anamorphic",
                    'description': "Widescreen cinematic look with enhanced contrast and blue lens flares",
                    'score': 6,
                    'reasoning': ["Low light scene detected", "Cinematic look enhances mood"]
                })
        
        if scene_type == "indoor":
            # Indoor scene
            matches.append({
                'style': "Vintage",
                'description': "Classic film-inspired look with faded colors",
                'score': 5,
                'reasoning': ["Detected indoor scene", "Vintage style works well with indoor settings"]
            })
        
        # Add auto-enhance as a fallback
        if not matches:
            matches.append({
                'style': "Auto-Enhance",
                'description': "Balanced automatic enhancement for most photos",
                'score': 5,
                'reasoning': ["General purpose enhancement"]
            })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:3]  # Return top 3 matches
    
    def apply_style(self, image: np.ndarray, style_name: str) -> np.ndarray:
        """Apply the specified style to an image.
        
        Args:
            image: Input image
            style_name: Name of the style to apply
            
        Returns:
            Processed image
        """
        # Import here to avoid circular imports
        from .executor import ImageExecutor
        
        # Get the style preset
        try:
            preset = get_style_preset(style_name)
            
            # Apply the style using the executor
            executor = ImageExecutor()
            processed = executor.apply(image, [], style_name)
            
            return processed
        except Exception as e:
            logger.error(f"Error applying style '{style_name}': {e}")
            return image
    
    def add_style_knowledge(self, style_name: str, knowledge: Dict[str, Any]) -> bool:
        """Add new style knowledge to the knowledge base.
        
        Args:
            style_name: Name of the style
            knowledge: Style knowledge data
            
        Returns:
            Success status
        """
        try:
            # Check if style already exists
            existing = next((entry for entry in self.knowledge_base 
                            if entry['style_name'].lower() == style_name.lower()), None)
            
            if existing:
                # Update existing entry
                for key, value in knowledge.items():
                    existing[key] = value
            else:
                # Add new entry
                self.knowledge_base.append({
                    'style_name': style_name,
                    **knowledge
                })
            
            # Save updated knowledge base
            with open(self.config['knowledge_base_path'], 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            
            # In a real implementation, we would update embeddings here
            logger.info(f"Added/updated style knowledge for '{style_name}'")
            return True
        except Exception as e:
            logger.error(f"Error adding style knowledge: {e}")
            return False
    
    def get_style_knowledge(self, style_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get style knowledge from the knowledge base.
        
        Args:
            style_name: Optional name of specific style to retrieve
            
        Returns:
            List of style knowledge entries
        """
        if style_name:
            # Return specific style
            matches = [entry for entry in self.knowledge_base 
                      if entry['style_name'].lower() == style_name.lower()]
            return matches
        else:
            # Return all styles
            return self.knowledge_base
    
    def search_style_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Search the style knowledge base for matching styles.
        
        Args:
            query: Search query
            
        Returns:
            List of matching style entries
        """
        query_lower = query.lower()
        matches = []
        
        for entry in self.knowledge_base:
            # Check name
            if query_lower in entry['style_name'].lower():
                matches.append(entry)
                continue
                
            # Check description
            if 'description' in entry and query_lower in entry['description'].lower():
                matches.append(entry)
                continue
                
            # Check keywords
            if 'keywords' in entry and any(query_lower in keyword.lower() for keyword in entry['keywords']):
                matches.append(entry)
                continue
                
            # Check examples
            if 'examples' in entry and any(query_lower in example.lower() for example in entry['examples']):
                matches.append(entry)
                continue
                
            # Check techniques
            if 'techniques' in entry and any(query_lower in technique.lower() for technique in entry['techniques']):
                matches.append(entry)
                continue
        
        return matches 
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

class ModelStorage:
    """Handles saving and loading trained models."""
    
    def __init__(self, base_dir: str = "models"):
        """Initialize model storage.
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file if it doesn't exist
        self.metadata_file = self.base_dir / "metadata.json"
        if not self.metadata_file.exists():
            self._save_metadata({})
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return {}
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def save_model(self, 
                  agent_name: str, 
                  model_data: Dict[str, Any], 
                  description: str = "",
                  version: str = "1.0.0") -> str:
        """Save a trained model.
        
        Args:
            agent_name: Name of the agent
            model_data: Model data to save
            description: Description of the model
            version: Version number
            
        Returns:
            Model ID
        """
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{agent_name}_{timestamp}"
        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model data
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata[model_id] = {
            "agent_name": agent_name,
            "timestamp": timestamp,
            "description": description,
            "version": version,
            "path": str(model_file)
        }
        self._save_metadata(metadata)
        
        return model_id
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a trained model.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded model data
        """
        metadata = self._load_metadata()
        if model_id not in metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_file = Path(metadata[model_id]["path"])
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    
    def list_models(self, agent_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List all saved models.
        
        Args:
            agent_name: Optional filter by agent name
            
        Returns:
            Dictionary of model metadata
        """
        metadata = self._load_metadata()
        if agent_name:
            return {k: v for k, v in metadata.items() 
                   if v["agent_name"] == agent_name}
        return metadata
    
    def delete_model(self, model_id: str) -> None:
        """Delete a saved model.
        
        Args:
            model_id: ID of the model to delete
        """
        metadata = self._load_metadata()
        if model_id not in metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Delete model files
        model_dir = self.base_dir / model_id
        if model_dir.exists():
            for file in model_dir.iterdir():
                file.unlink()
            model_dir.rmdir()
        
        # Update metadata
        del metadata[model_id]
        self._save_metadata(metadata) 
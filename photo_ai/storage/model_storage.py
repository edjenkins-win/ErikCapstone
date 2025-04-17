import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    """Metadata for a saved model."""
    model_id: str
    agent_name: str
    version: str
    description: str
    created_at: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]

class ModelStorage:
    """Handles model persistence and versioning."""
    
    def __init__(self, base_dir: str = "models"):
        """Initialize the model storage.
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_model_id(self, agent_name: str) -> str:
        """Generate a unique model ID.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Unique model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{agent_name}_{timestamp}"
    
    def save_model(self,
                  agent_name: str,
                  model_data: Any,
                  description: str,
                  version: str,
                  metrics: Optional[Dict[str, float]] = None,
                  parameters: Optional[Dict[str, Any]] = None) -> str:
        """Save a model to disk.
        
        Args:
            agent_name: Name of the agent
            model_data: Model state to save
            description: Description of the model
            version: Version string
            metrics: Optional training metrics
            parameters: Optional model parameters
            
        Returns:
            Model ID
        """
        model_id = self._generate_model_id(agent_name)
        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = model_dir / "model.pt"
        torch.save(model_data, model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            agent_name=agent_name,
            version=version,
            description=description,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
            parameters=parameters or {}
        )
        
        # Update storage metadata
        self.metadata[model_id] = {
            "agent_name": metadata.agent_name,
            "version": metadata.version,
            "description": metadata.description,
            "created_at": metadata.created_at,
            "metrics": metadata.metrics,
            "parameters": metadata.parameters
        }
        self._save_metadata()
        
        return model_id
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model from disk.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Dictionary containing model data and metadata
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_dir = self.base_dir / model_id
        model_path = model_dir / "model.pt"
        
        if not model_path.exists():
            raise ValueError(f"Model file for {model_id} not found")
        
        return {
            "model_data": torch.load(model_path),
            "metadata": self.metadata[model_id]
        }
    
    def list_models(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all saved models.
        
        Args:
            agent_name: Optional filter by agent name
            
        Returns:
            List of model metadata
        """
        models = []
        for model_id, metadata in self.metadata.items():
            if agent_name is None or metadata["agent_name"] == agent_name:
                models.append({
                    "model_id": model_id,
                    **metadata
                })
        
        # Sort by creation date, newest first
        return sorted(
            models,
            key=lambda x: x["created_at"],
            reverse=True
        )
    
    def delete_model(self, model_id: str) -> None:
        """Delete a model.
        
        Args:
            model_id: ID of the model to delete
        """
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Remove model files
        model_dir = self.base_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Update metadata
        del self.metadata[model_id]
        self._save_metadata()
    
    def get_latest_version(self, agent_name: str) -> Optional[str]:
        """Get the latest version for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Latest version string or None if no models exist
        """
        models = self.list_models(agent_name)
        return models[0]["version"] if models else None 
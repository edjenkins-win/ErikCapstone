import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import json

from PIL import Image

from photo_ai.core.vector_store import VectorStore
from photo_ai.core.feature_extractor import FeatureExtractor
from photo_ai.agents.base_agent import BaseAgent

class RAGStyleAgent(BaseAgent):
    """Style transfer agent enhanced with RAG capabilities."""

    def __init__(self):
        """Initialize the RAG-enhanced style transfer agent."""
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(device=str(self.device))

        # Initialize vector store
        self.vector_store = VectorStore(
            dimension=self.feature_extractor.get_feature_dimension(),
            index_path="style_vectors.index"
        )

        # Load VGG19 for style transfer
        self._load_model()

    def _load_model(self) -> None:
        """Load the VGG19 model for style transfer."""
        try:
            self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device)
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Error loading VGG19 model: {e}")
            self.model = models.vgg19(weights=None).features.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False

    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        self.config.setdefault('style_weight', 1e6)
        self.config.setdefault('content_weight', 1)
        self.config.setdefault('num_steps', 300)
        self.config.setdefault('learning_rate', 0.01)
        self.config.setdefault('image_size', 512)
        self.config.setdefault('k_nearest', 3)  # Number of similar styles to consider

    def add_style(self, style_path: str, metadata: Dict[str, Any]) -> None:
        """Add a style image to the vector store.

        Args:
            style_path: Path to the style image
            metadata: Additional metadata about the style
        """
        # Extract features
        features = self.feature_extractor.extract_single_feature(style_path)

        # Add to vector store
        self.vector_store.add_vectors(
            np.array([features]),
            [{'path': style_path, **metadata}]
        )

    def find_similar_styles(self, query_image_path: str, k: int = None) -> List[Dict[str, Any]]:
        """Find similar styles to a query image.

        Args:
            query_image_path: Path to the query image
            k: Number of similar styles to return

        Returns:
            List of similar styles with metadata
        """
        k = k or self.config['k_nearest']

        # Extract features from query image
        query_features = self.feature_extractor.extract_single_feature(query_image_path)

        # Search for similar styles
        return self.vector_store.search(query_features, k)

    def process(self, image: np.ndarray, style_image: Optional[np.ndarray] = None, 
                style_path: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process an image with style transfer.

        Args:
            image: Content image as numpy array
            style_image: Style image as numpy array (optional)
            style_path: Path to style image (optional)

        Returns:
            Tuple containing:
                - Stylized image as numpy array
                - Dictionary with processing metrics
        """
        if style_image is None and style_path is None:
            raise ValueError("Either style_image or style_path must be provided")

        if style_image is None:
            # Load style image
            style_image = np.array(Image.open(style_path).convert('RGB'))

        # Convert images to tensors
        content_tensor = self._image_to_tensor(image)
        style_tensor = self._image_to_tensor(style_image)

        # Initialize target image
        target = content_tensor.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=self.config['learning_rate'])

        # Get features
        content_features = self._get_features(content_tensor)
        style_features = self._get_features(style_tensor)

        # Style transfer loop
        for step in range(self.config['num_steps']):
            optimizer.zero_grad()
            target_features = self._get_features(target)

            # Calculate losses
            content_loss = self._content_loss(target_features, content_features)
            style_loss = self._style_loss(target_features, style_features)
            total_loss = self.config['content_weight'] * content_loss + \
                        self.config['style_weight'] * style_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Clamp values
            with torch.no_grad():
                target.clamp_(0, 1)

        # Convert back to numpy array
        result = self._tensor_to_image(target)

        # Return result and metrics
        metrics = {
            'content_loss': float(content_loss),
            'style_loss': float(style_loss),
            'total_loss': float(total_loss)
        }

        return result, metrics

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image."""
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (image * 255).astype(np.uint8)

    def _get_features(self, image: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get features from specified layers."""
        features = {}
        x = image
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in [4, 9, 18, 27, 36]:  # ReLU layers
                features[i] = x
        return features

    def _content_loss(self, target_features: Dict[int, torch.Tensor],
                     content_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Calculate content loss."""
        loss = 0
        for layer in content_features:
            loss += torch.mean((target_features[layer] - content_features[layer]) ** 2)
        return loss

    def _style_loss(self, target_features: Dict[int, torch.Tensor],
                   style_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Calculate style loss."""
        loss = 0
        for layer in style_features:
            target_gram = self._gram_matrix(target_features[layer])
            style_gram = self._gram_matrix(style_features[layer])
            loss += torch.mean((target_gram - style_gram) ** 2)
        return loss

    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix."""
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)

    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn from a before/after image pair.

        Args:
            before_image: Original image
            after_image: Edited image with style applied
        """
        # Extract features from both images
        before_tensor = self._image_to_tensor(before_image)
        after_tensor = self._image_to_tensor(after_image)

        # Extract style features from the after image
        after_features = self._get_features(after_tensor)
        style_features = {layer: self._gram_matrix(features) for layer, features in after_features.items()}

        # Store the extracted style in the vector store with metadata
        style_vector = np.mean([tensor.cpu().numpy().flatten() for tensor in style_features.values()], axis=0)
        style_vector = style_vector / np.linalg.norm(style_vector)  # Normalize

        # Add to vector store with metadata about the learning sample
        self.vector_store.add_vectors(
            np.array([style_vector]),
            [{'learned': True, 'timestamp': np.datetime64('now')}]
        )

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            'name': self.__class__.__name__,
            'device': str(self.device),
            'num_styles': len(self.vector_store.metadata),
            'config': self.config
        } 

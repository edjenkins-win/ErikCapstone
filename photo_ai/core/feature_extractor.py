import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Optional

class FeatureExtractor:
    """Extracts features from images using a pre-trained model."""
    
    def __init__(self, model_name: str = "resnet50", device: Optional[str] = None):
        """Initialize the feature extractor.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on (cuda, mps, or cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load pre-trained model
        self.model = getattr(models, model_name)(weights="DEFAULT")
        
        # Remove the last layer (classification layer)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from a list of images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Array of feature vectors
        """
        features = []
        
        for image_path in image_paths:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                feature = self.model(image_tensor)
                feature = feature.squeeze().cpu().numpy()
            
            features.append(feature)
        
        return np.array(features)
    
    def extract_single_feature(self, image_path: str) -> np.ndarray:
        """Extract features from a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Feature vector
        """
        return self.extract_features([image_path])[0]
    
    def get_feature_dimension(self) -> int:
        """Get the dimension of the feature vectors."""
        # Create a dummy input to get the output dimension
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        return output.shape[1] 
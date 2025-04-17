import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import ssl
import logging
import os
from ..utils.training_visualizer import TrainingVisualizer
from .base_agent import BaseAgent
from torchvision.models.vgg import VGG19_Weights
from ..core.model_loader import LazyModelLoader
from ..utils.quantization import ModelQuantizer, QuantizationLevel
from ..utils.progress_tracker import ProgressContext
from ..core.image_processor import ImageProcessor

class StyleAgent(BaseAgent):
    """Agent for neural style transfer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the style transfer agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.visualizer = TrainingVisualizer()
        self.model_loader = LazyModelLoader()
        self.model_quantizer = ModelQuantizer()
        # Model will be loaded lazily when needed, not at initialization

    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        self.config.setdefault('style_weight', 1e6)
        self.config.setdefault('content_weight', 1)
        self.config.setdefault('num_steps', 300)
        self.config.setdefault('learning_rate', 0.03)
        self.config.setdefault('image_size', 512)
        self.config.setdefault('use_quantization', True)
        self.config.setdefault('quantization_level', 'FP16')
        self.config.setdefault('default_output_format', '.jpg')
        self.config.setdefault('default_quality', 95)
    
    def _load_model(self) -> nn.Module:
        """Load the VGG19 model for feature extraction.
        
        Returns:
            The loaded model
        """
        def model_loader():
            try:
                # Set torch to use a single thread
                torch.set_num_threads(1)
                
                # Try loading with DEFAULT weights first
                model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
            except Exception as e:
                self.logger.warning(f"Error loading VGG19 model with DEFAULT weights: {e}")
                try:
                    # Fallback to IMAGENET1K_V1 weights
                    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
                except Exception as e:
                    self.logger.warning(f"Error loading VGG19 model with IMAGENET1K_V1 weights: {e}")
                    try:
                        # Final fallback to loading without weights
                        model = models.vgg19(weights=None).features
                        self.logger.info("Loaded VGG19 model without pretrained weights")
                    except Exception as e:
                        self.logger.error(f"Failed to load VGG19 model: {e}")
                        raise
            
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            # Set model to evaluation mode
            model.eval()
            
            return model
        
        # Use the LazyModelLoader to get or load the model
        model = self.model_loader.get_model("vgg19_style_transfer", model_loader)
        
        # Apply quantization if enabled
        if self.config.get('use_quantization', True):
            try:
                # Convert quantization level string to enum
                level_str = self.config.get('quantization_level', 'FP16')
                level = getattr(QuantizationLevel, level_str)
                
                # Apply quantization
                model = self.model_quantizer.quantize_model(model, level)
                self.logger.info(f"Applied {level_str} quantization to VGG19 model")
            except Exception as e:
                self.logger.warning(f"Failed to apply quantization: {e}")
        
        return model
    
    def _get_features(self, image: torch.Tensor, layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """Extract features from specified layers.
        
        Args:
            image: Input image tensor
            layers: List of layer indices to extract features from
            
        Returns:
            Dictionary of layer indices to feature tensors
        """
        if layers is None:
            layers = [0, 5, 10, 19, 28]  # Default layers for style and content
        
        # Ensure model is loaded
        model = self._load_model()
        
        features = {}
        x = image
        
        for i, layer in enumerate(model):
            x = layer(x)
            if i in layers:
                features[i] = x
        
        return features
    
    def _gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Calculate the Gram matrix for style loss.
        
        Args:
            tensor: Feature tensor
            
        Returns:
            Gram matrix
        """
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def _style_loss(self, 
                   target_features: Dict[int, torch.Tensor],
                   style_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Calculate style loss.
        
        Args:
            target_features: Features from target image
            style_features: Features from style image
            
        Returns:
            Style loss
        """
        loss = 0
        for layer in target_features:
            target_gram = self._gram_matrix(target_features[layer])
            style_gram = self._gram_matrix(style_features[layer])
            loss += torch.mean((target_gram - style_gram) ** 2)
        return loss
    
    def _content_loss(self,
                     target_features: Dict[int, torch.Tensor],
                     content_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Calculate content loss.
        
        Args:
            target_features: Features from target image
            content_features: Features from content image
            
        Returns:
            Content loss
        """
        return torch.mean((target_features[28] - content_features[28]) ** 2)
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Image tensor
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image.
        
        Args:
            tensor: Image tensor
            
        Returns:
            Image as numpy array
        """
        image = tensor.cpu().clone().squeeze(0)
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()
        return (image * 255).astype(np.uint8)
    
    def _prepare_images(self, content_image: np.ndarray, style_image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Prepare images for style transfer.
        
        Args:
            content_image: Content image as numpy array
            style_image: Style image as numpy array
            
        Returns:
            Tuple of (content_tensor, style_tensor, target_tensor, content_features, style_features)
        """
        # Convert images to tensors
        content_tensor = self._image_to_tensor(content_image)
        style_tensor = self._image_to_tensor(style_image)
        target_tensor = content_tensor.clone().requires_grad_(True)
        
        # Get features
        content_features = self._get_features(content_tensor)
        style_features = self._get_features(style_tensor)
        
        return content_tensor, style_tensor, target_tensor, content_features, style_features
    
    def _calculate_losses(self, target_features: Dict[int, torch.Tensor], 
                         content_features: Dict[int, torch.Tensor], 
                         style_features: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate losses for style transfer.
        
        Args:
            target_features: Features from target image
            content_features: Features from content image
            style_features: Features from style image
            
        Returns:
            Tuple of (style_loss, content_loss, total_loss)
        """
        style_loss = self._style_loss(target_features, style_features)
        content_loss = self._content_loss(target_features, content_features)
        
        total_loss = (self.config['style_weight'] * style_loss + 
                     self.config['content_weight'] * content_loss)
        
        return style_loss, content_loss, total_loss
    
    def process(self, 
               content_image: np.ndarray,
               style_image: np.ndarray,
               num_steps: Optional[int] = None,
               operation_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply style transfer to content image.
        
        Args:
            content_image: Content image as numpy array
            style_image: Style image as numpy array
            num_steps: Optional number of optimization steps
            operation_id: Optional identifier for progress tracking
            
        Returns:
            Tuple of (stylized image, metrics)
        """
        if num_steps is None:
            num_steps = self.config['num_steps']
            
        if operation_id is None:
            operation_id = f"style_transfer_{id(content_image)}"
            
        # Prepare the progress tracker
        progress_description = "Applying neural style transfer"
        
        # Track the overall style transfer process
        with ProgressContext(operation_id, progress_description, num_steps) as progress:
            # Prepare images and feature extraction (preprocessing stage)
            progress.update(0, "Preparing images and extracting features")
            content_tensor, style_tensor, target_tensor, content_features, style_features = self._prepare_images(content_image, style_image)
            
            # Initialize optimizer
            optimizer = optim.Adam([target_tensor], lr=self.config['learning_rate'])
            
            # Training loop
            for step in range(num_steps):
                optimizer.zero_grad()
                
                # Get target features
                target_features = self._get_features(target_tensor)
                
                # Calculate losses
                style_loss, content_loss, total_loss = self._calculate_losses(
                    target_features, content_features, style_features)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Update metrics for visualization
                self.visualizer.update_metrics(
                    agent_name=self.__class__.__name__,
                    loss=total_loss.item(),
                    accuracy=0.0,  # Not applicable for style transfer
                    learning_rate=optimizer.param_groups[0]['lr']
                )
                
                # Update progress with detailed information
                step_metrics = {
                    'style_loss': style_loss.item(),
                    'content_loss': content_loss.item(),
                    'total_loss': total_loss.item()
                }
                progress_message = f"Step {step+1}/{num_steps}: Loss = {total_loss.item():.4f}"
                progress.update(step + 1, progress_message)
            
            # Convert result to numpy
            progress.update(num_steps, "Finalizing stylized image")
            result_image = self._tensor_to_image(target_tensor)
            
            # Return metrics
            metrics = {
                'final_loss': total_loss.item(),
                'style_loss': style_loss.item(),
                'content_loss': content_loss.item(),
                'num_steps': num_steps,
                'style_weight': self.config['style_weight'],
                'content_weight': self.config['content_weight']
            }
            
            return result_image, metrics
    
    def save_stylized_image(self, 
                           stylized_image: np.ndarray, 
                           output_path: str,
                           formats: Optional[List[str]] = None,
                           quality: Optional[int] = None,
                           operation_id: Optional[str] = None) -> Dict[str, str]:
        """Save a stylized image in multiple formats.
        
        Args:
            stylized_image: The stylized image as numpy array
            output_path: Base path for saving the image (without extension)
            formats: List of format extensions to save (e.g., ['.jpg', '.png'])
            quality: Quality for lossy formats (0-100)
            operation_id: Optional identifier for progress tracking
            
        Returns:
            Dictionary mapping format to saved file path
        """
        if formats is None:
            formats = [self.config.get('default_output_format', '.jpg')]
            
        if quality is None:
            quality = self.config.get('default_quality', 95)
            
        if operation_id is None:
            operation_id = f"save_stylized_{id(stylized_image)}"
            
        # Get the directory and filename from the output path
        output_dir = os.path.dirname(output_path)
        filename_base = os.path.basename(output_path)
        # Remove extension if present
        filename_base = os.path.splitext(filename_base)[0]
        
        # Ensure only supported formats are used
        supported_formats = ImageProcessor.get_supported_formats()
        valid_formats = [fmt for fmt in formats if fmt.lower() in supported_formats]
        
        if not valid_formats:
            valid_formats = [self.config.get('default_output_format', '.jpg')]
            self.logger.warning(f"No valid formats specified, using default: {valid_formats[0]}")
            
        # Track saving progress
        with ProgressContext(operation_id, "Saving stylized image", len(valid_formats)) as progress:
            saved_files = {}
            
            for i, fmt in enumerate(valid_formats):
                # Construct output filename with format extension
                if not fmt.startswith('.'):
                    fmt = '.' + fmt
                output_file = os.path.join(output_dir, f"{filename_base}{fmt}")
                
                # Save image in the current format
                progress.update(i, f"Saving image as {fmt.upper()} format")
                try:
                    ImageProcessor.save_image(stylized_image, output_file, quality)
                    saved_files[fmt] = output_file
                    self.logger.info(f"Saved stylized image as {fmt} to {output_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save image as {fmt}: {str(e)}")
            
            progress.update(len(valid_formats), f"Saved stylized image in {len(saved_files)} formats")
            return saved_files
    
    def process_and_save(self,
                        content_image: np.ndarray,
                        style_image: np.ndarray,
                        output_path: str,
                        formats: Optional[List[str]] = None,
                        num_steps: Optional[int] = None,
                        quality: Optional[int] = None) -> Dict[str, Any]:
        """Process a content image with style transfer and save the result in multiple formats.
        
        Args:
            content_image: Content image as numpy array
            style_image: Style image as numpy array
            output_path: Base path for saving the image (without extension)
            formats: List of format extensions to save (e.g., ['.jpg', '.png'])
            num_steps: Optional number of optimization steps
            quality: Quality for lossy formats (0-100)
            
        Returns:
            Dictionary with processing metrics and saved file paths
        """
        # Create a unique operation ID for tracking
        operation_id = f"style_transfer_{id(content_image)}_{id(style_image)}"
        
        # Process the image with style transfer
        stylized_image, metrics = self.process(
            content_image, 
            style_image, 
            num_steps=num_steps, 
            operation_id=operation_id
        )
        
        # Save the stylized image in multiple formats
        save_operation_id = f"{operation_id}_save"
        saved_files = self.save_stylized_image(
            stylized_image,
            output_path,
            formats=formats,
            quality=quality,
            operation_id=save_operation_id
        )
        
        # Return combined results
        result = {
            'metrics': metrics,
            'saved_files': saved_files
        }
        
        return result
    
    def _get_model_data(self) -> Dict[str, Any]:
        """Get the current model state.
        
        Returns:
            Dictionary containing model state
        """
        return {
            'config': self.config,
            'device': str(self.device)
        }
    
    def _set_model_data(self, model_data: Dict[str, Any]) -> None:
        """Set the model state.
        
        Args:
            model_data: Dictionary containing model state
        """
        self.config.update(model_data.get('config', {}))
        self.device = torch.device(model_data.get('device', 'cpu'))
        self._load_model()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent.
        
        Returns:
            Dictionary containing agent status
        """
        status = super().get_status()
        status['visualizer'] = self.visualizer
        return status
    
    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn from before/after image pairs.
        
        For style transfer, we don't need to learn from image pairs as we use
        a pre-trained VGG model for feature extraction.
        
        Args:
            before_image: Original image
            after_image: Edited image
        """
        pass  # Style transfer uses a pre-trained model, no learning needed 
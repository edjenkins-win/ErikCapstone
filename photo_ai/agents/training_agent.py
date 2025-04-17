from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from photo_ai.agents.base_agent import BaseAgent
from photo_ai.utils.training_visualizer import TrainingVisualizer

class TrainingAgent(BaseAgent):
    """Agent responsible for training and fine-tuning models."""
    
    def __init__(self):
        """Initialize the training agent."""
        super().__init__()
        self.visualizer = TrainingVisualizer()
        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    def _validate_config(self) -> None:
        """Validate and set default configuration parameters."""
        self.config.setdefault('learning_rate', 0.001)
        self.config.setdefault('batch_size', 32)
        self.config.setdefault('num_epochs', 10)
        self.config.setdefault('weight_decay', 0.0001)
        self.config.setdefault('patience', 5)
    
    def process(self, input_data: Any) -> Any:
        """Process input data using the trained model.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            output = self.model(input_tensor)
            return output.cpu().numpy()
    
    def learn(self, input_data: Any, target_data: Any) -> Dict[str, float]:
        """Train the model on the given data.
        
        Args:
            input_data: Input data for training
            target_data: Target data for training
            
        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        self.model.train()
        criterion = nn.MSELoss()
        
        # Convert data to tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_data, dtype=torch.float32).to(self.device)
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.visualizer.update_metrics(
                self.__class__.__name__,
                loss=loss.item(),
                accuracy=0.0,  # Placeholder for now
                validation_loss=0.0,  # Placeholder for now
                validation_accuracy=0.0,  # Placeholder for now
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
        
        return {
            'final_loss': loss.item(),
            'epochs': self.config['num_epochs']
        }
    
    def setup_model(self, model: nn.Module) -> None:
        """Set up the model and optimizer for training.
        
        Args:
            model: PyTorch model to train
        """
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            'name': self.__class__.__name__,
            'device': str(self.device),
            'model_initialized': self.model is not None,
            'optimizer_initialized': self.optimizer is not None,
            'config': self.config,
            'visualizer': self.visualizer
        } 
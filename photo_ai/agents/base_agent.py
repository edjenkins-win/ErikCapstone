from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from ..utils.model_storage import ModelStorage
from ..utils.performance import PerformanceOptimizer, ProcessingMode

class BaseAgent(ABC):
    """Abstract base class for all photo editing agents following SRP."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._validate_config()
        self.performance_optimizer = PerformanceOptimizer(
            max_workers=self.config.get('max_workers'),
            cache_size=self.config.get('cache_size', 100),
            processing_mode=ProcessingMode(self.config.get('processing_mode', ProcessingMode.CPU))
        )
        self.model_storage = ModelStorage()
        self._model_id: Optional[str] = None

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate agent configuration."""
        pass

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> tuple[np.ndarray, Dict[str, Any]]:
        """Process an image and return the result with metrics.

        Args:
            image: Input image as numpy array
            **kwargs: Additional processing parameters

        Returns:
            Tuple containing:
                - Processed image as numpy array
                - Dictionary with processing metrics
        """
        pass

    @abstractmethod
    def learn(self, before_image: np.ndarray, after_image: np.ndarray) -> None:
        """Learn from a before/after image pair.

        Args:
            before_image: Original image
            after_image: Edited image
        """
        pass

    def save_model(self, 
                  description: str,
                  version: Optional[str] = None,
                  metrics: Optional[Dict[str, float]] = None) -> str:
        """Save the current model state.

        Args:
            description: Description of the model state
            version: Optional version string
            metrics: Optional training metrics

        Returns:
            Model ID
        """
        # Get model data
        model_data = self._get_model_data()

        # Generate version if not provided
        if version is None:
            latest_version = self.model_storage.get_latest_version(self.__class__.__name__)
            if latest_version is None:
                version = "1.0.0"
            else:
                major, minor, patch = map(int, latest_version.split('.'))
                version = f"{major}.{minor}.{patch + 1}"

        # Save model
        model_id = self.model_storage.save_model(
            agent_name=self.__class__.__name__,
            model_data=model_data,
            description=description,
            version=version,
            metrics=metrics,
            parameters=self.config
        )

        self._model_id = model_id
        return model_id

    def load_model(self, model_id: str) -> None:
        """Load a saved model state.

        Args:
            model_id: ID of the model to load
        """
        model_info = self.model_storage.load_model(model_id)
        self._set_model_data(model_info["model_data"])
        self.config.update(model_info["metadata"]["parameters"])
        self._model_id = model_id

    def _get_model_data(self) -> Dict[str, Any]:
        """Get the current model state.

        Returns:
            Dictionary containing model state
        """
        raise NotImplementedError("Subclasses must implement _get_model_data")

    def _set_model_data(self, model_data: Dict[str, Any]) -> None:
        """Set the model state.

        Args:
            model_data: Dictionary containing model state
        """
        raise NotImplementedError("Subclasses must implement _set_model_data")

    def _get_model_state(self) -> Dict[str, Any]:
        """Get the internal model state.

        Returns:
            Dictionary containing internal model state
        """
        return {}

    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set the internal model state.

        Args:
            state: Dictionary containing internal model state
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent.

        Returns:
            Dictionary containing agent status
        """
        return {
            "name": self.__class__.__name__,
            "config": self.config,
            "model_id": self._model_id,
            "performance": self.performance_optimizer.get_performance_report()
        } 

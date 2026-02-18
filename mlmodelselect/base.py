"""Base model interface for all MLModelSelect models."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Tuple


class BaseModel(ABC):
    """Abstract base class for all ML models in MLModelSelect.
    
    This provides a unified interface for all models, making them
    easy to swap and compare.
    """
    
    def __init__(self, **kwargs):
        """Initialize the model with given parameters.
        
        Args:
            **kwargs: Model-specific parameters
        """
        self.params = kwargs
        self.is_fitted = False
        self.model = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit the model to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        pass
        
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the model and return predictions on training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Predictions on training data
        """
        self.fit(X, y)
        return self.predict(X)
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.params.copy()
        
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The model with updated parameters
        """
        self.params.update(params)
        return self
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model information like type, parameters, etc.
        """
        pass

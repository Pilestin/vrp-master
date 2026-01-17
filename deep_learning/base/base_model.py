from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseVRPModel(nn.Module, ABC):
    """
    Abstract base class for all VRP neural network models.
    All models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass of the model.
        
        Args:
            inputs: (batch_size, num_nodes, input_dim) - Node features (x, y, demand)
            deterministic: If True, use greedy selection. If False, sample from distribution.
            
        Returns:
            tour_indices: (batch_size, num_nodes-1) - Sequence of visited customer indices
            tour_log_probs: List of log probabilities for each selection (empty if deterministic)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> dict:
        """
        Return model configuration as a dictionary.
        Useful for saving/loading and logging.
        """
        pass
    
    def get_model_name(self) -> str:
        """Return the name of the model."""
        return self.__class__.__name__

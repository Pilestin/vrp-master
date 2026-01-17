import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.base.base_trainer import BaseTrainer
from deep_learning.utils.data_utils import generate_vrp_data, calculate_tour_length
from .model import PointerNetworkModel


class PointerNetworkTrainer(BaseTrainer):
    """
    Trainer for the Pointer Network model.
    Uses REINFORCE with moving average baseline.
    """
    
    def __init__(self, model=None, device=None, lr=1e-3):
        if model is None:
            model = PointerNetworkModel(input_dim=3, hidden_dim=128)
        super().__init__(model, device, lr)
    
    def generate_data(self, batch_size: int, num_nodes: int) -> torch.Tensor:
        return generate_vrp_data(batch_size, num_nodes)
    
    def calculate_reward(self, tour_indices: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return calculate_tour_length(tour_indices, inputs)


def train_model(num_epochs=100, batch_size=32, num_nodes=20, 
                save_path="deep_learning/checkpoints/pointer_network/vrp_model.pth"):
    """
    Train the Pointer Network model.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        num_nodes: Number of nodes (including depot)
        save_path: Path to save the trained model
    """
    trainer = PointerNetworkTrainer()
    trainer.train(num_epochs, batch_size, num_nodes, save_path)


if __name__ == "__main__":
    os.makedirs("deep_learning/checkpoints/pointer_network", exist_ok=True)
    train_model(num_epochs=500, batch_size=32, num_nodes=30)

"""
Trainer for the GNN Model.

Uses REINFORCE with greedy rollout baseline.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.base.base_trainer import BaseTrainer
from deep_learning.utils.data_utils import generate_vrp_data, calculate_tour_length
from .model import GNNModel


class GNNTrainer(BaseTrainer):
    """
    Trainer for the GNN Model.
    
    Uses REINFORCE with greedy rollout baseline for reduced variance.
    """
    
    def __init__(self, model=None, device=None, lr=1e-4,
                 hidden_dim=128, num_layers=3, num_heads=4):
        if model is None:
            model = GNNModel(
                input_dim=3,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads
            )
        super().__init__(model, device, lr)
        
        self.baseline_model = None
        self.baseline_update_freq = 50
    
    def generate_data(self, batch_size: int, num_nodes: int) -> torch.Tensor:
        return generate_vrp_data(batch_size, num_nodes)
    
    def calculate_reward(self, tour_indices: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return calculate_tour_length(tour_indices, inputs)
    
    def train_epoch(self, batch_size: int, num_nodes: int, epoch: int) -> float:
        """Train for one epoch with greedy rollout baseline."""
        self.model.train()
        inputs = self.generate_data(batch_size, num_nodes).to(self.device)
        
        # Sample solution
        tour_indices, tour_log_probs = self.model(inputs, deterministic=False)
        reward = self.calculate_reward(tour_indices, inputs)
        
        # Compute baseline
        if self.baseline_model is None:
            if epoch == 0:
                self.moving_avg_reward = reward.mean().item()
            baseline = self.moving_avg_reward
        else:
            with torch.no_grad():
                baseline_tour, _ = self.baseline_model(inputs, deterministic=True)
                baseline = self.calculate_reward(baseline_tour, inputs).mean().item()
        
        self.moving_avg_reward = 0.9 * self.moving_avg_reward + 0.1 * reward.mean().item()
        
        # REINFORCE loss
        advantage = reward - baseline
        loss = (advantage.detach() * sum(tour_log_probs)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return reward.mean().item()
    
    def train(self, num_epochs: int, batch_size: int, num_nodes: int,
              save_path: str, save_interval: int = 100):
        """Full training loop with periodic baseline updates."""
        print(f"Training {self.model.get_model_name()} on {self.device}...")
        print(f"Config: {self.model.get_config()}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(num_epochs):
            avg_reward = self.train_epoch(batch_size, num_nodes, epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Tour Length = {avg_reward:.4f}")
            
            # Update baseline model periodically
            if (epoch + 1) % self.baseline_update_freq == 0:
                if self.baseline_model is None:
                    self.baseline_model = GNNModel(
                        input_dim=self.model.input_dim,
                        hidden_dim=self.model.hidden_dim,
                        num_layers=self.model.num_layers,
                        num_heads=self.model.num_heads
                    ).to(self.device)
                self.baseline_model.load_state_dict(self.model.state_dict())
                self.baseline_model.eval()
                print(f"  → Updated baseline model at epoch {epoch + 1}")
            
            if (epoch + 1) % save_interval == 0:
                torch.save(self.model.state_dict(), save_path)
        
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def train_model(num_epochs=100, batch_size=32, num_nodes=20,
                save_path="deep_learning/checkpoints/gnn_model/vrp_model.pth",
                hidden_dim=128, num_layers=3, num_heads=4):
    """Train the GNN Model."""
    trainer = GNNTrainer(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        lr=1e-4
    )
    trainer.train(num_epochs, batch_size, num_nodes, save_path)


if __name__ == "__main__":
    os.makedirs("deep_learning/checkpoints/gnn_model", exist_ok=True)
    train_model(
        num_epochs=500,
        batch_size=32,
        num_nodes=30,
        hidden_dim=128,
        num_layers=3,
        num_heads=4
    )

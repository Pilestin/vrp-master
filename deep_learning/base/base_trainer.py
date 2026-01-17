from abc import ABC, abstractmethod
import torch
import torch.optim as optim
import os


class BaseTrainer(ABC):
    """
    Abstract base class for all VRP model trainers.
    Provides common training loop structure with REINFORCE algorithm.
    """
    
    def __init__(self, model, device=None, lr=1e-3):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.moving_avg_reward = 0
    
    @abstractmethod
    def generate_data(self, batch_size: int, num_nodes: int) -> torch.Tensor:
        """
        Generate random VRP instances for training.
        
        Returns:
            data: (batch_size, num_nodes, 3) - Node features (x, y, demand)
        """
        pass
    
    @abstractmethod
    def calculate_reward(self, tour_indices: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the reward (negative tour length) for given tours.
        
        Returns:
            rewards: (batch_size,) - Reward for each instance
        """
        pass
    
    def train_epoch(self, batch_size: int, num_nodes: int, epoch: int) -> float:
        """
        Train for one epoch.
        
        Returns:
            avg_reward: Average reward for this epoch
        """
        self.model.train()
        inputs = self.generate_data(batch_size, num_nodes).to(self.device)
        
        tour_indices, tour_log_probs = self.model(inputs, deterministic=False)
        
        reward = self.calculate_reward(tour_indices, inputs)
        
        # REINFORCE with moving average baseline
        if epoch == 0:
            self.moving_avg_reward = reward.mean().item()
        else:
            self.moving_avg_reward = 0.9 * self.moving_avg_reward + 0.1 * reward.mean().item()
        
        advantage = reward - self.moving_avg_reward
        loss = (advantage * sum(tour_log_probs)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return reward.mean().item()
    
    def train(self, num_epochs: int, batch_size: int, num_nodes: int, 
              save_path: str, save_interval: int = 100):
        """
        Full training loop.
        """
        print(f"Training {self.model.get_model_name()} on {self.device}...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(num_epochs):
            avg_reward = self.train_epoch(batch_size, num_nodes, epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Tour Length = {avg_reward:.4f}")
            
            if (epoch + 1) % save_interval == 0:
                torch.save(self.model.state_dict(), save_path)
        
        # Final save
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

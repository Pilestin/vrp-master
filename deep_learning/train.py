import torch
import torch.optim as optim
import numpy as np
import os
from .model import AttentionModel
import math

def generate_data(batch_size, num_nodes):
    # Generate random VRP instances
    # Node 0 is depot (50, 50)
    # Others random (0-100)
    # Demand random (1-9)
    
    data = torch.zeros(batch_size, num_nodes, 3) # x, y, demand
    
    # Depot
    data[:, 0, 0] = 50.0 / 100.0 # Normalize
    data[:, 0, 1] = 50.0 / 100.0
    data[:, 0, 2] = 0.0
    
    # Customers
    data[:, 1:, 0] = torch.rand(batch_size, num_nodes - 1)
    data[:, 1:, 1] = torch.rand(batch_size, num_nodes - 1)
    data[:, 1:, 2] = torch.randint(1, 10, (batch_size, num_nodes - 1)).float() / 40.0 # Normalize by capacity
    
    return data

def calculate_tour_length(tour_indices, inputs):
    # inputs: (batch, nodes, 3)
    # tour_indices: (batch, num_customers) - indices of customers
    
    batch_size = inputs.size(0)
    
    # We need to reconstruct the full tour including depot returns based on capacity
    # But for this simplified "TSP-like" training, we will just minimize the TSP path length
    # and assume the splitter will handle the rest. 
    # This is a simplification. A real VRP model would handle capacity dynamically.
    # But for a "simple" implementation, learning a good TSP tour is 80% of the work.
    
    # Add depot at start
    depot_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=inputs.device)
    full_tour = torch.cat([depot_idx, tour_indices, depot_idx], dim=1)
    
    # Gather coordinates
    # inputs is (batch, nodes, 3)
    # full_tour is (batch, seq_len)
    
    coords = inputs[:, :, :2] # x, y
    
    # Gather
    # (batch, seq_len, 2)
    tour_coords = torch.gather(coords, 1, full_tour.unsqueeze(2).expand(-1, -1, 2))
    
    # Calculate distance
    # dist[i] = || tour[i] - tour[i+1] ||
    diff = tour_coords[:, 1:] - tour_coords[:, :-1]
    dist = torch.norm(diff, p=2, dim=2).sum(dim=1)
    
    return dist

def train_model(num_epochs=100, batch_size=32, num_nodes=20, save_path="deep_learning/checkpoints/vrp_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = AttentionModel(input_dim=3, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Baseline (Moving Average)
    moving_avg_reward = 0
    
    for epoch in range(num_epochs):
        model.train()
        inputs = generate_data(batch_size, num_nodes).to(device)
        
        tour_indices, tour_log_probs = model(inputs)
        
        # Calculate Reward (Negative Tour Length)
        tour_len = calculate_tour_length(tour_indices, inputs)
        reward = tour_len
        
        # REINFORCE with baseline
        if epoch == 0:
            moving_avg_reward = reward.mean().item()
        else:
            moving_avg_reward = 0.9 * moving_avg_reward + 0.1 * reward.mean().item()
            
        advantage = reward - moving_avg_reward
        
        loss = (advantage * sum(tour_log_probs)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Tour Length = {reward.mean().item():.4f}")
            
    # Save
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("deep_learning/checkpoints", exist_ok=True)
    train_model(num_epochs=500, batch_size=32, num_nodes=30)

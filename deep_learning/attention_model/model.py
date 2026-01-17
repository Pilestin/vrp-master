"""
Attention Model for Vehicle Routing Problem (CVRP)

Based on:
    Kool, W., van Hoof, H., & Welling, M. (2019).
    "Attention, Learn to Solve Routing Problems!"
    International Conference on Learning Representations (ICLR).

Key differences from simple Pointer Network:
    1. Multi-Head Self-Attention encoder (Transformer-style)
    2. Context embedding with current state
    3. Dynamic masking based on capacity constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.base.base_model import BaseVRPModel


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention layer."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, ff_dim: int = 512):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class AttentionModel(BaseVRPModel):
    """
    Attention Model for CVRP (Kool et al. 2019).
    
    Architecture:
        - Encoder: Linear embedding + N Transformer layers
        - Decoder: Context embedding + Multi-Head Attention for node selection
        
    Key Features:
        - Multi-head self-attention captures complex node relationships
        - Dynamic context includes current node and remaining capacity
        - Scaled dot-product attention for final node selection
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, 
                 num_heads: int = 8, num_layers: int = 3, ff_dim: int = 512):
        super().__init__(input_dim, hidden_dim)
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        
        # Input embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Decoder components
        # Context: [graph_embedding, current_node_embedding, remaining_capacity]
        self.context_dim = hidden_dim * 2 + 1
        self.context_projection = nn.Linear(self.context_dim, hidden_dim)
        
        # Attention for node selection
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        
        # Clipping coefficient for logits (helps exploration)
        self.C = 10.0
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode the input nodes using Transformer layers.
        
        Args:
            inputs: (batch, num_nodes, input_dim)
            
        Returns:
            node_embeddings: (batch, num_nodes, hidden_dim)
        """
        # Initial embedding
        x = self.node_embedding(inputs)
        
        # Apply Transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def get_context(self, embeddings: torch.Tensor, current_node: torch.Tensor,
                    remaining_capacity: torch.Tensor) -> torch.Tensor:
        """
        Create context vector for decoding.
        
        Args:
            embeddings: (batch, num_nodes, hidden_dim)
            current_node: (batch,) - Index of current node
            remaining_capacity: (batch,) - Remaining vehicle capacity
            
        Returns:
            context: (batch, hidden_dim)
        """
        batch_size = embeddings.size(0)
        
        # Graph embedding (mean of all node embeddings)
        graph_embed = embeddings.mean(dim=1)  # (batch, hidden)
        
        # Current node embedding
        current_embed = torch.gather(
            embeddings, 1,
            current_node.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
        ).squeeze(1)  # (batch, hidden)
        
        # Concatenate context
        context = torch.cat([
            graph_embed,
            current_embed,
            remaining_capacity.unsqueeze(1)
        ], dim=1)  # (batch, hidden*2 + 1)
        
        return self.context_projection(context)
    
    def forward(self, inputs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass with dynamic masking based on capacity.
        
        Args:
            inputs: (batch, num_nodes, 3) - (x, y, demand)
            deterministic: If True, use greedy selection
            
        Returns:
            tour_indices: (batch, num_customers)
            tour_log_probs: List of log probabilities
        """
        batch_size, num_nodes, _ = inputs.size()
        
        # Encode all nodes
        embeddings = self.encode(inputs)  # (batch, num_nodes, hidden)
        
        # Initialize state
        current_node = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)  # Start at depot
        remaining_capacity = torch.ones(batch_size, device=inputs.device)  # Normalized capacity
        
        # Visited mask
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=inputs.device)
        visited[:, 0] = True  # Mark depot as visited initially
        
        # Demand tensor for easy access
        demands = inputs[:, :, 2]  # (batch, num_nodes) - normalized demands
        
        tour_indices = []
        tour_log_probs = []
        
        # Visit all customers
        for step in range(num_nodes - 1):
            # Get context
            context = self.get_context(embeddings, current_node, remaining_capacity)
            
            # Compute attention scores
            query = self.W_q(context).unsqueeze(1)  # (batch, 1, hidden)
            keys = self.W_k(embeddings)  # (batch, num_nodes, hidden)
            
            # Scaled dot-product attention
            scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / math.sqrt(self.hidden_dim)
            # (batch, num_nodes)
            
            # Apply clipping for exploration
            scores = self.C * torch.tanh(scores)
            
            # Mask visited nodes
            scores = scores.masked_fill(visited, float('-inf'))
            
            # Optional: Mask nodes that exceed remaining capacity
            capacity_mask = demands > remaining_capacity.unsqueeze(1)
            
            # Only apply capacity mask if there are still feasible nodes
            feasible = ~(visited | capacity_mask)
            has_feasible = feasible.any(dim=1)
            
            # If no feasible nodes (all demand > capacity), we need to return to depot
            # For simplicity, we just select the minimum demand node or continue
            combined_mask = visited.clone()
            combined_mask[has_feasible] = combined_mask[has_feasible] | capacity_mask[has_feasible]
            
            scores = scores.masked_fill(combined_mask, float('-inf'))
            
            # Handle case where all nodes are masked (shouldn't happen with proper setup)
            all_masked = (scores == float('-inf')).all(dim=1)
            if all_masked.any():
                # Reset capacity for those instances (simulate depot return)
                remaining_capacity[all_masked] = 1.0
                scores[all_masked] = scores[all_masked].masked_fill(visited[all_masked], float('-inf'))
            
            # Softmax
            probs = F.softmax(scores, dim=-1)
            
            # Select next node
            if deterministic:
                selected = probs.argmax(dim=-1)
            else:
                m = torch.distributions.Categorical(probs)
                selected = m.sample()
                tour_log_probs.append(m.log_prob(selected))
            
            tour_indices.append(selected)
            
            # Update state
            visited = visited.clone()
            visited.scatter_(1, selected.unsqueeze(1), True)
            
            current_node = selected
            selected_demand = torch.gather(demands, 1, selected.unsqueeze(1)).squeeze(1)
            remaining_capacity = remaining_capacity - selected_demand
            
            # Reset capacity if it goes negative (shouldn't happen)
            remaining_capacity = torch.clamp(remaining_capacity, min=0.0)
        
        return torch.stack(tour_indices, dim=1), tour_log_probs
    
    def get_config(self) -> dict:
        return {
            'model_type': 'attention_model',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
        }

"""
Graph Neural Network Model for Vehicle Routing Problem (CVRP)

This implementation uses a Graph Attention Network (GAT) style encoder
followed by an attention-based decoder for node selection.

Key Features:
    - Message passing between nodes based on graph structure
    - Edge features (distances) incorporated into attention
    - Multiple GNN layers for multi-hop information aggregation

References:
    - Joshi et al. (2019) "An Efficient Graph Convolutional Network for TSP"
    - Veličković et al. (2018) "Graph Attention Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.base.base_model import BaseVRPModel


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT-style).
    
    Computes attention-weighted message passing between connected nodes.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # Linear transformations for query, key, value
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        
        # Edge feature projection (for distance information)
        self.edge_proj = nn.Linear(1, num_heads)
        
        # Output projection
        self.W_o = nn.Linear(out_dim, out_dim)
        
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, node_features: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (batch, num_nodes, in_dim)
            edge_weights: (batch, num_nodes, num_nodes) - Distance matrix
            
        Returns:
            updated_features: (batch, num_nodes, out_dim)
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Compute Q, K, V
        Q = self.W_q(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.W_k(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, nodes, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (batch, heads, nodes, nodes)
        
        # Add edge bias (distance information)
        edge_bias = self.edge_proj(edge_weights.unsqueeze(-1))  # (batch, nodes, nodes, heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (batch, heads, nodes, nodes)
        scores = scores + edge_bias
        
        # Softmax attention
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch, heads, nodes, head_dim)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        out = self.W_o(out)
        
        # Residual + LayerNorm
        out = self.norm(node_features + out) if node_features.size(-1) == out.size(-1) else self.norm(out)
        
        return out


class GNNEncoder(nn.Module):
    """
    GNN Encoder with multiple Graph Attention Layers.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        
        # Initial embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, node_features: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (batch, num_nodes, input_dim)
            edge_weights: (batch, num_nodes, num_nodes)
            
        Returns:
            node_embeddings: (batch, num_nodes, hidden_dim)
        """
        x = self.input_proj(node_features)
        
        for gnn_layer, ff_layer in zip(self.layers, self.ff_layers):
            x = gnn_layer(x, edge_weights)
            x = x + ff_layer(x)  # Residual
        
        return x


class GNNModel(BaseVRPModel):
    """
    Graph Neural Network Model for CVRP.
    
    Architecture:
        - GNN Encoder: Multi-layer Graph Attention Network
        - Decoder: Context-based attention for sequential node selection
        
    Key difference from Attention Model:
        - Explicitly uses graph structure (edge weights/distances)
        - Message passing aggregates neighbor information
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128,
                 num_layers: int = 3, num_heads: int = 4):
        super().__init__(input_dim, hidden_dim)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # GNN Encoder
        self.encoder = GNNEncoder(input_dim, hidden_dim, num_layers, num_heads)
        
        # Decoder components
        self.context_dim = hidden_dim * 2 + 1  # graph_embed + current_node + capacity
        self.context_proj = nn.Linear(self.context_dim, hidden_dim)
        
        # Attention for node selection
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        
        # Clipping for exploration
        self.C = 10.0
    
    def compute_edge_weights(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all nodes.
        
        Args:
            inputs: (batch, num_nodes, 3) - (x, y, demand)
            
        Returns:
            distances: (batch, num_nodes, num_nodes) - Normalized distances
        """
        coords = inputs[:, :, :2]  # (batch, nodes, 2)
        
        # Pairwise distance: ||a - b||
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, nodes, nodes, 2)
        distances = torch.norm(diff, p=2, dim=-1)  # (batch, nodes, nodes)
        
        # Normalize distances
        max_dist = distances.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        distances = distances / (max_dist + 1e-8)
        
        return distances
    
    def forward(self, inputs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass with GNN encoding.
        
        Args:
            inputs: (batch, num_nodes, 3) - (x, y, demand)
            deterministic: If True, use greedy selection
            
        Returns:
            tour_indices: (batch, num_customers)
            tour_log_probs: List of log probabilities
        """
        batch_size, num_nodes, _ = inputs.size()
        
        # Compute edge weights (distance matrix)
        edge_weights = self.compute_edge_weights(inputs)
        
        # GNN Encoding
        embeddings = self.encoder(inputs, edge_weights)  # (batch, nodes, hidden)
        
        # Initialize decoder state
        current_node = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)
        remaining_capacity = torch.ones(batch_size, device=inputs.device)
        
        # Visited mask
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=inputs.device)
        visited[:, 0] = True  # Mask depot
        
        demands = inputs[:, :, 2]
        
        tour_indices = []
        tour_log_probs = []
        
        # Decode sequence
        for _ in range(num_nodes - 1):
            # Context: graph embedding + current node + capacity
            graph_embed = embeddings.mean(dim=1)
            current_embed = torch.gather(
                embeddings, 1,
                current_node.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            ).squeeze(1)
            
            context = torch.cat([
                graph_embed,
                current_embed,
                remaining_capacity.unsqueeze(1)
            ], dim=1)
            context = self.context_proj(context)
            
            # Attention scores
            query = self.W_q(context).unsqueeze(1)
            keys = self.W_k(embeddings)
            scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / math.sqrt(self.hidden_dim)
            
            # Clip for exploration
            scores = self.C * torch.tanh(scores)
            
            # Mask visited and infeasible nodes
            scores = scores.masked_fill(visited, float('-inf'))
            
            # Capacity mask
            capacity_mask = demands > remaining_capacity.unsqueeze(1)
            feasible = ~(visited | capacity_mask)
            has_feasible = feasible.any(dim=1)
            
            combined_mask = visited.clone()
            combined_mask[has_feasible] = combined_mask[has_feasible] | capacity_mask[has_feasible]
            scores = scores.masked_fill(combined_mask, float('-inf'))
            
            # Handle all masked
            all_masked = (scores == float('-inf')).all(dim=1)
            if all_masked.any():
                remaining_capacity[all_masked] = 1.0
                scores[all_masked] = scores[all_masked].masked_fill(visited[all_masked], float('-inf'))
            
            # Softmax
            probs = F.softmax(scores, dim=-1)
            
            # Select
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
            remaining_capacity = torch.clamp(remaining_capacity - selected_demand, min=0.0)
        
        return torch.stack(tour_indices, dim=1), tour_log_probs
    
    def get_config(self) -> dict:
        return {
            'model_type': 'gnn_model',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
        }

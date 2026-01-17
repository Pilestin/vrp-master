import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.base.base_model import BaseVRPModel


class PointerNetworkModel(BaseVRPModel):
    """
    A simplified Attention Model for VRP based on Pointer Networks.
    
    Architecture:
        - Encoder: Linear embedding layer
        - Decoder: LSTM cell with additive (Bahdanau) attention
        
    This model learns a TSP-like tour over customers. The tour is then
    split based on vehicle capacity during inference.
    
    Reference:
        - Vinyals et al. (2015) "Pointer Networks"
        - Bello et al. (2016) "Neural Combinatorial Optimization with RL"
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128):
        super().__init__(input_dim, hidden_dim)
        
        # Encoder
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Decoder
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Attention (Bahdanau-style additive attention)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
        # First input to decoder (learnable)
        self.first_input = nn.Parameter(torch.randn(hidden_dim))
    
    def forward(self, inputs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass of the Pointer Network.
        
        Args:
            inputs: (batch_size, num_nodes, input_dim) - Node features
            deterministic: If True, use greedy selection
            
        Returns:
            tour_indices: (batch_size, num_customers) - Customer visit order
            tour_log_probs: List of log probabilities (empty if deterministic)
        """
        batch_size, num_nodes, _ = inputs.size()
        
        # Encode all nodes
        encoder_outputs = self.embedding(inputs)  # (batch, nodes, hidden)
        
        # Initialize decoder state
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        
        # Initial decoder input
        decoder_input = self.first_input.unsqueeze(0).expand(batch_size, -1)
        
        # Mask to track visited nodes
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=inputs.device)
        mask[:, 0] = True  # Mask depot - we only visit customers
        
        tour_indices = []
        tour_log_probs = []
        
        # Decode sequence - visit all customers
        for _ in range(num_nodes - 1):
            hx, cx = self.decoder_cell(decoder_input, (hx, cx))
            
            # Compute attention scores
            query = self.W_q(hx).unsqueeze(1)  # (batch, 1, hidden)
            keys = self.W_k(encoder_outputs)   # (batch, nodes, hidden)
            scores = self.V(torch.tanh(query + keys)).squeeze(-1)  # (batch, nodes)
            
            # Apply mask
            scores = scores.masked_fill(mask, float('-inf'))
            
            # Softmax to get probabilities
            probs = F.softmax(scores, dim=-1)
            
            # Select next node
            if deterministic:
                selected = probs.argmax(dim=-1)
            else:
                m = torch.distributions.Categorical(probs)
                selected = m.sample()
                tour_log_probs.append(m.log_prob(selected))
            
            tour_indices.append(selected)
            
            # Update mask (avoid in-place for autograd)
            mask = mask.clone()
            mask.scatter_(1, selected.unsqueeze(1), True)
            
            # Prepare next decoder input
            decoder_input = torch.gather(
                encoder_outputs, 1,
                selected.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            ).squeeze(1)
        
        return torch.stack(tour_indices, dim=1), tour_log_probs
    
    def get_config(self) -> dict:
        return {
            'model_type': 'pointer_network',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
        }


# Backwards compatibility alias
AttentionModel = PointerNetworkModel

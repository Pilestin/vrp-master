import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    """
    A simplified Attention Model for VRP.
    Encoder: Embeds inputs.
    Decoder: LSTM + Attention to select next node.
    """
    def __init__(self, input_dim=3, hidden_dim=128):
        super(AttentionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Decoder
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Attention
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
        # First input to decoder (learnable)
        self.first_input = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, inputs, deterministic=False):
        """
        inputs: (batch_size, num_nodes, input_dim)
        Returns: tour_indices, tour_log_probs
        """
        batch_size, num_nodes, _ = inputs.size()
        
        # Encoder
        encoder_outputs = self.embedding(inputs) # (batch, nodes, hidden)
        
        # Decoder State
        hx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        
        # Initial input
        decoder_input = self.first_input.unsqueeze(0).expand(batch_size, -1)
        
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=inputs.device)
        
        tour_indices = []
        tour_log_probs = []
        
        # Start at depot (node 0) - usually we force start at 0, but here let's let it pick 0 first or handle it.
        # For VRP, we usually start at 0. Let's assume we force start at 0.
        # But for simplicity of this generic pointer net, let's let it pick.
        # Actually, standard VRP starts at 0. Let's force 0 as visited and current node.
        
        # Simplified: We just predict a sequence of nodes. 
        # Handling capacity in the model is hard. 
        # Strategy: The model predicts a permutation (TSP-like). 
        # Then we split it based on capacity (Split delivery / greedy split).
        # This is a common strategy for VRP (e.g. "Learning to Solve TSP" applied to VRP via splitting).
        
        # Force start at node 0
        prev_node_index = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)
        mask[:, 0] = True # Mask depot initially so we don't pick it immediately again (unless we want multiple trips)
        # Wait, for VRP we need to return to depot. 
        # Let's stick to: Model predicts a TSP tour over customers. We split it later.
        # So we mask node 0 (depot) and never pick it.
        
        for _ in range(num_nodes - 1): # Pick all customers
            hx, cx = self.decoder_cell(decoder_input, (hx, cx))
            
            # Attention mechanism
            # query = hx
            # keys = encoder_outputs
            
            query = self.W_q(hx).unsqueeze(1) # (batch, 1, hidden)
            keys = self.W_k(encoder_outputs)  # (batch, nodes, hidden)
            
            # Tanh exploration
            scores = self.V(torch.tanh(query + keys)).squeeze(-1) # (batch, nodes)
            
            # Masking
            scores = scores.masked_fill(mask, -float('inf'))
            
            # Softmax
            probs = F.softmax(scores, dim=-1)
            
            if deterministic:
                selected = probs.argmax(dim=-1)
            else:
                m = torch.distributions.Categorical(probs)
                selected = m.sample()
                tour_log_probs.append(m.log_prob(selected))
            
            tour_indices.append(selected)
            
            # Update mask
            mask.scatter_(1, selected.unsqueeze(1), True)
            
            # Next input
            # Gather embedding of selected node
            decoder_input = torch.gather(encoder_outputs, 1, selected.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)).squeeze(1)
            
        return torch.stack(tour_indices, dim=1), tour_log_probs

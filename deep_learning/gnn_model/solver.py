"""
Solver using the GNN Model.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrp_system.solvers.base_solver import BaseSolver
from .model import GNNModel


class GNNSolver(BaseSolver):
    """
    VRP Solver using the Graph Neural Network model.
    """
    
    def __init__(self, graph, capacity=40,
                 model_path="deep_learning/checkpoints/gnn_model/vrp_model.pth"):
        super().__init__(graph)
        self.capacity = capacity
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.model = None
    
    def load_model(self) -> bool:
        """Load the trained model from checkpoint."""
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}. Please run training first.")
            return False
        
        try:
            self.model = GNNModel(
                input_dim=3,
                hidden_dim=128,
                num_layers=3,
                num_heads=4
            ).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded GNN Model from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def solve(self):
        """Solve VRP using the GNN model."""
        if self.model is None:
            if not self.load_model():
                yield []
                return
        
        # Prepare input data
        nodes = list(self.graph.nodes(data=True))
        nodes.sort(key=lambda x: x[0])
        
        num_nodes = len(nodes)
        input_data = torch.zeros(1, num_nodes, 3)
        
        for i, (node_id, data) in enumerate(nodes):
            input_data[0, i, 0] = data['pos'][0] / 100.0
            input_data[0, i, 1] = data['pos'][1] / 100.0
            input_data[0, i, 2] = data['demand'] / 40.0
        
        input_data = input_data.to(self.device)
        
        # Get tour from model
        with torch.no_grad():
            tour_indices, _ = self.model(input_data, deterministic=True)
        
        tsp_tour = tour_indices[0].tolist()
        
        # Split tour into routes based on capacity
        routes = []
        current_route = [0]
        current_load = 0
        
        yield [current_route]
        
        for node_idx in tsp_tour:
            node_id = nodes[node_idx][0]
            demand = self.graph.nodes[node_id]['demand']
            
            if current_load + demand <= self.capacity:
                current_route.append(node_id)
                current_load += demand
                yield routes + [current_route]
            else:
                current_route.append(0)
                routes.append(current_route)
                yield routes + [current_route]
                
                current_route = [0, node_id]
                current_load = demand
                yield routes + [current_route]
        
        current_route.append(0)
        routes.append(current_route)
        
        yield routes

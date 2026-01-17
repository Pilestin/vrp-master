import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrp_system.solvers.base_solver import BaseSolver
from .model import PointerNetworkModel


class PointerNetworkSolver(BaseSolver):
    """
    VRP Solver using the Pointer Network model.
    
    The model predicts a TSP-like tour which is then split into
    feasible routes based on vehicle capacity.
    """
    
    def __init__(self, graph, capacity=40, 
                 model_path="deep_learning/checkpoints/pointer_network/vrp_model.pth"):
        super().__init__(graph)
        self.capacity = capacity
        self.model_path = model_path
        self.device = torch.device("cpu")  # CPU for inference
        self.model = None
    
    def load_model(self) -> bool:
        """Load the trained model from checkpoint."""
        # Try new path first, then fall back to legacy path
        paths_to_try = [
            self.model_path,
            "deep_learning/checkpoints/vrp_model.pth"  # Legacy path
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    self.model = PointerNetworkModel(input_dim=3, hidden_dim=128).to(self.device)
                    self.model.load_state_dict(torch.load(path, map_location=self.device))
                    self.model.eval()
                    print(f"Loaded model from {path}")
                    return True
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
        
        print(f"Model not found. Please run training first.")
        return False
    
    def solve(self):
        """
        Solve VRP using the neural network model.
        
        Yields intermediate solutions for visualization.
        """
        if self.model is None:
            if not self.load_model():
                yield []
                return
        
        # Prepare input data
        nodes = list(self.graph.nodes(data=True))
        nodes.sort(key=lambda x: x[0])
        
        num_nodes = len(nodes)
        input_data = torch.zeros(1, num_nodes, 3)
        
        # Normalize inputs
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

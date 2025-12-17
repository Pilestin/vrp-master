from vrp_system.solvers.base_solver import BaseSolver
import torch
import numpy as np
import os
from .model import AttentionModel
import networkx as nx

class NeuralSolver(BaseSolver):
    def __init__(self, graph, capacity=40, model_path="deep_learning/checkpoints/vrp_model.pth"):
        super().__init__(graph)
        self.capacity = capacity
        self.model_path = model_path
        self.device = torch.device("cpu") # Inference on CPU is fine for small instances
        self.model = None
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}. Please run deep_learning/train.py first.")
            return False
            
        try:
            self.model = AttentionModel(input_dim=3, hidden_dim=128).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def solve(self):
        if self.model is None:
            if not self.load_model():
                yield [] # Fail gracefully
                return

        # Prepare Input
        nodes = list(self.graph.nodes(data=True))
        # Sort by ID to ensure order matches
        nodes.sort(key=lambda x: x[0])
        
        num_nodes = len(nodes)
        input_data = torch.zeros(1, num_nodes, 3)
        
        # Normalize inputs as in training
        for i, (node_id, data) in enumerate(nodes):
            input_data[0, i, 0] = data['pos'][0] / 100.0
            input_data[0, i, 1] = data['pos'][1] / 100.0
            input_data[0, i, 2] = data['demand'] / 40.0 # Assuming 40 is max capacity used in training normalization
            
        input_data = input_data.to(self.device)
        
        # Inference
        with torch.no_grad():
            tour_indices, _ = self.model(input_data, deterministic=True)
            
        # tour_indices contains indices of customers (1..N-1) in visited order
        # We need to split this giant tour into feasible routes based on capacity
        
        tsp_tour = tour_indices[0].tolist() # List of node indices
        
        # Greedy Split
        routes = []
        current_route = [0]
        current_load = 0
        
        # Yield initial
        yield [current_route]
        
        for node_idx in tsp_tour:
            # Map index back to node ID (assuming 0..N-1 mapping holds)
            node_id = nodes[node_idx][0]
            demand = self.graph.nodes[node_id]['demand']
            
            if current_load + demand <= self.capacity:
                current_route.append(node_id)
                current_load += demand
            else:
                current_route.append(0)
                routes.append(current_route)
                yield routes + [current_route] # Visualization update
                
                current_route = [0, node_id]
                current_load = demand
                
        current_route.append(0)
        routes.append(current_route)
        
        yield routes

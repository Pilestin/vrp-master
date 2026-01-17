"""
Solver Factory for Deep Learning VRP Models.

Provides a unified interface to create solvers for different neural network models.
"""

from typing import Literal
import networkx as nx


ModelType = Literal["pointer_network", "attention_model"]


def get_neural_solver(model_type: ModelType, graph: nx.Graph, capacity: int = 40):
    """
    Factory function to create a neural VRP solver.
    
    Args:
        model_type: Type of model to use ("pointer_network" or "attention_model")
        graph: NetworkX graph representing the VRP instance
        capacity: Vehicle capacity
        
    Returns:
        A solver instance (PointerNetworkSolver or AttentionModelSolver)
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "pointer_network":
        from .pointer_network.solver import PointerNetworkSolver
        return PointerNetworkSolver(graph, capacity)
    elif model_type == "attention_model":
        from .attention_model.solver import AttentionModelSolver
        return AttentionModelSolver(graph, capacity)
    elif model_type == "gnn_model":
        from .gnn_model.solver import GNNSolver
        return GNNSolver(graph, capacity)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: pointer_network, attention_model, gnn_model")


def get_available_models() -> list:
    """Return list of available model types."""
    return ["pointer_network", "attention_model", "gnn_model"]


def get_model_info(model_type: ModelType) -> dict:
    """
    Get information about a specific model.
    
    Returns:
        Dictionary with model name, description, and paper reference
    """
    info = {
        "pointer_network": {
            "name": "Pointer Network",
            "description": "LSTM-based sequence-to-sequence model with additive attention",
            "paper": "Vinyals et al. (2015) 'Pointer Networks' + Bello et al. (2016)",
            "complexity": "Low",
            "training_time": "Fast",
        },
        "attention_model": {
            "name": "Attention Model (Kool 2019)",
            "description": "Transformer encoder with multi-head self-attention",
            "paper": "Kool et al. (2019) 'Attention, Learn to Solve Routing Problems!'",
            "complexity": "Medium",
            "training_time": "Medium",
        },
        "gnn_model": {
            "name": "Graph Neural Network",
            "description": "Graph Attention Network with edge-based message passing",
            "paper": "Joshi et al. (2019) + Veličković et al. (2018) 'Graph Attention Networks'",
            "complexity": "High",
            "training_time": "Slow",
        }
    }
    return info.get(model_type, {})

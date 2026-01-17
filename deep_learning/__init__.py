"""
Deep Learning Module for VRP Solvers.

This module provides neural network-based solvers for the Vehicle Routing Problem.

Available Models:
    - pointer_network: Simple LSTM + Attention model (Vinyals 2015, Bello 2016)
    - attention_model: Transformer-based model (Kool et al. 2019)

Usage:
    # Using the factory
    from deep_learning import get_neural_solver
    solver = get_neural_solver("attention_model", graph, capacity=40)
    
    # Direct import
    from deep_learning.pointer_network import PointerNetworkSolver
    from deep_learning.attention_model import AttentionModelSolver
"""

# Solver factory (recommended API)
from .solver_factory import get_neural_solver, get_available_models, get_model_info

# Legacy imports for backwards compatibility
from .pointer_network.model import PointerNetworkModel
from .pointer_network.model import AttentionModel  # Alias
from .pointer_network.solver import PointerNetworkSolver

# New model imports
from .attention_model.model import AttentionModel as KoolAttentionModel
from .attention_model.solver import AttentionModelSolver

# Legacy neural_solver import (points to pointer_network)
from .neural_solver import NeuralSolver

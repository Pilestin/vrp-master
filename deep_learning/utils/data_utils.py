import torch


def generate_vrp_data(batch_size: int, num_nodes: int, 
                      depot_pos: tuple = (50.0, 50.0),
                      demand_range: tuple = (1, 9),
                      capacity: int = 40) -> torch.Tensor:
    """
    Generate random VRP instances.
    
    Args:
        batch_size: Number of instances to generate
        num_nodes: Total number of nodes (including depot at index 0)
        depot_pos: (x, y) position of depot (before normalization)
        demand_range: (min, max) demand for customers
        capacity: Vehicle capacity (used for demand normalization)
        
    Returns:
        data: (batch_size, num_nodes, 3) - Normalized node features (x, y, demand)
    """
    data = torch.zeros(batch_size, num_nodes, 3)
    
    # Depot at index 0
    data[:, 0, 0] = depot_pos[0] / 100.0  # Normalize x
    data[:, 0, 1] = depot_pos[1] / 100.0  # Normalize y
    data[:, 0, 2] = 0.0  # Depot has no demand
    
    # Customers at indices 1 to num_nodes-1
    data[:, 1:, 0] = torch.rand(batch_size, num_nodes - 1)  # x coordinates
    data[:, 1:, 1] = torch.rand(batch_size, num_nodes - 1)  # y coordinates
    data[:, 1:, 2] = torch.randint(
        demand_range[0], demand_range[1] + 1, 
        (batch_size, num_nodes - 1)
    ).float() / capacity  # Normalized demand
    
    return data


def calculate_tour_length(tour_indices: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Calculate the total tour length for given tours.
    
    This is a simplified TSP-like calculation that assumes depot returns
    are handled separately (by splitting the tour based on capacity).
    
    Args:
        tour_indices: (batch_size, num_customers) - Indices of customers in visit order
        inputs: (batch_size, num_nodes, 3) - Node features (x, y, demand)
        
    Returns:
        distances: (batch_size,) - Total tour length for each instance
    """
    batch_size = inputs.size(0)
    
    # Add depot at start and end
    depot_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=inputs.device)
    full_tour = torch.cat([depot_idx, tour_indices, depot_idx], dim=1)
    
    # Extract coordinates only (x, y)
    coords = inputs[:, :, :2]
    
    # Gather coordinates for the tour
    tour_coords = torch.gather(
        coords, 1, 
        full_tour.unsqueeze(2).expand(-1, -1, 2)
    )
    
    # Calculate distances between consecutive nodes
    diff = tour_coords[:, 1:] - tour_coords[:, :-1]
    dist = torch.norm(diff, p=2, dim=2).sum(dim=1)
    
    return dist

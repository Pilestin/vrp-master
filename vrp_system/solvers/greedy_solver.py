from .base_solver import BaseSolver
import networkx as nx

class GreedySolver(BaseSolver):
    def __init__(self, graph, capacity=40):
        super().__init__(graph)
        self.capacity = capacity

    def solve(self):
        """
        Implements a Nearest Neighbor heuristic for CVRP.
        Yields the current set of routes at each step.
        """
        unvisited = set(self.graph.nodes())
        unvisited.remove(0) # Remove depot
        
        routes = []
        current_route = [0]
        current_load = 0
        current_node = 0
        
        # Yield initial state
        yield routes + [current_route]

        while unvisited:
            # Find nearest unvisited neighbor that fits capacity
            nearest_node = None
            min_dist = float('inf')
            
            for node in unvisited:
                demand = self.graph.nodes[node]['demand']
                if current_load + demand <= self.capacity:
                    dist = self.graph[current_node][node]['weight']
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node
            
            if nearest_node is not None:
                # Move to node
                current_node = nearest_node
                current_load += self.graph.nodes[current_node]['demand']
                unvisited.remove(current_node)
                current_route.append(current_node)
                yield routes + [current_route]
            else:
                # Return to depot and start new route
                current_route.append(0)
                routes.append(current_route)
                yield routes # Yield completed route
                
                # Start new route
                current_node = 0
                current_load = 0
                current_route = [0]
                yield routes + [current_route]
        
        # Finish last route
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        
        yield routes

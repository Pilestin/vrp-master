from .base_solver import BaseSolver
import math
import random
import copy

class SimulatedAnnealingSolver(BaseSolver):
    def __init__(self, graph, capacity=40, initial_temp=1000, cooling_rate=0.995, max_iterations=2000):
        super().__init__(graph)
        self.capacity = capacity
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

    def solve(self):
        # 1. Generate Initial Solution (Random valid)
        current_routes = self._generate_initial_solution()
        current_cost = self._calculate_cost(current_routes)
        
        best_routes = copy.deepcopy(current_routes)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        yield current_routes

        for i in range(self.max_iterations):
            # 2. Create Neighbor
            neighbor_routes = self._get_neighbor(current_routes)
            neighbor_cost = self._calculate_cost(neighbor_routes)
            
            # 3. Acceptance Probability
            delta = neighbor_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_routes = neighbor_routes
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_routes = copy.deepcopy(current_routes)
                    best_cost = current_cost
            
            temp *= self.cooling_rate
            
            # Yield periodically to visualize progress
            if i % 50 == 0:
                yield current_routes
        
        yield best_routes

    def _generate_initial_solution(self):
        # Simple greedy-like construction just to get a valid start
        nodes = list(self.graph.nodes())
        nodes.remove(0)
        random.shuffle(nodes)
        
        routes = []
        current_route = [0]
        current_load = 0
        
        for node in nodes:
            demand = self.graph.nodes[node]['demand']
            if current_load + demand <= self.capacity:
                current_route.append(node)
                current_load += demand
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, node]
                current_load = demand
        
        current_route.append(0)
        routes.append(current_route)
        return routes

    def _get_neighbor(self, routes):
        # Deep copy to avoid modifying current state
        new_routes = copy.deepcopy(routes)
        
        # Flatten routes to list of customers
        customers = []
        for route in new_routes:
            customers.extend(route[1:-1]) # Exclude depots
            
        if not customers:
            return new_routes

        # Operator: Swap two random customers
        if len(customers) > 1:
            idx1, idx2 = random.sample(range(len(customers)), 2)
            customers[idx1], customers[idx2] = customers[idx2], customers[idx1]
            
        # Reconstruct routes with capacity constraint
        reconstructed_routes = []
        current_route = [0]
        current_load = 0
        
        for node in customers:
            demand = self.graph.nodes[node]['demand']
            if current_load + demand <= self.capacity:
                current_route.append(node)
                current_load += demand
            else:
                current_route.append(0)
                reconstructed_routes.append(current_route)
                current_route = [0, node]
                current_load = demand
        
        current_route.append(0)
        reconstructed_routes.append(current_route)
        
        return reconstructed_routes

    def _calculate_cost(self, routes):
        total_dist = 0
        for route in routes:
            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                total_dist += self.graph[u][v]['weight']
        return total_dist

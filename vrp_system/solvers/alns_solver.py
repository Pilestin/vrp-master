from .base_solver import BaseSolver
from .greedy_solver import GreedySolver
import random
import copy
import math

class ALNSSolver(BaseSolver):
    def __init__(self, graph, capacity=40, iterations=1000, remove_count=4):
        super().__init__(graph)
        self.capacity = capacity
        self.iterations = iterations
        self.remove_count = remove_count
        self.best_solution = None
        self.best_cost = float('inf')

    def solve(self):
        # 1. Generate Initial Solution using Greedy
        greedy = GreedySolver(self.graph, self.capacity)
        initial_routes = []
        for routes in greedy.solve():
            initial_routes = routes
        
        current_routes = initial_routes
        current_cost = self._calculate_cost(current_routes)
        
        self.best_routes = copy.deepcopy(current_routes)
        self.best_cost = current_cost
        
        yield current_routes

        # ALNS Loop
        for i in range(self.iterations):
            # Destroy
            destroyed_routes, removed_nodes = self._destroy(current_routes)
            
            # Repair
            new_routes = self._repair(destroyed_routes, removed_nodes)
            new_cost = self._calculate_cost(new_routes)
            
            # Acceptance (Simple Hill Climbing + slight randomness/SA-like can be added)
            # Here we accept if better, or with small probability if worse (Simulated Annealing-like acceptance)
            temperature = 100 * (0.995 ** i)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / (temperature + 1e-6)):
                current_routes = new_routes
                current_cost = new_cost
                
                if current_cost < self.best_cost:
                    self.best_routes = copy.deepcopy(current_routes)
                    self.best_cost = current_cost
            
            if i % 10 == 0:
                yield current_routes
        
        yield self.best_routes

    def _destroy(self, routes):
        """Randomly remove 'remove_count' nodes from routes."""
        current_routes = copy.deepcopy(routes)
        removed_nodes = []
        
        # Flatten all customers
        all_customers = []
        for r_idx, route in enumerate(current_routes):
            for n_idx, node in enumerate(route):
                if node != 0:
                    all_customers.append((r_idx, n_idx, node))
        
        if not all_customers:
            return current_routes, []

        # Select nodes to remove
        num_to_remove = min(self.remove_count, len(all_customers))
        targets = random.sample(all_customers, num_to_remove)
        
        # Sort targets by route index and node index descending to remove safely
        targets.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        for r_idx, n_idx, node in targets:
            current_routes[r_idx].pop(n_idx)
            removed_nodes.append(node)
            
        # Clean up empty routes (routes with only depot 0, 0 or just 0)
        current_routes = [r for r in current_routes if len(r) > 2] # >2 because [0, 0] is empty route
        
        return current_routes, removed_nodes

    def _repair(self, routes, removed_nodes):
        """Greedy insertion of removed nodes."""
        current_routes = copy.deepcopy(routes)
        
        # Shuffle removed nodes to avoid bias
        random.shuffle(removed_nodes)
        
        for node in removed_nodes:
            best_cost_increase = float('inf')
            best_position = None # (route_idx, insert_idx)
            
            demand = self.graph.nodes[node]['demand']
            
            # Try to insert in existing routes
            for r_idx, route in enumerate(current_routes):
                # Check capacity
                current_load = sum(self.graph.nodes[n]['demand'] for n in route)
                if current_load + demand <= self.capacity:
                    # Try all positions
                    for i in range(1, len(route)): # Insert before index i (1 to len-1)
                        # Cost increase = (u->node + node->v) - (u->v)
                        u = route[i-1]
                        v = route[i]
                        increase = (self.graph[u][node]['weight'] + 
                                    self.graph[node][v]['weight'] - 
                                    self.graph[u][v]['weight'])
                        
                        if increase < best_cost_increase:
                            best_cost_increase = increase
                            best_position = (r_idx, i)
            
            # Also consider creating a new route
            # Cost = 0->node + node->0
            new_route_cost = (self.graph[0][node]['weight'] + self.graph[node][0]['weight'])
            if new_route_cost < best_cost_increase:
                best_cost_increase = new_route_cost
                best_position = ("new", 0)
            
            # Apply insertion
            if best_position[0] == "new":
                current_routes.append([0, node, 0])
            else:
                r_idx, insert_idx = best_position
                current_routes[r_idx].insert(insert_idx, node)
                
        return current_routes

    def _calculate_cost(self, routes):
        total_dist = 0
        for route in routes:
            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                total_dist += self.graph[u][v]['weight']
        return total_dist

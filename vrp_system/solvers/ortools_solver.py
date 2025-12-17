from .base_solver import BaseSolver
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import networkx as nx

class ORToolsSolver(BaseSolver):
    def __init__(self, graph, vehicle_capacity=40, num_vehicles=5):
        super().__init__(graph)
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

    def solve(self):
        """
        Solves the VRP using OR-Tools.
        Yields the final solution (list of routes).
        """
        # Data transformation
        nodes = list(self.graph.nodes(data=True))
        # Map node ID to index (0 to N-1)
        node_map = {node_id: i for i, (node_id, _) in enumerate(nodes)}
        reverse_map = {i: node_id for node_id, i in node_map.items()}
        
        num_nodes = len(nodes)
        depot_index = node_map[0]

        manager = pywrapcp.RoutingIndexManager(num_nodes, self.num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)

        # Distance Callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Get original node IDs
            u = reverse_map[from_node]
            v = reverse_map[to_node]
            if u == v:
                return 0
            return int(self.graph[u][v]['weight'])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity Constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            u = reverse_map[from_node]
            return self.graph.nodes[u]['demand']

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.vehicle_capacity] * self.num_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity"
        )

        # Search Parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 5

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            routes = []
            for vehicle_id in range(self.num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route.append(reverse_map[node_index])
                    index = solution.Value(routing.NextVar(index))
                route.append(reverse_map[manager.IndexToNode(index)]) # End at depot
                
                if len(route) > 2: # Only add non-empty routes (depot -> ... -> depot)
                    routes.append(route)
            
            yield routes
        else:
            print("No solution found by OR-Tools")
            yield []

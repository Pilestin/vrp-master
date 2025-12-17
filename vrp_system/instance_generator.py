import networkx as nx
import numpy as np
import math

class InstanceGenerator:
    def __init__(self, num_nodes=20, seed=None):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)

    def generate_vrp_instance(self):
        """
        Generates a VRP instance as a NetworkX graph.
        Node 0 is the depot.
        Nodes have attributes: 'pos' (x, y), 'demand'.
        Edges have attribute: 'weight' (distance).
        """
        G = nx.DiGraph()

        # 1. Generate Nodes
        # Depot
        depot_pos = (50.0, 50.0)
        G.add_node(0, pos=depot_pos, demand=0, type="depot")

        # Customers
        for i in range(1, self.num_nodes):
            x = self.rng.uniform(0, 100)
            y = self.rng.uniform(0, 100)
            demand = self.rng.integers(1, 10)
            G.add_node(i, pos=(x, y), demand=demand, type="customer")

        # 2. Generate Edges (Complete Graph)
        nodes = list(G.nodes(data=True))
        for i, data_i in nodes:
            for j, data_j in nodes:
                if i == j:
                    continue
                dist = math.dist(data_i['pos'], data_j['pos'])
                G.add_edge(i, j, weight=dist)

        return G

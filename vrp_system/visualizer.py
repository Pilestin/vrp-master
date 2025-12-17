import matplotlib.pyplot as plt
import networkx as nx

class Visualizer:
    def __init__(self, graph):
        self.graph = graph
        self.pos = nx.get_node_attributes(graph, 'pos')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.ion() # Interactive mode

    def draw_base(self):
        self.ax.clear()
        # Draw nodes
        # Depot
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=[0], node_color='red', node_size=200, label='Depot', ax=self.ax)
        # Customers
        customers = [n for n in self.graph.nodes() if n != 0]
        nx.draw_networkx_nodes(self.graph, self.pos, nodelist=customers, node_color='blue', node_size=100, label='Customer', ax=self.ax)
        
        # Labels
        # nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax, font_size=8)
        
        self.ax.set_title("VRP Visualization")
        self.ax.legend()

    def update(self, routes, title="Solution"):
        self.draw_base()
        
        colors = ['g', 'c', 'm', 'y', 'k', 'orange', 'purple']
        
        total_dist = 0
        
        for i, route in enumerate(routes):
            if not route: continue
            
            # Edges for this route
            edges = [(route[j], route[j+1]) for j in range(len(route)-1)]
            
            color = colors[i % len(colors)]
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=edges, edge_color=color, width=2, ax=self.ax)
            
            # Calculate distance
            for u, v in edges:
                total_dist += self.graph[u][v]['weight']

        self.ax.set_title(f"{title} - Total Distance: {total_dist:.2f}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.show()

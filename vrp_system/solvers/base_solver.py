from abc import ABC, abstractmethod
import networkx as nx

class BaseSolver(ABC):
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @abstractmethod
    def solve(self):
        """
        Generator that yields the current solution state at each step.
        The solution state can be a list of routes or a partial route.
        """
        pass

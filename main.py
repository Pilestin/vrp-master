from vrp_system.instance_generator import InstanceGenerator
from vrp_system.solvers.greedy_solver import GreedySolver
from vrp_system.solvers.ortools_solver import ORToolsSolver
from vrp_system.visualizer import Visualizer
import time

def main():
    # 1. Generate Instance
    print("Generating Instance...")
    generator = InstanceGenerator(num_nodes=30, seed=42)
    graph = generator.generate_vrp_instance()

    # 2. Initialize Visualizer
    viz = Visualizer(graph)

    # 3. Solve with Greedy Solver (Nearest Neighbor)
    print("Running Greedy Solver (Nearest Neighbor)...")
    greedy_solver = GreedySolver(graph, capacity=40)
    
    for step, routes in enumerate(greedy_solver.solve()):
        viz.update(routes, title=f"Greedy Solver - Step {step}")
        # time.sleep(0.05) # Optional delay

    print("Greedy Solver Finished.")
    time.sleep(1)

    # 4. Solve with OR-Tools
    print("Running OR-Tools Solver...")
    ortools_solver = ORToolsSolver(graph, vehicle_capacity=40, num_vehicles=5)
    
    # OR-Tools usually yields just the final result in this implementation
    for routes in ortools_solver.solve():
        viz.update(routes, title="OR-Tools Solution")

    print("OR-Tools Solver Finished.")
    
    # Keep window open
    viz.close()

if __name__ == "__main__":
    main()

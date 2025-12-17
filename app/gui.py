import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
import networkx as nx

from vrp_system.instance_generator import InstanceGenerator
from vrp_system.solvers.greedy_solver import GreedySolver
from vrp_system.solvers.ortools_solver import ORToolsSolver
from vrp_system.solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from vrp_system.solvers.alns_solver import ALNSSolver
from vrp_system.visualizer import Visualizer
import vrp_config as config

class VRPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VRP Solver System")
        self.root.geometry("1000x700")
        
        # Layout
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.viz_frame = ttk.Frame(root, padding="10")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Controls
        ttk.Label(self.control_frame, text="VRP Solver Control", font=("Helvetica", 14, "bold")).pack(pady=(0, 20))
        
        # Node Count Input
        ttk.Label(self.control_frame, text="Number of Nodes:").pack(pady=5, anchor=tk.W)
        self.node_count_var = tk.IntVar(value=config.DEFAULT_NUM_NODES)
        self.node_count_entry = ttk.Entry(self.control_frame, textvariable=self.node_count_var)
        self.node_count_entry.pack(pady=5, anchor=tk.W, fill=tk.X)

        ttk.Label(self.control_frame, text="Solver Selection:").pack(pady=5, anchor=tk.W)
        self.solver_var = tk.StringVar(value="Greedy")
        ttk.Radiobutton(self.control_frame, text="Greedy (Nearest Neighbor)", variable=self.solver_var, value="Greedy").pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(self.control_frame, text="Simulated Annealing", variable=self.solver_var, value="SimulatedAnnealing").pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(self.control_frame, text="ALNS (Adaptive Large Neighborhood)", variable=self.solver_var, value="ALNS").pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(self.control_frame, text="OR-Tools", variable=self.solver_var, value="ORTools").pack(anchor=tk.W, padx=10)
        
        self.show_optimal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Calculate Optimal Solution (Benchmark)", variable=self.show_optimal_var).pack(pady=20, anchor=tk.W)
        
        self.run_btn = ttk.Button(self.control_frame, text="Run Simulation", command=self.run_simulation)
        self.run_btn.pack(pady=10, fill=tk.X)
        
        self.reset_btn = ttk.Button(self.control_frame, text="Reset / New Instance", command=self.reset_simulation, state=tk.DISABLED)
        self.reset_btn.pack(pady=5, fill=tk.X)

        self.view_mode_var = tk.StringVar(value="Solver")
        self.view_toggle_frame = ttk.LabelFrame(self.control_frame, text="View Mode")
        self.view_toggle_frame.pack(pady=10, fill=tk.X)
        
        self.view_solver_rb = ttk.Radiobutton(self.view_toggle_frame, text="Solver Solution", variable=self.view_mode_var, value="Solver", command=self.toggle_view, state=tk.DISABLED)
        self.view_solver_rb.pack(anchor=tk.W, padx=5, pady=2)
        
        self.view_optimal_rb = ttk.Radiobutton(self.view_toggle_frame, text="Optimal Solution", variable=self.view_mode_var, value="Optimal", command=self.toggle_view, state=tk.DISABLED)
        self.view_optimal_rb.pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20)
        
        self.status_label = ttk.Label(self.control_frame, text="Ready", wraplength=200, font=("Helvetica", 10))
        self.status_label.pack(pady=5)

        self.result_label = ttk.Label(self.control_frame, text="", wraplength=200, justify=tk.LEFT)
        self.result_label.pack(pady=10)
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.graph = None
        self.viz = None
        self.running = False
        self.optimal_routes = None
        self.final_routes = None

    def reset_simulation(self):
        self.ax.clear()
        self.canvas.draw()
        self.status_label.config(text="Ready")
        self.result_label.config(text="")
        self.view_solver_rb.config(state=tk.DISABLED)
        self.view_optimal_rb.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.NORMAL)
        self.view_mode_var.set("Solver")

    def run_simulation(self):
        if self.running:
            return
        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        self.view_solver_rb.config(state=tk.DISABLED)
        self.view_optimal_rb.config(state=tk.DISABLED)
        
        self.status_label.config(text="Generating Instance...")
        self.result_label.config(text="")
        self.optimal_routes = None
        self.final_routes = None
        self.view_mode_var.set("Solver")
        
        # Generate Instance
        try:
            num_nodes = self.node_count_var.get()
        except:
            num_nodes = config.DEFAULT_NUM_NODES
            
        generator = InstanceGenerator(num_nodes=num_nodes, seed=config.SEED) 
        self.graph = generator.generate_vrp_instance()
        
        # Setup Visualizer
        self.viz = Visualizer(self.graph, ax=self.ax)
        self.viz.draw_base()
        self.canvas.draw()
        
        self.solver_name = self.solver_var.get()
        self.calculate_optimal = self.show_optimal_var.get()
        
        # Start processing in a thread
        threading.Thread(target=self.solve_logic, daemon=True).start()

    def solve_logic(self):
        optimal_cost = None
        
        if self.calculate_optimal:
            self.update_status("Calculating Optimal Solution (Benchmark)...")
            # Use OR-Tools as optimal solver
            solver = ORToolsSolver(self.graph, vehicle_capacity=config.VEHICLE_CAPACITY, num_vehicles=config.NUM_VEHICLES)
            for routes in solver.solve():
                self.optimal_routes = routes
            
            if self.optimal_routes:
                optimal_cost = self.calculate_cost(self.optimal_routes)
        
        self.update_status(f"Running {self.solver_name}...")
        
        if self.solver_name == "Greedy":
            solver = GreedySolver(self.graph, capacity=config.VEHICLE_CAPACITY)
        elif self.solver_name == "SimulatedAnnealing":
            solver = SimulatedAnnealingSolver(self.graph, capacity=config.VEHICLE_CAPACITY, 
                                              initial_temp=config.SA_INITIAL_TEMP, 
                                              cooling_rate=config.SA_COOLING_RATE, 
                                              max_iterations=config.SA_MAX_ITERATIONS)
        elif self.solver_name == "ALNS":
            solver = ALNSSolver(self.graph, capacity=config.VEHICLE_CAPACITY, 
                                iterations=config.ALNS_ITERATIONS, 
                                remove_count=config.ALNS_REMOVE_COUNT)
        else:
            solver = ORToolsSolver(self.graph, vehicle_capacity=config.VEHICLE_CAPACITY, num_vehicles=config.NUM_VEHICLES)
            
        final_routes = []
        for step, routes in enumerate(solver.solve()):
            final_routes = routes
            # Update GUI from thread
            self.root.after(0, self.update_viz, routes, f"{self.solver_name} - Step {step}")
            time.sleep(config.ANIMATION_DELAY) # Animation delay
            
        self.final_routes = final_routes
        final_cost = self.calculate_cost(final_routes)
        
        result_text = f"--- Results ---\n\n{self.solver_name} Cost: {final_cost:.2f}"
        
        if optimal_cost is not None:
            result_text += f"\n\nOptimal Cost: {optimal_cost:.2f}"
            gap = ((final_cost - optimal_cost) / optimal_cost) * 100
            result_text += f"\nGap: {gap:.2f}%"
            
            if gap < 0.001:
                result_text += "\n(Optimal Found!)"
            
        self.root.after(0, self.finish_run, result_text)

    def update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))

    def update_viz(self, routes, title, linestyle='-'):
        self.viz.update(routes, title, linestyle=linestyle)

    def finish_run(self, result_text):
        self.running = False
        self.status_label.config(text="Finished")
        self.result_label.config(text=result_text)
        self.reset_btn.config(state=tk.NORMAL)
        self.view_solver_rb.config(state=tk.NORMAL)
        
        if self.optimal_routes:
            self.view_optimal_rb.config(state=tk.NORMAL)

    def toggle_view(self):
        mode = self.view_mode_var.get()
        if mode == "Solver" and self.final_routes:
            self.viz.update(self.final_routes, f"{self.solver_name} Solution", linestyle='-')
        elif mode == "Optimal" and self.optimal_routes:
            self.viz.update(self.optimal_routes, "Optimal Solution", linestyle='--')

    def calculate_cost(self, routes):
        total_dist = 0
        for route in routes:
            if not route: continue
            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                total_dist += self.graph[u][v]['weight']
        return total_dist

if __name__ == "__main__":
    root = tk.Tk()
    app = VRPApp(root)
    root.mainloop()

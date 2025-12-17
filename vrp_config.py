# VRP Problem Configuration

# Problem Settings
DEFAULT_NUM_NODES = 30
VEHICLE_CAPACITY = 40
NUM_VEHICLES = 5
DEPOT_POS = (50.0, 50.0)
SEED = None # Set to an integer for reproducible results

# Solver Settings

# Simulated Annealing
SA_INITIAL_TEMP = 1000
SA_COOLING_RATE = 0.995
SA_MAX_ITERATIONS = 200

# ALNS (Adaptive Large Neighborhood Search)
ALNS_ITERATIONS = 100
ALNS_REMOVE_COUNT = 6 # Number of nodes to remove/reinsert in each step

# Visualization
ANIMATION_DELAY = 0.05

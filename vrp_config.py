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

# Deep Learning
DL_MODEL_TYPE = "attention_model"  # Options: "pointer_network", "attention_model"
DL_MODEL_PATHS = {
    "pointer_network": "deep_learning/checkpoints/pointer_network/vrp_model.pth",
    "attention_model": "deep_learning/checkpoints/attention_model/vrp_model.pth",
}
# Legacy path for backwards compatibility
DL_MODEL_PATH = DL_MODEL_PATHS.get(DL_MODEL_TYPE, "deep_learning/checkpoints/vrp_model.pth")

# Visualization
ANIMATION_DELAY = 0.3


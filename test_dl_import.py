import sys
sys.path.append('.')

try:
    from deep_learning import get_neural_solver, get_available_models, get_model_info
    print("DL_AVAILABLE = True")
    print("Models:", get_available_models())
except ImportError as e:
    print("DL_AVAILABLE = False")
    print("ImportError:", e)
except Exception as e:
    print("DL_AVAILABLE = False")
    print("Exception:", type(e).__name__, e)

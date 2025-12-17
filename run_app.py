import sys
import os

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.gui import VRPApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = VRPApp(root)
    root.mainloop()

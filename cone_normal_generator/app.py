"""
Main entry point for the Cone Normal Map Generator.
"""
import os
import sys
import tkinter as tk
from PIL import Image

from cone_normal_generator.gui import ConeNormalMapApp
from cone_normal_generator.helpers import ensure_folders_exist, create_simple_icon

def main():
    """Main entry point."""
    print(f"Starting Cone Normal Map Generator (Python {sys.version})")
    
    # Ensure folders exist
    ensure_folders_exist()
    
    # Create and configure Tkinter root
    root = tk.Tk()
    root.title("Cone Normal Map Generator")
    
    # Create application
    app = ConeNormalMapApp(root)
    
    # Set window icon if available
    try:
        # Windows and Linux
        icon_path = "normal_map_icon.png"
        if not os.path.exists(icon_path):
            # Create a simple icon
            icon = create_simple_icon()
            icon.save(icon_path)
        
        # Set icon
        icon_img = tk.PhotoImage(file=icon_path)
        root.iconphoto(True, icon_img)
    except Exception as e:
        print(f"Could not set application icon: {e}")
    
    # Start the GUI loop
    print("Starting Tkinter main loop")
    root.mainloop()

if __name__ == "__main__":
    main()
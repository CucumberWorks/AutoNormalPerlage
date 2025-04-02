"""
Helper functions for the Cone Normal Map Generator.
"""
import os
import sys
from PIL import Image

from cone_normal_generator.config import TEMP_FOLDER, OUTPUT_FOLDER

def ensure_folders_exist():
    """Create necessary folders if they don't exist."""
    for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

def open_folder(folder_path):
    """Open the specified folder in the file explorer."""
    try:
        # Make sure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Open the folder using the appropriate method for the OS
        if sys.platform == 'win32':
            os.startfile(os.path.abspath(folder_path))
        elif sys.platform == 'darwin':  # macOS
            os.system(f'open "{os.path.abspath(folder_path)}"')
        else:  # Linux
            os.system(f'xdg-open "{os.path.abspath(folder_path)}"')
        return True
    except Exception as e:
        print(f"Error opening folder: {str(e)}")
        return False

def clean_folder(folder_path):
    """Delete all files in the specified folder."""
    if not os.path.exists(folder_path):
        return False
        
    try:
        # Delete all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error cleaning folder: {str(e)}")
        return False

def create_simple_icon(size=64):
    """Create a simple normal map icon."""
    import numpy as np
    
    icon = Image.new("RGB", (size, size), color="blue")
    pixels = icon.load()
    
    for i in range(size):
        for j in range(size):
            x = (i / size) * 2 - 1
            y = (j / size) * 2 - 1
            dist = min(1, x*x + y*y)
            z = np.sqrt(1 - dist)
            r = int((x * 0.5 + 0.5) * 255)
            g = int((y * 0.5 + 0.5) * 255)
            b = int((z * 0.5 + 0.5) * 255)
            pixels[i, j] = (r, g, b)
    
    return icon

def validate_numeric(value, is_int=False):
    """Validate if a string is a valid number."""
    if value == "":
        return True
    try:
        if is_int:
            int(value)
        else:
            float(value)
        return True
    except ValueError:
        return False 
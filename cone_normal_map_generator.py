#!/usr/bin/env python3
"""
Launcher for the Cone Normal Map Generator.

This is the main entry point for the application.
"""
import sys
import os

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main function from the refactored module
from cone_normal_generator.app import main

if __name__ == "__main__":
    main() 
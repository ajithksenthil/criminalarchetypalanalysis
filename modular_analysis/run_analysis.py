#!/usr/bin/env python3
"""
run_analysis.py

Launcher script for the modular criminal archetypal analysis system.
This script can be run from within the modular_analysis directory.
"""

import sys
import os

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import and run the main script
if __name__ == "__main__":
    try:
        # Import the main function from the parent directory
        sys.path.insert(0, parent_dir)
        from run_modular_analysis import main
        main()
    except ImportError:
        print("Error: Could not import the main analysis script.")
        print("Please ensure you're running from the correct directory.")
        print("\nTry running from the parent directory:")
        print("cd .. && python run_modular_analysis.py [arguments]")
        sys.exit(1)

#!/usr/bin/env python3
"""
Simple demo script for Data Quality Co-pilot
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the console demo
from demo.console_demo import main

if __name__ == "__main__":
    sys.exit(main()) 
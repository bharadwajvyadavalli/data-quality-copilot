#!/usr/bin/env python3
"""
Startup script for Data Quality Co-pilot Web Application
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Change to web directory
os.chdir(Path(__file__).parent / "web")

# Import and run the Flask app
from app import app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("=" * 60)
    print("Data Quality Co-pilot Web Application")
    print("=" * 60)
    print(f"Starting server on port {port}")
    print(f"Open your browser to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug) 
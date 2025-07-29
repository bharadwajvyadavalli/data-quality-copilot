#!/usr/bin/env python3
"""
Test script to verify Data Quality Co-pilot installation
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
        
    try:
        import openai
        print("✓ openai imported successfully")
    except ImportError as e:
        print(f"✗ openai import failed: {e}")
        return False
        
    try:
        import flask
        print("✓ flask imported successfully")
    except ImportError as e:
        print(f"✗ flask import failed: {e}")
        return False
        
    try:
        import yaml
        print("✓ pyyaml imported successfully")
    except ImportError as e:
        print(f"✗ pyyaml import failed: {e}")
        return False
        
    return True

def test_src_imports():
    """Test that our source modules can be imported."""
    print("\nTesting source module imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    try:
        from agents.base_agent import BaseAgent
        print("✓ BaseAgent imported successfully")
    except ImportError as e:
        print(f"✗ BaseAgent import failed: {e}")
        return False
        
    try:
        from agents.ingestion_agent import IngestionAgent
        print("✓ IngestionAgent imported successfully")
    except ImportError as e:
        print(f"✗ IngestionAgent import failed: {e}")
        return False
        
    try:
        from mcp.client import MCPClient
        print("✓ MCPClient imported successfully")
    except ImportError as e:
        print(f"✗ MCPClient import failed: {e}")
        return False
        
    try:
        from pipeline import DataQualityPipeline
        print("✓ DataQualityPipeline imported successfully")
    except ImportError as e:
        print(f"✗ DataQualityPipeline import failed: {e}")
        return False
        
    return True

def test_sample_data_creation():
    """Test that we can create sample data."""
    print("\nTesting sample data creation...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create simple sample data
        data = {
            'id': range(1, 11),
            'name': [f'Test_{i}' for i in range(1, 11)],
            'value': np.random.randn(10)
        }
        
        df = pd.DataFrame(data)
        print(f"✓ Sample data created: {len(df)} rows, {len(df.columns)} columns")
        
        # Save to CSV
        sample_file = "test_sample.csv"
        df.to_csv(sample_file, index=False)
        print(f"✓ Sample data saved to {sample_file}")
        
        # Clean up
        os.remove(sample_file)
        print("✓ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation failed: {e}")
        return False

def test_openai_key():
    """Test OpenAI API key configuration."""
    print("\nTesting OpenAI API key...")
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        print("✓ OpenAI API key found in environment")
        return True
    else:
        print("⚠ OpenAI API key not found (schema generation will be limited)")
        print("  Set OPENAI_API_KEY environment variable for full functionality")
        return True  # Not a failure, just a warning

def main():
    """Run all tests."""
    print("=" * 60)
    print("Data Quality Co-pilot - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Source Module Imports", test_src_imports),
        ("Sample Data Creation", test_sample_data_creation),
        ("OpenAI API Key", test_openai_key)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Run console demo: python demo.py --sample")
        print("2. Start web app: python run_web.py")
        print("3. Open browser to: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify file permissions")
    
    print("=" * 60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main()) 
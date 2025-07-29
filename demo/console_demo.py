#!/usr/bin/env python3
"""
Console Demo for Data Quality Co-pilot
Demonstrates the end-to-end data quality inspection pipeline.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import DataQualityPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_sample_csv():
    """Create a sample CSV file for demonstration."""
    import pandas as pd
    import numpy as np
    
    # Create sample data with various quality issues
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, 101),
        'name': [f'Customer_{i}' for i in range(1, 101)],
        'email': [f'customer{i}@example.com' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'purchase_amount': np.random.exponential(100, 100),
        'last_purchase_date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'customer_type': np.random.choice(['Premium', 'Standard', 'Basic'], 100),
        'satisfaction_score': np.random.randint(1, 11, 100),
        'is_active': np.random.choice([True, False], 100)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some quality issues
    # Missing values
    df.loc[10:15, 'email'] = None
    df.loc[20:25, 'age'] = None
    
    # Duplicate rows
    df = pd.concat([df, df.iloc[0:3]], ignore_index=True)
    
    # Outliers
    df.loc[50, 'income'] = 1000000  # Extreme outlier
    df.loc[51, 'purchase_amount'] = 50000  # Another outlier
    
    # Invalid data
    df.loc[60, 'age'] = 150  # Invalid age
    df.loc[61, 'satisfaction_score'] = 15  # Invalid score
    
    # Mixed data types
    df.loc[70:75, 'customer_id'] = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    
    # Save to CSV
    sample_file = "sample_customer_data.csv"
    df.to_csv(sample_file, index=False)
    
    print(f"Created sample CSV file: {sample_file}")
    print(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
    print("Quality issues introduced:")
    print("- Missing values in email and age columns")
    print("- Duplicate rows")
    print("- Outliers in income and purchase_amount")
    print("- Invalid values in age and satisfaction_score")
    print("- Mixed data types in customer_id")
    print()
    
    return sample_file


def run_demo(file_path: str, openai_api_key: str = None, verbose: bool = False):
    """Run the data quality inspection demo."""
    
    print("=" * 80)
    print("DATA QUALITY CO-PILOT - CONSOLE DEMO")
    print("=" * 80)
    print()
    
    # Setup logging
    setup_logging(verbose)
    
    # Initialize pipeline
    print("Initializing Data Quality Pipeline...")
    pipeline = DataQualityPipeline(openai_api_key=openai_api_key)
    print("✓ Pipeline initialized successfully")
    print()
    
    # Run pipeline
    print(f"Running data quality inspection on: {file_path}")
    print("-" * 60)
    
    try:
        results = pipeline.run_pipeline(file_path)
        
        if 'error' in results:
            print(f"❌ Pipeline failed: {results['error']}")
            return False
            
        # Display results
        print("\n" + "=" * 80)
        print("PIPELINE RESULTS")
        print("=" * 80)
        
        # Pipeline summary
        summary = results['pipeline_summary']
        print(f"Status: {summary['status']}")
        print(f"Duration: {summary['duration']:.2f} seconds")
        print(f"Quality Score: {summary['quality_score']:.2f}/100")
        print(f"Total Issues: {summary['total_issues']}")
        print(f"Total Warnings: {summary['total_warnings']}")
        print()
        
        # Display console report
        if 'reporting_results' in results:
            console_report = results['reporting_results']['console_report']
            print(console_report)
        
        # Display agent thoughts
        if verbose:
            print("\n" + "=" * 80)
            print("AGENT THOUGHTS (Verbose Mode)")
            print("=" * 80)
            
            thoughts = pipeline.get_agent_thoughts()
            for agent_name, agent_thoughts in thoughts.items():
                if agent_thoughts:
                    print(f"\n{agent_name.upper()}:")
                    for thought in agent_thoughts[-5:]:  # Show last 5 thoughts
                        print(f"  [{thought['timestamp']}] {thought['thought']}")
        
        # Display MCP context summary
        if 'mcp_context' in results:
            mcp_context = results['mcp_context']
            print(f"\nMCP Context Summary:")
            print(f"  Total thoughts: {mcp_context['summary']['total_thoughts']}")
            print(f"  Total tool calls: {mcp_context['summary']['total_tool_calls']}")
            print(f"  Context history length: {mcp_context['summary']['context_history_length']}")
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        logging.error(f"Demo error: {str(e)}", exc_info=True)
        return False


def main():
    """Main function for the console demo."""
    parser = argparse.ArgumentParser(description="Data Quality Co-pilot Console Demo")
    parser.add_argument("--file", "-f", help="Path to CSV/Excel file to analyze")
    parser.add_argument("--sample", "-s", action="store_true", help="Create and use sample data")
    parser.add_argument("--openai-key", "-k", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Determine file path
    file_path = None
    
    if args.sample:
        print("Creating sample data...")
        file_path = create_sample_csv()
    elif args.file:
        file_path = args.file
    else:
        print("No file specified. Creating sample data...")
        file_path = create_sample_csv()
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return 1
    
    # Get OpenAI API key
    openai_api_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("⚠️  Warning: No OpenAI API key provided. Schema generation will be limited.")
        print("   Set OPENAI_API_KEY environment variable or use --openai-key option.")
        print()
    
    # Run demo
    success = run_demo(file_path, openai_api_key, args.verbose)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 
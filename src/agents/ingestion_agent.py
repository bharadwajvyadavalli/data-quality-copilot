"""
Ingestion Agent for Data Quality Co-pilot
Handles file reading, data loading, and initial preprocessing.
"""

import pandas as pd
import os
from typing import Dict, Any, Optional
from .base_agent import BaseAgent


class IngestionAgent(BaseAgent):
    """Agent responsible for data ingestion and preprocessing."""
    
    def __init__(self):
        super().__init__("IngestionAgent")
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and load the dataset.
        
        Args:
            data: Dictionary containing 'file_path' key
            
        Returns:
            Dictionary containing loaded dataframe and metadata
        """
        self.start_processing()
        
        file_path = data.get('file_path')
        if not file_path:
            raise ValueError("file_path is required in input data")
            
        self.add_thought(f"Starting ingestion of file: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            self.add_thought(error_msg, {"error": True})
            raise FileNotFoundError(error_msg)
            
        # Validate file format
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.supported_formats:
            error_msg = f"Unsupported file format: {file_extension}. Supported: {self.supported_formats}"
            self.add_thought(error_msg, {"error": True})
            raise ValueError(error_msg)
            
        self.add_thought(f"File format validated: {file_extension}")
        
        try:
            # Load the data
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
                self.add_thought("Successfully loaded CSV file using pandas.read_csv")
            else:  # Excel files
                df = pd.read_excel(file_path)
                self.add_thought("Successfully loaded Excel file using pandas.read_excel")
                
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.add_thought(error_msg, {"error": True, "exception": str(e)})
            raise
            
        # Basic data preprocessing
        self.add_thought("Starting data preprocessing")
        
        # Remove completely empty rows and columns
        initial_shape = df.shape
        df = df.dropna(how='all').dropna(axis=1, how='all')
        final_shape = df.shape
        
        if initial_shape != final_shape:
            self.add_thought(
                f"Removed empty rows/columns. Shape changed from {initial_shape} to {final_shape}",
                {"initial_shape": initial_shape, "final_shape": final_shape}
            )
            
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        self.add_thought("Reset dataframe index after cleaning")
        
        # Generate basic metadata
        metadata = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "file_extension": file_extension,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum()
        }
        
        self.add_thought(
            f"Generated metadata: {len(df)} rows, {len(df.columns)} columns",
            {"metadata": metadata}
        )
        
        # Check for potential data quality issues
        quality_issues = []
        
        # Check for high missing value percentages
        for col, missing_count in metadata["missing_values"].items():
            missing_pct = (missing_count / len(df)) * 100
            if missing_pct > 50:
                quality_issues.append(f"Column '{col}' has {missing_pct:.1f}% missing values")
                
        # Check for duplicate rows
        if metadata["duplicate_rows"] > 0:
            quality_issues.append(f"Found {metadata['duplicate_rows']} duplicate rows")
            
        # Check for single-value columns
        for col in df.columns:
            if df[col].nunique() == 1:
                quality_issues.append(f"Column '{col}' has only one unique value")
                
        if quality_issues:
            self.add_thought(
                f"Identified {len(quality_issues)} potential quality issues",
                {"quality_issues": quality_issues}
            )
        else:
            self.add_thought("No obvious quality issues detected during initial scan")
            
        self.end_processing()
        
        return {
            "dataframe": df,
            "metadata": metadata,
            "quality_issues": quality_issues,
            "processing_summary": self.get_processing_summary()
        } 
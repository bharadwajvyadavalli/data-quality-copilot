"""
Schema Agent for Data Quality Co-pilot
Uses LLM to generate comprehensive data schemas and data dictionaries.
"""

import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os
from .base_agent import BaseAgent
from ..mcp.client import MCPClient


class SchemaAgent(BaseAgent):
    """Agent responsible for LLM-powered schema generation and validation."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        super().__init__("SchemaAgent")
        self.mcp_client = MCPClient()
        
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass to constructor.")
            
        self.client = OpenAI(api_key=api_key)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and generate a comprehensive schema.
        
        Args:
            data: Dictionary containing 'dataframe' and 'metadata' keys
            
        Returns:
            Dictionary containing generated schema and validation results
        """
        self.start_processing()
        
        df = data.get('dataframe')
        metadata = data.get('metadata', {})
        
        if df is None:
            raise ValueError("dataframe is required in input data")
            
        self.add_thought("Starting schema generation using LLM")
        
        # Prepare data sample for LLM analysis
        sample_data = self._prepare_data_sample(df)
        self.add_thought(f"Prepared data sample with {len(sample_data)} rows for LLM analysis")
        
        # Generate schema using LLM
        schema = self._generate_schema_with_llm(df, sample_data, metadata)
        self.add_thought("Successfully generated schema using LLM")
        
        # Validate schema against actual data
        validation_results = self._validate_schema(df, schema)
        self.add_thought(f"Schema validation completed with {len(validation_results['issues'])} issues found")
        
        # Generate data dictionary
        data_dictionary = self._generate_data_dictionary(df, schema)
        self.add_thought("Generated comprehensive data dictionary")
        
        # Create MCP exchange
        self.mcp_client.exchange_mcp_message(
            agent_name=self.name,
            message="Generate schema for dataset",
            response=f"Generated schema with {len(schema['columns'])} columns and {len(validation_results['issues'])} validation issues",
            metadata={"schema_size": len(schema['columns']), "validation_issues": len(validation_results['issues'])}
        )
        
        self.end_processing()
        
        return {
            "schema": schema,
            "data_dictionary": data_dictionary,
            "validation_results": validation_results,
            "mcp_context": self.mcp_client.export_context(),
            "processing_summary": self.get_processing_summary()
        }
        
    def _prepare_data_sample(self, df: pd.DataFrame, sample_size: int = 10) -> List[Dict[str, Any]]:
        """Prepare a sample of data for LLM analysis."""
        # Take a stratified sample if possible
        if len(df) > sample_size:
            sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        else:
            sample = df
            
        # Convert to list of dictionaries
        sample_data = sample.to_dict('records')
        
        # Add basic statistics
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return {
            "sample_data": sample_data,
            "statistics": stats
        }
        
    def _generate_schema_with_llm(self, df: pd.DataFrame, sample_data: Dict[str, Any], 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate schema using OpenAI GPT."""
        
        # Create prompt for schema generation
        prompt = self._create_schema_prompt(df, sample_data, metadata)
        
        # Chunk prompt if needed
        prompt_chunks = self.mcp_client.chunk_prompt(prompt)
        
        schema_parts = []
        for i, chunk in enumerate(prompt_chunks):
            self.add_thought(f"Processing schema prompt chunk {i+1}/{len(prompt_chunks)}")
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a data schema expert. Generate detailed, accurate JSON schemas for datasets."},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                schema_part = response.choices[0].message.content
                schema_parts.append(schema_part)
                
                self.add_thought(f"Successfully processed chunk {i+1}")
                
            except Exception as e:
                self.add_thought(f"Error processing chunk {i+1}: {str(e)}", {"error": True})
                raise
                
        # Combine schema parts and parse
        full_schema_text = "\n".join(schema_parts)
        
        # Extract JSON from response
        schema_json = self._extract_json_from_response(full_schema_text)
        
        # Parse and validate schema structure
        schema = self._parse_and_validate_schema(schema_json, df)
        
        return schema
        
    def _create_schema_prompt(self, df: pd.DataFrame, sample_data: Dict[str, Any], 
                            metadata: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for schema generation."""
        
        prompt = f"""
        Analyze the following dataset and generate a comprehensive JSON schema.
        
        Dataset Information:
        - Total rows: {metadata.get('rows', len(df))}
        - Total columns: {metadata.get('columns', len(df.columns))}
        - Column names: {list(df.columns)}
        - Data types: {df.dtypes.to_dict()}
        - Missing values: {df.isnull().sum().to_dict()}
        
        Sample Data (first few rows):
        {json.dumps(sample_data['sample_data'][:3], indent=2, default=str)}
        
        For each column, provide:
        1. Data type (string, integer, float, boolean, date, etc.)
        2. Whether it's nullable
        3. Constraints (unique, range, pattern, etc.)
        4. Business meaning/description
        5. Expected format (if applicable)
        6. Quality rules that should apply
        
        Return a JSON object with this structure:
        {{
            "dataset_name": "string",
            "description": "string",
            "columns": [
                {{
                    "name": "string",
                    "data_type": "string",
                    "nullable": boolean,
                    "description": "string",
                    "constraints": {{
                        "unique": boolean,
                        "min_value": "number or null",
                        "max_value": "number or null",
                        "pattern": "regex or null",
                        "enum_values": ["array of valid values or null"]
                    }},
                    "quality_rules": ["array of quality rules"],
                    "format": "string or null"
                }}
            ],
            "relationships": ["array of column relationships if any"],
            "business_rules": ["array of business rules"]
        }}
        
        Be thorough and accurate in your analysis. Consider the business context and data patterns.
        """
        
        return prompt
        
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except json.JSONDecodeError as e:
            self.add_thought(f"JSON parsing error: {str(e)}", {"error": True})
            # Fallback: create basic schema
            return self._create_fallback_schema()
            
    def _create_fallback_schema(self) -> Dict[str, Any]:
        """Create a basic fallback schema if LLM fails."""
        return {
            "dataset_name": "Unknown Dataset",
            "description": "Schema generated from fallback method",
            "columns": [],
            "relationships": [],
            "business_rules": []
        }
        
    def _parse_and_validate_schema(self, schema_json: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Parse and validate the generated schema against actual data."""
        
        # Ensure all columns in the dataframe are represented in the schema
        schema_columns = {col['name'] for col in schema_json.get('columns', [])}
        actual_columns = set(df.columns)
        
        # Add missing columns to schema
        missing_columns = actual_columns - schema_columns
        for col in missing_columns:
            schema_json['columns'].append({
                "name": col,
                "data_type": str(df[col].dtype),
                "nullable": df[col].isnull().any(),
                "description": f"Column {col} - auto-generated",
                "constraints": {
                    "unique": df[col].nunique() == len(df),
                    "min_value": None,
                    "max_value": None,
                    "pattern": None,
                    "enum_values": None
                },
                "quality_rules": [],
                "format": None
            })
            
        # Validate data types
        for col_info in schema_json['columns']:
            col_name = col_info['name']
            if col_name in df.columns:
                # Validate nullable constraint
                actual_nullable = df[col_name].isnull().any()
                col_info['nullable'] = actual_nullable
                
                # Validate unique constraint
                actual_unique = df[col_name].nunique() == len(df)
                col_info['constraints']['unique'] = actual_unique
                
        return schema_json
        
    def _validate_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated schema against the actual data."""
        
        issues = []
        warnings = []
        
        for col_info in schema.get('columns', []):
            col_name = col_info['name']
            
            if col_name not in df.columns:
                issues.append(f"Schema column '{col_name}' not found in actual data")
                continue
                
            # Validate data type
            expected_type = col_info.get('data_type', 'unknown')
            actual_type = str(df[col_name].dtype)
            
            if not self._is_type_compatible(expected_type, actual_type):
                warnings.append(f"Column '{col_name}': expected type '{expected_type}', got '{actual_type}'")
                
            # Validate nullable constraint
            expected_nullable = col_info.get('nullable', False)
            actual_nullable = df[col_name].isnull().any()
            
            if expected_nullable != actual_nullable:
                warnings.append(f"Column '{col_name}': nullable constraint mismatch")
                
            # Validate unique constraint
            expected_unique = col_info.get('constraints', {}).get('unique', False)
            actual_unique = df[col_name].nunique() == len(df)
            
            if expected_unique != actual_unique:
                warnings.append(f"Column '{col_name}': unique constraint mismatch")
                
        return {
            "issues": issues,
            "warnings": warnings,
            "total_issues": len(issues),
            "total_warnings": len(warnings)
        }
        
    def _is_type_compatible(self, expected: str, actual: str) -> bool:
        """Check if expected and actual types are compatible."""
        type_mapping = {
            'string': ['object', 'string'],
            'integer': ['int64', 'int32'],
            'float': ['float64', 'float32'],
            'boolean': ['bool'],
            'date': ['datetime64[ns]', 'object']  # dates can be stored as objects
        }
        
        expected_lower = expected.lower()
        for compatible_types in type_mapping.values():
            if expected_lower in compatible_types and actual in compatible_types:
                return True
                
        return False
        
    def _generate_data_dictionary(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive data dictionary."""
        
        data_dict = {
            "dataset_name": schema.get('dataset_name', 'Unknown'),
            "description": schema.get('description', ''),
            "columns": {}
        }
        
        for col_info in schema.get('columns', []):
            col_name = col_info['name']
            
            if col_name in df.columns:
                # Calculate additional statistics
                col_stats = {
                    "data_type": str(df[col_name].dtype),
                    "nullable": df[col_name].isnull().any(),
                    "missing_count": df[col_name].isnull().sum(),
                    "missing_percentage": (df[col_name].isnull().sum() / len(df)) * 100,
                    "unique_count": df[col_name].nunique(),
                    "unique_percentage": (df[col_name].nunique() / len(df)) * 100
                }
                
                # Add numeric statistics if applicable
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    col_stats.update({
                        "min_value": float(df[col_name].min()) if not df[col_name].isnull().all() else None,
                        "max_value": float(df[col_name].max()) if not df[col_name].isnull().all() else None,
                        "mean": float(df[col_name].mean()) if not df[col_name].isnull().all() else None,
                        "std": float(df[col_name].std()) if not df[col_name].isnull().all() else None
                    })
                    
                # Add sample values
                non_null_values = df[col_name].dropna()
                if len(non_null_values) > 0:
                    col_stats["sample_values"] = non_null_values.head(5).tolist()
                    
                data_dict["columns"][col_name] = {
                    **col_info,
                    "statistics": col_stats
                }
                
        return data_dict 
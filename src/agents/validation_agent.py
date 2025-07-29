"""
Validation Agent for Data Quality Co-pilot
Applies quality rules and performs comprehensive data validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re
from datetime import datetime
from .base_agent import BaseAgent
from ..mcp.client import MCPClient


class ValidationAgent(BaseAgent):
    """Agent responsible for data quality validation and rule application."""
    
    def __init__(self):
        super().__init__("ValidationAgent")
        self.mcp_client = MCPClient()
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and perform comprehensive validation.
        
        Args:
            data: Dictionary containing 'dataframe', 'schema', and 'data_dictionary' keys
            
        Returns:
            Dictionary containing validation results and quality metrics
        """
        self.start_processing()
        
        df = data.get('dataframe')
        schema = data.get('schema', {})
        data_dictionary = data.get('data_dictionary', {})
        
        if df is None:
            raise ValueError("dataframe is required in input data")
            
        self.add_thought("Starting comprehensive data validation")
        
        # Perform basic quality checks
        basic_checks = self._perform_basic_quality_checks(df)
        self.add_thought(f"Completed basic quality checks with {len(basic_checks['issues'])} issues found")
        
        # Perform schema-based validation
        schema_validation = self._perform_schema_validation(df, schema)
        self.add_thought(f"Completed schema validation with {len(schema_validation['issues'])} issues found")
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(df)
        self.add_thought("Completed statistical analysis")
        
        # Perform outlier detection
        outlier_analysis = self._detect_outliers(df)
        self.add_thought(f"Completed outlier detection with {len(outlier_analysis['outliers'])} outliers found")
        
        # Perform data consistency checks
        consistency_checks = self._check_data_consistency(df, schema)
        self.add_thought(f"Completed consistency checks with {len(consistency_checks['issues'])} issues found")
        
        # Generate quality score
        quality_score = self._calculate_quality_score(
            basic_checks, schema_validation, outlier_analysis, consistency_checks
        )
        self.add_thought(f"Calculated overall quality score: {quality_score:.2f}/100")
        
        # Create comprehensive validation report
        validation_report = self._create_validation_report(
            basic_checks, schema_validation, statistical_analysis, 
            outlier_analysis, consistency_checks, quality_score
        )
        
        # Create MCP exchange
        self.mcp_client.exchange_mcp_message(
            agent_name=self.name,
            message="Validate dataset quality",
            response=f"Validation completed with quality score {quality_score:.2f}/100 and {len(validation_report['total_issues'])} total issues",
            metadata={"quality_score": quality_score, "total_issues": len(validation_report['total_issues'])}
        )
        
        self.end_processing()
        
        return {
            "validation_report": validation_report,
            "quality_score": quality_score,
            "basic_checks": basic_checks,
            "schema_validation": schema_validation,
            "statistical_analysis": statistical_analysis,
            "outlier_analysis": outlier_analysis,
            "consistency_checks": consistency_checks,
            "mcp_context": self.mcp_client.export_context(),
            "processing_summary": self.get_processing_summary()
        }
        
    def _perform_basic_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic data quality checks."""
        
        issues = []
        warnings = []
        metrics = {}
        
        # Check for missing values
        missing_values = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = (missing_values.sum() / total_cells) * 100
        
        metrics['missing_percentage'] = missing_percentage
        metrics['missing_values_by_column'] = missing_values.to_dict()
        
        if missing_percentage > 20:
            issues.append(f"High percentage of missing values: {missing_percentage:.2f}%")
        elif missing_percentage > 5:
            warnings.append(f"Moderate percentage of missing values: {missing_percentage:.2f}%")
            
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        metrics['duplicate_percentage'] = duplicate_percentage
        metrics['duplicate_count'] = duplicate_count
        
        if duplicate_percentage > 10:
            issues.append(f"High percentage of duplicate rows: {duplicate_percentage:.2f}%")
        elif duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}%)")
            
        # Check for empty columns
        empty_columns = []
        for col in df.columns:
            if df[col].isnull().all() or (df[col].astype(str).str.strip() == '').all():
                empty_columns.append(col)
                
        metrics['empty_columns'] = empty_columns
        
        if empty_columns:
            issues.append(f"Found {len(empty_columns)} empty columns: {empty_columns}")
            
        # Check for single-value columns
        single_value_columns = []
        for col in df.columns:
            if df[col].nunique() == 1:
                single_value_columns.append(col)
                
        metrics['single_value_columns'] = single_value_columns
        
        if single_value_columns:
            warnings.append(f"Found {len(single_value_columns)} columns with single value: {single_value_columns}")
            
        return {
            "issues": issues,
            "warnings": warnings,
            "metrics": metrics
        }
        
    def _perform_schema_validation(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Perform validation based on the generated schema."""
        
        issues = []
        warnings = []
        validation_results = {}
        
        for col_info in schema.get('columns', []):
            col_name = col_info['name']
            
            if col_name not in df.columns:
                issues.append(f"Schema column '{col_name}' not found in data")
                continue
                
            col_issues = []
            col_warnings = []
            
            # Validate data type
            expected_type = col_info.get('data_type', '').lower()
            actual_type = str(df[col_name].dtype)
            
            if not self._is_type_compatible(expected_type, actual_type):
                col_warnings.append(f"Type mismatch: expected '{expected_type}', got '{actual_type}'")
                
            # Validate nullable constraint
            expected_nullable = col_info.get('nullable', False)
            actual_nullable = df[col_name].isnull().any()
            
            if expected_nullable != actual_nullable:
                col_warnings.append(f"Nullable constraint mismatch: expected {expected_nullable}, got {actual_nullable}")
                
            # Validate constraints
            constraints = col_info.get('constraints', {})
            
            # Unique constraint
            if constraints.get('unique', False):
                if df[col_name].nunique() != len(df):
                    col_issues.append("Unique constraint violated")
                    
            # Range constraints
            if pd.api.types.is_numeric_dtype(df[col_name]):
                min_val = constraints.get('min_value')
                max_val = constraints.get('max_value')
                
                if min_val is not None and df[col_name].min() < min_val:
                    col_issues.append(f"Min value constraint violated: {df[col_name].min()} < {min_val}")
                    
                if max_val is not None and df[col_name].max() > max_val:
                    col_issues.append(f"Max value constraint violated: {df[col_name].max()} > {max_val}")
                    
            # Pattern constraints
            pattern = constraints.get('pattern')
            if pattern and pd.api.types.is_string_dtype(df[col_name]):
                non_null_values = df[col_name].dropna()
                if len(non_null_values) > 0:
                    pattern_matches = non_null_values.astype(str).str.match(pattern)
                    if not pattern_matches.all():
                        col_issues.append(f"Pattern constraint violated: {pattern}")
                        
            # Enum constraints
            enum_values = constraints.get('enum_values')
            if enum_values and len(enum_values) > 0:
                non_null_values = df[col_name].dropna()
                if len(non_null_values) > 0:
                    invalid_values = set(non_null_values) - set(enum_values)
                    if invalid_values:
                        col_issues.append(f"Enum constraint violated: invalid values {list(invalid_values)[:5]}")
                        
            # Quality rules
            quality_rules = col_info.get('quality_rules', [])
            for rule in quality_rules:
                rule_result = self._apply_quality_rule(df[col_name], rule)
                if rule_result['violated']:
                    col_issues.append(f"Quality rule violated: {rule} - {rule_result['message']}")
                    
            validation_results[col_name] = {
                "issues": col_issues,
                "warnings": col_warnings,
                "passed": len(col_issues) == 0
            }
            
            issues.extend(col_issues)
            warnings.extend(col_warnings)
            
        return {
            "issues": issues,
            "warnings": warnings,
            "validation_results": validation_results
        }
        
    def _apply_quality_rule(self, series: pd.Series, rule: str) -> Dict[str, Any]:
        """Apply a specific quality rule to a column."""
        
        rule_lower = rule.lower()
        
        if "no nulls" in rule_lower or "not null" in rule_lower:
            return {
                "violated": series.isnull().any(),
                "message": "Contains null values"
            }
        elif "unique" in rule_lower:
            return {
                "violated": series.nunique() != len(series),
                "message": "Contains duplicate values"
            }
        elif "positive" in rule_lower and pd.api.types.is_numeric_dtype(series):
            return {
                "violated": (series < 0).any(),
                "message": "Contains negative values"
            }
        elif "email" in rule_lower:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                valid_emails = non_null_values.astype(str).str.match(email_pattern)
                return {
                    "violated": not valid_emails.all(),
                    "message": "Contains invalid email formats"
                }
        elif "phone" in rule_lower:
            phone_pattern = r'^[\+]?[1-9][\d]{0,15}$'
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                valid_phones = non_null_values.astype(str).str.replace(r'[\s\-\(\)]', '', regex=True).str.match(phone_pattern)
                return {
                    "violated": not valid_phones.all(),
                    "message": "Contains invalid phone formats"
                }
                
        return {
            "violated": False,
            "message": "Rule not recognized"
        }
        
    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on the dataset."""
        
        analysis = {
            "summary_statistics": {},
            "correlations": {},
            "distributions": {}
        }
        
        # Summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            analysis["summary_statistics"] = df[numeric_columns].describe().to_dict()
            
        # Correlation analysis
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr()
            # Get high correlations (> 0.8 or < -0.8)
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append({
                            "column1": correlation_matrix.columns[i],
                            "column2": correlation_matrix.columns[j],
                            "correlation": corr_value
                        })
            analysis["correlations"]["high_correlations"] = high_correlations
            
        # Distribution analysis
        for col in df.columns:
            if df[col].dtype == 'object':
                # Categorical distribution
                value_counts = df[col].value_counts()
                analysis["distributions"][col] = {
                    "type": "categorical",
                    "unique_values": len(value_counts),
                    "top_values": value_counts.head(5).to_dict()
                }
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Numeric distribution
                analysis["distributions"][col] = {
                    "type": "numeric",
                    "skewness": float(df[col].skew()) if len(df[col].dropna()) > 0 else None,
                    "kurtosis": float(df[col].kurtosis()) if len(df[col].dropna()) > 0 else None
                }
                
        return analysis
        
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns using IQR method."""
        
        outliers = {}
        outlier_summary = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outlier_mask].index.tolist()
            
            outliers[col] = {
                "indices": outlier_indices,
                "values": series[outlier_mask].tolist(),
                "count": len(outlier_indices),
                "percentage": (len(outlier_indices) / len(series)) * 100,
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
            }
            
            outlier_summary[col] = {
                "count": len(outlier_indices),
                "percentage": (len(outlier_indices) / len(series)) * 100
            }
            
        return {
            "outliers": outliers,
            "summary": outlier_summary
        }
        
    def _check_data_consistency(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Check for data consistency issues."""
        
        issues = []
        warnings = []
        
        # Check for logical inconsistencies
        for col_info in schema.get('columns', []):
            col_name = col_info['name']
            
            if col_name not in df.columns:
                continue
                
            # Check for mixed data types in string columns
            if df[col_name].dtype == 'object':
                # Try to detect mixed types
                non_null_values = df[col_name].dropna()
                if len(non_null_values) > 0:
                    # Check if some values can be converted to numbers
                    numeric_count = 0
                    for val in non_null_values:
                        try:
                            float(val)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass
                            
                    if 0 < numeric_count < len(non_null_values):
                        warnings.append(f"Column '{col_name}' contains mixed string and numeric values")
                        
        # Check for date consistency
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
                
        for col in date_columns:
            try:
                pd.to_datetime(df[col], errors='coerce')
            except:
                warnings.append(f"Column '{col}' appears to be a date column but contains invalid dates")
                
        return {
            "issues": issues,
            "warnings": warnings
        }
        
    def _calculate_quality_score(self, basic_checks: Dict[str, Any], 
                               schema_validation: Dict[str, Any],
                               outlier_analysis: Dict[str, Any],
                               consistency_checks: Dict[str, Any]) -> float:
        """Calculate an overall quality score (0-100)."""
        
        score = 100.0
        deductions = 0.0
        
        # Deduct for basic quality issues
        deductions += len(basic_checks['issues']) * 5  # 5 points per critical issue
        deductions += len(basic_checks['warnings']) * 2  # 2 points per warning
        
        # Deduct for schema validation issues
        deductions += len(schema_validation['issues']) * 3  # 3 points per validation issue
        deductions += len(schema_validation['warnings']) * 1  # 1 point per validation warning
        
        # Deduct for outliers (capped at 10 points)
        total_outliers = sum(summary['count'] for summary in outlier_analysis['summary'].values())
        outlier_deduction = min(total_outliers * 0.1, 10)
        deductions += outlier_deduction
        
        # Deduct for consistency issues
        deductions += len(consistency_checks['issues']) * 4  # 4 points per consistency issue
        deductions += len(consistency_checks['warnings']) * 1  # 1 point per consistency warning
        
        # Calculate final score
        final_score = max(0, score - deductions)
        
        return round(final_score, 2)
        
    def _create_validation_report(self, basic_checks: Dict[str, Any],
                                schema_validation: Dict[str, Any],
                                statistical_analysis: Dict[str, Any],
                                outlier_analysis: Dict[str, Any],
                                consistency_checks: Dict[str, Any],
                                quality_score: float) -> Dict[str, Any]:
        """Create a comprehensive validation report."""
        
        total_issues = (
            basic_checks['issues'] + 
            schema_validation['issues'] + 
            consistency_checks['issues']
        )
        
        total_warnings = (
            basic_checks['warnings'] + 
            schema_validation['warnings'] + 
            consistency_checks['warnings']
        )
        
        return {
            "quality_score": quality_score,
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "issue_count": len(total_issues),
            "warning_count": len(total_warnings),
            "outlier_count": sum(summary['count'] for summary in outlier_analysis['summary'].values()),
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "critical_issues": len(total_issues),
                "warnings": len(total_warnings),
                "quality_level": self._get_quality_level(quality_score)
            }
        }
        
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Very Poor" 
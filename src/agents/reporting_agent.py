"""
Reporting Agent for Data Quality Co-pilot
Generates comprehensive reports and outputs from all agent processing.
"""

import json
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from .base_agent import BaseAgent
from ..mcp.client import MCPClient


class ReportingAgent(BaseAgent):
    """Agent responsible for generating comprehensive reports and outputs."""
    
    def __init__(self, output_dir: str = "reports"):
        super().__init__("ReportingAgent")
        self.mcp_client = MCPClient()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all agent results and generate comprehensive reports.
        
        Args:
            data: Dictionary containing results from all agents
            
        Returns:
            Dictionary containing generated reports and file paths
        """
        self.start_processing()
        
        # Extract data from all agents
        ingestion_results = data.get('ingestion_results', {})
        schema_results = data.get('schema_results', {})
        validation_results = data.get('validation_results', {})
        
        self.add_thought("Starting comprehensive report generation")
        
        # Generate console report
        console_report = self._generate_console_report(
            ingestion_results, schema_results, validation_results
        )
        self.add_thought("Generated console report")
        
        # Generate JSON report
        json_report = self._generate_json_report(
            ingestion_results, schema_results, validation_results
        )
        json_file_path = self._save_json_report(json_report)
        self.add_thought(f"Generated and saved JSON report to {json_file_path}")
        
        # Generate YAML report
        yaml_report = self._generate_yaml_report(
            ingestion_results, schema_results, validation_results
        )
        yaml_file_path = self._save_yaml_report(yaml_report)
        self.add_thought(f"Generated and saved YAML report to {yaml_file_path}")
        
        # Generate detailed HTML report
        html_report = self._generate_html_report(
            ingestion_results, schema_results, validation_results
        )
        html_file_path = self._save_html_report(html_report)
        self.add_thought(f"Generated and saved HTML report to {html_file_path}")
        
        # Generate summary dashboard data
        dashboard_data = self._generate_dashboard_data(
            ingestion_results, schema_results, validation_results
        )
        self.add_thought("Generated dashboard data for web interface")
        
        # Create MCP exchange
        self.mcp_client.exchange_mcp_message(
            agent_name=self.name,
            message="Generate comprehensive reports",
            response=f"Generated console, JSON, YAML, and HTML reports with quality score {validation_results.get('quality_score', 0):.2f}/100",
            metadata={"reports_generated": ["console", "json", "yaml", "html"], "quality_score": validation_results.get('quality_score', 0)}
        )
        
        self.end_processing()
        
        return {
            "console_report": console_report,
            "json_report": json_report,
            "yaml_report": yaml_report,
            "html_report": html_report,
            "dashboard_data": dashboard_data,
            "file_paths": {
                "json": json_file_path,
                "yaml": yaml_file_path,
                "html": html_file_path
            },
            "mcp_context": self.mcp_client.export_context(),
            "processing_summary": self.get_processing_summary()
        }
        
    def _generate_console_report(self, ingestion_results: Dict[str, Any],
                               schema_results: Dict[str, Any],
                               validation_results: Dict[str, Any]) -> str:
        """Generate a formatted console report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA QUALITY INSPECTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset Overview
        metadata = ingestion_results.get('metadata', {})
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"File: {metadata.get('file_path', 'Unknown')}")
        report_lines.append(f"Rows: {metadata.get('rows', 0):,}")
        report_lines.append(f"Columns: {metadata.get('columns', 0)}")
        report_lines.append(f"File Size: {metadata.get('file_size', 0):,} bytes")
        report_lines.append("")
        
        # Quality Score
        quality_score = validation_results.get('quality_score', 0)
        report_lines.append("QUALITY ASSESSMENT")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Quality Score: {quality_score:.2f}/100")
        report_lines.append(f"Quality Level: {self._get_quality_level(quality_score)}")
        report_lines.append("")
        
        # Issues Summary
        validation_report = validation_results.get('validation_report', {})
        report_lines.append("ISSUES SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Critical Issues: {validation_report.get('issue_count', 0)}")
        report_lines.append(f"Warnings: {validation_report.get('warning_count', 0)}")
        report_lines.append(f"Outliers: {validation_report.get('outlier_count', 0)}")
        report_lines.append("")
        
        # Schema Information
        schema = schema_results.get('schema', {})
        report_lines.append("SCHEMA INFORMATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Dataset Name: {schema.get('dataset_name', 'Unknown')}")
        report_lines.append(f"Description: {schema.get('description', 'No description available')}")
        report_lines.append(f"Columns: {len(schema.get('columns', []))}")
        report_lines.append("")
        
        # Column Details
        report_lines.append("COLUMN DETAILS")
        report_lines.append("-" * 40)
        for col_info in schema.get('columns', [])[:10]:  # Show first 10 columns
            col_name = col_info.get('name', 'Unknown')
            data_type = col_info.get('data_type', 'Unknown')
            nullable = col_info.get('nullable', False)
            description = col_info.get('description', 'No description')[:50]
            
            report_lines.append(f"{col_name:<20} | {data_type:<15} | {'NULL' if nullable else 'NOT NULL':<10} | {description}")
            
        if len(schema.get('columns', [])) > 10:
            report_lines.append(f"... and {len(schema.get('columns', [])) - 10} more columns")
        report_lines.append("")
        
        # Critical Issues
        total_issues = validation_report.get('total_issues', [])
        if total_issues:
            report_lines.append("CRITICAL ISSUES")
            report_lines.append("-" * 40)
            for issue in total_issues[:10]:  # Show first 10 issues
                report_lines.append(f"• {issue}")
            if len(total_issues) > 10:
                report_lines.append(f"... and {len(total_issues) - 10} more issues")
            report_lines.append("")
            
        # Warnings
        total_warnings = validation_report.get('total_warnings', [])
        if total_warnings:
            report_lines.append("WARNINGS")
            report_lines.append("-" * 40)
            for warning in total_warnings[:10]:  # Show first 10 warnings
                report_lines.append(f"• {warning}")
            if len(total_warnings) > 10:
                report_lines.append(f"... and {len(total_warnings) - 10} more warnings")
            report_lines.append("")
            
        # Agent Processing Summary
        report_lines.append("AGENT PROCESSING SUMMARY")
        report_lines.append("-" * 40)
        
        agents = ['IngestionAgent', 'SchemaAgent', 'ValidationAgent', 'ReportingAgent']
        for agent_name in agents:
            if f'{agent_name.lower().replace("agent", "")}_results' in [ingestion_results, schema_results, validation_results]:
                # Extract processing time if available
                processing_summary = None
                if 'processing_summary' in ingestion_results:
                    processing_summary = ingestion_results['processing_summary']
                elif 'processing_summary' in schema_results:
                    processing_summary = schema_results['processing_summary']
                elif 'processing_summary' in validation_results:
                    processing_summary = validation_results['processing_summary']
                    
                if processing_summary and processing_summary.get('agent_name') == agent_name:
                    duration = processing_summary.get('duration', 0)
                    thoughts_count = processing_summary.get('thoughts_count', 0)
                    report_lines.append(f"{agent_name:<20} | {duration:.2f}s | {thoughts_count} thoughts")
                else:
                    report_lines.append(f"{agent_name:<20} | Completed | N/A")
                    
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("Report generation completed successfully.")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
        
    def _generate_json_report(self, ingestion_results: Dict[str, Any],
                            schema_results: Dict[str, Any],
                            validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive JSON report."""
        
        return {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "comprehensive_data_quality",
                "version": "1.0"
            },
            "dataset_overview": {
                "metadata": ingestion_results.get('metadata', {}),
                "quality_issues": ingestion_results.get('quality_issues', [])
            },
            "schema_analysis": {
                "schema": schema_results.get('schema', {}),
                "data_dictionary": schema_results.get('data_dictionary', {}),
                "validation_results": schema_results.get('validation_results', {})
            },
            "quality_validation": {
                "validation_report": validation_results.get('validation_report', {}),
                "quality_score": validation_results.get('quality_score', 0),
                "basic_checks": validation_results.get('basic_checks', {}),
                "schema_validation": validation_results.get('schema_validation', {}),
                "statistical_analysis": validation_results.get('statistical_analysis', {}),
                "outlier_analysis": validation_results.get('outlier_analysis', {}),
                "consistency_checks": validation_results.get('consistency_checks', {})
            },
            "agent_processing": {
                "ingestion_agent": ingestion_results.get('processing_summary', {}),
                "schema_agent": schema_results.get('processing_summary', {}),
                "validation_agent": validation_results.get('processing_summary', {}),
                "reporting_agent": self.get_processing_summary()
            },
            "mcp_context": {
                "ingestion": ingestion_results.get('mcp_context', {}),
                "schema": schema_results.get('mcp_context', {}),
                "validation": validation_results.get('mcp_context', {}),
                "reporting": self.mcp_client.export_context()
            }
        }
        
    def _generate_yaml_report(self, ingestion_results: Dict[str, Any],
                            schema_results: Dict[str, Any],
                            validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a YAML-formatted report."""
        
        # Convert JSON report to YAML format
        json_report = self._generate_json_report(ingestion_results, schema_results, validation_results)
        
        # Add YAML-specific formatting
        yaml_report = {
            "report_info": {
                "format": "yaml",
                "generated_at": json_report["report_metadata"]["generated_at"],
                "version": json_report["report_metadata"]["version"]
            },
            "summary": {
                "quality_score": validation_results.get('quality_score', 0),
                "total_issues": len(validation_results.get('validation_report', {}).get('total_issues', [])),
                "total_warnings": len(validation_results.get('validation_report', {}).get('total_warnings', [])),
                "dataset_size": {
                    "rows": ingestion_results.get('metadata', {}).get('rows', 0),
                    "columns": ingestion_results.get('metadata', {}).get('columns', 0)
                }
            },
            "detailed_results": json_report
        }
        
        return yaml_report
        
    def _generate_html_report(self, ingestion_results: Dict[str, Any],
                            schema_results: Dict[str, Any],
                            validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive HTML report."""
        
        quality_score = validation_results.get('quality_score', 0)
        quality_level = self._get_quality_level(quality_score)
        quality_color = self._get_quality_color(quality_score)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Data Quality Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .quality-score {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: {quality_color};
                    margin: 10px 0;
                }}
                .quality-level {{
                    font-size: 1.2em;
                    color: #666;
                    margin-bottom: 20px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .section h2 {{
                    color: #007bff;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    text-align: center;
                    min-width: 120px;
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: #666;
                }}
                .issue-list {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .warning-list {{
                    background-color: #d1ecf1;
                    border: 1px solid #bee5eb;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                .table th, .table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .table th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Data Quality Inspection Report</h1>
                    <div class="quality-score">{quality_score:.1f}/100</div>
                    <div class="quality-level">{quality_level}</div>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="metric">
                        <div class="metric-value">{ingestion_results.get('metadata', {}).get('rows', 0):,}</div>
                        <div class="metric-label">Rows</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{ingestion_results.get('metadata', {}).get('columns', 0)}</div>
                        <div class="metric-label">Columns</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{ingestion_results.get('metadata', {}).get('file_size', 0):,}</div>
                        <div class="metric-label">File Size (bytes)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{validation_results.get('validation_report', {}).get('issue_count', 0)}</div>
                        <div class="metric-label">Issues</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Quality Issues</h2>
                    {self._generate_issues_html(validation_results)}
                </div>
                
                <div class="section">
                    <h2>Schema Information</h2>
                    {self._generate_schema_html(schema_results)}
                </div>
                
                <div class="section">
                    <h2>Processing Summary</h2>
                    {self._generate_processing_html(ingestion_results, schema_results, validation_results)}
                </div>
                
                <div class="footer">
                    <p>Report generated by Data Quality Co-pilot</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    def _generate_issues_html(self, validation_results: Dict[str, Any]) -> str:
        """Generate HTML for issues section."""
        
        total_issues = validation_results.get('validation_report', {}).get('total_issues', [])
        total_warnings = validation_results.get('validation_report', {}).get('total_warnings', [])
        
        html = ""
        
        if total_issues:
            html += '<div class="issue-list">'
            html += '<h3>Critical Issues</h3>'
            html += '<ul>'
            for issue in total_issues[:10]:  # Show first 10 issues
                html += f'<li>{issue}</li>'
            if len(total_issues) > 10:
                html += f'<li>... and {len(total_issues) - 10} more issues</li>'
            html += '</ul>'
            html += '</div>'
            
        if total_warnings:
            html += '<div class="warning-list">'
            html += '<h3>Warnings</h3>'
            html += '<ul>'
            for warning in total_warnings[:10]:  # Show first 10 warnings
                html += f'<li>{warning}</li>'
            if len(total_warnings) > 10:
                html += f'<li>... and {len(total_warnings) - 10} more warnings</li>'
            html += '</ul>'
            html += '</div>'
            
        return html
        
    def _generate_schema_html(self, schema_results: Dict[str, Any]) -> str:
        """Generate HTML for schema section."""
        
        schema = schema_results.get('schema', {})
        columns = schema.get('columns', [])
        
        html = f'<p><strong>Dataset Name:</strong> {schema.get("dataset_name", "Unknown")}</p>'
        html += f'<p><strong>Description:</strong> {schema.get("description", "No description available")}</p>'
        
        if columns:
            html += '<table class="table">'
            html += '<thead><tr><th>Column</th><th>Type</th><th>Nullable</th><th>Description</th></tr></thead>'
            html += '<tbody>'
            
            for col in columns[:10]:  # Show first 10 columns
                html += f'<tr>'
                html += f'<td>{col.get("name", "Unknown")}</td>'
                html += f'<td>{col.get("data_type", "Unknown")}</td>'
                html += f'<td>{"Yes" if col.get("nullable", False) else "No"}</td>'
                html += f'<td>{col.get("description", "No description")[:50]}</td>'
                html += f'</tr>'
                
            html += '</tbody></table>'
            
            if len(columns) > 10:
                html += f'<p><em>... and {len(columns) - 10} more columns</em></p>'
                
        return html
        
    def _generate_processing_html(self, ingestion_results: Dict[str, Any],
                                schema_results: Dict[str, Any],
                                validation_results: Dict[str, Any]) -> str:
        """Generate HTML for processing summary section."""
        
        html = '<table class="table">'
        html += '<thead><tr><th>Agent</th><th>Duration</th><th>Thoughts</th><th>Status</th></tr></thead>'
        html += '<tbody>'
        
        agents_data = [
            ("Ingestion Agent", ingestion_results.get('processing_summary', {})),
            ("Schema Agent", schema_results.get('processing_summary', {})),
            ("Validation Agent", validation_results.get('processing_summary', {})),
            ("Reporting Agent", self.get_processing_summary())
        ]
        
        for agent_name, summary in agents_data:
            duration = summary.get('duration', 0)
            thoughts = summary.get('thoughts_count', 0)
            status = "Completed" if summary else "N/A"
            
            html += f'<tr>'
            html += f'<td>{agent_name}</td>'
            html += f'<td>{duration:.2f}s</td>'
            html += f'<td>{thoughts}</td>'
            html += f'<td>{status}</td>'
            html += f'</tr>'
            
        html += '</tbody></table>'
        
        return html
        
    def _generate_dashboard_data(self, ingestion_results: Dict[str, Any],
                               schema_results: Dict[str, Any],
                               validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for web dashboard."""
        
        return {
            "overview": {
                "quality_score": validation_results.get('quality_score', 0),
                "quality_level": self._get_quality_level(validation_results.get('quality_score', 0)),
                "rows": ingestion_results.get('metadata', {}).get('rows', 0),
                "columns": ingestion_results.get('metadata', {}).get('columns', 0),
                "file_size": ingestion_results.get('metadata', {}).get('file_size', 0)
            },
            "quality_metrics": {
                "issues_count": len(validation_results.get('validation_report', {}).get('total_issues', [])),
                "warnings_count": len(validation_results.get('validation_report', {}).get('total_warnings', [])),
                "outliers_count": validation_results.get('validation_report', {}).get('outlier_count', 0)
            },
            "schema_summary": {
                "dataset_name": schema_results.get('schema', {}).get('dataset_name', 'Unknown'),
                "columns_count": len(schema_results.get('schema', {}).get('columns', [])),
                "description": schema_results.get('schema', {}).get('description', 'No description')
            },
            "agent_thoughts": {
                "ingestion": ingestion_results.get('processing_summary', {}).get('thoughts', []),
                "schema": schema_results.get('processing_summary', {}).get('thoughts', []),
                "validation": validation_results.get('processing_summary', {}).get('thoughts', [])
            }
        }
        
    def _save_json_report(self, report: Dict[str, Any]) -> str:
        """Save JSON report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data_quality_report_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return filepath
        
    def _save_yaml_report(self, report: Dict[str, Any]) -> str:
        """Save YAML report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data_quality_report_{timestamp}.yaml"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
            
        return filepath
        
    def _save_html_report(self, report: str) -> str:
        """Save HTML report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data_quality_report_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
            
        return filepath
        
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
            
    def _get_quality_color(self, score: float) -> str:
        """Get color for quality score."""
        if score >= 90:
            return "#28a745"  # Green
        elif score >= 80:
            return "#17a2b8"  # Blue
        elif score >= 70:
            return "#ffc107"  # Yellow
        elif score >= 60:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red 
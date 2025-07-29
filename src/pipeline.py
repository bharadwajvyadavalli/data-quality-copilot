"""
Data Quality Pipeline Orchestrator
Coordinates all agents in the data quality inspection workflow.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .agents.ingestion_agent import IngestionAgent
from .agents.schema_agent import SchemaAgent
from .agents.validation_agent import ValidationAgent
from .agents.reporting_agent import ReportingAgent
from .mcp.client import MCPClient


class DataQualityPipeline:
    """Main pipeline orchestrator for data quality inspection."""
    
    def __init__(self, openai_api_key: Optional[str] = None, output_dir: str = "reports"):
        self.logger = logging.getLogger("pipeline")
        self.mcp_client = MCPClient()
        
        # Initialize agents
        self.ingestion_agent = IngestionAgent()
        self.schema_agent = SchemaAgent(openai_api_key)
        self.validation_agent = ValidationAgent()
        self.reporting_agent = ReportingAgent(output_dir)
        
        self.logger.info("Data Quality Pipeline initialized with all agents")
        
    def run_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        Run the complete data quality inspection pipeline.
        
        Args:
            file_path: Path to the data file to analyze
            
        Returns:
            Dictionary containing results from all agents
        """
        self.logger.info(f"Starting data quality pipeline for file: {file_path}")
        
        pipeline_start = datetime.now()
        results = {}
        
        try:
            # Step 1: Data Ingestion
            self.logger.info("Step 1: Running Ingestion Agent")
            ingestion_results = self.ingestion_agent.process({"file_path": file_path})
            results['ingestion_results'] = ingestion_results
            
            # Add to MCP context
            self.mcp_client.add_thought(
                "Pipeline",
                f"Ingestion completed: {ingestion_results['metadata']['rows']} rows, {ingestion_results['metadata']['columns']} columns",
                {"agent": "IngestionAgent", "rows": ingestion_results['metadata']['rows'], "columns": ingestion_results['metadata']['columns']}
            )
            
            # Step 2: Schema Generation
            self.logger.info("Step 2: Running Schema Agent")
            schema_results = self.schema_agent.process({
                "dataframe": ingestion_results['dataframe'],
                "metadata": ingestion_results['metadata']
            })
            results['schema_results'] = schema_results
            
            # Add to MCP context
            self.mcp_client.add_thought(
                "Pipeline",
                f"Schema generation completed: {len(schema_results['schema']['columns'])} columns analyzed",
                {"agent": "SchemaAgent", "columns": len(schema_results['schema']['columns'])}
            )
            
            # Step 3: Data Validation
            self.logger.info("Step 3: Running Validation Agent")
            validation_results = self.validation_agent.process({
                "dataframe": ingestion_results['dataframe'],
                "schema": schema_results['schema'],
                "data_dictionary": schema_results['data_dictionary']
            })
            results['validation_results'] = validation_results
            
            # Add to MCP context
            self.mcp_client.add_thought(
                "Pipeline",
                f"Validation completed: Quality score {validation_results['quality_score']:.2f}/100",
                {"agent": "ValidationAgent", "quality_score": validation_results['quality_score']}
            )
            
            # Step 4: Report Generation
            self.logger.info("Step 4: Running Reporting Agent")
            reporting_results = self.reporting_agent.process(results)
            results['reporting_results'] = reporting_results
            
            # Add to MCP context
            self.mcp_client.add_thought(
                "Pipeline",
                "Report generation completed: Console, JSON, YAML, and HTML reports created",
                {"agent": "ReportingAgent", "reports": ["console", "json", "yaml", "html"]}
            )
            
            # Calculate pipeline metrics
            pipeline_end = datetime.now()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()
            
            # Create pipeline summary
            pipeline_summary = {
                "file_path": file_path,
                "start_time": pipeline_start.isoformat(),
                "end_time": pipeline_end.isoformat(),
                "duration": pipeline_duration,
                "quality_score": validation_results['quality_score'],
                "total_issues": len(validation_results['validation_report']['total_issues']),
                "total_warnings": len(validation_results['validation_report']['total_warnings']),
                "agents_completed": ["IngestionAgent", "SchemaAgent", "ValidationAgent", "ReportingAgent"],
                "status": "success"
            }
            
            results['pipeline_summary'] = pipeline_summary
            results['mcp_context'] = self.mcp_client.export_context()
            
            self.logger.info(f"Pipeline completed successfully in {pipeline_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            
            # Create error summary
            pipeline_end = datetime.now()
            pipeline_duration = (pipeline_end - pipeline_start).total_seconds()
            
            error_summary = {
                "file_path": file_path,
                "start_time": pipeline_start.isoformat(),
                "end_time": pipeline_end.isoformat(),
                "duration": pipeline_duration,
                "error": str(e),
                "status": "failed"
            }
            
            results['pipeline_summary'] = error_summary
            results['error'] = str(e)
            
            # Add error to MCP context
            self.mcp_client.add_thought(
                "Pipeline",
                f"Pipeline failed with error: {str(e)}",
                {"error": True, "error_message": str(e)}
            )
            
        return results
        
    def get_agent_thoughts(self) -> Dict[str, Any]:
        """Get thoughts from all agents."""
        return {
            "ingestion_agent": self.ingestion_agent.get_thoughts(),
            "schema_agent": self.schema_agent.get_thoughts(),
            "validation_agent": self.validation_agent.get_thoughts(),
            "reporting_agent": self.reporting_agent.get_thoughts(),
            "pipeline": self.mcp_client.get_thoughts()
        }
        
    def get_processing_summaries(self) -> Dict[str, Any]:
        """Get processing summaries from all agents."""
        return {
            "ingestion_agent": self.ingestion_agent.get_processing_summary(),
            "schema_agent": self.schema_agent.get_processing_summary(),
            "validation_agent": self.validation_agent.get_processing_summary(),
            "reporting_agent": self.reporting_agent.get_processing_summary()
        }
        
    def clear_context(self):
        """Clear all agent contexts."""
        self.ingestion_agent.clear_thoughts()
        self.schema_agent.clear_thoughts()
        self.validation_agent.clear_thoughts()
        self.reporting_agent.clear_thoughts()
        self.mcp_client.clear_context()
        self.logger.info("All agent contexts cleared") 
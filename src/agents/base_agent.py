"""
Base Agent Class for Data Quality Co-pilot
Provides common functionality for all agents in the multi-agent architecture.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BaseAgent(ABC):
    """Base class for all agents in the data quality pipeline."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.thoughts = []
        self.start_time = None
        self.end_time = None
        
    def start_processing(self):
        """Mark the start of processing."""
        self.start_time = datetime.now()
        self.logger.info(f"{self.name} started processing")
        
    def end_processing(self):
        """Mark the end of processing."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"{self.name} completed processing in {duration:.2f}s")
        
    def add_thought(self, thought: str, data: Optional[Dict[str, Any]] = None):
        """Add a thought/step to the agent's processing log."""
        thought_entry = {
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "data": data or {}
        }
        self.thoughts.append(thought_entry)
        self.logger.info(f"Thought: {thought}")
        
    def get_thoughts(self) -> list:
        """Get all thoughts from this agent."""
        return self.thoughts
        
    def clear_thoughts(self):
        """Clear all thoughts."""
        self.thoughts = []
        
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and return results.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary containing processing results
        """
        pass
        
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's processing."""
        return {
            "agent_name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "thoughts_count": len(self.thoughts),
            "thoughts": self.thoughts
        } 
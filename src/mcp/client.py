"""
Model Context Protocol (MCP) Client
Handles prompt chunking, tool call injection, and thought collection for agent communication.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class MCPClient:
    """MCP-compatible client for managing agent communication and context."""
    
    def __init__(self):
        self.logger = logging.getLogger("mcp.client")
        self.context_history = []
        self.tool_calls = []
        self.thoughts = []
        
    def chunk_prompt(self, prompt: str, max_chunk_size: int = 4000) -> List[str]:
        """
        Chunk a large prompt into smaller pieces for better context management.
        
        Args:
            prompt: The full prompt to chunk
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of prompt chunks
        """
        if len(prompt) <= max_chunk_size:
            return [prompt]
            
        chunks = []
        current_chunk = ""
        
        # Split by sentences to maintain context
        sentences = prompt.split('. ')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        self.logger.info(f"Chunked prompt into {len(chunks)} pieces")
        return chunks
        
    def inject_tool_call(self, tool_name: str, parameters: Dict[str, Any], 
                        description: str = "") -> Dict[str, Any]:
        """
        Inject a tool call into the context for agent use.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool
            description: Description of what the tool does
            
        Returns:
            Tool call dictionary
        """
        tool_call = {
            "id": f"tool_{len(self.tool_calls) + 1}",
            "name": tool_name,
            "description": description,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }
        
        self.tool_calls.append(tool_call)
        self.logger.info(f"Injected tool call: {tool_name}")
        
        return tool_call
        
    def add_thought(self, agent_name: str, thought: str, 
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a thought to the MCP context.
        
        Args:
            agent_name: Name of the agent generating the thought
            thought: The thought content
            context: Additional context data
            
        Returns:
            Thought entry dictionary
        """
        thought_entry = {
            "id": f"thought_{len(self.thoughts) + 1}",
            "agent": agent_name,
            "thought": thought,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.thoughts.append(thought_entry)
        self.logger.info(f"Added thought from {agent_name}: {thought[:100]}...")
        
        return thought_entry
        
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current MCP context.
        
        Returns:
            Context summary dictionary
        """
        return {
            "total_thoughts": len(self.thoughts),
            "total_tool_calls": len(self.tool_calls),
            "context_history_length": len(self.context_history),
            "latest_thought": self.thoughts[-1] if self.thoughts else None,
            "latest_tool_call": self.tool_calls[-1] if self.tool_calls else None
        }
        
    def create_agent_prompt(self, agent_name: str, task: str, 
                           relevant_context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Create a prompt for an agent with relevant context and tool calls.
        
        Args:
            agent_name: Name of the agent
            task: Task description
            relevant_context: Relevant context from previous thoughts
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"You are the {agent_name} in a data quality inspection pipeline.",
            f"Your task: {task}",
            "\nRelevant context from previous agents:"
        ]
        
        if relevant_context:
            for ctx in relevant_context[-5:]:  # Last 5 relevant thoughts
                prompt_parts.append(f"- {ctx['agent']}: {ctx['thought']}")
        else:
            prompt_parts.append("- No previous context available")
            
        prompt_parts.append("\nAvailable tools:")
        for tool in self.tool_calls[-3:]:  # Last 3 tool calls
            prompt_parts.append(f"- {tool['name']}: {tool['description']}")
            
        prompt_parts.append("\nPlease proceed with your analysis and provide your thoughts.")
        
        return "\n".join(prompt_parts)
        
    def exchange_mcp_message(self, agent_name: str, message: str, 
                           response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record an MCP message exchange between agents.
        
        Args:
            agent_name: Name of the agent
            message: Input message
            response: Agent response
            metadata: Additional metadata
            
        Returns:
            Exchange record
        """
        exchange = {
            "id": f"exchange_{len(self.context_history) + 1}",
            "agent": agent_name,
            "message": message,
            "response": response,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.context_history.append(exchange)
        self.logger.info(f"Recorded MCP exchange with {agent_name}")
        
        return exchange
        
    def get_agent_context(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get context relevant to a specific agent.
        
        Args:
            agent_name: Name of the agent
            limit: Maximum number of context items to return
            
        Returns:
            List of relevant context items
        """
        relevant_context = []
        
        # Get thoughts from this agent
        agent_thoughts = [t for t in self.thoughts if t['agent'] == agent_name]
        relevant_context.extend(agent_thoughts[-limit//2:])
        
        # Get recent exchanges involving this agent
        agent_exchanges = [e for e in self.context_history if e['agent'] == agent_name]
        relevant_context.extend(agent_exchanges[-limit//2:])
        
        return sorted(relevant_context, key=lambda x: x['timestamp'])[-limit:]
        
    def clear_context(self):
        """Clear all context history."""
        self.context_history = []
        self.tool_calls = []
        self.thoughts = []
        self.logger.info("Cleared all MCP context")
        
    def export_context(self) -> Dict[str, Any]:
        """
        Export the current MCP context for persistence or analysis.
        
        Returns:
            Complete context dictionary
        """
        return {
            "context_history": self.context_history,
            "tool_calls": self.tool_calls,
            "thoughts": self.thoughts,
            "summary": self.get_context_summary(),
            "export_timestamp": datetime.now().isoformat()
        } 
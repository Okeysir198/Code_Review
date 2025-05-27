# ./src/Agents/call_center_agent/utility_agents.py
"""
Utility Agents - Query Resolution, Cancellation, and Escalation handlers.
"""
from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    add_client_note,
    save_call_disposition,
    get_client_contracts
)


def create_query_resolution_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False
) -> CompiledGraph:
    """Create a query resolution agent for handling client questions."""
    
    agent_tools = [add_client_note]
    if tools:
        agent_tools.extend(tools)
    
    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Post-process to log query resolution and reset query flag."""
        
        try:
            # Add note about query handled
            add_client_note.invoke({
                "user_id": user_id,
                "note_text": f"Client query resolved during {state.current_step} step"
            })
            
            return {
                "query_detected": False,  # Reset query flag
                "call_info": {"query_resolution": "completed"}
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in query resolution post-processing: {e}")
            return {"query_detected": False, "call_info": {"query_resolution": "error"}}

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
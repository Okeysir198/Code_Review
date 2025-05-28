# ./src/Agents/call_center_agent/step11_further_assistance.py
"""
Further Assistance Agent - Final check for additional concerns.
SIMPLIFIED: No query detection - router handles all routing decisions.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import add_client_note


def create_further_assistance_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a further assistance agent for debt collection calls."""
    
    agent_tools = [add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to check for additional client concerns."""
        
        # Check if client has additional questions or concerns
        recent_messages = state.get("messages", [])[-2:] if state.get("messages") else []
        
        # Look for client responses indicating they have questions
        has_concerns = False
        for msg in recent_messages:
            if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "human":
                content = msg.content.lower()
                concern_indicators = ["yes", "actually", "one more", "also", "question", "problem"]
                if any(indicator in content for indicator in concern_indicators):
                    has_concerns = True
                    break
        
        return Command(
            update={
                "has_concerns": has_concerns,
                "current_step": CallStep.FURTHER_ASSISTANCE.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for further assistance step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.FURTHER_ASSISTANCE.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.FURTHER_ASSISTANCE.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="FurtherAssistanceAgent"
    )
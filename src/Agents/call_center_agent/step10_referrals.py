# ./src/Agents/call_center_agent/referrals_agent.py
"""
Referrals Agent - Briefly mentions referral program.
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


def create_referrals_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a referrals agent for debt collection calls."""
    
    agent_tools = [add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["__end__", "agent"]]:
        """Pre-process to check for queries and prepare referral info."""
        
        # Check for queries first
        recent_messages = state.get("messages", [])[-2:] if state.get("messages") else []
        query_detected = False
        
        for msg in recent_messages:
            if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "human":
                content = msg.content.lower()
                query_indicators = ["why", "what", "how", "when", "who", "where", "?"]
                if any(indicator in content for indicator in query_indicators):
                    query_detected = True
                    break
        
        if query_detected:
            return Command(
                update={
                    "query_detected": True,
                    "current_step": CallStep.QUERY_RESOLUTION.value,
                    "return_to_step": CallStep.REFERRALS.value
                },
                goto="__end__"
            )
        
        return Command(
            update={"current_step": CallStep.REFERRALS.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for referrals step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.REFERRALS.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.REFERRALS.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="ReferralsAgent"
    )
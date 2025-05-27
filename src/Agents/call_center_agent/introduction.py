# ./src/Agents/call_center_agent/introduction.py
"""
Introduction Agent for Call Center.
"""
from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.call_scripts import ScriptType



def create_introduction_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = ScriptType.RATIO_1_INFLOW.value,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False
) -> CompiledGraph:
    """
    Create an introduction agent for debt collection calls.
    
    Args:
        model: Language model to use
        client_data: client information
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled introduction agent workflow
    """
    
    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for introduction step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.INTRODUCTION.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.INTRODUCTION.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="IntroductionAgent"
    )
# ./src/Agents/call_center_agent/introduction.py
"""
Introduction Agent for Call Center.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.graph import END
from langgraph.types import Command

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
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create an introduction agent for debt collection calls."""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["__end__", "agent"]]:
        client_full_name = client_data["profile"]["client_info"]['client_full_name']
        client_title = client_data["profile"]["client_info"]['title']

        messages = HumanMessage(content=f"Good day, you are speaking to {agent_name} from Cartrack Accounts Department. May I speak to {client_title} {client_full_name}, please?")
        
        return Command(
            # state update
            update={
                "messages": [messages],
                "current_step": CallStep.INTRODUCTION.value,
                "next_step": CallStep.NAME_VERIFICATION.value,
                },
            goto="__end__"
        )


    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for introduction step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.INTRODUCTION.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else {},
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.INTRODUCTION.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    # Configure basic agent
    kwargs = {
        "model": model,
        "prompt": dynamic_prompt,
        "tools": tools or [],
        "pre_processing_node": pre_processing_node,
        "state_schema": CallCenterAgentState,
        "verbose": verbose,
        "name": "IntroductionAgent"
    }
    
    # Add memory if configured
    if config and config.get('configurable', {}).get('use_memory'):
        from langgraph.checkpoint.memory import MemorySaver
        kwargs["checkpointer"] = MemorySaver()
    
    return create_basic_agent(**kwargs)
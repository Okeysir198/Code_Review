# src/Agents/call_center_agent/step00_introduction.py
"""
Introduction Agent - Self-contained with own prompt
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value

def get_introduction_prompt(client_data: Dict[str, Any], agent_name: str) -> str:
    """Generate introduction specific prompt."""
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    client_title = get_safe_value(client_data, "profile.client_info.title", "Mr/Ms")
    
    return f"""<role>
You are {agent_name}, a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<task>
Deliver professional greeting and request specific client. MAXIMUM 15 words.
</task>

<response>
"Good day, you are speaking to {agent_name} from Cartrack Accounts Department. May I speak to {client_title} {client_full_name}, please?"
</response>

<style>
- Professional and confident
- Clear company identification
- Direct request for specific person
- MAXIMUM 15 words
</style>"""

def create_introduction_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create an introduction agent for debt collection calls."""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
        client_title = get_safe_value(client_data, "profile.client_info.title", "Mr/Ms")

        messages = AIMessage(content=f"Good day, you are speaking to {agent_name} from Cartrack Accounts Department. May I speak to {client_title} {client_full_name}, please?")
        
        return Command(
            update={
                "messages": [messages],
                "current_step": CallStep.NAME_VERIFICATION.value,
            },
            goto="__end__"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for introduction step."""
        prompt_content = get_introduction_prompt(client_data, agent_name)
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
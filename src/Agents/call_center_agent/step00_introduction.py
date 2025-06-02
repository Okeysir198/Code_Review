# ===============================================================================
# STEP 00: INTRODUCTION AGENT - Updated with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step00_introduction.py
"""
Introduction Agent - Enhanced with aging-aware script integration
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
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

def get_introduction_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware introduction prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    
    # Get client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    client_title = get_safe_value(client_data, "profile.client_info.title", "Mr/Ms")
    
    # Base prompt
    base_prompt = """<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
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
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.INTRODUCTION,
        client_data=client_data,
        state=state or {}
    )

def create_introduction_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create an introduction agent with aging-aware scripts."""
    
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
        prompt_content = get_introduction_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    kwargs = {
        "model": model,
        "prompt": dynamic_prompt,
        "tools": tools or [],
        "pre_processing_node": pre_processing_node,
        "state_schema": CallCenterAgentState,
        "verbose": verbose,
        "name": "IntroductionAgent"
    }
    
    if config and config.get('configurable', {}).get('use_memory'):
        from langgraph.checkpoint.memory import MemorySaver
        kwargs["checkpointer"] = MemorySaver()
    
    return create_basic_agent(**kwargs)
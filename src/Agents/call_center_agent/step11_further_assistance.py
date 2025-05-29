# src/Agents/call_center_agent/step11_further_assistance.py
"""
Further Assistance Agent - Self-contained with own prompt
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import add_client_note

def get_further_assistance_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate further assistance specific prompt."""
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Check for other concerns. MAXIMUM 15 words.
</task>

<approach>
"Is there anything else regarding your account I can help you with today?"
</approach>

<follow_up_options>
- Account questions
- Service issues
- Payment queries
- General assistance
</follow_up_options>

<style>
- MAXIMUM 15 words
- Genuine concern for client needs
- Complete resolution focus
- Professional care
</style>"""

def create_further_assistance_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a further assistance agent."""
    
    agent_tools = [add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to check for additional client concerns."""
        
        return Command(
            update={
                "assistance_offer": "Is there anything else regarding your account I can help you with?",
                "current_step": CallStep.FURTHER_ASSISTANCE.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_further_assistance_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="FurtherAssistanceAgent"
    )
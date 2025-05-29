# src/Agents/call_center_agent/step09_client_details_update.py
"""
Client Details Update Agent - Self-contained with own prompt
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    update_client_contact_number,
    update_client_email,
    add_client_note
)

def get_client_details_update_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate client details update specific prompt."""
    # Extract current contact information
    current_mobile = get_safe_value(client_data, "profile.client_info.contact.mobile", "")
    current_email = get_safe_value(client_data, "profile.client_info.email_address", "")
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Update client contact information. MAXIMUM 20 words.
</task>

<approach>
"As part of standard account maintenance, let me verify your contact details."
</approach>

<current_details>
- Mobile: {current_mobile}
- Email: {current_email}
</current_details>

<verification_process>
1. "Can you confirm your mobile number?"
2. "And your email address?"
3. "Thank you, I've updated your details"
</verification_process>

<style>
- MAXIMUM 20 words
- Position as beneficial service
- Be efficient but thorough
- Professional routine maintenance
</style>"""

def create_client_details_update_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a client details update agent."""
    
    agent_tools = [update_client_contact_number, update_client_email, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to identify what details need updating."""
        
        # Extract current contact information
        current_mobile = get_safe_value(client_data, "profile.client_info.contact.mobile", "")
        current_email = get_safe_value(client_data, "profile.client_info.email_address", "")
        
        return Command(
            update={
                "current_mobile": current_mobile,
                "current_email": current_email,
                "update_request": "Let me verify your contact details",
                "current_step": CallStep.CLIENT_DETAILS_UPDATE.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_client_details_update_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ClientDetailsUpdateAgent"
    )
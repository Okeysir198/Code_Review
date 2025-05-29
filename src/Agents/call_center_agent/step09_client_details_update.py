# src/Agents/call_center_agent/step09_client_details_update.py
"""
Client Details Update Agent - Enhanced with aging-aware script integration
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
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

from src.Database.CartrackSQLDatabase import (
    update_client_contact_number,
    update_client_email,
    add_client_note
)

def get_client_details_update_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate aging-aware client details update prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Extract current contact information
    current_mobile = get_safe_value(client_data, "profile.client_info.contact.mobile", "")
    current_email = get_safe_value(client_data, "profile.client_info.email_address", "")
    
    # Build aging-specific approach to updates
    update_approaches_by_category = {
        "First Missed Payment": {
            "justification": "As part of standard account maintenance",
            "tone": "routine and helpful",
            "urgency": "to ensure you receive important account notifications"
        },
        "Failed Promise to Pay": {
            "justification": "To ensure reliable communication for payment arrangements",
            "tone": "solution-focused",
            "urgency": "so we can coordinate future payments effectively"
        },
        "2-3 Months Overdue": {
            "justification": "For critical account communications",
            "tone": "professional and direct",
            "urgency": "to ensure you receive urgent account updates"
        },
        "Pre-Legal 120+ Days": {
            "justification": "For essential legal correspondence",
            "tone": "formal and necessary",
            "urgency": "to ensure you receive all legal notifications"
        },
        "Legal 150+ Days": {
            "justification": "Required for legal proceedings",
            "tone": "mandatory compliance",
            "urgency": "for proper legal service of documents"
        }
    }
    
    category = aging_context['category']
    approach = update_approaches_by_category.get(category, update_approaches_by_category["First Missed Payment"])
    
    # Build urgency-appropriate verification process
    verification_processes_by_urgency = {
        "Medium": [
            "Can you confirm your mobile number?",
            "And your email address?",
            "Thank you, I've updated your details"
        ],
        "High": [
            "I need to verify your mobile number for urgent communications",
            "And confirm your email address",
            "Details updated - you'll receive important account notifications"
        ],
        "Very High": [
            "I must verify your current mobile number for critical updates",
            "And your current email address",
            "Updated - essential you receive all account communications"
        ],
        "Critical": [
            "Legal requirement to verify your current mobile number",
            "And current email address for legal correspondence",
            "Details updated for legal compliance"
        ]
    }
    
    urgency_level = aging_context['urgency']
    verification_process = verification_processes_by_urgency.get(urgency_level, verification_processes_by_urgency["Medium"])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<context>
- Current Mobile: {current_mobile}
- Current Email: {current_email}
- Aging Category: {category}
- Urgency Level: {urgency_level}
</context>

<task>
Update client contact information using aging-appropriate justification and urgency.
</task>

<aging_specific_approach>
**Justification**: "{approach['justification']}"
**Purpose**: "{approach['urgency']}"
**Tone**: "{approach['tone']}"
</aging_specific_approach>

<verification_process>
{chr(10).join([f"{i+1}. \"{step}\"" for i, step in enumerate(verification_process)])}
</verification_process>

<urgency_context>
{aging_context['approach']}
</urgency_context>

<positioning_strategy>
- Position as beneficial service appropriate to account status
- Frame as {approach['tone']} maintenance
- Emphasize {approach['urgency']}
- Be efficient but thorough
</positioning_strategy>

<style>
- {aging_context['tone']}
- Professional {approach['tone']}
- Appropriate urgency for {urgency_level.lower()} priority account
- Clear benefit messaging
- Efficient process
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.CLIENT_DETAILS_UPDATE,
        client_data=client_data,
        state=state
    )

def create_client_details_update_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a client details update agent with aging-aware scripts."""
    
    agent_tools = [update_client_contact_number, update_client_email, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Extract current contact information
        current_mobile = get_safe_value(client_data, "profile.client_info.contact.mobile", "")
        current_email = get_safe_value(client_data, "profile.client_info.email_address", "")
        
        # Determine script type and urgency
        account_aging = client_data.get("account_aging", {})
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        return Command(
            update={
                "current_mobile": current_mobile,
                "current_email": current_email,
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
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
# ===============================================================================
# STEP 03: REASON FOR CALL AGENT - Updated with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step03_reason_for_call.py
"""
Reason for Call Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value, calculate_outstanding_amount, format_currency
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

def get_reason_for_call_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware reason for call prompt."""
    
    # Determine script type from aging
    user_id = get_safe_value(client_data, "profile.user_id", "")     
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    
    # Calculate outstanding amount
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Get account status
    account_overview = client_data.get("account_overview", {})
    account_status = account_overview.get("account_status", "Overdue") if account_overview else "Overdue"
    
    # Build aging-specific approach
    aging_approaches = {
        "First Missed Payment": f"We didn't receive your subscription payment. Your account is overdue by {outstanding_amount}. Can we debit this today?",
        "Failed Promise to Pay": f"We didn't receive your payment of {outstanding_amount} as arranged. Your account remains overdue.",
        "New Installation Pro-Rata": f"We haven't received your pro-rata payment since fitment. Services activate once we receive {outstanding_amount}.",
        "2-3 Months Overdue": f"Your account is overdue for over 2 months. An immediate payment of {outstanding_amount} is required.",
        "2-3 Months Failed PTP": f"We didn't receive your payment of {outstanding_amount} as arranged. Your account remains seriously overdue.",
        "Pre-Legal 120+ Days": f"Your account is 4+ months overdue and in our pre-legal department. A letter of demand was sent. Immediate payment of {outstanding_amount} is required.",
        "Legal 150+ Days": f"Cartrack has handed your account to us as attorneys. Your arrears are {outstanding_amount}. Do you acknowledge this debt?"
    }
    
    category = aging_context['category']
    specific_approach = aging_approaches.get(category, aging_approaches["First Missed Payment"])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
</role>

<client_context>
- Client VERIFIED: {client_full_name}
- Outstanding Amount: {outstanding_amount}
- Account Status: {account_status}
- Aging Category: {category}
- Urgency Level: {aging_context['urgency']}
- Client user_id: {user_id}
</client_context>

<task>
Clearly communicate account status and required payment using aging-appropriate approach.
</task>

<aging_specific_approach>
{specific_approach}
</aging_specific_approach>

<communication_strategy>
1. **State status directly**: Clear, factual account status appropriate to aging
2. **Specify amount**: Exact outstanding amount
3. **Create appropriate urgency**: Match tone to account severity
4. **Request immediate action**: Direct payment request
</communication_strategy>

<consequences_to_emphasize>
{aging_context['consequences']}
</consequences_to_emphasize>

<tone_guidance>
{aging_context['tone']} - {aging_context['approach']}
</tone_guidance>

<style>
- Adapt formality to account severity
- State amount clearly without hesitation
- Create urgency matching account status
- Professional but {aging_context['urgency'].lower()} priority
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.REASON_FOR_CALL,
        client_data=client_data,
        state=state
    )

def create_reason_for_call_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a reason for call agent with aging-aware scripts."""
    
  
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        account_aging = client_data.get("account_aging", {})
        account_overview = client_data.get("account_overview", {})
        
        outstanding_amount = calculate_outstanding_amount(account_aging)
        outstanding_formatted = format_currency(outstanding_amount)
        
        account_status = account_overview.get("account_status", "Overdue") if account_overview else "Overdue"
        
        # Determine script type for urgency level
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        urgency_level = aging_context['urgency'].lower()
        
        return Command(
            update={
                "outstanding_amount": outstanding_formatted,
                "account_status": account_status,
                "urgency_level": urgency_level,
                "outstanding_float": outstanding_amount,
                "aging_category": aging_context['category'],
                "current_step": CallStep.REASON_FOR_CALL.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_reason_for_call_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(f"Prompt: {prompt_content}")
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ReasonForCallAgent"
    )

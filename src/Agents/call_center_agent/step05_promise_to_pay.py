# src/Agents/call_center_agent/step05_promise_to_pay.py
"""
Promise to Pay Agent - Enhanced with aging-aware script integration
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

from src.Database.CartrackSQLDatabase import (
    get_client_banking_details,
    add_client_note
)

def get_promise_to_pay_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware promise to pay prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    client_name = get_safe_value(client_data, "profile.client_info.first_name", "Client")
    
    # Calculate outstanding amount
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    amount_with_fee = format_currency(outstanding_float + 10)
    
    # Extract state info
    payment_willingness = state.get("payment_willingness", "unknown")
    has_banking_details = len(client_data.get("banking_details", {})) > 0
    
    # Build aging-specific payment approaches
    payment_approaches_by_category = {
        "First Missed Payment": {
            "primary": f"Can we debit {outstanding_amount} from your account today?",
            "debicheck": f"I'll set up secure bank payment. Total {amount_with_fee} including R10 processing fee",
            "portal": f"I'm sending you a payment link. You can pay {outstanding_amount} while we're talking"
        },
        "Failed Promise to Pay": {
            "primary": f"You previously agreed to pay. Can we debit {outstanding_amount} immediately to honor that commitment?",
            "debicheck": f"Let's secure this with bank authentication. Total {amount_with_fee} including processing fee",
            "portal": f"I'm sending an immediate payment link for {outstanding_amount}. Please complete now"
        },
        "2-3 Months Overdue": {
            "primary": f"To prevent legal action, can we debit {outstanding_amount} from your account today?",
            "debicheck": f"Secure bank payment prevents escalation. Total {amount_with_fee} including R10 fee",
            "portal": f"Immediate payment link for {outstanding_amount} to stop account escalation"
        },
        "Pre-Legal 120+ Days": {
            "primary": f"To prevent court proceedings, we need {outstanding_amount} debited immediately",
            "debicheck": f"Final opportunity - secure payment of {amount_with_fee} to prevent legal action",
            "portal": f"Emergency payment link for {outstanding_amount} - must be completed today"
        },
        "Legal 150+ Days": {
            "primary": f"Legal demand for {outstanding_amount}. Authorize immediate debit to prevent court judgment",
            "debicheck": f"Court proceedings stopped only with {amount_with_fee} authenticated payment",
            "portal": f"Final legal payment opportunity - {outstanding_amount} required immediately"
        }
    }
    
    category = aging_context['category']
    approaches = payment_approaches_by_category.get(category, payment_approaches_by_category["First Missed Payment"])
    
    # Build urgency-based persistence levels
    persistence_by_urgency = {
        "Medium": "Professional persistence - offer alternatives if initial approach declined",
        "High": "Firm persistence - emphasize consequences of delay, limited alternatives",
        "Very High": "Urgent persistence - stress immediate action required, final opportunity messaging",
        "Critical": "Legal persistence - payment demanded, no alternatives without legal consequences"
    }
    
    urgency_level = aging_context['urgency']
    persistence_level = persistence_by_urgency.get(urgency_level, persistence_by_urgency["Medium"])
    
    # Base prompt
    base_prompt = f"""<role>
You are {agent_name}, a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<client_context>
- Client: {client_full_name}
- Outstanding: {outstanding_amount}
- Payment Willingness: {payment_willingness}
- Has Banking Details: {has_banking_details}
- Aging Category: {category}
- Urgency Level: {urgency_level}
</client_context>

<task>
Secure payment arrangement using aging-appropriate urgency and persistence.
</task>

<aging_specific_payment_hierarchy>
1. Primary Approach: "{approaches['primary']}"
2. DebiCheck Setup: "{approaches['debicheck']}"
3. Payment Portal: "{approaches['portal']}"
</aging_specific_payment_hierarchy>

<persistence_strategy>
{persistence_level}
</persistence_strategy>

<no_exit_rule>
Must secure SOME arrangement before ending. Urgency level determines flexibility:
- Medium: Multiple options, flexible timing
- High: Limited options, immediate timing preferred
- Very High: Immediate action required, minimal alternatives
- Critical: Payment demanded now, no delay acceptable
</no_exit_rule>

<urgency_adaptation>
{aging_context['approach']}
</urgency_adaptation>

<consequences_for_delay>
{aging_context['consequences']}
</consequences_for_delay>

<style>
- {aging_context['tone']}
- Assume they'll pay (positive framing appropriate to urgency)
- Direct questions requiring yes/no answers
- Persistence matching account severity
- Professional but {urgency_level.lower()} priority
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.PROMISE_TO_PAY,
        client_data=client_data,
        state=state
    )

def create_promise_to_pay_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a promise to pay agent with aging-aware scripts."""
    
    agent_tools = [get_client_banking_details, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Get banking details availability
        banking_details = client_data.get('banking_details', {})
        has_banking_details = bool(banking_details and len(banking_details) > 0)
        
        # Get outstanding amount for calculations
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        
        # Determine script type and recommended approach
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        # Determine recommended approach based on urgency and banking details
        if aging_context['urgency'] in ['Very High', 'Critical']:
            recommended_approach = "immediate_debit"  # Most urgent accounts need immediate action
        elif has_banking_details:
            recommended_approach = "immediate_debit"
        else:
            recommended_approach = "payment_portal"
        
        return Command(
            update={
                "has_banking_details": has_banking_details,
                "outstanding_float": outstanding_amount,
                "recommended_approach": recommended_approach,
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
                "current_step": CallStep.PROMISE_TO_PAY.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_promise_to_pay_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(f"Prompt: {prompt_content}")
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="PromiseToPayAgent"
    )
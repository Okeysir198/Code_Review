# src/Agents/call_center_agent/step05_promise_to_pay.py
"""
Promise to Pay Agent - Self-contained with own prompt
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

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    get_client_banking_details,
    add_client_note
)

def get_promise_to_pay_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate promise to pay specific prompt."""
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    client_name = get_safe_value(client_data, "profile.client_info.first_name", "Client")
    
    # Calculate outstanding amount
    account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    amount_with_fee = format_currency(outstanding_float + 10)
    
    # Extract state info
    payment_willingness = state.get("payment_willingness", "unknown")
    has_banking_details = len(client_data.get("banking_details", {})) > 0
    
    return f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<client_context>
- Client: {client_full_name}
- Outstanding: {outstanding_amount}
- Payment Willingness: {payment_willingness}
- Has Banking Details: {has_banking_details}
</client_context>

<task>
Secure payment arrangement. Try immediate debit first, then alternatives. MAXIMUM 20 words.
</task>

<payment_hierarchy>
1. "Can we debit {outstanding_amount} from your account today?"
2. "I'll set up secure bank payment. Total {amount_with_fee} including R10 processing fee"
3. "I'm sending you a payment link. You can pay while we're talking"
</payment_hierarchy>

<no_exit_rule>
Must secure SOME arrangement before ending. Keep offering alternatives.
</no_exit_rule>

<style>
- MAXIMUM 20 words per response
- Assume they'll pay (positive framing)
- Direct questions requiring yes/no answers
- Professional persistence
</style>"""

def create_promise_to_pay_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a promise to pay agent."""
    
    agent_tools = [get_client_banking_details, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare payment context."""
        
        # Get banking details availability
        banking_details = client_data.get('banking_details', {})
        has_banking_details = bool(banking_details and len(banking_details) > 0)
        
        # Get outstanding amount for calculations
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        
        # Determine recommended approach
        if has_banking_details:
            recommended_approach = "immediate_debit"
        else:
            recommended_approach = "payment_portal"
        
        return Command(
            update={
                "has_banking_details": has_banking_details,
                "outstanding_float": outstanding_amount,
                "recommended_approach": recommended_approach,
                "current_step": CallStep.PROMISE_TO_PAY.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for promise to pay step."""
        prompt_content = get_promise_to_pay_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
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
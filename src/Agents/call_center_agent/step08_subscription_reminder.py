# src/Agents/call_center_agent/step08_subscription_reminder.py
"""
Subscription Reminder Agent - Self-contained with own prompt
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import calculate_outstanding_amount, format_currency

from src.Database.CartrackSQLDatabase import get_client_subscription_amount, add_client_note

def get_subscription_reminder_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate subscription reminder specific prompt."""
    # Calculate outstanding amount
    account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Get subscription amount
    subscription = client_data.get("subscription", {})
    subscription_amount = subscription.get("subscription_amount", "199.00")
    try:
        subscription_str = format_currency(float(subscription_amount))
    except (ValueError, TypeError):
        subscription_str = "R 199.00"
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Clarify arrears vs ongoing subscription. MAXIMUM 20 words.
</task>

<message>
"Today's {outstanding_amount} covers arrears. Your regular {subscription_str} continues on the 5th of each month."
</message>

<clarification>
- Today's payment = overdue arrears
- Regular subscription = ongoing monthly payment
- Two separate things
- Prevent confusion about double charging
</clarification>

<style>
- MAXIMUM 20 words
- Clear differentiation
- Prevent double-payment confusion
- Professional explanation
</style>"""

def create_subscription_reminder_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a subscription reminder agent."""
    
    agent_tools = [get_client_subscription_amount, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare subscription information."""
        
        # Get subscription information
        subscription = client_data.get("subscription", {})
        subscription_amount = subscription.get("subscription_amount", "199.00")
        
        try:
            subscription_str = format_currency(float(subscription_amount))
        except (ValueError, TypeError):
            subscription_str = "R 199.00"
        
        return Command(
            update={
                "subscription_amount": subscription_str,
                "subscription_date": "5th of each month",
                "clarification_message": "Today's payment covers arrears, regular subscription continues",
                "current_step": CallStep.SUBSCRIPTION_REMINDER.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_subscription_reminder_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="SubscriptionReminderAgent"
    )
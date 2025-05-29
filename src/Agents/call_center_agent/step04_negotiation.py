# src/Agents/call_center_agent/step04_negotiation.py
"""
Negotiation Agent - Self-contained with own prompt
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
    get_client_payment_history,
    get_client_failed_payments,
    add_client_note
)

def get_negotiation_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate negotiation specific prompt."""
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    
    # Calculate outstanding amount
    account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    return f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<client_context>
- Client: {client_full_name}
- Outstanding: {outstanding_amount}
- Account Status: Overdue
</client_context>

<task>
Handle objections and explain consequences. MAXIMUM 20 words per response.
</task>

<consequences_framework>
**Without Payment**: "Your tracking stops working and you lose vehicle security"
**With Payment**: "Pay now and everything works immediately"
</consequences_framework>

<objection_responses>
- "No money": "I understand. What amount can you manage today to keep services active?"
- "Dispute amount": "Let's verify while arranging payment to prevent service suspension"
- "Will pay later": "Services suspend today without payment. Can we arrange something now?"
- "Already paid": "When was this paid? I need to locate it and arrange immediate payment"
</objection_responses>

<style>
- MAXIMUM 20 words per response
- Natural, conversational tone
- Focus on solutions, not problems
- Create urgency through benefits
</style>"""

def create_negotiation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a negotiation agent."""
    
    agent_tools = [
        get_client_payment_history,
        get_client_failed_payments,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare negotiation context."""
        
        # Get outstanding amount
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        
        # Determine urgency level based on amount
        urgency_level = "standard"
        if outstanding_amount > 1000:
            urgency_level = "high"
        elif outstanding_amount > 500:
            urgency_level = "medium"
        
        return Command(
            update={
                "outstanding_float": outstanding_amount,
                "urgency_level": urgency_level,
                "current_step": CallStep.NEGOTIATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for negotiation step."""
        prompt_content = get_negotiation_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="NegotiationAgent"
    )
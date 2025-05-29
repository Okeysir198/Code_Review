# src/Agents/call_center_agent/step03_reason_for_call.py
"""
Reason for Call Agent - Self-contained with own prompt
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
    get_client_account_aging,
    get_client_account_overview,
    add_client_note
)

def get_reason_for_call_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate reason for call specific prompt."""
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    
    # Calculate outstanding amount
    account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Get account status
    account_overview = client_data.get("account_overview", {})
    account_status = account_overview.get("account_status", "Overdue") if account_overview else "Overdue"
    
    return f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<client_context>
- Client VERIFIED: {client_full_name}
- Outstanding Amount: {outstanding_amount}
- Account Status: {account_status}
</client_context>

<task>
Clearly communicate account status and required payment. MAXIMUM 20 words.
</task>

<optimized_approach>
"We didn't receive your subscription payment. Your account is overdue by {outstanding_amount}. Can we debit this today?"
</optimized_approach>

<communication_strategy>
1. **State status directly**: Clear, factual account status
2. **Specify amount**: Exact outstanding amount
3. **Ask for immediate action**: Direct payment request
</communication_strategy>

<style>
- MAXIMUM 20 words
- Factual, not apologetic
- State amount clearly without hesitation
- Create urgency without aggression
</style>"""

def create_reason_for_call_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a reason for call agent."""
    
    agent_tools = [
        get_client_account_aging,
        get_client_account_overview,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare account status information."""
        
        # Extract account information from client data
        account_aging = client_data.get("account_aging", {})
        account_overview = client_data.get("account_overview", {})
        
        # Calculate outstanding amount
        outstanding_amount = calculate_outstanding_amount(account_aging)
        outstanding_formatted = format_currency(outstanding_amount)
        
        # Determine account status
        account_status = account_overview.get("account_status", "Overdue") if account_overview else "Overdue"
        
        # Create urgency level based on amount
        urgency_level = "standard"
        if outstanding_amount > 1000:
            urgency_level = "high"
        elif outstanding_amount > 500:
            urgency_level = "medium"
        
        return Command(
            update={
                "outstanding_amount": outstanding_formatted,
                "account_status": account_status,
                "urgency_level": urgency_level,
                "outstanding_float": outstanding_amount,
                "current_step": CallStep.REASON_FOR_CALL.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for reason for call step."""
        prompt_content = get_reason_for_call_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ReasonForCallAgent"
    )
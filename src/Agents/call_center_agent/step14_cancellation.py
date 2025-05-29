# src/Agents/call_center_agent/step14_cancellation.py
"""
Cancellation Agent - Self-contained with own prompt
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
import uuid
from datetime import datetime

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import calculate_outstanding_amount, format_currency

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    add_client_note,
    get_client_account_aging
)

def get_cancellation_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate cancellation specific prompt."""
    # Get cancellation details
    cancellation_fee = state.get("cancellation_fee", "R 0.00")
    total_balance = state.get("total_balance", "R 0.00")
    ticket_number = state.get("ticket_number", "CAN12345")
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Process cancellation professionally. MAXIMUM 20 words.
</task>

<approach>
"I understand you want to cancel. The cancellation fee is {cancellation_fee}. Your total balance is {total_balance}."
</approach>

<cancellation_process>
1. Acknowledge request
2. Explain fees
3. State total balance
4. Create ticket reference
5. Set expectations
</cancellation_process>

<follow_up>
"I'll escalate this to our cancellations team. Reference: {ticket_number}"
</follow_up>

<style>
- MAXIMUM 20 words
- Professional acceptance
- Clear fee explanation
- No retention attempts
</style>"""

def create_cancellation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a cancellation agent."""
    
    agent_tools = [add_client_note, get_client_account_aging] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to calculate cancellation fees and create ticket."""
        
        # Calculate cancellation fee and total balance
        account_aging = client_data.get("account_aging", {})
        outstanding_balance = calculate_outstanding_amount(account_aging)
        
        # Standard cancellation fee (adjust based on business rules)
        cancellation_fee = 0.0  # Set based on your business rules
        total_balance = outstanding_balance + cancellation_fee
        
        # Generate cancellation ticket
        ticket_number = f"CAN{datetime.now().strftime('%Y%m%d%H%M')}{str(uuid.uuid4())[:4].upper()}"
        
        # Add cancellation note
        user_id = client_data.get("user_id")
        if user_id:
            try:
                add_client_note.invoke({
                    "user_id": user_id,
                    "note_text": f"Cancellation requested. Ticket: {ticket_number}, Total balance: {format_currency(total_balance)}"
                })
            except Exception as e:
                if verbose:
                    print(f"Error adding cancellation note: {e}")
        
        return Command(
            update={
                "cancellation_fee": format_currency(cancellation_fee),
                "total_balance": format_currency(total_balance),
                "ticket_number": ticket_number,
                "cancellation_requested": True,
                "current_step": CallStep.CANCELLATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_cancellation_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="CancellationAgent"
    )
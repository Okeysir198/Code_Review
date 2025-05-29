# ./src/Agents/call_center_agent/step14_cancellation.py
"""
Cancellation Agent - Handles cancellation requests professionally.
SIMPLIFIED: Keep fee logic, no query detection.
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
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    add_client_note,
    get_client_account_aging
)


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
        
        import uuid
        from datetime import datetime
        
        # Calculate cancellation fee and total balance
        account_aging = client_data.get("account_aging", {})
        outstanding_balance = 0.0
        
        if account_aging:
            balance = account_aging.get("xbalance", "0")
            try:
                outstanding_balance = float(balance) if balance else 0.0
            except (ValueError, TypeError):
                outstanding_balance = 0.0
        
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
                    "note_text": f"Cancellation requested. Ticket: {ticket_number}, Total balance: R{total_balance:.2f}"
                })
            except Exception as e:
                if verbose:
                    print(f"Error adding cancellation note: {e}")
        
        return Command(
            update={
                "cancellation_fee": f"R {cancellation_fee:.2f}",
                "total_balance": f"R {total_balance:.2f}",
                "ticket_number": ticket_number,
                "cancellation_requested": True,
                "current_step": CallStep.CANCELLATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.CANCELLATION.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        prompt_content = get_step_prompt(CallStep.CANCELLATION.value, parameters)
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
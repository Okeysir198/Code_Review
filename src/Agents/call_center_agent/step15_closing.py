# src/Agents/call_center_agent/step15_closing.py
"""
Closing Agent - Self-contained with own prompt
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
    save_call_disposition,
    get_disposition_types,
    add_client_note,
    update_payment_arrangements
)

def get_closing_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate closing specific prompt."""
    # Extract client info
    client_name = get_safe_value(client_data, "profile.client_info.first_name", "Client")
    
    # Get call outcome
    call_outcome = state.get("call_outcome", "incomplete")
    outstanding_amount = state.get("outstanding_amount", "R 0.00")
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
End call professionally with summary. MAXIMUM 20 words.
</task>

<summary_options>
**Payment Secured**: "Perfect. We've arranged payment of {outstanding_amount}"
**Escalation**: "I've escalated to supervisor with reference {state.get('ticket_number', 'TKT12345')}"
**Cancellation**: "Cancellation request logged with reference {state.get('ticket_number', 'TKT12345')}"
**Incomplete**: "Thank you for your time today, {client_name}. Please call us back at 011 250 3000"
</summary_options>

<style>
- MAXIMUM 20 words
- Professional and courteous
- Clear outcome summary
- Thank the client
</style>"""

def create_closing_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a closing agent."""
    
    agent_tools = [save_call_disposition, get_disposition_types, add_client_note, update_payment_arrangements] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to determine call outcome and prepare summary."""
        
        # Determine call outcome based on state
        call_outcome = "incomplete"
        outcome_summary = "Call completed"
        
        # Analyze call progress and results
        if state.get("payment_secured"):
            call_outcome = "payment_secured"
            outcome_summary = "Payment arrangement secured"
        elif state.get("cancellation_requested"):
            call_outcome = "cancelled"
            outcome_summary = "Client requested cancellation"
        elif state.get("escalation_requested"):
            call_outcome = "escalated"
            outcome_summary = "Call escalated to supervisor"
        elif state.get("name_verification_status") == "THIRD_PARTY":
            call_outcome = "wrong_person"
            outcome_summary = "Third party contact - message left"
        elif not state.get("name_verification_status") == "VERIFIED":
            call_outcome = "uncontactable"
            outcome_summary = "Unable to verify client identity"
        
        # Create call summary
        call_summary = {
            "verification_completed": state.get("name_verification_status") == "VERIFIED" and 
                                    state.get("details_verification_status") == "VERIFIED",
            "payment_secured": state.get("payment_secured", False),
            "final_outcome": call_outcome
        }
        
        return Command(
            update={
                "call_outcome": call_outcome,
                "outcome_summary": outcome_summary,
                "call_summary": call_summary,
                "is_call_ended": True,
                "current_step": CallStep.CLOSING.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_closing_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ClosingAgent"
    )
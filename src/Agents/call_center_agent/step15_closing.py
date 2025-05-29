# ./src/Agents/call_center_agent/closing.py
"""
Closing Agent - Finalizes the call with summary and disposition.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
from enum import Enum

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

class CallOutcome(Enum):
    """Possible call outcomes"""
    PAYMENT_SECURED = "payment_secured"
    PTP_ARRANGED = "ptp_arranged"
    PARTIAL_PAYMENT = "partial_payment"
    CALLBACK_SCHEDULED = "callback_scheduled"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"
    UNCONTACTABLE = "uncontactable"
    REFUSAL = "refusal"
    DISPUTE = "dispute"
    WRONG_PERSON = "wrong_person"
    INCOMPLETE = "incomplete"


# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    save_call_disposition,
    get_disposition_types,
    add_client_note,
    update_payment_arrangements
)


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
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.CLOSING.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        prompt_content = get_step_prompt(CallStep.CLOSING.value, parameters)
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
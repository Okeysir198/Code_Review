# ./src/Agents/call_center_agent/closing.py
"""
Closing Agent - Finalizes the call with summary and disposition.
"""
from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
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
    """
    Create a closing agent for debt collection calls.
    
    Args:
        model: Language model to use
        client_data: client information
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled closing agent workflow
    """
    
    # Add relevant database tools
    agent_tools = [
        save_call_disposition,
        get_disposition_types,
        add_client_note,
        update_payment_arrangements
    ]
    if tools:
        agent_tools.extend(tools)
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process to determine call outcome and prepare summary."""
        
        try:
            # Determine call outcome based on state
            call_outcome = CallOutcome.INCOMPLETE.value
            outcome_summary = "Call completed"
            
            # Analyze call progress and results
            if state.payment_secured:
                if state.payment_arrangement and state.payment_arrangement.get('arrangement_created'):
                    call_outcome = CallOutcome.PTP_ARRANGED.value
                    outcome_summary = f"Payment arrangement secured: {state.payment_arrangement.get('payment_method', 'Unknown method')}"
                else:
                    call_outcome = CallOutcome.PAYMENT_SECURED.value
                    outcome_summary = "Payment secured"
            
            elif state.cancellation_requested:
                call_outcome = CallOutcome.CANCELLED.value
                outcome_summary = "Client requested cancellation"
            
            elif state.escalation_requested:
                call_outcome = CallOutcome.ESCALATED.value
                outcome_summary = "Call escalated to supervisor"
            
            elif state.name_verification_status == "THIRD_PARTY":
                call_outcome = CallOutcome.WRONG_PERSON.value
                outcome_summary = "Third party contact - message left"
            
            elif state.name_verification_status == "UNAVAILABLE":
                call_outcome = CallOutcome.UNCONTACTABLE.value
                outcome_summary = "Client unavailable - callback requested"
            
            elif not state.is_verified():
                call_outcome = CallOutcome.UNCONTACTABLE.value
                outcome_summary = "Unable to verify client identity"
            
            elif state.objections_raised and len(state.objections_raised) > 0:
                if "no_money" in state.objections_raised or "cant_afford" in state.objections_raised:
                    call_outcome = CallOutcome.REFUSAL.value
                    outcome_summary = "Client unable to pay - financial hardship"
                elif "dispute_amount" in state.objections_raised:
                    call_outcome = CallOutcome.DISPUTE.value
                    outcome_summary = "Client disputes outstanding amount"
                else:
                    call_outcome = CallOutcome.CALLBACK_SCHEDULED.value
                    outcome_summary = "Follow-up required"
            
            # Map call outcome to disposition type ID
            disposition_mapping = {
                CallOutcome.PAYMENT_SECURED.value: "1",  # Payment Secured
                CallOutcome.PTP_ARRANGED.value: "2",     # PTP Arranged
                CallOutcome.PARTIAL_PAYMENT.value: "3",  # Partial Payment
                CallOutcome.CALLBACK_SCHEDULED.value: "4", # Callback Required
                CallOutcome.ESCALATED.value: "5",        # Escalated
                CallOutcome.CANCELLED.value: "6",        # Cancellation Request
                CallOutcome.UNCONTACTABLE.value: "7",    # Uncontactable
                CallOutcome.REFUSAL.value: "8",          # Refusal to Pay
                CallOutcome.DISPUTE.value: "9",          # Dispute
                CallOutcome.WRONG_PERSON.value: "10",    # Wrong Person
                CallOutcome.INCOMPLETE.value: "11"       # Incomplete Call
            }
            
            disposition_type_id = disposition_mapping.get(call_outcome, "11")
            
            # Create comprehensive call summary
            call_summary = {
                "verification_completed": state.is_verified(),
                "payment_secured": state.payment_secured,
                "objections_count": len(state.objections_raised),
                "steps_completed": state.metrics.steps_completed if hasattr(state, 'metrics') else 0,
                "emotional_state": state.emotional_state,
                "final_outcome": call_outcome
            }
            
            return {
                "call_outcome": call_outcome,
                "outcome_summary": outcome_summary,
                "disposition_type_id": disposition_type_id,
                "call_summary": call_summary,
                "disposition_saved": False,
                "is_call_ended": True,
                "call_info": {
                    "final_status": call_outcome,
                    "verification_status": "verified" if state.is_verified() else "unverified"
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in closing pre-processing: {e}")
            
            return {
                "call_outcome": CallOutcome.INCOMPLETE.value,
                "outcome_summary": "Call completed with errors",
                "disposition_type_id": "11",
                "call_summary": {},
                "disposition_saved": False,
                "is_call_ended": True,
                "call_info": {"final_status": "error"}
            }

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Post-process to save call disposition and final notes."""
        
        try:
            disposition_saved = False
            note_added = False
            user_id = client_data['user_id']
            
            # Save call disposition
            try:
                disposition_result = save_call_disposition.invoke({
                    "client_id": user_id,
                    "disposition_type_id": state.get('disposition_type_id', '11'),
                    "note_text": state.get('outcome_summary', 'Call completed')
                })
                
                disposition_saved = disposition_result.get("success", False)
                
            except Exception as disposition_error:
                if verbose:
                    print(f"Error saving disposition: {disposition_error}")
            
            # Add final call summary note
            try:
                call_summary = state.get('call_summary', {})
                summary_parts = []
                
                if call_summary.get('verification_completed'):
                    summary_parts.append("Client verified")
                
                if call_summary.get('payment_secured'):
                    summary_parts.append("Payment secured")
                
                if call_summary.get('objections_count', 0) > 0:
                    summary_parts.append(f"{call_summary['objections_count']} objections handled")
                
                final_note = f"Call completed: {', '.join(summary_parts) if summary_parts else 'No specific outcomes'}. Final outcome: {state.get('call_outcome', 'Unknown')}"
                
                note_result = add_client_note.invoke({
                    "user_id": user_id,
                    "note_text": final_note
                })
                
                note_added = note_result.get("success", False)
                
            except Exception as note_error:
                if verbose:
                    print(f"Error adding final note: {note_error}")
            
            # Update payment arrangements status if needed
            if state.payment_secured:
                try:
                    update_payment_arrangements.invoke(user_id)
                except Exception as update_error:
                    if verbose:
                        print(f"Error updating payment arrangements: {update_error}")
            
            return {
                "disposition_saved": disposition_saved,
                "final_note_added": note_added,
                "call_completed": True,
                "call_info": {
                    "disposition_status": "saved" if disposition_saved else "failed",
                    "note_status": "added" if note_added else "failed"
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in closing post-processing: {e}")
            
            return {
                "disposition_saved": False,
                "final_note_added": False,
                "call_completed": True,
                "call_info": {"status": "error"}
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for closing step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.CLOSING.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.CLOSING.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        post_processing_node=post_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ClosingAgent"
    )
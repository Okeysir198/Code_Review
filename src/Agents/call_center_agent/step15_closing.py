# src/Agents/call_center_agent/step15_closing.py
"""
Enhanced Closing Agent - Comprehensive logging and professional call conclusion
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
from src.Database.CartrackSQLDatabase import (
    add_client_note, save_call_disposition, get_disposition_types,
    update_payment_arrangements, get_current_date_time
)

CLOSING_PROMPT = """
You're {agent_name} from Cartrack concluding the call with {client_name} about their {outstanding_amount} account.

TODAY: {current_date}
OBJECTIVE: Professionally end the call with clear summary and next steps.

CALL OUTCOME: {call_outcome}
PAYMENT SECURED: {payment_secured}
ESCALATION: {escalation_requested}
CANCELLATION: {cancellation_requested}

TOOL USAGE FOR COMPREHENSIVE LOGGING:
1. FIRST: add_client_note with comprehensive call summary
   - Parameters: user_id={user_id}, note_text=detailed_summary
   - Include: verification status, payment outcome, client cooperation, next steps
   - Format: "Call completed [date]: Identity verified, Payment [secured/not secured], Method [debit/online], Client [cooperative/difficult], Outcome [summary]"

2. THEN: save_call_disposition 
   - Parameters: client_id={user_id}, disposition_type_id=outcome, note_text=brief_summary
   - Disposition types: COMPLETED, PAYMENT_SECURED, ESCALATED, CANCELLED, VERIFICATION_FAILED

3. IF payment secured: update_payment_arrangements
   - Parameters: user_id={user_id}
   - Updates arrangement status and confirms payment setup

4. ALWAYS: get_current_date_time for accurate timestamps

CLOSING SUMMARY BY OUTCOME:
- Payment Secured: "Perfect, {client_name}. Your payment of {outstanding_amount} is arranged. You'll see the debit/receive the link as discussed."
- Escalation: "I've escalated your request to our {department}. You'll hear back within {response_time}."
- Cancellation: "Cancellation request logged. Your total balance of {total_balance} must be settled before cancellation can proceed."
- Verification Failed: "For security reasons, I can't proceed without proper verification. Please call us at 011 250 3000 with ID available."
- Incomplete: "Thank you for your time, {client_name}. Please call us at 011 250 3000 to complete your payment arrangement."

PROFESSIONAL CLOSING PHRASES:
- Success: "Thank you for resolving this today, {client_name}. Have a great day."
- Escalation: "Thank you for your patience. Our team will be in touch soon."
- Incomplete: "We appreciate your time today. Please call us back when convenient."

URGENCY LEVEL: {urgency_level}

Create comprehensive documentation while providing professional closure. Maximum 30 words for closing statement.
"""

def create_closing_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced closing agent with comprehensive logging"""
    
    # Comprehensive logging tools
    closing_tools = [
        add_client_note,              # Detailed call summary
        save_call_disposition,        # Call outcome
        get_disposition_types,        # Available dispositions  
        update_payment_arrangements,  # Payment status updates
        get_current_date_time        # Accurate timestamps
    ]
    
    def _determine_call_outcome(state: CallCenterAgentState) -> str:
        """Determine call outcome based on state"""
        if state.get("payment_secured"):
            return "payment_secured"
        elif state.get("escalation_requested"):
            return "escalated"
        elif state.get("cancellation_requested"):
            return "cancelled"
        elif state.get("name_verification_status") in ["WRONG_PERSON", "VERIFICATION_FAILED"]:
            return "verification_failed"
        elif state.get("details_verification_status") == "VERIFICATION_FAILED":
            return "verification_failed"
        else:
            return "incomplete"
    
    def _create_comprehensive_note(state: CallCenterAgentState, call_outcome: str) -> str:
        """Create detailed note summarizing entire call"""
        note_parts = []
        
        # Basic call info
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        note_parts.append(f"Call completed {current_time}")
        note_parts.append(f"Client: {state.get('client_name', 'Unknown')}")
        note_parts.append(f"Outstanding: {state.get('outstanding_amount', 'Unknown')}")
        
        # Verification outcome
        name_status = state.get('name_verification_status', 'PENDING')
        details_status = state.get('details_verification_status', 'PENDING')
        if name_status == 'VERIFIED' and details_status == 'VERIFIED':
            note_parts.append("✓ Identity fully verified")
        elif name_status == 'VERIFIED':
            note_parts.append("✓ Name verified, details incomplete")
        else:
            note_parts.append(f"✗ Verification failed: {name_status}")
        
        # Payment outcome
        if state.get('payment_secured'):
            method = state.get('payment_method_preference', 'Unknown method')
            note_parts.append(f"✓ Payment secured via {method}")
        else:
            note_parts.append("✗ No payment arrangement made")
        
        # Special situations
        if state.get('escalation_requested'):
            note_parts.append("⚠ Escalation requested")
        if state.get('cancellation_requested'):
            note_parts.append("⚠ Cancellation requested")
        
        # Conversation quality indicators
        attempts = state.get('name_verification_attempts', 0) + state.get('details_verification_attempts', 0)
        if attempts <= 2:
            note_parts.append("😊 Smooth interaction")
        elif attempts > 5:
            note_parts.append("😤 Difficult interaction")
        
        # Call outcome
        note_parts.append(f"Outcome: {call_outcome}")
        
        return " | ".join(note_parts)
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Determine call outcome and prepare for logging"""
        
        call_outcome = _determine_call_outcome(state)
        
        return Command(
            update={
                "call_outcome": call_outcome,
                "is_call_ended": True,
                "current_step": CallStep.CLOSING.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate comprehensive closing prompt with tool guidance"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Add call outcome details
        params["call_outcome"] = state.get("call_outcome", "incomplete")
        params["payment_secured"] = "Yes" if state.get("payment_secured") else "No"
        params["escalation_requested"] = "Yes" if state.get("escalation_requested") else "No"
        params["cancellation_requested"] = "Yes" if state.get("cancellation_requested") else "No"
        
        # Get aging context
        aging_context = ScriptManager.get_aging_context(script_type)
        params["urgency_level"] = aging_context['urgency']
        
        # Format prompt
        prompt_content = CLOSING_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Closing Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=closing_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedClosingAgent"
    )
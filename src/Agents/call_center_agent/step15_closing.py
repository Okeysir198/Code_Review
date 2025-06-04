# src/Agents/call_center_agent/step15_closing.py
"""
Enhanced Closing Agent - Professional call conclusion with comprehensive logging
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
from src.Agents.call_center_agent.call_scripts import ScriptManager, ScriptType
from src.Database.CartrackSQLDatabase import (
    add_client_note, save_call_disposition, get_disposition_types,
    update_payment_arrangements, get_current_date_time
)

import logging
logger = logging.getLogger(__name__)

CLOSING_PROMPT = """You are {agent_name} from Cartrack Accounts Department concluding an OUTBOUND PHONE CALL with {client_title} {client_full_name} about their {outstanding_amount} overdue account.

<phone_conversation_rules>
- This is a LIVE OUTBOUND PHONE CALL - you initiated this call to them about their debt
- Each agent handles ONE conversation turn, then waits for the client's response  
- Keep responses conversational length - not too brief (robotic) or too long (overwhelming)
- Match your tone to the client's cooperation level and the account urgency
- Listen to what they're actually saying and respond appropriately
- Don't assume their mood or intent - respond to their actual words
- If they ask questions, acknowledge briefly but stay focused on your step's objective
- Remember: phone conversations flow naturally - avoid scripted, mechanical responses
- End your turn when you've accomplished your step's goal or need their input
</phone_conversation_rules>

<context>
Today: {current_date} | Account: {aging_category} ({urgency_level} urgency)
Call outcome: {call_outcome} | Payment secured: {payment_secured}
Verification status: Name: {name_verification_status} | Details: {details_verification_status}
Your goal: Professional call closure appropriate to verification outcome
</context>

<tool_usage_sequence>
1. FIRST: add_client_note with comprehensive call summary
   - Parameters: user_id={user_id}, note_text="Call completed {current_date}: [verification status], [payment outcome], [client cooperation], [next steps]"

2. THEN: save_call_disposition based on outcome
   - Parameters: client_id={user_id}, disposition_type_id=[outcome_code], note_text="Brief summary"

3. IF payment secured: update_payment_arrangements
   - Parameters: user_id={user_id}

Use tools to document everything, then deliver professional closing.
</tool_usage_sequence>

<closing_by_outcome>
Payment Secured: "Perfect, {client_name}. Your payment of {outstanding_amount} is arranged. You'll see the debit/receive the link as discussed. Thank you for resolving this today."

Escalation: "I've escalated your request to our team. You'll hear back within 24-48 hours. Thank you for your patience."

Cancellation: "Cancellation request logged. Your total balance must be settled before cancellation proceeds. Thank you for your time."

Verification Failed: "For security reasons, I can't proceed without proper verification. Please call us at 011 250 3000 with ID available."

Incomplete: "Thank you for your time, {client_name}. Please call us at 011 250 3000 when you're ready to complete your payment arrangement."
</closing_by_outcome>

<verification_failure_closings>
WRONG_PERSON (complete stranger):
"My apologies, I have the wrong number. Thank you for your time. Have a good day."

THIRD_PARTY (knows the client) - Standard urgency:
"Please ask {client_title} {client_full_name} to call Cartrack urgently at 011 250 3000 about their outstanding account. It needs immediate attention. Thank you."

THIRD_PARTY (knows the client) - High urgency:
"This is urgent - {client_title} {client_full_name} needs to call Cartrack immediately at 011 250 3000 about their overdue account. Please make sure they get this message today."

THIRD_PARTY (knows the client) - Critical urgency:
"This is critical - {client_title} {client_full_name} must call Cartrack immediately at 011 250 3000 about their seriously overdue account. Legal action may be considered if not resolved urgently."

UNAVAILABLE (right person, can't talk):
"I understand you can't talk now. Please call Cartrack back at 011 250 3000 when convenient to discuss your account. Thank you."

VERIFICATION_FAILED (too many attempts):
"For security reasons, I can't proceed without proper verification. Please call us at 011 250 3000 with your ID available. Thank you."
</verification_failure_closings>

<natural_conversation_rules>
- Speak naturally like a real phone conversation
- NO brackets [ ], asterisks *, or placeholder formatting
- NO internal system variables or markdown in your response
- Use actual names or speak generally if you don't know specifics
- Just natural spoken words as if talking to a real person
</natural_conversation_rules>

Document the call comprehensively using tools, then deliver appropriate professional closing based on {call_outcome}.
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
    """Create enhanced closing agent with comprehensive logging and professional closure"""
    
    # Comprehensive logging tools
    closing_tools = [
        add_client_note,
        save_call_disposition,
        get_disposition_types,
        update_payment_arrangements,
        get_current_date_time
    ]
    
    def _determine_call_outcome(state: CallCenterAgentState) -> str:
        """Determine call outcome based on conversation state"""
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
    
    def _check_closing_completion(messages: List) -> bool:
        """Check if professional closing was delivered"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                closing_indicators = [
                    "thank you", "have a great day", "good day", "goodbye",
                    "call us at", "arranged", "escalated", "logged"
                ]
                if any(indicator in content for indicator in closing_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Determine call outcome and handle completion"""
        
        messages = state.get("messages", [])
        
        # Check if closing already completed
        if len(messages) >= 2:
            closing_completed = _check_closing_completion(messages)
            
            if closing_completed:
                logger.info("Call closing completed - ending conversation")
                return Command(
                    update={
                        "is_call_ended": True
                    },
                    goto="__end__"
                )
        
        # Prepare for closing
        call_outcome = _determine_call_outcome(state)
        
        return Command(
            update={
                "call_outcome": call_outcome,
                "current_step": CallStep.CLOSING.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced closing prompt with tool guidance"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, script_type, agent_name)
        
        # Add call outcome details
        params["call_outcome"] = state.get("call_outcome", "incomplete")
        params["payment_secured"] = "Yes" if state.get("payment_secured") else "No"
        params["escalation_requested"] = "Yes" if state.get("escalation_requested") else "No"
        params["cancellation_requested"] = "Yes" if state.get("cancellation_requested") else "No"
        
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
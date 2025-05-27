# # ./src/Agents/call_center/utils.py
# ./src/Agents/call_center/utils.py

"""
Utility functions for call center agent script formatting and prompt generation.

This module provides helper functions for formatting scripts and generating
dynamic system prompts for each call step, ensuring that the LLM gets the
appropriate context and guidance to maintain natural conversation flow.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from src.Agents.call_center.call_scripts import ScriptManager, ScriptType
from src.Agents.call_center.state import CallStep, VerificationStatus, PaymentMethod
import src.Agents.call_center.prompts as prompts
from app_config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_script_parameters(state: Dict[str, Any], client_info: Dict[str, Any], config:Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare comprehensive parameters dictionary for script and prompt formatting.
    Handles all possible parameters including field_to_verify for verification steps.
    """
    # Extract core client information
    client_name = client_info.get('full_name', 'Client')
    client_first_name = client_name.split()[0] if ' ' in client_name else client_name
    
    # Build parameters dictionary with all possible parameters
    # This ensures templates don't break if they reference any parameter
    parameters = {
        # Agent information
        "agent_name": config.get("agent_name", "AI Agent"),
        
        # Client identification
        "client_full_name": client_name,
        "client_name": client_first_name,
        "client_title": client_info.get("title", "Mr./Ms."),
        "salutation": "Sir/Madam",
        
        # Account information
        "outstanding_amount": client_info.get("outstanding_amount", "the outstanding amount"),
        "subscription_amount": client_info.get("subscription_amount", "your regular subscription"),
        "subscription_date": client_info.get("subscription_date", "the usual date"),
        "cancellation_fee": client_info.get("cancellation_fee", "the cancellation fee"),
        "total_balance": client_info.get("total_balance", "the total balance"),
        
        # Verification status - name verification
        "name_verification_status": state.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
        "name_verification_attempts": state.get("name_verification_attempts", 1),
        "max_name_verification_attempts": config.get("max_name_verification_attempts", 3),
        
        # Verification status - details verification
        "details_verification_status": state.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
        "details_verification_attempts": state.get("details_verification_attempts", 1),
        "max_details_verification_attempts": config.get("max_details_verification_attempts", 3),
        "matched_fields": state.get("matched_fields", []),
        
        # Field to verify - critical for details verification step
        "field_to_verify": state.get("field_to_verify", "verification information"),
        "current_verification_field": state.get("current_verification_field", ""),
        
        # Payment processing
        "ptp_amount_plus_fee": f"{client_info.get('outstanding_amount', 'the balance')} plus R10",
        "amount_with_fee": _calculate_amount_with_fee(client_info.get('outstanding_amount')),
        
        # Call tracking
        "current_step": state.get("current_call_step", CallStep.INTRODUCTION.value),
        "step_number": _get_step_number(state.get("current_call_step")),
        
        # State-based information
        "payment_secured": state.get("payment_secured", False),
        "call_outcome": state.get("call_outcome", "UNKNOWN"),
        
        # For cancellation/escalation
        "department": state.get("escalation_department", "relevant"),
        "priority_level": state.get("priority_level", "standard"),
        "case_type": state.get("case_type", "general"),
        "ticket_number": state.get("ticket_number", ""),
        "response_time": _get_response_time(state.get("priority_level", "standard")),
        
        # Additional consequences based on script type
        "additional_consequences": _get_additional_consequences(state.get("script_type", "")),
    }
    
    return parameters

def _calculate_amount_with_fee(amount: Union[str, float, int]) -> str:
    """Helper to calculate amount with R10 fee."""
    try:
        if isinstance(amount, (int, float)):
            total = float(amount) + 10
            return f"R {total:.2f}"
        elif isinstance(amount, str) and amount.replace("R", "").strip().replace(".", "").isdigit():
            amount_val = float(amount.replace("R", "").strip())
            total = amount_val + 10
            return f"R {total:.2f}"
        return f"{amount} plus R10 fee"
    except:
        return f"{amount} plus R10 fee"

def _get_step_number(current_step: Optional[str]) -> int:
    """Get numeric position in call flow."""
    step_positions = {
        CallStep.INTRODUCTION.value: 1,
        CallStep.NAME_VERIFICATION.value: 2,
        CallStep.DETAILS_VERIFICATION.value: 3,
        CallStep.REASON_FOR_CALL.value: 4,
        CallStep.NEGOTIATION.value: 5,
        CallStep.PROMISE_TO_PAY.value: 6,
        CallStep.DEBICHECK_SETUP.value: 7,
        CallStep.PAYMENT_PORTAL.value: 7,
        CallStep.SUBSCRIPTION_REMINDER.value: 8,
        CallStep.CLIENT_DETAILS_UPDATE.value: 9,
        CallStep.REFERRALS.value: 10,
        CallStep.FURTHER_ASSISTANCE.value: 11,
        CallStep.CLOSING.value: 12,
    }
    return step_positions.get(current_step, 1)

def _get_response_time(priority: str) -> str:
    """Get estimated response time based on priority."""
    response_times = {
        "critical": "1-2 hours",
        "urgent": "4-8 hours",
        "high": "12-24 hours",
        "standard": "24-48 hours"
    }
    return response_times.get(priority.lower(), "24-48 hours")

def _get_additional_consequences(script_type: str) -> str:
    """Get additional consequences based on script type."""
    if "2_3" in script_type:
        return "Potential recovery fee of up to R25,000 if vehicle is stolen"
    elif "legal" in script_type:
        return "Legal action and credit listing as default payer"
    return ""

def build_client_info_block(state: Dict[str, Any], client_info: Dict[str, Any]) -> str:
    """
    Build formatted client information block for prompts.
    Only shows detailed info after verification.
    """
    # Check verification state
    verification_status = state.get('name_verification_status')
    current_step = state.get('current_call_step')
    
    if verification_status != VerificationStatus.VERIFIED.value or current_step in [
        CallStep.INTRODUCTION.value,
        CallStep.NAME_VERIFICATION.value,
        CallStep.DETAILS_VERIFICATION.value
    ]:
        return "Client details not yet verified."
    
    # Build verified client info
    info_parts = []
    
    # Basic information
    client_name = state.get('verified_client_name', client_info.get('full_name', 'Client'))
    info_parts.extend([
        f"Full Name: {client_name}",
        f"Email: {client_info.get('email', 'Not available')}",
        f"Outstanding Amount: {client_info.get('outstanding_amount', 'Not available')}",
        f"Account Status: {client_info.get('account_status', 'Not available')}"
    ])
    
    # Add phone if available
    phone = client_info.get('phone', client_info.get('contact', {}).get('mobile'))
    if phone:
        info_parts.append(f"Phone: {phone}")
    
    # Add vehicle information if available
    vehicles = client_info.get("vehicles", [])
    if vehicles:
        info_parts.append("Vehicles:")
        for vehicle in vehicles:
            info_parts.append(
                f"- {vehicle.get('make', '')} {vehicle.get('model', '')}, "
                f"Registration: {vehicle.get('registration', '')}, "
                f"Color: {vehicle.get('color', '')}"
            )
    
    # Add payment history for relevant steps
    if current_step in [CallStep.NEGOTIATION.value, CallStep.PROMISE_TO_PAY.value]:
        payment_history = state.get('payment_history', client_info.get('payment_history', []))
        if payment_history:
            info_parts.append("Recent Payment History:")
            for payment in payment_history[:3]:  # Last 3 payments
                info_parts.append(
                    f"- {payment.get('date', 'Unknown')}: "
                    f"{payment.get('amount', 'Unknown')} "
                    f"({payment.get('status', 'Unknown')})"
                )
    
    return "\n".join(info_parts)

def get_call_flow_position(current_step: str) -> str:
    """
    Create formatted call flow progress with current step highlighted.
    Returns a formatted string showing all steps with the current one marked.
    """
    step_positions = {
        CallStep.INTRODUCTION.value: 1,
        CallStep.NAME_VERIFICATION.value: 2,
        CallStep.DETAILS_VERIFICATION.value: 3,
        CallStep.REASON_FOR_CALL.value: 4,
        CallStep.NEGOTIATION.value: 5,
        CallStep.PROMISE_TO_PAY.value: 6,
        CallStep.DEBICHECK_SETUP.value: 7,
        CallStep.PAYMENT_PORTAL.value: 7,
        CallStep.SUBSCRIPTION_REMINDER.value: 8,
        CallStep.CLIENT_DETAILS_UPDATE.value: 9,
        CallStep.REFERRALS.value: 10,
        CallStep.FURTHER_ASSISTANCE.value: 11,
        CallStep.CLOSING.value: 12,
    }
    
    position = step_positions.get(current_step, 0)
    
    if position == 0:
        if current_step == CallStep.QUERY_RESOLUTION.value:
            return "SPECIAL PHASE: HANDLING CLIENT QUERY (Will return to main flow)"
        elif current_step == CallStep.CANCELLATION.value:
            return "SPECIAL PHASE: PROCESSING CANCELLATION REQUEST"
        elif current_step == CallStep.ESCALATION.value:
            return "SPECIAL PHASE: PROCESSING ESCALATION REQUEST"
        return "SPECIAL PHASE: CASE HANDLING"
    
    flow_steps = [
        "1. Introduction",
        "2. Name Verification",
        "3. Details Verification",
        "4. Reason for Call",
        "5. Negotiation",
        "6. Promise to Pay",
        "7. Payment Processing",
        "8. Subscription Reminder",
        "9. Client Details Update",
        "10. Referrals",
        "11. Further Assistance",
        "12. Closing"
    ]
    
    # Format each step, highlighting the current one
    formatted_steps = []
    for i, step in enumerate(flow_steps, 1):
        if i < position:
            formatted_steps.append(f"✓ {step}")
        elif i == position:
            formatted_steps.append(f"➤ {step} (CURRENT)")
        else:
            formatted_steps.append(f"  {step}")
    
    return "\n".join(formatted_steps)

def create_system_prompt(
    state: Dict[str, Any],
    script_type: str,
    client_info: Dict[str, Any],
) -> str:
    """
    Create complete system prompt with all necessary context.
    Handles all call steps including verification steps with field-specific guidance.
    """
    # Get the current step
    current_step = state.get('current_call_step', CallStep.INTRODUCTION.value)

    # Step to section mapping
    section_map = {
        CallStep.INTRODUCTION.value: "INTRODUCTION",
        CallStep.NAME_VERIFICATION.value: "NAME_VERIFICATION",
        CallStep.DETAILS_VERIFICATION.value: "DETAILS_VERIFICATION",
        CallStep.REASON_FOR_CALL.value: "REASON_FOR_CALL",
        CallStep.NEGOTIATION.value: "NEGOTIATION",
        CallStep.PROMISE_TO_PAY.value: "PROMISE_TO_PAY",
        CallStep.DEBICHECK_SETUP.value: "DEBICHECK_SETUP",
        CallStep.SUBSCRIPTION_REMINDER.value: "SUBSCRIPTION_REMINDER",
        CallStep.PAYMENT_PORTAL.value: "PAYMENT_PORTAL",
        CallStep.CLIENT_DETAILS_UPDATE.value: "CLIENT_DETAILS_UPDATE",
        CallStep.REFERRALS.value: "REFERRALS",
        CallStep.FURTHER_ASSISTANCE.value: "FURTHER_ASSISTANCE",
        CallStep.CANCELLATION.value: "CANCELLATION",
        CallStep.CLOSING.value: "CLOSING",
        CallStep.QUERY_RESOLUTION.value: "NEGOTIATION_QUERY_HANDLING",
        CallStep.ESCALATION.value: "ESCALATION",
    }
    
    # Get script section and format parameters
    script_section = section_map.get(current_step, "INTRODUCTION")
    
    # Ensure we're working with the correct ScriptType enum
    if isinstance(script_type, str):
        try:
            script_type_enum = ScriptType(script_type)
        except ValueError:
            logger.warning(f"Invalid script type string: {script_type}, using default RATIO_1_INFLOW")
            script_type_enum = ScriptType.RATIO_1_INFLOW
    else:
        script_type_enum = script_type
    
    # Prepare parameters for script and prompt formatting
    parameters = prepare_script_parameters(state, client_info, config=CONFIG["verification"].copy())
    
    # Get script text
    script_text = ScriptManager.get_script_text(script_type_enum, script_section)
    
    # Safely format script text with parameters, handling missing placeholders
    try:
        formatted_script = script_text.format(**parameters)
    except KeyError as e:
        logger.warning(f"Missing parameter in script formatting: {e}")
        # Provide script text without formatting rather than failing
        formatted_script = script_text
    
    # Build client info block
    client_info_block = build_client_info_block(state, client_info)
    call_flow_position = get_call_flow_position(current_step)
    
    # Get step-specific guidance
    step_guidance_attr = f"{current_step.upper()}_GUIDANCE"
    step_guidance = getattr(prompts, step_guidance_attr, "")
    
    if step_guidance:
        try:
            step_guidance = step_guidance.format(**parameters)
        except KeyError as e:
            logger.warning(f"Missing parameter in guidance formatting: {e}")
            # Provide unformatted guidance rather than failing
    
    # Create complete prompt
    try:
        complete_prompt = prompts.DETAILED_SYSTEM_PROMPT.format(
            call_flow_position=call_flow_position,
            client_info_block=client_info_block,
            script_text=formatted_script,
            contextual_features=step_guidance,
            **parameters
        )
        logger.info(f"complete_prompt: {complete_prompt}")
        return complete_prompt
    except KeyError as e:
        logger.error(f"Error formatting prompt: {e}")
        # Fallback prompt with minimal formatting requirements
        fallback_prompt = f"""ERROR: Missing parameter in prompt formatting: {e}"""
        return fallback_prompt


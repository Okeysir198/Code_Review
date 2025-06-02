# src/Agents/call_center_agent/step02_details_verification.py
"""
Enhanced Details Verification Agent - Natural conversation with fast ID verification
"""
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
from src.Agents.call_center_agent.tools.verify_client_details import verify_client_details

logger = logging.getLogger(__name__)

# Enhanced conversational prompt for details verification
DETAILS_VERIFICATION_PROMPT = """
You're {agent_name} from Cartrack continuing the conversation with {client_full_name} about their {outstanding_amount} overdue account.

TODAY: {current_date}
OBJECTIVE: Verify {client_name}'s {field_to_verify} for security before discussing the account details.

VERIFIED SO FAR: {matched_fields_display}

RESPOND NATURALLY FOR SECURITY VERIFICATION:
- If they're cooperative: "For security, could you confirm your {field_to_verify}?"
- If they're hesitant: "I understand your caution. I need to verify your {field_to_verify} to protect your account before we can proceed."
- If they're resistant: "This is standard security protocol. Your {field_to_verify} helps ensure I'm speaking with the account holder."
- If they provide details: "Thank you. Let me verify that information."

URGENCY LEVEL: {urgency_level} - {aging_approach}

You're building on the successful name verification. Be professional but reassuring - explain that this protects their account information. If they've been cooperative so far, keep the momentum going. If they're cautious, take time to explain why verification is necessary.

Keep responses natural and under 25 words unless security explanation is needed. Focus on getting the specific information you need: {field_to_verify}.
"""

def create_details_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced details verification agent with fast ID detection"""
    
    FIELD_PRIORITY = ["id_number", "passport_number", "vehicle_registration", 
                     "vehicle_make", "vehicle_model", "vehicle_color", "email", "username"]
    
    def _get_available_fields(client_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract verification fields"""
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {})
        vehicles = profile.get("vehicles", [])
        
        fields = {}
        if client_info.get("id_number"): 
            fields["id_number"] = client_info["id_number"]
        if profile.get("user_name"): 
            fields["username"] = profile["user_name"]
        if client_info.get("email_address"): 
            fields["email"] = client_info["email_address"]
        
        if vehicles and isinstance(vehicles[0], dict):
            v = vehicles[0]
            if v.get("registration"): 
                fields["vehicle_registration"] = v["registration"]
            if v.get("make"): 
                fields["vehicle_make"] = v["make"]
            if v.get("model"): 
                fields["vehicle_model"] = v["model"]
            if v.get("color"): 
                fields["vehicle_color"] = v["color"]
        
        return fields
    
    def _select_next_field(available_fields: Dict[str, str], matched_fields: List[str]) -> str:
        """Select next verification field with priority"""
        remaining = [f for f in FIELD_PRIORITY if f in available_fields and f not in matched_fields]
        if not remaining: 
            return "id_number"
        
        # Prioritize ID number and passport
        high_priority = [f for f in FIELD_PRIORITY[:2] if f in remaining]
        if high_priority: 
            return high_priority[0]
        
        # Then vehicle details
        vehicle_fields = [f for f in remaining if f.startswith("vehicle_")]
        if vehicle_fields:
            random.shuffle(vehicle_fields)
            return vehicle_fields[0]
        
        # Then other fields
        other_fields = [f for f in remaining if not f.startswith("vehicle_")]
        if other_fields:
            random.shuffle(other_fields)
            return other_fields[0]
            
        return "id_number"
    
    def _get_last_client_message(messages: List) -> str:
        """Extract last human message"""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message.content.strip()
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                return message.content.strip()
        return ""
    
    def _quick_details_check(messages: List, available_fields: Dict[str, str]) -> bool:
        """Fast check if client provided verification details"""
        last_msg = _get_last_client_message(messages)
        
        if not last_msg:
            return False
        
        # Check for ID number patterns (13 digits for SA ID)
        import re
        id_pattern = r'\b\d{13}\b'
        if re.search(id_pattern, last_msg):
            return True
        
        # Check for vehicle registration patterns
        reg_patterns = [r'\b[A-Z]{2,3}[\s\-]?\d{3,4}[\s\-]?[A-Z]{2,3}\b', r'\b\d{3}[\s\-]?\d{3}[\s\-]?\d{3}\b']
        if any(re.search(pattern, last_msg.upper()) for pattern in reg_patterns):
            return True
        
        # Check for email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, last_msg):
            return True
        
        # Check if they mentioned specific vehicle details
        for field, value in available_fields.items():
            if field.startswith("vehicle_") and value.lower() in last_msg.lower():
                return True
        
        return False
    
    def _format_matched_fields(matched_fields: List[str]) -> str:
        """Format matched fields for display"""
        if not matched_fields:
            return "None yet"
        
        field_names = {
            "id_number": "ID Number", "email": "Email", "username": "Username",
            "vehicle_registration": "Vehicle Registration", "vehicle_make": "Vehicle Make",
            "vehicle_model": "Vehicle Model", "vehicle_color": "Vehicle Color"
        }
        
        return ", ".join([field_names.get(field, field.title()) for field in matched_fields])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Enhanced preprocessing with fast details detection"""
        
        attempts = state.get("details_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_details_verification_attempts", 5)
        matched_fields = state.get("matched_fields", [])
        available_fields = _get_available_fields(client_data)
        
        # Select field to verify
        field_to_verify = _select_next_field(available_fields, matched_fields)
        
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        all_matched = matched_fields.copy()
        
        messages = state.get("messages", [])
        
        # Fast check if client provided details
        if len(messages) >= 2 and _quick_details_check(messages, available_fields):
            try:
                # Use existing verification tool for accuracy
                result = verify_client_details.invoke({
                    "client_details": available_fields,
                    "messages": messages,
                    "required_match_count": 3,
                    "max_failed_attempts": max_attempts
                })
                
                new_matched = result.get("matched_fields", [])
                all_matched = list(set(matched_fields + new_matched))
                verification_status = result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
                
            except Exception as e:
                if verbose: 
                    logger.error(f"Verification error: {e}")
        
        # Auto-fail if max attempts reached
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        # Format field name for display
        field_names = {
            "id_number": "ID number", "passport_number": "passport number",
            "username": "username", "email": "email address",
            "vehicle_registration": "vehicle registration", "vehicle_make": "vehicle make",
            "vehicle_model": "vehicle model", "vehicle_color": "vehicle color"
        }
        
        # Determine next action
        if verification_status == VerificationStatus.VERIFIED.value:
            logger.info("Details verification VERIFIED - jumping to reason for call")
            return Command(
                update={
                    "details_verification_attempts": attempts,
                    "details_verification_status": verification_status,
                    "matched_fields": all_matched,
                    "field_to_verify": field_names.get(field_to_verify, field_to_verify),
                    "current_step": CallStep.REASON_FOR_CALL.value
                },
                goto="__end__"  # Direct jump!
            )
        
        elif verification_status in [
            VerificationStatus.THIRD_PARTY.value,
            VerificationStatus.UNAVAILABLE.value,
            VerificationStatus.WRONG_PERSON.value,
            VerificationStatus.VERIFICATION_FAILED.value
        ]:
            logger.info(f"Details verification terminal: {verification_status} - ending call")
            return Command(
                update={
                    "details_verification_attempts": attempts,
                    "details_verification_status": verification_status,
                    "matched_fields": all_matched,
                    "is_call_ended": True,
                    "current_step": CallStep.CLOSING.value
                },
                goto="__end__"
            )
        
        else:
            # Continue verification
            logger.info(f"Details verification continuing - attempt {attempts}")
            return Command(
                update={
                    "details_verification_attempts": attempts,
                    "details_verification_status": verification_status,
                    "matched_fields": all_matched,
                    "field_to_verify": field_names.get(field_to_verify, field_to_verify),
                    "current_step": CallStep.DETAILS_VERIFICATION.value
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced conversational prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        params["matched_fields_display"] = _format_matched_fields(state.get("matched_fields", []))
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format enhanced prompt
        prompt_content = DETAILS_VERIFICATION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Details Verification Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed - verification logic handles details
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedDetailsVerificationAgent"
    )
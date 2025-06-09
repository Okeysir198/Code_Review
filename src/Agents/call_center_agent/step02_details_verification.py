# src/Agents/call_center_agent/step02_details_verification.py
"""
Enhanced Details Verification Agent - Natural phone conversation with security verification and fuzzy matching
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

# Natural conversation prompt for details verification
DETAILS_VERIFICATION_PROMPT = """You are {agent_name} from Cartrack Accounts Department on an OUTBOUND PHONE CALL to {client_title} {client_full_name} about their {outstanding_amount} overdue account.

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
Verification attempt: {details_verification_attempts}/{max_details_verification_attempts}
Currently verified: {matched_fields_display} | Need to verify: {field_to_verify}
Verification status: {details_verification_status}
Your goal: Get {field_to_verify} to complete security verification before discussing account
</context>

<verification_requirements>
Security requires EITHER:
- ID number or passport number (most secure - single item sufficient)
OR  
- Three account details: username, vehicle info (registration/make/model/color), email
Currently verified: {matched_fields_display}
</verification_requirements>

<security_approach>
First attempt (with recording notice):
Standard urgency: "This call is recorded for security. I need to verify your {field_to_verify} before we discuss your account"
High urgency: "This call is recorded for security. I need your {field_to_verify} immediately to discuss your urgent account matter"
Critical urgency: "This call is recorded for security. I must verify your {field_to_verify} now for this critical account matter"

Follow-up attempts (no recording notice):
Standard urgency: "I still need your {field_to_verify} to proceed securely"
High urgency: "Your {field_to_verify} is required for this urgent account matter"
Critical urgency: "Provide your {field_to_verify} immediately - this is urgent"

Near match retry (when close but not exact):
"I have {field_to_verify} as something similar - can you repeat that clearly?"
"Can you give me your {field_to_verify} again, with all the digits clearly?"
"Let me get that {field_to_verify} one more time to make sure I have it right"

If resistant:
Standard: "This protects your account information - your {field_to_verify} please"
High/Critical: "Security verification is mandatory for overdue accounts"
</security_approach>

<conversation_adaptation>
If they ask why: "It's standard security before discussing account details"
If they're concerned about sharing: "This verifies you're the account holder"
If they ask what happens next: "Once verified, we'll discuss your account situation"
If they seem confused: "I just need your {field_to_verify} to confirm your identity"

For ID number specifically: "Your 13-digit ID number please"
For birth date specifically: "Your date of birth please"
For vehicle details: "Your vehicle registration number" or "What make is your tracked vehicle?"
For email: "The email address on your account"

For near matches (retry scenarios):
- If close but not exact: "I have {field_to_verify} as something similar - can you repeat that clearly?"
- If formatting issue: "Can you give me your {field_to_verify} again, with all the digits/letters?"
- If unclear: "Let me get that {field_to_verify} one more time to make sure"
</conversation_adaptation>

<natural_conversation_rules>
- Speak naturally like a real phone conversation
- NO brackets [ ], asterisks *, or placeholder formatting
- NO internal system variables or markdown in your response
- Use actual names or speak generally if you don't know specifics
- Just natural spoken words as if talking to a real person
- Be professional but human - you're helping them access their account securely
</natural_conversation_rules>


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
    """Create enhanced details verification agent with natural conversation flow and fuzzy matching"""
    
    FIELD_PRIORITY = ["id_number", "passport_number", "birth_date", "vehicle_registration", 
                     "vehicle_make", "vehicle_model", "vehicle_color", "email", "username"]
    
    def _get_available_fields(client_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract verification fields from client data"""
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {})
        vehicles = profile.get("vehicles", [])
        
        fields = {}
        if client_info.get("id_number"): 
            fields["id_number"] = client_info["id_number"]
        if client_info.get("passport_number"):
            fields["passport_number"] = client_info["passport_number"]
        if client_info.get("birth_date") or client_info.get("date_of_birth"):
            # Handle different possible field names for birth date
            birth_date = client_info.get("birth_date") or client_info.get("date_of_birth")
            if birth_date:
                fields["birth_date"] = str(birth_date)
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
        """Select next verification field with strategic priority and randomization"""
        remaining = [f for f in FIELD_PRIORITY if f in available_fields and f not in matched_fields]
        if not remaining: 
            return "id_number"
        
        # Tier 1: ID/Passport (single verification sufficient) - always prioritize
        tier1 = [f for f in ["id_number", "passport_number"] if f in remaining]
        if tier1:
            return random.choice(tier1)  # Random if both available
        
        # Tier 2: Birth date (highly secure, memorable) - high priority
        if "birth_date" in remaining:
            return "birth_date"
        
        # Tier 3: Vehicle registration (concrete, memorable) - high priority
        if "vehicle_registration" in remaining:
            return "vehicle_registration"
        
        # Tier 4: Other vehicle details - randomize within tier
        tier4 = [f for f in ["vehicle_make", "vehicle_model", "vehicle_color"] if f in remaining]
        if tier4:
            return random.choice(tier4)
        
        # Tier 5: Email/username - randomize within tier
        tier5 = [f for f in ["email", "username"] if f in remaining]
        if tier5:
            return random.choice(tier5)
            
        return "id_number"
    
    def _fuzzy_match_field(client_input: str, system_value: str, field_type: str) -> tuple[bool, float]:
        """
        Fuzzy matching for verification fields with confidence score
        Returns (is_match, confidence_score)
        """
        if not client_input or not system_value:
            return False, 0.0
        
        # Normalize inputs
        client_clean = client_input.strip().lower()
        system_clean = system_value.strip().lower()
        
        # Exact match
        if client_clean == system_clean:
            return True, 1.0
        
        # Field-specific fuzzy matching
        if field_type == "id_number":
            # Remove common formatting (spaces, dashes)
            client_digits = ''.join(filter(str.isdigit, client_clean))
            system_digits = ''.join(filter(str.isdigit, system_clean))
            
            if client_digits == system_digits:
                return True, 0.95  # High confidence for format difference
            
            # Check if only 1-2 digit differences (typos)
            if len(client_digits) == len(system_digits) == 13:
                differences = sum(c != s for c, s in zip(client_digits, system_digits))
                if differences <= 2:
                    return True, 0.7  # Medium confidence for typos
        
        elif field_type == "birth_date":
            # Try to parse different date formats
            import re
            from datetime import datetime
            
            date_patterns = [
                (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', '%d/%m/%Y'),  # DD/MM/YYYY
                (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', '%Y/%m/%d'),  # YYYY/MM/DD
                (r'(\d{1,2})\s+(\w+)\s+(\d{4})', '%d %B %Y'),        # DD Month YYYY
            ]
            
            def parse_date(date_str):
                for pattern, fmt in date_patterns:
                    match = re.search(pattern, date_str)
                    if match:
                        try:
                            if 'B' in fmt:  # Month name
                                return datetime.strptime(date_str, fmt).date()
                            else:
                                return datetime.strptime('/'.join(match.groups()), fmt.replace('-', '/')).date()
                        except:
                            continue
                return None
            
            client_date = parse_date(client_input)
            system_date = parse_date(system_value)
            
            if client_date and system_date and client_date == system_date:
                return True, 0.9  # High confidence for date format difference
        
        elif field_type in ["vehicle_registration", "vehicle_make", "vehicle_model", "vehicle_color"]:
            # Remove spaces and common formatting
            client_alpha = ''.join(filter(str.isalnum, client_clean))
            system_alpha = ''.join(filter(str.isalnum, system_clean))
            
            if client_alpha == system_alpha:
                return True, 0.9  # High confidence for formatting difference
            
            # Partial match for vehicle details
            if len(client_alpha) >= 3 and len(system_alpha) >= 3:
                if client_alpha in system_alpha or system_alpha in client_alpha:
                    return True, 0.75  # Medium-high confidence for partial match
        
        elif field_type == "email":
            # Remove common variations
            client_email = client_clean.replace('.', '').replace('-', '').replace('_', '')
            system_email = system_clean.replace('.', '').replace('-', '').replace('_', '')
            
            if client_email == system_email:
                return True, 0.8  # Good confidence for formatting difference
        
        return False, 0.0
    
    def _enhanced_verification_check(messages: List, available_fields: Dict[str, str]) -> Dict[str, Any]:
        """
        Enhanced verification with fuzzy matching and retry logic
        Returns detailed verification results
        """
        if not messages:
            return {"has_details": False, "matches": [], "near_matches": [], "confidence": 0.0}
        
        # Get last client message
        last_msg = ""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                last_msg = message.content.strip()
                break
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                last_msg = message.content.strip()
                break
        
        if not last_msg:
            return {"has_details": False, "matches": [], "near_matches": [], "confidence": 0.0}
        
        matches = []
        near_matches = []
        max_confidence = 0.0
        
        # Check each available field for matches
        for field_name, field_value in available_fields.items():
            is_match, confidence = _fuzzy_match_field(last_msg, field_value, field_name)
            
            if is_match:
                if confidence >= 0.85:  # High confidence threshold
                    matches.append({
                        "field": field_name,
                        "confidence": confidence,
                        "type": "exact" if confidence == 1.0 else "high_fuzzy"
                    })
                elif confidence >= 0.6:  # Medium confidence threshold
                    near_matches.append({
                        "field": field_name,
                        "confidence": confidence,
                        "client_input": last_msg,
                        "expected_value": field_value,
                        "type": "medium_fuzzy"
                    })
                
                max_confidence = max(max_confidence, confidence)
        
        # Also check for pattern matches (fallback)
        has_pattern_match = _quick_details_check(messages, available_fields)
        
        return {
            "has_details": len(matches) > 0 or len(near_matches) > 0 or has_pattern_match,
            "matches": matches,
            "near_matches": near_matches,
            "confidence": max_confidence,
            "pattern_detected": has_pattern_match
        }

    def _quick_details_check(messages: List, available_fields: Dict[str, str]) -> bool:
        """Fast pattern detection for common verification info"""
        if not messages:
            return False
            
        # Get last client message
        last_msg = ""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                last_msg = message.content.strip()
                break
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                last_msg = message.content.strip()
                break
        
        if not last_msg:
            return False
        
        # Check for common patterns
        import re
        
        # ID number patterns (13 digits for SA ID)
        if re.search(r'\b\d{13}\b', last_msg):
            return True
        
        # Date patterns (birth date)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b'  # DD Month YYYY
        ]
        if any(re.search(pattern, last_msg, re.IGNORECASE) for pattern in date_patterns):
            return True
        
        # Vehicle registration patterns
        reg_patterns = [r'\b[A-Z]{2,3}[\s\-]?\d{3,4}[\s\-]?[A-Z]{2,3}\b', r'\b\d{3}[\s\-]?\d{3}[\s\-]?\d{3}\b']
        if any(re.search(pattern, last_msg.upper()) for pattern in reg_patterns):
            return True
        
        # Email patterns
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', last_msg):
            return True
        
        # Vehicle details mentioned
        for field, value in available_fields.items():
            if field.startswith("vehicle_") and value and value.lower() in last_msg.lower():
                return True
        
        return False
    
    def _format_matched_fields(matched_fields: List[str]) -> str:
        """Format matched fields for natural display"""
        if not matched_fields:
            return "None yet"
        
        field_names = {
            "id_number": "ID Number", 
            "passport_number": "Passport Number",
            "birth_date": "Birth Date",
            "email": "Email", 
            "username": "Username",
            "vehicle_registration": "Vehicle Registration", 
            "vehicle_make": "Vehicle Make",
            "vehicle_model": "Vehicle Model", 
            "vehicle_color": "Vehicle Color"
        }
        
        return ", ".join([field_names.get(field, field.title()) for field in matched_fields])
    
    def _format_field_request(field: str) -> str:
        """Format field name for natural conversation"""
        field_names = {
            "id_number": "ID number", 
            "passport_number": "passport number",
            "birth_date": "date of birth",
            "username": "username", 
            "email": "email address",
            "vehicle_registration": "vehicle registration number", 
            "vehicle_make": "vehicle make",
            "vehicle_model": "vehicle model", 
            "vehicle_color": "vehicle color"
        }
        return field_names.get(field, field.replace("_", " "))
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Enhanced preprocessing with natural conversation flow and fuzzy matching"""
        
        attempts = state.get("details_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_details_verification_attempts", 5)
        matched_fields = state.get("matched_fields", [])
        available_fields = _get_available_fields(client_data)
        
        # Select field to verify
        field_to_verify = _select_next_field(available_fields, matched_fields)
        
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        all_matched = matched_fields.copy()
        
        messages = state.get("messages", [])
        
        # Enhanced verification check with fuzzy matching
        verification_result = _enhanced_verification_check(messages, available_fields)
        
        if verification_result["has_details"]:
            try:
                # Use verification tool for comprehensive analysis
                result = verify_client_details.invoke({
                    "client_details": available_fields,
                    "messages": messages,
                    "required_match_count": 3,
                    "max_failed_attempts": max_attempts
                })
                
                new_matched = result.get("matched_fields", [])
                all_matched = list(set(matched_fields + new_matched))
                verification_status = result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
                
                # Handle near matches that need clarification
                if (verification_status == VerificationStatus.INSUFFICIENT_INFO.value and 
                    verification_result["near_matches"] and 
                    attempts < max_attempts):
                    
                    # Add retry context for near matches
                    near_match = verification_result["near_matches"][0]  # Take the best near match
                    field_to_verify = near_match["field"]
                    
                    # Set state for retry with clarification
                    return Command(
                        update={
                            "details_verification_attempts": attempts,
                            "details_verification_status": "NEAR_MATCH_RETRY",
                            "matched_fields": all_matched,
                            "field_to_verify": _format_field_request(field_to_verify),
                            "near_match_info": near_match,
                            "current_step": CallStep.DETAILS_VERIFICATION.value
                        },
                        goto="agent"
                    )
                
                if verbose:
                    logger.info(f"Verification tool result: {verification_status}, matched: {all_matched}")
                
            except Exception as e:
                if verbose: 
                    logger.error(f"Verification tool error: {e}")
        
        # Auto-fail if max attempts reached without success
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        # Route based on verification outcome
        if verification_status == VerificationStatus.VERIFIED.value:
            logger.info("Details verification VERIFIED - moving to reason for call")
            return Command(
                update={
                    "details_verification_attempts": attempts,
                    "details_verification_status": verification_status,
                    "matched_fields": all_matched,
                    "field_to_verify": _format_field_request(field_to_verify),
                    "current_step": CallStep.REASON_FOR_CALL.value
                },
                goto="__end__"
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
            logger.info(f"Details verification continuing - attempt {attempts}/{max_attempts}")
            return Command(
                update={
                    "details_verification_attempts": attempts,
                    "details_verification_status": verification_status,
                    "matched_fields": all_matched,
                    "field_to_verify": _format_field_request(field_to_verify),
                    "current_step": CallStep.DETAILS_VERIFICATION.value
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate natural conversation prompt for security verification"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, script_type, agent_name)
        params["matched_fields_display"] = _format_matched_fields(state.get("matched_fields", []))
        
        # Format prompt for natural conversation
        prompt_content = DETAILS_VERIFICATION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Details Verification Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # Verification logic handled in preprocessing
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedDetailsVerificationAgent"
    )
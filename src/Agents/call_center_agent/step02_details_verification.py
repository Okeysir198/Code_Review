# ===============================================================================
# STEP 02: DETAILS VERIFICATION AGENT - Updated with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step02_details_verification.py
"""
Details Verification Agent - Enhanced with aging-aware script integration
"""
import random
import logging
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value
from src.Agents.call_center_agent.tools.verify_client_details import verify_client_details
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

logger = logging.getLogger(__name__)

def get_details_verification_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate aging-aware details verification prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Extract verification info
    field_to_verify = state.get("field_to_verify", "ID number")
    status = state.get("details_verification_status", "INSUFFICIENT_INFO")
    attempts = state.get("details_verification_attempts", 1)
    max_attempts = 5
    matched_fields = state.get("matched_fields", [])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<verification_context>
- Status: {status}
- Attempt: {attempts}/{max_attempts}
- Requesting: {field_to_verify}
- Already Verified: {matched_fields}
- Urgency Level: {aging_context['urgency']}
- Account Category: {aging_context['category']}
</verification_context>

<task>
Complete identity verification for data protection compliance. Adapt urgency to account status.
</task>

<verification_requirements>
**Option A**: Full ID number (sufficient alone)
**Option B**: THREE items from available fields
</verification_requirements>

<approach_by_urgency>
**Standard/Medium Urgency**: 
- Security Notice: "This call is recorded for quality and security purposes"
- Request: "Please provide your {field_to_verify}"
- If Resistant: "This protects your account information"

**High Urgency**: 
- Security Notice: "This call is recorded. Due to the urgency of your account status"
- Request: "I need your {field_to_verify} immediately to proceed"
- If Resistant: "Security verification is required for overdue accounts"

**Legal/Critical Urgency**:
- Security Notice: "This is a legal matter. I must verify your identity"
- Request: "Provide your {field_to_verify} now to proceed with this matter"
- If Resistant: "Legal proceedings require proper identification"
</approach_by_urgency>

<critical_rules>
- ONE field at a time: currently {field_to_verify}
- NO account details until FULLY verified
- Adapt persistence to urgency level
- Security is non-negotiable regardless of urgency
</critical_rules>

<urgency_guidance>
{aging_context['approach']}
</urgency_guidance>

<style>
- {aging_context['tone']}
- Adapt formality to urgency level
- Clear, specific requests
- Acknowledge each successful verification
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.DETAILS_VERIFICATION,
        client_data=client_data,
        state=state
    )

def create_details_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a details verification agent with aging-aware scripts."""
    
    # Verification field priority - most secure to least secure
    FIELD_PRIORITY = [
        "id_number", "passport_number",  
        "username", "vehicle_registration", "vehicle_make", "vehicle_model", 
        "vehicle_color", "email"
    ]
    
    def _get_available_fields(client_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract available verification fields from client data."""
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {}) if isinstance(profile, dict) else {}
        vehicles = profile.get("vehicles", []) if isinstance(profile, dict) else []
        
        verification_info = {}
        
        # ID/Passport
        if client_info.get("id_number"):
            verification_info["id_number"] = client_info["id_number"]
        
        # Username
        if profile.get("user_name"):
            verification_info["username"] = profile["user_name"]
        
        # Email
        if client_info.get("email_address"):
            verification_info["email"] = client_info["email_address"]
        
        # Vehicle information
        if vehicles and isinstance(vehicles[0], dict):
            vehicle = vehicles[0]
            field_mappings = [
                ("vehicle_registration", "registration"),
                ("vehicle_make", "make"),
                ("vehicle_model", "model"),
                ("vehicle_color", "color")
            ]
            for field, key in field_mappings:
                if vehicle.get(key):
                    verification_info[field] = vehicle[key]
        
        return verification_info
    
    def _select_next_field(available_fields: Dict[str, str], matched_fields: List[str]) -> str:
        """Select next field to verify based on priority."""
        remaining_fields = [f for f in FIELD_PRIORITY if f in available_fields and f not in matched_fields]

        if not remaining_fields:
            return "id_number"

        # Prioritize certain fields, then randomly select from the rest
        high_priority_fields = [f for f in FIELD_PRIORITY[:2] if f in remaining_fields]
        if high_priority_fields:
            return high_priority_fields[0]

        # If high priority fields are not available, shuffle the rest
        other_remaining_fields = [f for f in remaining_fields if f not in FIELD_PRIORITY[:2]]
        if other_remaining_fields:
            random.shuffle(other_remaining_fields)
            return other_remaining_fields[0]
        
        return "id_number"
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        attempts = state.get("details_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_details_verification_attempts", 5)
        
        matched_fields = state.get("matched_fields", [])
        available_fields = _get_available_fields(client_data)
        field_to_verify = _select_next_field(available_fields, matched_fields)
        
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        all_matched = matched_fields
        
        try:
            verification_result = verify_client_details.invoke({
                "client_details": available_fields,
                "messages": state.get("messages", []),
                "required_match_count": 3,
                "max_failed_attempts": max_attempts
            })
            
            new_matched = verification_result.get("matched_fields", [])
            all_matched = list(set(matched_fields + new_matched))
            verification_status = verification_result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
            
            if verbose:
                logger.info(f"Details verification - Status: {verification_status}, Matched: {all_matched}")
            
        except Exception as e:
            if verbose:
                logger.error(f"Details verification error: {e}")
        
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        field_display_names = {
            "id_number": "ID number",
            "passport_number": "passport number",
            "username": "username", 
            "email": "email address",
            "vehicle_registration": "vehicle registration",
            "vehicle_make": "vehicle make",
            "vehicle_model": "vehicle model", 
            "vehicle_color": "vehicle color"
        }
        
        return Command(
            update={
                "details_verification_attempts": attempts,
                "details_verification_status": verification_status,
                "matched_fields": all_matched,
                "field_to_verify": field_display_names.get(field_to_verify, field_to_verify),
                "available_fields": list(available_fields.keys()),
                "current_step": CallStep.DETAILS_VERIFICATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_details_verification_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="DetailsVerificationAgent"
    )

# src/Agents/call_center_agent/step02_details_verification.py
"""
Details Verification Agent - Lean version with concise responses
"""
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate 
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
from src.Agents.call_center_agent.tools.verify_client_details import verify_client_details

logger = logging.getLogger(__name__)

# Optimized prompt template - ensures concise responses
DETAILS_VERIFICATION_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today time: {current_date}
</role>
                                                               
<context>
Target client: {client_full_name} | Outstanding amount: {outstanding_amount} | Details Verification Status: {details_verification_status}
Verification Attempt: {details_verification_attempts}/{max_details_verification_attempts} 
Need to verify: {field_to_verify} | Verified items: {matched_fields}
Urgency: {urgency_level} | Category: {aging_category} 
user_id: {user_id}
</context>

<script>
{formatted_script}
</script>

<task>
Get {field_to_verify} for verification. Match urgency to account severity.
</task>

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

<rules>
- Request ONE field only: {field_to_verify}
- NO account details until verified
- Match tone to {urgency_level} urgency
- Security non-negotiable
</rules>

<response_style>
CRITICAL: Keep responses under 15 words. Be direct. No explanations unless asked.
Examples:
✓ "Your ID number please"
✓ "I need your email address for verification"  
✗ "For security purposes and to ensure I'm speaking with the right person, could you please provide..."
</response_style>
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
    """Create details verification agent with concise responses"""
    
    FIELD_PRIORITY = ["id_number", "passport_number", "vehicle_registration", 
                     "vehicle_make", "vehicle_model", "vehicle_color", "email", "username"]
    
    def _get_available_fields(client_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract verification fields"""
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {})
        vehicles = profile.get("vehicles", [])
        
        fields = {}
        if client_info.get("id_number"): fields["id_number"] = client_info["id_number"]
        if profile.get("user_name"): fields["username"] = profile["user_name"]
        if client_info.get("email_address"): fields["email"] = client_info["email_address"]
        
        if vehicles and isinstance(vehicles[0], dict):
            v = vehicles[0]
            if v.get("registration"): fields["vehicle_registration"] = v["registration"]
            if v.get("make"): fields["vehicle_make"] = v["make"]
            if v.get("model"): fields["vehicle_model"] = v["model"]
            if v.get("color"): fields["vehicle_color"] = v["color"]
        
        return fields
    
    def _select_next_field(available_fields: Dict[str, str], matched_fields: List[str]) -> str:
        """Select next verification field"""
        remaining = [f for f in FIELD_PRIORITY if f in available_fields and f not in matched_fields]
        if not remaining: return "id_number"
        
        high_priority = [f for f in FIELD_PRIORITY[:2] if f in remaining]
        if high_priority: return high_priority[0]
        
        other_fields = [f for f in remaining if f not in FIELD_PRIORITY[:2]]
        if other_fields:
            random.shuffle(other_fields)
            return other_fields[0]
        return "id_number"
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Process verification attempt"""
        attempts = state.get("details_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_details_verification_attempts", 5)
        
        matched_fields = state.get("matched_fields", [])
        available_fields = _get_available_fields(client_data)
        field_to_verify = _select_next_field(available_fields, matched_fields)
        
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        all_matched = matched_fields
        
        try:
            result = verify_client_details.invoke({
                "client_details": available_fields,
                "messages": state.get("messages", []),
                "required_match_count": 3,
                "max_failed_attempts": max_attempts
            })
            
            new_matched = result.get("matched_fields", [])
            all_matched = list(set(matched_fields + new_matched))
            verification_status = result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
            
        except Exception as e:
            if verbose: logger.error(f"Verification error: {e}")
        
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        field_names = {
            "id_number": "ID number", "passport_number": "passport number",
            "username": "username", "email": "email address",
            "vehicle_registration": "vehicle registration", "vehicle_make": "vehicle make",
            "vehicle_model": "vehicle model", "vehicle_color": "vehicle color"
        }
        
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
        """Generate concise verification prompt"""
        # Step 1: Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
         # Step 2: Format script
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.NAME_VERIFICATION)
        formatted_script = script_template.format(**params) if script_template else f"Are you {params['client_full_name']}?"
        params["formatted_script"] = formatted_script
        
        # Step 3: Format prompt
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        params["tone"] = aging_context['tone']
        
        prompt_content = DETAILS_VERIFICATION_PROMPT.format(**params)
        
        if verbose: print(f"Details Verification Prompt: {prompt_content}")
        
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
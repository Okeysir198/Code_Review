# ./src/Agents/call_center_agent/step02_details_verification.py
"""
Details Verification Agent - Simplified for security verification process.
"""
import random
import logging
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.call_scripts import ScriptType
from src.Agents.call_center_agent.tools.verify_client_details import verify_client_details
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_details_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = ScriptType.RATIO_1_INFLOW.value,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a simplified details verification agent."""
    
    # Verification field priority - most secure to least secure
    FIELD_PRIORITY = [
        "id_number", "passport_number", "vehicle_registration", 
        "username", "vehicle_make", "vehicle_model", 
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
        # Remove already matched fields
        remaining_fields = [f for f in FIELD_PRIORITY if f in available_fields and f not in matched_fields]
        
        # Return highest priority remaining field
        return remaining_fields[0] if remaining_fields else "id_number"
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Command[Literal["__end__", "agent"]]:
        """Verify client details and determine next step."""
        
        # Increment attempt counter
        attempts = state.get("details_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_details_verification_attempts", 5)
        
        # Get matched fields and select next field to verify
        matched_fields = state.get("matched_fields", [])
        available_fields = _get_available_fields(client_data)
        field_to_verify = _select_next_field(available_fields, matched_fields)
        
        # Perform verification using the tool
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        all_matched = matched_fields
        
        try:
            logger.info(f"available_fields: {available_fields}")
            verification_result = verify_client_details.invoke({
                "client_details": available_fields,
                "messages": state.get("messages", []),
                "required_match_count": 3,
                "max_failed_attempts": max_attempts
            })
            
            # Update matched fields
            new_matched = verification_result.get("matched_fields", [])
            all_matched = list(set(matched_fields + new_matched))
            verification_status = verification_result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
            
        except Exception as e:
            if verbose:
                print(f"Details verification error: {e}")
        
        # Handle max attempts reached
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        # Map field to human-readable format for prompt
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
        
        # Determine routing: VERIFIED goes to reason_for_call, otherwise continue verification
        if verification_status == VerificationStatus.VERIFIED.value:
            next_step = CallStep.REASON_FOR_CALL.value
            goto = "__end__"
        elif verification_status == VerificationStatus.VERIFICATION_FAILED.value:
            next_step = CallStep.CLOSING.value
            goto = "__end__"
        else:
            next_step = CallStep.DETAILS_VERIFICATION.value
            goto = "agent"
        
        return Command(
            update={
                "details_verification_attempts": attempts,
                "details_verification_status": verification_status,
                "matched_fields": all_matched,
                "field_to_verify": field_display_names.get(field_to_verify, field_to_verify),
                "current_step": CallStep.DETAILS_VERIFICATION.value,
                "next_step": next_step
            },
            goto=goto
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for details verification step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.DETAILS_VERIFICATION.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.DETAILS_VERIFICATION.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
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
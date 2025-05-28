# ./src/Agents/call_center_agent/step02_details_verification.py
"""
Details Verification Agent for Call Center.
"""
import random
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


def create_details_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = ScriptType.RATIO_1_INFLOW.value,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a details verification agent for debt collection calls."""
    
    # Verification field priority
    FIELD_PRIORITY = [
        "id_number", "passport_number", "vehicle_registration", 
        "username", "vehicle_make", "vehicle_model", 
        "vehicle_color", "email"
    ]
    
    def _select_verification_field(client_data: Dict[str, Any], matched_fields: List[str]) -> str:
        """Select next verification field based on available data."""
        # Get available fields from client data
        available_fields = []
        
        # Extract verification info from client data
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {}) if isinstance(profile, dict) else {}
        vehicles = profile.get("vehicles", []) if isinstance(profile, dict) else []
        
        # Check available verification fields
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
            for field, key in [
                ("vehicle_registration", "registration"),
                ("vehicle_make", "make"),
                ("vehicle_model", "model"),
                ("vehicle_color", "color")
            ]:
                if vehicle.get(key):
                    verification_info[field] = vehicle[key]
        
        # Build available fields list based on priority
        for field in FIELD_PRIORITY:
            if field in verification_info and verification_info[field]:
                available_fields.append(field)
        
        # Remove already matched fields
        remaining_fields = [f for f in available_fields if f not in matched_fields]
        
        # Return random choice from remaining or first available
        return random.choice(remaining_fields) if remaining_fields else available_fields[0] if available_fields else "id_number"
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Command[Literal["__end__", "agent"]]:
        """Pre-process state to verify client details."""
        
        # Increment attempt counter
        attempts = state.get("details_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_details_verification_attempts", 5)
        
        # Select next field to verify
        matched_fields = state.get("matched_fields", [])
        field_to_verify = _select_verification_field(client_data, matched_fields)
        
        # Perform details verification
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        current_step = CallStep.DETAILS_VERIFICATION.value
        goto = "agent"
        
        try:
            verification_result = verify_client_details.invoke({
                "client_details": client_data,
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
            all_matched = matched_fields
        
        # Handle max attempts
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        # Determine next step - only VERIFIED proceeds to reason for call
        if verification_status == VerificationStatus.VERIFIED.value:
            next_step = CallStep.REASON_FOR_CALL.value
            goto = "__end__"
        elif verification_status == VerificationStatus.VERIFICATION_FAILED.value:
            next_step = CallStep.CLOSING.value
            goto = "__end__"
        else:
            # Continue with details verification (retry)
            next_step = CallStep.DETAILS_VERIFICATION.value
            goto = "agent"
        
        return Command(
            update={
                "details_verification_attempts": attempts,
                "details_verification_status": verification_status,
                "matched_fields": all_matched,
                "field_to_verify": field_to_verify,
                "current_step": current_step,
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
    
    # Configure basic agent
    kwargs = {
        "model": model,
        "prompt": dynamic_prompt,
        "tools": tools or [],
        "pre_processing_node": pre_processing_node,
        "state_schema": CallCenterAgentState,
        "verbose": verbose,
        "name": "DetailsVerificationAgent"
    }
    
    # Add memory if configured
    if config and config.get('configurable', {}).get('use_memory'):
        from langgraph.checkpoint.memory import MemorySaver
        kwargs["checkpointer"] = MemorySaver()
    
    return create_basic_agent(**kwargs)
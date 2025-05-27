# ./src/Agents/call_center_agent/details_verification.py
"""
Details Verification Agent for Call Center.
"""
import random
from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph

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
    verbose: bool = False
) -> CompiledGraph:
    """
    Create a details verification agent for debt collection calls.
    
    Args:
        model: Language model to use
        client_data: client information
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled details verification agent workflow
    """
    
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
        
        for field in FIELD_PRIORITY:
            if field in client_data.get("verification_info", {}) and client_data["verification_info"][field]:
                available_fields.append(field)
        
        # Remove already matched fields
        remaining_fields = [f for f in available_fields if f not in matched_fields]
        
        # Return random choice from remaining or first available
        return random.choice(remaining_fields) if remaining_fields else available_fields[0] if available_fields else "id_number"
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process state to verify client details."""
        # Increment attempt counter
        attempts = state.get("details_verification_attempts", -1) + 1
        
        # Select next field to verify
        field_to_verify = _select_verification_field(client_data, state.get("matched_fields",[]))
        
        # Perform details verification
        try:
            verification_result = verify_client_details.invoke({
                "client_details": client_data,
                "messages": state.get("messages",[]),
                "required_match_count": 3,
                "max_failed_attempts": state.get("max_details_verification_attempts", 5)
            })
            
            # Update matched fields
            new_matched = verification_result.get("matched_fields", [])
            all_matched = list(set(state.get("matched_fields",[]) + new_matched))
            
            return {
                "details_verification_attempts": attempts,
                "details_verification_status": verification_result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value),
                "matched_fields": all_matched,
                "field_to_verify": field_to_verify,
            }
            
        except Exception as e:
            if verbose:
                print(f"Details verification error: {e}")
            
            return {
                "details_verification_attempts": attempts,
                "details_verification_status": VerificationStatus.INSUFFICIENT_INFO.value,
                "matched_fields": state.get("matched_fields",[]),
                "field_to_verify": field_to_verify,
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for details verification step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.DETAILS_VERIFICATION.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.DETAILS_VERIFICATION.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="DetailsVerificationAgent"
    )
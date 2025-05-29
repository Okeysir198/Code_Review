# ./src/Agents/call_center_agent/step01_name_verification.py
"""
Name Verification Agent - Optimized with pre-processing only.
Confirms client identity through name verification.
"""
from typing import Dict, Any, Optional, List, Literal
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.call_scripts import ScriptType
from src.Agents.call_center_agent.tools.verify_client_name import verify_client_name

logger = logging.getLogger(__name__)

def create_name_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = ScriptType.RATIO_1_INFLOW.value,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a name verification agent for debt collection calls."""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to analyze conversation and update verification status."""
        
        # Extract client info
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {}) if profile else {}
        client_full_name = client_info.get("client_full_name", "Client")
        
        # Increment attempts
        attempts = state.get("name_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_name_verification_attempts", 5)
        
        # Perform verification using conversation analysis
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        
        try:
            result = verify_client_name.invoke({
                "client_full_name": client_full_name,
                "messages": state.get("messages", []),
                "max_failed_attempts": max_attempts
            })
            verification_status = result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
            
            if verbose:
                logger.info(f"Name verification result: {verification_status}")
                
        except Exception as e:
            if verbose:
                logger.error(f"Name verification error: {e}")
        
        # Handle max attempts reached
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        return Command(
            update={
                "name_verification_status": verification_status,
                "name_verification_attempts": attempts,
                "client_full_name": client_full_name,
                "current_step": CallStep.NAME_VERIFICATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for name verification step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.NAME_VERIFICATION.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.NAME_VERIFICATION.value, parameters)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="NameVerificationAgent"
    )
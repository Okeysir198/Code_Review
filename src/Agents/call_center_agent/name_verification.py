# ./src/Agents/call_center_agent/name_verification.py
"""
Optimized Name Verification Agent for Call Center.
"""
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
from src.Agents.call_center_agent.tools.verify_client_name import verify_client_name


def create_name_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = ScriptType.RATIO_1_INFLOW.value,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False
) -> CompiledGraph:
    """
    Create a name verification agent for debt collection calls.
    
    Args:
        model: Language model to use
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled name verification agent workflow
    """
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process state to verify client name."""
        # Increment attempt counter
        attempts = state.get("name_verification_attempts", -1) + 1
        

        
        # Extract client name safely
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {}) if profile else {}
        client_full_name = client_info.get("client_full_name", "Client")
        
        # Perform name verification
        try:
            verification_result = verify_client_name.invoke({
                "client_full_name": client_full_name,
                "messages": state.get("messages",[]),
                "max_failed_attempts": state.get("max_name_verification_attempts",5)
            })
            
            return {
                "client_full_name": client_full_name,
                "name_verification_status": verification_result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value),
                "name_verification_attempts": attempts,
            }
            
        except Exception as e:
            if verbose:
                print(f"Name verification error: {e}")
            
            return {
                "client_full_name": client_full_name,
                "name_verification_status": VerificationStatus.INSUFFICIENT_INFO.value,
                "name_verification_attempts": attempts,
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for name verification step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.NAME_VERIFICATION.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.NAME_VERIFICATION.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="NameVerificationAgent"
    )
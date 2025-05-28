"""
Call Center Agent Workflow - Orchestrates specialized sub-agents for debt collection calls.
"""
import logging
from typing import Literal, Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# Import existing components
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.introduction import create_introduction_agent
from src.Agents.call_center_agent.name_verification import create_name_verification_agent
from src.Agents.call_center_agent.details_verification import create_details_verification_agent
from src.Agents.call_center_agent.reason_for_call import create_reason_for_call_agent
from src.Agents.call_center_agent.negotiation import create_negotiation_agent
from src.Agents.call_center_agent.data_parameter_builder import get_client_data, get_client_data_async

logger = logging.getLogger(__name__)



def create_call_center_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """
    Create the main call center agent workflow that orchestrates specialized sub-agents.
    
    Args:
        model: Language model for all agents
        client_data: Client information from database
        script_type: Script type for the call
        agent_name: Name of the agent
        config: Configuration options
        
    Returns:
        Compiled main workflow graph
    """
    config = config or {}
    
    # Create specialized sub-agents
    introduction_agent = create_introduction_agent(model, client_data, script_type, agent_name, config=config)
    name_verification_agent = create_name_verification_agent(model, client_data, script_type, agent_name, config=config)
    details_verification_agent = create_details_verification_agent(model, client_data, script_type, agent_name, config=config)
    reason_for_call_agent = create_reason_for_call_agent(model, client_data, script_type, agent_name, config=config)
    negotiation_agent = create_negotiation_agent(model, client_data, script_type, agent_name, config=config)

    # Router function
    def route_to_agent(state: CallCenterAgentState) -> str:
        """Route to appropriate agent based on conversation state."""
        next_step = state.get("next_step") or state.get("current_step", CallStep.INTRODUCTION.value)
        
        step_mapping = {
            CallStep.INTRODUCTION.value: CallStep.INTRODUCTION.value,
            CallStep.NAME_VERIFICATION.value: CallStep.NAME_VERIFICATION.value,
            CallStep.DETAILS_VERIFICATION.value: CallStep.DETAILS_VERIFICATION.value,
            CallStep.REASON_FOR_CALL.value: CallStep.REASON_FOR_CALL.value,
            CallStep.NEGOTIATION.value: CallStep.NEGOTIATION.value
        }
        
        return step_mapping.get(next_step, CallStep.INTRODUCTION.value)

    # Node functions
    def router_node(state: CallCenterAgentState) -> Dict[str, Any]:
        """Router determines next step."""
        return {"current_step": state.get("current_step", CallStep.INTRODUCTION.value)}

    def introduction_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Introduction step - always ends to wait for response."""
        result = introduction_agent.invoke(state)
        
        return Command(
            update={
                "messages": result.get("messages", state.get("messages", [])),
                "current_step": result.get("current_step", CallStep.INTRODUCTION.value),
                "next_step": result.get("next_step")
            },
            goto="__end__"
        )

    def name_verification_node(state: CallCenterAgentState) -> Command[Literal["details_verification", "__end__"]]:
        """Name verification step."""
        result = name_verification_agent.invoke(state)
        
        status = result.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        
        update = {
            "messages": result.get("messages", state.get("messages", [])),
            "name_verification_status": status,
            "name_verification_attempts": result.get("name_verification_attempts", 0),
            "current_step": CallStep.NAME_VERIFICATION.value
        }
        
        # Route based on verification status
        if status == VerificationStatus.VERIFIED.value:
            update["current_step"] = CallStep.DETAILS_VERIFICATION.value
            return Command(update=update, goto=CallStep.DETAILS_VERIFICATION.value)
        elif status in [VerificationStatus.THIRD_PARTY.value, VerificationStatus.WRONG_PERSON.value, VerificationStatus.VERIFICATION_FAILED.value]:
            return Command(update=update, goto="__end__")
        else:
            return Command(update=update, goto="__end__")

    def details_verification_node(state: CallCenterAgentState) -> Command[Literal["reason_for_call", "__end__"]]:
        """Details verification step."""
        result = details_verification_agent.invoke(state)
        
        status = result.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        
        update = {
            "messages": result.get("messages", state.get("messages", [])),
            "details_verification_status": status,
            "details_verification_attempts": result.get("details_verification_attempts", 0),
            "matched_fields": result.get("matched_fields", []),
            "field_to_verify": result.get("field_to_verify", "id_number"),
            "current_step": CallStep.DETAILS_VERIFICATION.value
        }
        
        # Route based on verification status
        if status == VerificationStatus.VERIFIED.value:
            update["current_step"] = CallStep.REASON_FOR_CALL.value
            return Command(update=update, goto=CallStep.REASON_FOR_CALL.value)
        elif status == VerificationStatus.VERIFICATION_FAILED.value:
            return Command(update=update, goto="__end__")
        else:
            return Command(update=update, goto="__end__")

    def reason_for_call_node(state: CallCenterAgentState) -> Command[Literal["negotiation"]]:
        """Reason for call step - explains debt and proceeds to negotiation."""
        result = reason_for_call_agent.invoke(state)
        
        return Command(
            update={
                "messages": result.get("messages", state.get("messages", [])),
                "current_step": CallStep.NEGOTIATION.value,
                "outstanding_amount": result.get("outstanding_amount"),
                "account_status": result.get("account_status")
            },
            goto=CallStep.NEGOTIATION.value
        )

    def negotiation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Negotiation step - handles objections and seeks payment commitment."""
        result = negotiation_agent.invoke(state)
        
        return Command(
            update={
                "messages": result.get("messages", state.get("messages", [])),
                "current_step": CallStep.NEGOTIATION.value,
                "emotional_state": result.get("emotional_state"),
                "objections_raised": result.get("objections_raised", []),
                "payment_willingness": result.get("payment_willingness")
            },
            goto="__end__"
        )
    
    # Build workflow
    workflow = StateGraph(CallCenterAgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node(CallStep.INTRODUCTION.value, introduction_node)
    workflow.add_node(CallStep.NAME_VERIFICATION.value, name_verification_node)
    workflow.add_node(CallStep.DETAILS_VERIFICATION.value, details_verification_node)
    workflow.add_node(CallStep.REASON_FOR_CALL.value, reason_for_call_node)
    workflow.add_node(CallStep.NEGOTIATION.value, negotiation_node)
    
    # Set entry point and routing
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            CallStep.INTRODUCTION.value: CallStep.INTRODUCTION.value,
            CallStep.NAME_VERIFICATION.value: CallStep.NAME_VERIFICATION.value,
            CallStep.DETAILS_VERIFICATION.value: CallStep.DETAILS_VERIFICATION.value,
            CallStep.REASON_FOR_CALL.value: CallStep.REASON_FOR_CALL.value,
            CallStep.NEGOTIATION.value: CallStep.NEGOTIATION.value
        }
    )
    
    # Compile with optional memory
    compile_kwargs = {}
    if config.get('configurable', {}).get('use_memory'):
        compile_kwargs["checkpointer"] = MemorySaver()
    
    return workflow.compile(**compile_kwargs)


####################################################################################
from langchain_ollama import ChatOllama
from src.Agents.call_center_agent.call_scripts import ScriptType

from app_config import CONFIG

llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0, num_ctx=32000)
def create_call_center_agent_with_client_data(user_id:str) -> CompiledGraph:
    client_data = get_client_data(user_id=user_id)
    return create_call_center_agent(
        model=llm,
        client_data=client_data,    
        script_type=ScriptType.RATIO_1_INFLOW.value,
        config=CONFIG
    )
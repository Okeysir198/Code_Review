"""
Call Center Agent Workflow - Orchestrates specialized sub-agents for debt collection calls
"""
from typing import Literal, Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver

# Import existing components
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.introduction import create_introduction_agent
from src.Agents.call_center_agent.name_verification import create_name_verification_agent
from src.Agents.call_center_agent.details_verification import create_details_verification_agent
from src.Agents.call_center_agent.reason_for_call import create_reason_for_call_agent
from src.Agents.call_center_agent.negotiation import create_negotiation_agent


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
        script_type: Script type for the call (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        config: Configuration options
        
    Returns:
        Compiled main workflow graph
    """
    config = config or {}
    
    # Create all specialized sub-agents
    introduction_agent = create_introduction_agent(
        model=model, 
        client_data=client_data, 
        script_type=script_type, 
        agent_name=agent_name, 
        config=config
    )
    
    name_verification_agent = create_name_verification_agent(
        model=model, 
        client_data=client_data, 
        script_type=script_type, 
        agent_name=agent_name, 
        config=config
    )
    
    details_verification_agent = create_details_verification_agent(
        model=model, 
        client_data=client_data, 
        script_type=script_type, 
        agent_name=agent_name, 
        config=config
    )
    
    reason_for_call_agent = create_reason_for_call_agent(
        model=model, 
        client_data=client_data, 
        script_type=script_type, 
        agent_name=agent_name, 
        config=config
    )
    
    negotiation_agent = create_negotiation_agent(
        model=model, 
        client_data=client_data, 
        script_type=script_type, 
        agent_name=agent_name, 
        config=config
    )

    # Router and workflow node functions
    def router_node(state: CallCenterAgentState) -> Dict[str, Any]:
        """Router tracks conversation state and determines next step."""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        return {
            "current_step": current_step,
        }
    
    def route_to_agent(state: CallCenterAgentState) -> str:
        """Route to appropriate agent based on conversation state."""
        # Check if we have a specific next_step from a sub-agent
        next_step = state.get("next_step")
        current_step = next_step if next_step else state.get("current_step", CallStep.INTRODUCTION.value)
        
        # Map steps to node names using enum values
        step_to_node_map = {
            CallStep.INTRODUCTION.value: CallStep.INTRODUCTION.value,
            CallStep.NAME_VERIFICATION.value: CallStep.NAME_VERIFICATION.value, 
            CallStep.DETAILS_VERIFICATION.value: CallStep.DETAILS_VERIFICATION.value,
            CallStep.REASON_FOR_CALL.value: CallStep.REASON_FOR_CALL.value,
            CallStep.NEGOTIATION.value: CallStep.NEGOTIATION.value
        }
        
        return step_to_node_map.get(current_step, CallStep.INTRODUCTION.value)
    
    def introduction_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Introduction step"""
        result = introduction_agent.invoke(state)
        
        update = {
            "messages": result.get("messages", state.get("messages", [])),
            "current_step": result.get("current_step", state.get("current_step", [])),
        }
        
        return Command(update=update, goto="__end__")
    
    def name_verification_node(state: CallCenterAgentState) -> Command[Literal["details_verification", "__end__"]]:
        """Name verification - proceeds to details verification on success, ends on failure."""
        result = name_verification_agent.invoke(state)
        
        # Extract verification results
        status = result.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        attempts = result.get("name_verification_attempts", 0)
        messages = result.get("messages", state.get("messages", []))
        
        updated_state = {
            "messages": messages,
            "name_verification_status": status,
            "name_verification_attempts": attempts,
            "current_step": CallStep.NAME_VERIFICATION.value
        }
        
        if status == VerificationStatus.VERIFIED.value:
            # Verified - continue to details verification
            updated_state["current_step"] = CallStep.DETAILS_VERIFICATION.value
            return Command(update=updated_state, goto=CallStep.DETAILS_VERIFICATION.value)
        elif status in [
            VerificationStatus.THIRD_PARTY.value, 
            VerificationStatus.WRONG_PERSON.value, 
            VerificationStatus.VERIFICATION_FAILED.value
        ]:
            # End call for these scenarios
            return Command(update=updated_state, goto="__end__")
        else:
            # Continue with name verification (more attempts needed)
            return Command(update=updated_state, goto="__end__")
    
    def details_verification_node(state: CallCenterAgentState) -> Command[Literal["reason_for_call", "__end__"]]:
        """Details verification - proceeds to reason for call on success."""
        result = details_verification_agent.invoke(state)
        
        # Extract verification results
        status = result.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        attempts = result.get("details_verification_attempts", 0)
        matched_fields = result.get("matched_fields", [])
        field_to_verify = result.get("field_to_verify", "id_number")
        messages = result.get("messages", state.get("messages", []))

        updated_state = {
            "messages": messages,
            "details_verification_status": status,
            "details_verification_attempts": attempts,
            "matched_fields": matched_fields,
            "field_to_verify": field_to_verify,
            "current_step": CallStep.DETAILS_VERIFICATION.value
        }
        
        if status == VerificationStatus.VERIFIED.value:
            # Fully verified - proceed to explain debt
            updated_state["current_step"] = CallStep.REASON_FOR_CALL.value
            return Command(update=updated_state, goto=CallStep.REASON_FOR_CALL.value)
        elif status == VerificationStatus.VERIFICATION_FAILED.value:
            # Too many failed attempts
            return Command(update=updated_state, goto="__end__")
        else:
            # More verification needed - wait for debtor response
            return Command(update=updated_state, goto="__end__")
    
    def reason_for_call_node(state: CallCenterAgentState) -> Command[Literal["negotiation"]]:
        """Reason for call - explains debt situation and proceeds to negotiation."""
        result = reason_for_call_agent.invoke(state)
        
        updated_state = {
            # "messages": result.get("messages", state.get("messages", [])),
            # "current_step": CallStep.NEGOTIATION.value,
            # "outstanding_amount": result.get("outstanding_amount", state.get("outstanding_amount")),
            # "account_status": result.get("account_status", state.get("account_status"))
        }
        
        return Command(update=updated_state, goto=CallStep.NEGOTIATION.value)
    
    def negotiation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Negotiation - handles objections and seeks payment commitment."""
        result = negotiation_agent.invoke(state)
        
        updated_state = {
            # "messages": result.get("messages", state.get("messages", [])),
            # "current_step": CallStep.NEGOTIATION.value,
            # "emotional_state": result.get("emotional_state", state.get("emotional_state")),
            # "objections_raised": result.get("objections_raised", state.get("objections_raised", [])),
            # "payment_willingness": result.get("payment_willingness", state.get("payment_willingness"))
        }
        
        # For now, negotiation ends the conversation
        # In a full implementation, this would route to promise_to_pay or other steps
        return Command(update=updated_state, goto="__end__")
    
    # Build main workflow
    workflow = StateGraph(CallCenterAgentState)
    
    # Add all nodes
    workflow.add_node("router", router_node)
    workflow.add_node(CallStep.INTRODUCTION.value, introduction_node)
    workflow.add_node(CallStep.NAME_VERIFICATION.value, name_verification_node)
    workflow.add_node(CallStep.DETAILS_VERIFICATION.value, details_verification_node)
    workflow.add_node(CallStep.REASON_FOR_CALL.value, reason_for_call_node)
    workflow.add_node(CallStep.NEGOTIATION.value, negotiation_node)
    
    # Set entry point
    workflow.add_edge(START, "router")
    
    # Router uses conditional edges to determine next agent
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
    
    # Compile with checkpointer if memory is enabled
    compile_kwargs = {}
    if config.get('configurable', {}).get('use_memory'):
        compile_kwargs["checkpointer"] = MemorySaver()
    
    return workflow.compile(**compile_kwargs)
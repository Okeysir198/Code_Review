# ./src/Agents/call_center/call_center_agent.py
"""
Integrated Call Center Agent for Debt Collection.

Orchestrates the complete debt collection call flow using specialized step agents.
"""
import logging
from typing import Dict, Any, Optional, List, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus, CallOutcome
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters, get_client_data
from src.Agents.call_center_agent.prompts import get_step_prompt

# Import step agents
from src.Agents.call_center_agent.introduction import create_introduction_agent
from src.Agents.call_center_agent.name_verification import create_name_verification_agent
from src.Agents.call_center_agent.details_verification import create_details_verification_agent

# Import database tools
from src.Database.CartrackSQLDatabase import (
    create_debicheck_payment,
    create_payment_arrangement_payment_portal,
    add_client_note,
    save_call_disposition
)

logger = logging.getLogger(__name__)


class CallCenterAgent:
    """
    Integrated call center agent for debt collection calls.
    
    Manages the complete call flow using specialized step agents and state management.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        user_id: str,
        script_type: str = "ratio_1_inflow",
        agent_name: str = "AI Agent",
        tools: Optional[List[BaseTool]] = None,
        checkpointer: Optional[Any] = None,
        verbose: bool = False
    ):
        """
        Initialize the call center agent.
        
        Args:
            model: Language model for all agents
            user_id: Client's unique identifier
            script_type: Script type for the call
            agent_name: Name of the agent
            tools: Additional tools for agents
            checkpointer: State persistence (defaults to MemorySaver)
            verbose: Enable verbose logging
        """
        self.model = model
        self.user_id = user_id
        self.script_type = script_type
        self.agent_name = agent_name
        self.tools = tools or []
        self.verbose = verbose
        self.checkpointer = checkpointer or MemorySaver()
        
        # Create the workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> CompiledGraph:
        """Build the complete call center workflow."""
        
        # Create workflow with state management
        workflow = StateGraph(CallCenterAgentState)
        
        # Add all call step nodes
        workflow.add_node("introduction", self._create_step_agent(CallStep.INTRODUCTION))
        workflow.add_node("name_verification", self._create_step_agent(CallStep.NAME_VERIFICATION))
        workflow.add_node("details_verification", self._create_step_agent(CallStep.DETAILS_VERIFICATION))
        workflow.add_node("reason_for_call", self._create_step_agent(CallStep.REASON_FOR_CALL))
        workflow.add_node("negotiation", self._create_step_agent(CallStep.NEGOTIATION))
        workflow.add_node("promise_to_pay", self._create_step_agent(CallStep.PROMISE_TO_PAY))
        workflow.add_node("debicheck_setup", self._create_step_agent(CallStep.DEBICHECK_SETUP))
        workflow.add_node("payment_portal", self._create_step_agent(CallStep.PAYMENT_PORTAL))
        workflow.add_node("subscription_reminder", self._create_step_agent(CallStep.SUBSCRIPTION_REMINDER))
        workflow.add_node("client_details_update", self._create_step_agent(CallStep.CLIENT_DETAILS_UPDATE))
        workflow.add_node("referrals", self._create_step_agent(CallStep.REFERRALS))
        workflow.add_node("further_assistance", self._create_step_agent(CallStep.FURTHER_ASSISTANCE))
        workflow.add_node("closing", self._create_step_agent(CallStep.CLOSING))
        
        # Special nodes
        workflow.add_node("query_resolution", self._create_step_agent(CallStep.QUERY_RESOLUTION))
        workflow.add_node("cancellation", self._create_step_agent(CallStep.CANCELLATION))
        workflow.add_node("escalation", self._create_step_agent(CallStep.ESCALATION))
        workflow.add_node("router", self._router_node)
        
        # Set entry point
        workflow.add_edge(START, "introduction")
        
        # Define call flow with router for dynamic routing
        workflow.add_edge("introduction", "router")
        workflow.add_edge("name_verification", "router")
        workflow.add_edge("details_verification", "router")
        workflow.add_edge("reason_for_call", "router")
        workflow.add_edge("negotiation", "router")
        workflow.add_edge("promise_to_pay", "router")
        workflow.add_edge("debicheck_setup", "router")
        workflow.add_edge("payment_portal", "router")
        workflow.add_edge("subscription_reminder", "router")
        workflow.add_edge("client_details_update", "router")
        workflow.add_edge("referrals", "router")
        workflow.add_edge("further_assistance", "router")
        workflow.add_edge("query_resolution", "router")
        workflow.add_edge("cancellation", "router")
        workflow.add_edge("escalation", "router")
        workflow.add_edge("closing", END)
        
        # Router conditional edges
        workflow.add_conditional_edges(
            "router",
            self._route_next_step,
            {
                "name_verification": "name_verification",
                "details_verification": "details_verification",
                "reason_for_call": "reason_for_call",
                "negotiation": "negotiation",
                "promise_to_pay": "promise_to_pay",
                "debicheck_setup": "debicheck_setup",
                "payment_portal": "payment_portal",
                "subscription_reminder": "subscription_reminder",
                "client_details_update": "client_details_update",
                "referrals": "referrals",
                "further_assistance": "further_assistance",
                "query_resolution": "query_resolution",
                "cancellation": "cancellation",
                "escalation": "escalation",
                "closing": "closing",
                END: END
            }
        )
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _create_step_agent(self, step: CallStep):
        """Create an agent for a specific call step."""
        
        def step_agent_node(state: CallCenterAgentState) -> Dict[str, Any]:
            """Execute a call step using dynamic prompt generation."""
            
            # Update current step
            state.current_step = step.value
            
            # Handle specialized agents
            if step == CallStep.INTRODUCTION:
                agent = create_introduction_agent(
                    self.model, self.user_id, self.script_type, 
                    self.agent_name, self.tools, self.verbose
                )
            elif step == CallStep.NAME_VERIFICATION:
                agent = create_name_verification_agent(
                    self.model, self.user_id, self.script_type,
                    self.agent_name, self.tools, self.verbose
                )
            elif step == CallStep.DETAILS_VERIFICATION:
                agent = create_details_verification_agent(
                    self.model, self.user_id, self.script_type,
                    self.agent_name, self.tools, self.verbose
                )
            else:
                # Create generic agent for other steps
                agent = self._create_generic_step_agent(step)
            
            # Execute the agent
            result = agent.invoke({"messages": state.messages})
            
            # Extract new messages and update state
            new_messages = result.get("messages", [])
            
            return {
                "messages": new_messages,
                "current_step": step.value,
                "previous_step": state.current_step
            }
        
        return step_agent_node
    
    def _create_generic_step_agent(self, step: CallStep):
        """Create a generic agent for steps without specialized logic."""
        
        def dynamic_prompt(state: CallCenterAgentState):
            """Generate dynamic prompt for the step."""
            parameters = prepare_parameters(
                user_id=self.user_id,
                current_step=step.value,
                state=state.to_dict(),
                script_type=self.script_type,
                agent_name=self.agent_name
            )
            
            return get_step_prompt(step.value, parameters)
        
        return create_basic_agent(
            model=self.model,
            prompt=dynamic_prompt,
            tools=self.tools,
            state_schema=CallCenterAgentState,
            verbose=self.verbose,
            name=f"{step.value.title().replace('_', '')}Agent"
        )
    
    def _router_node(self, state: CallCenterAgentState) -> Dict[str, Any]:
        """Route to the next appropriate step based on current state."""
        
        # Determine next step based on call flow logic
        next_step = self._determine_next_step(state)
        
        return {
            "next_step": next_step,
            "recommended_next_node": next_step
        }
    
    def _route_next_step(self, state: CallCenterAgentState) -> str:
        """Conditional router function."""
        return state.recommended_next_node or END
    
    def _determine_next_step(self, state: CallCenterAgentState) -> str:
        """Determine the next step in the call flow."""
        current = state.current_step
        
        # Handle special cases first
        if state.query_detected:
            return "query_resolution"
        if state.cancellation_requested:
            return "cancellation"
        if state.escalation_requested:
            return "escalation"
        
        # Standard call flow
        flow_map = {
            CallStep.INTRODUCTION.value: "name_verification",
            CallStep.NAME_VERIFICATION.value: self._after_name_verification(state),
            CallStep.DETAILS_VERIFICATION.value: self._after_details_verification(state),
            CallStep.REASON_FOR_CALL.value: "negotiation",
            CallStep.NEGOTIATION.value: "promise_to_pay",
            CallStep.PROMISE_TO_PAY.value: self._after_payment_step(state),
            CallStep.DEBICHECK_SETUP.value: "subscription_reminder",
            CallStep.PAYMENT_PORTAL.value: "subscription_reminder",
            CallStep.SUBSCRIPTION_REMINDER.value: "client_details_update",
            CallStep.CLIENT_DETAILS_UPDATE.value: "referrals",
            CallStep.REFERRALS.value: "further_assistance",
            CallStep.FURTHER_ASSISTANCE.value: "closing",
            CallStep.QUERY_RESOLUTION.value: current,  # Return to previous step
            CallStep.CANCELLATION.value: "closing",
            CallStep.ESCALATION.value: "closing"
        }
        
        return flow_map.get(current, "closing")
    
    def _after_name_verification(self, state: CallCenterAgentState) -> str:
        """Determine next step after name verification."""
        if state.name_verification_status == VerificationStatus.VERIFIED.value:
            return "details_verification"
        elif state.name_verification_status in [VerificationStatus.THIRD_PARTY.value, VerificationStatus.UNAVAILABLE.value]:
            return "closing"
        elif state.name_verification_attempts >= state.max_name_verification_attempts:
            return "closing"
        else:
            return "name_verification"  # Retry
    
    def _after_details_verification(self, state: CallCenterAgentState) -> str:
        """Determine next step after details verification."""
        if state.details_verification_status == VerificationStatus.VERIFIED.value:
            return "reason_for_call"
        elif state.details_verification_attempts >= state.max_details_verification_attempts:
            return "closing"
        else:
            return "details_verification"  # Retry
    
    def _after_payment_step(self, state: CallCenterAgentState) -> str:
        """Determine next step after payment arrangement."""
        if state.payment_secured:
            if state.payment_arrangement.payment_method == "debicheck":
                return "debicheck_setup"
            elif state.payment_arrangement.payment_method == "payment_portal":
                return "payment_portal"
            else:
                return "subscription_reminder"
        else:
            return "closing"  # No payment secured
    
    # Public interface methods
    def start_call(self, initial_message: Optional[str] = None) -> Dict[str, Any]:
        """Start a new debt collection call."""
        messages = []
        if initial_message:
            messages.append(HumanMessage(content=initial_message))
        
        config = {"configurable": {"thread_id": f"call_{self.user_id}"}}
        
        return self.workflow.invoke({
            "messages": messages,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "script_type": self.script_type,
            "current_step": CallStep.INTRODUCTION.value
        }, config=config)
    
    def continue_call(self, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Continue an existing call conversation."""
        config = {"configurable": {"thread_id": thread_id or f"call_{self.user_id}"}}
        
        return self.workflow.invoke({
            "messages": [HumanMessage(content=message)]
        }, config=config)
    
    def stream_call(self, message: str, thread_id: Optional[str] = None):
        """Stream call responses for real-time interaction."""
        config = {"configurable": {"thread_id": thread_id or f"call_{self.user_id}"}}
        
        return self.workflow.stream({
            "messages": [HumanMessage(content=message)]
        }, config=config)
    
    def get_call_state(self, thread_id: Optional[str] = None) -> CallCenterAgentState:
        """Get current call state."""
        config = {"configurable": {"thread_id": thread_id or f"call_{self.user_id}"}}
        
        state = self.workflow.get_state(config)
        return state.values if state else CallCenterAgentState()
    
    def end_call(self, outcome: str, notes: Optional[str] = None, thread_id: Optional[str] = None) -> bool:
        """End the call and save disposition."""
        try:
            # Add final notes if provided
            if notes:
                add_client_note.invoke({
                    "user_id": self.user_id,
                    "note_text": f"Call ended - {outcome}. {notes}"
                })
            
            # Save call disposition (you'll need to map outcomes to disposition IDs)
            disposition_result = save_call_disposition.invoke({
                "client_id": self.user_id,
                "disposition_type_id": "1",  # Map outcome to appropriate ID
                "note_text": notes
            })
            
            return disposition_result.get("success", False)
            
        except Exception as e:
            logger.error(f"Error ending call: {e}")
            return False


# Convenience function for easy agent creation
def create_call_center_agent(
    model: BaseChatModel,
    user_id: str,
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False
) -> CallCenterAgent:
    """
    Create a call center agent for debt collection.
    
    Args:
        model: Language model to use
        user_id: Client's unique identifier
        script_type: Script type for the call
        agent_name: Name of the agent
        tools: Additional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Configured CallCenterAgent ready for use
    """
    return CallCenterAgent(
        model=model,
        user_id=user_id,
        script_type=script_type,
        agent_name=agent_name,
        tools=tools,
        verbose=verbose
    )
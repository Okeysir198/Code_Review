"""
Complete Call Center Agent Workflow with Integrated LLM Router
Updated to follow proper patterns: Router only updates state, execution on next message
"""
import logging
from typing import Literal, Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# Import existing components
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.prompts import get_router_prompt, parse_router_decision

# Import all specialized sub-agents
from src.Agents.call_center_agent.step00_introduction import create_introduction_agent
from src.Agents.call_center_agent.step01_name_verification import create_name_verification_agent
from src.Agents.call_center_agent.step02_details_verification import create_details_verification_agent
from src.Agents.call_center_agent.step03_reason_for_call import create_reason_for_call_agent
from src.Agents.call_center_agent.step04_negotiation import create_negotiation_agent
from src.Agents.call_center_agent.step05_promise_to_pay import create_promise_to_pay_agent
from src.Agents.call_center_agent.step06_debicheck_setup import create_debicheck_setup_agent
from src.Agents.call_center_agent.step07_payment_portal import create_payment_portal_agent
from src.Agents.call_center_agent.step08_subscription_reminder import create_subscription_reminder_agent
from src.Agents.call_center_agent.step09_client_details_update import create_client_details_update_agent
from src.Agents.call_center_agent.step10_referrals import create_referrals_agent
from src.Agents.call_center_agent.step11_further_assistance import create_further_assistance_agent
from src.Agents.call_center_agent.step12_query_resolution import create_query_resolution_agent
from src.Agents.call_center_agent.step13_escalation import create_escalation_agent
from src.Agents.call_center_agent.step14_cancellation import create_cancellation_agent
from src.Agents.call_center_agent.step15_closing import create_closing_agent
from app_config import CONFIG

logger = logging.getLogger(__name__)


def create_call_center_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """
    Create the complete call center agent workflow with integrated LLM router.
    """
    config = config or {}
    max_name_attempts = CONFIG.get("verification", {}).get("max_name_verification_attempts", 5)
    max_details_attempts = CONFIG.get("verification", {}).get("max_details_verification_attempts", 5)

    # ===== CREATE ALL SPECIALIZED SUB-AGENTS =====
    introduction_agent = create_introduction_agent(model, client_data, script_type, agent_name, config=config)
    name_verification_agent = create_name_verification_agent(model, client_data, script_type, agent_name, config=config)
    details_verification_agent = create_details_verification_agent(model, client_data, script_type, agent_name, config=config)
    reason_for_call_agent = create_reason_for_call_agent(model, client_data, script_type, agent_name, config=config)
    negotiation_agent = create_negotiation_agent(model, client_data, script_type, agent_name, config=config)
    promise_to_pay_agent = create_promise_to_pay_agent(model, client_data, script_type, agent_name, config=config)
    debicheck_setup_agent = create_debicheck_setup_agent(model, client_data, script_type, agent_name, config=config)
    payment_portal_agent = create_payment_portal_agent(model, client_data, script_type, agent_name, config=config)
    subscription_reminder_agent = create_subscription_reminder_agent(model, client_data, script_type, agent_name, config=config)
    client_details_update_agent = create_client_details_update_agent(model, client_data, script_type, agent_name, config=config)
    referrals_agent = create_referrals_agent(model, client_data, script_type, agent_name, config=config)
    further_assistance_agent = create_further_assistance_agent(model, client_data, script_type, agent_name, config=config)
    query_resolution_agent = create_query_resolution_agent(model, client_data, script_type, agent_name, config=config)
    escalation_agent = create_escalation_agent(model, client_data, script_type, agent_name, config=config)
    cancellation_agent = create_cancellation_agent(model, client_data, script_type, agent_name, config=config)
    closing_agent = create_closing_agent(model, client_data, script_type, agent_name, config=config)

    # ===== LLM ROUTER FUNCTIONS =====
    
    def classify_message_intent(state: CallCenterAgentState) -> str:
        """Use small LLM to classify if message is step-related or needs query resolution."""
        
        # Use fast 3B model for classification
        router_llm = ChatOllama(model="qwen2.5:3b-instruct", temperature=0)
        
        # Build classification prompt
        prompt_content = get_router_prompt(state.to_dict() if hasattr(state, 'to_dict') else state)
        prompt = [SystemMessage(content=prompt_content)]
        
        try:
            # Get LLM decision
            response = router_llm.invoke(prompt)
            
            # Parse response for routing decision
            classification = parse_router_decision(response, state)
            
            logger.info(f"Router classified message as: {classification}")
            return classification
            
        except Exception as e:
            logger.warning(f"Router LLM failed: {e}, defaulting to STEP_RELATED")
            return "STEP_RELATED"
    
    def has_hard_state_override(state: CallCenterAgentState) -> bool:
        """Check if state requires hard override (verification failures, etc.)"""
        
        # Verification failures
        if (state.get("name_verification_attempts", 0) >= max_name_attempts and 
            state.get("name_verification_status") != VerificationStatus.VERIFIED.value):
            return True
            
        if (state.get("details_verification_attempts", 0) >= max_details_attempts and 
            state.get("details_verification_status") != VerificationStatus.VERIFIED.value):
            return True
        
        # Route override set
        if state.get("route_override"):
            return True
            
        # Call ended
        if state.get("is_call_ended"):
            return True
            
        return False
    
    def get_state_override_route(state: CallCenterAgentState) -> str:
        """Get route from hard state overrides."""

        # Route override takes priority
        if state.get("route_override"):
            return state.get("route_override")
        
        # Verification failures go to closing
        if (state.get("name_verification_attempts", 0) >= max_name_attempts or 
            state.get("details_verification_attempts", 0) >= max_details_attempts):
            return CallStep.CLOSING.value
        
        # Call ended
        if state.get("is_call_ended"):
            return CallStep.CLOSING.value
        
        return state.get("current_step", CallStep.INTRODUCTION.value)
    
    def get_default_next_step(current_step: str, state: CallCenterAgentState) -> str:
        """Get default next step in call sequence."""
        
        # Normal call flow progression
        flow_sequence = {
            CallStep.INTRODUCTION.value: CallStep.NAME_VERIFICATION.value,
            CallStep.NAME_VERIFICATION.value: CallStep.DETAILS_VERIFICATION.value if state.get("name_verification_status") == VerificationStatus.VERIFIED.value else CallStep.CLOSING.value,
            CallStep.DETAILS_VERIFICATION.value: CallStep.REASON_FOR_CALL.value if state.get("details_verification_status") == VerificationStatus.VERIFIED.value else CallStep.CLOSING.value,
            CallStep.REASON_FOR_CALL.value: CallStep.NEGOTIATION.value,
            CallStep.NEGOTIATION.value: CallStep.PROMISE_TO_PAY.value,
            CallStep.PROMISE_TO_PAY.value: CallStep.SUBSCRIPTION_REMINDER.value,  # Will be overridden by payment method
            CallStep.DEBICHECK_SETUP.value: CallStep.SUBSCRIPTION_REMINDER.value,
            CallStep.PAYMENT_PORTAL.value: CallStep.SUBSCRIPTION_REMINDER.value,
            CallStep.SUBSCRIPTION_REMINDER.value: CallStep.CLIENT_DETAILS_UPDATE.value,
            CallStep.CLIENT_DETAILS_UPDATE.value: CallStep.REFERRALS.value,
            CallStep.REFERRALS.value: CallStep.FURTHER_ASSISTANCE.value,
            CallStep.FURTHER_ASSISTANCE.value: CallStep.CLOSING.value,
            CallStep.QUERY_RESOLUTION.value: state.get("return_to_step", CallStep.CLOSING.value),
            CallStep.ESCALATION.value: CallStep.CLOSING.value,
            CallStep.CANCELLATION.value: CallStep.CLOSING.value,
            CallStep.CLOSING.value: CallStep.CLOSING.value
        }
        
        return flow_sequence.get(current_step, CallStep.CLOSING.value)
    
    def determine_next_step(state: CallCenterAgentState) -> str:
        """Determine which step should execute next based on router logic."""
        
        # 1. Hard state overrides (business rules)
        if has_hard_state_override(state):
            route = get_state_override_route(state)
            logger.info(f"State override routing to: {route}")
            return route
        
        # 2. Check for emergency keywords (bypass LLM)
        if state.get("messages"):
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                content_lower = last_message.content.lower()
                if any(word in content_lower for word in ["supervisor", "manager", "cancel", "complaint"]):
                    logger.info("Emergency keyword detected, routing to escalation")
                    return CallStep.ESCALATION.value
        
        # 3. LLM classification for normal flow
        if state.get("messages") and len(state["messages"]) > 1:  # Skip for first message
            classification = classify_message_intent(state)
            
            # Route based on LLM classification
            if classification == "ESCALATION":
                return CallStep.ESCALATION.value
            elif classification == "QUERY_UNRELATED":
                return CallStep.QUERY_RESOLUTION.value
            elif classification in ["AGREEMENT", "STEP_RELATED", "OBJECTION"]:
                # Stay in current step or move to next based on step logic
                return state.get("current_step", CallStep.INTRODUCTION.value)
        
        # 4. Default progression
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        return get_default_next_step(current_step, state)

    # ===== ROUTER NODE - ONLY UPDATES STATE =====
    
    def router_node(state: CallCenterAgentState) :
        """Router: Updates state with next step, then goes to __end__. Next message triggers execution."""
        if len(state.get("messages",[])) < 2: 
            return {"current_step":CallStep.INTRODUCTION.value}
        
        next_step = determine_next_step(state)
        return_to_step = None
        
        # Set return step for query resolution
        if next_step == CallStep.QUERY_RESOLUTION.value:
            return_to_step = state.get("current_step", CallStep.INTRODUCTION.value)
            logger.info(f"Setting return_to_step: {return_to_step}")
        
        logger.info(f"Router: Setting current_step to {next_step}")
        return {
            'current_step':next_step,
            'return_to_step':return_to_step
        }

    # ===== EXECUTION ROUTER - EXECUTES THE DETERMINED STEP =====
    
    def execution_router(state: CallCenterAgentState) -> str:
        """Routes to the step that should execute based on current_step."""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        # If call ended, no execution needed
        if state.get("is_call_ended"):
            return END
            
        logger.info(f"Execution router: Routing to {current_step}")
        return current_step

    # ===== NODE FUNCTIONS =====

    def introduction_node(state: CallCenterAgentState) -> Command[Literal[ "__end__"]]:
        """Introduction step - direct handover to name verification."""
        result = introduction_agent.invoke(state)
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")
        
        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"
        )

    def name_verification_node(state: CallCenterAgentState) -> Command[Literal["details_verification", "__end__"]]:
        """Name verification step - one turn then end or direct handover if verified."""
        result = name_verification_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        name_verification_status = result.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        name_verification_attempts = result.get("name_verification_attempts", 0)
        current_step = result.get("current_step")

        update = {
            "messages": messages,
            "name_verification_status": name_verification_status,
            "name_verification_attempts": name_verification_attempts,
            "current_step": current_step
        }
        
        # Smart routing: If verified, direct handover to details verification
        if name_verification_status == VerificationStatus.VERIFIED.value:
            update["current_step"] = CallStep.DETAILS_VERIFICATION.value
            return Command(update=update, goto=CallStep.DETAILS_VERIFICATION.value)
        
        # Otherwise end and wait for debtor response
        return Command(update=update, goto="__end__")

    def details_verification_node(state: CallCenterAgentState) -> Command[Literal["reason_for_call", "__end__"]]:
        """Details verification step - one turn then end or direct handover if verified."""
        result = details_verification_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        details_verification_status = result.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        details_verification_attempts = result.get("details_verification_attempts", 0)
        matched_fields = result.get("matched_fields", [])
        field_to_verify = result.get("field_to_verify", "id_number")
        current_step = result.get("current_step")

        update = {
            "messages": messages,
            "details_verification_status": details_verification_status,
            "details_verification_attempts": details_verification_attempts,
            "matched_fields": matched_fields,
            "field_to_verify": field_to_verify,
            "current_step": current_step
        }
        
        # Smart routing: If verified, direct handover to reason for call
        if details_verification_status == VerificationStatus.VERIFIED.value:
            update["current_step"] = CallStep.REASON_FOR_CALL.value
            return Command(update=update, goto=CallStep.REASON_FOR_CALL.value)
        
        # Otherwise end and wait for debtor response
        return Command(update=update, goto="__end__")

    def reason_for_call_node(state: CallCenterAgentState) -> Command[Literal["negotiation", "__end__"]]:
        """Reason for call step - one turn then end to wait for response."""
        result = reason_for_call_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        outstanding_amount = result.get("outstanding_amount")
        current_step = result.get("current_step")

        return Command(
            update={
                "messages": messages,
                "current_step": CallStep.NEGOTIATION.value,
                "outstanding_amount": outstanding_amount
            },
            goto=CallStep.NEGOTIATION.value  
        )

    def negotiation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Negotiation step - one turn then end to wait for response."""
        result = negotiation_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")

        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"  # End and wait for debtor response
        )

    def promise_to_pay_node(state: CallCenterAgentState) -> Command[Literal["debicheck_setup", "payment_portal", "__end__"]]:
        """Promise to pay step - one turn then route based on payment method or end."""
        result = promise_to_pay_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        payment_secured = result.get("payment_secured", False)
        payment_arrangement = result.get("payment_arrangement", {})
        current_step = result.get("current_step")
        update = {
            "messages": messages,
            "payment_secured": payment_secured,
            "payment_method": payment_arrangement.get("payment_method", "none"),
            "current_step": current_step
        }
        
        # Route based on payment method if payment secured
        if payment_secured:
            payment_method = payment_arrangement.get("payment_method", "")
            if payment_method == "debicheck":
                update["current_step"] = CallStep.DEBICHECK_SETUP.value
                return Command(update=update, goto=CallStep.DEBICHECK_SETUP.value)
            elif payment_method == "payment_portal":
                update["current_step"] = CallStep.PAYMENT_PORTAL.value
                return Command(update=update, goto=CallStep.PAYMENT_PORTAL.value)
        
        # Default: end and wait for response
        return Command(update=update, goto="__end__")

    def debicheck_setup_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """DebiCheck setup step - one turn then end."""
        result = debicheck_setup_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")
        
        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"  # End and wait for response
        )

    def payment_portal_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Payment portal step - one turn then end."""
        result = payment_portal_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")
        
        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"  # End and wait for response
        )

    def subscription_reminder_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Subscription reminder step - one turn then end."""
        result = subscription_reminder_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")
        
        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"  # End and wait for response
        )

    def client_details_update_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Client details update step - one turn then end."""
        result = client_details_update_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")
        
        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"  # End and wait for response
        )

    def referrals_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Referrals step - one turn then end."""
        result = referrals_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")
        
        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"  # End and wait for response
        )

    def further_assistance_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Further assistance step - one turn then end."""
        result = further_assistance_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        current_step = result.get("current_step")
        
        return Command(
            update={
                "messages": messages,
                "current_step": current_step
            },
            goto="__end__"  # End and wait for response
        )

    def query_resolution_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Query resolution - answer briefly then return to main goal."""
        result = query_resolution_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        return_to_step = state.get("return_to_step", CallStep.CLOSING.value)
        
        # Clear the return step and set up return to main goal
        return Command(
            update={
                "messages": messages,
                "current_step": return_to_step,
                "return_to_step": None  # Clear return step
            },
            goto="__end__"  # End - next message will trigger execution of return_to_step
        )

    def escalation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Escalation step - handle request then determine next step."""
        result = escalation_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        
        # After escalation, determine where to return based on call progress
        if state.get("details_verification_status") == VerificationStatus.VERIFIED.value:
            if not state.get("payment_secured"):
                # Return to payment discussion
                next_step = CallStep.NEGOTIATION.value
            else:
                # Payment secured, close call
                next_step = CallStep.CLOSING.value
        else:
            # Not verified, close call
            next_step = CallStep.CLOSING.value
        
        return Command(
            update={
                "messages": messages,
                "current_step": next_step
            },
            goto="__end__"  # End - next message will trigger execution of next_step
        )

    def cancellation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Cancellation step - handle then close call."""
        result = cancellation_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        
        return Command(
            update={
                "messages": messages,
                "current_step": CallStep.CLOSING.value
            },
            goto="__end__"  # End - next message will trigger closing
        )

    def closing_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Closing step - ends the call."""
        result = closing_agent.invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        
        return Command(
            update={
                "messages": messages,
                "is_call_ended": True,
                "current_step": CallStep.CLOSING.value
            },
            goto="__end__"
        )
    
    # ===== BUILD WORKFLOW =====
    
    workflow = StateGraph(CallCenterAgentState)
    
    # Add router node (entry point)
    workflow.add_node("router", router_node)
    
    # Add all step execution nodes
    workflow.add_node(CallStep.INTRODUCTION.value, introduction_node)
    workflow.add_node(CallStep.NAME_VERIFICATION.value, name_verification_node)
    workflow.add_node(CallStep.DETAILS_VERIFICATION.value, details_verification_node)
    workflow.add_node(CallStep.REASON_FOR_CALL.value, reason_for_call_node)
    workflow.add_node(CallStep.NEGOTIATION.value, negotiation_node)
    workflow.add_node(CallStep.PROMISE_TO_PAY.value, promise_to_pay_node)
    workflow.add_node(CallStep.DEBICHECK_SETUP.value, debicheck_setup_node)
    workflow.add_node(CallStep.PAYMENT_PORTAL.value, payment_portal_node)
    workflow.add_node(CallStep.SUBSCRIPTION_REMINDER.value, subscription_reminder_node)
    workflow.add_node(CallStep.CLIENT_DETAILS_UPDATE.value, client_details_update_node)
    workflow.add_node(CallStep.REFERRALS.value, referrals_node)
    workflow.add_node(CallStep.FURTHER_ASSISTANCE.value, further_assistance_node)
    workflow.add_node(CallStep.QUERY_RESOLUTION.value, query_resolution_node)
    workflow.add_node(CallStep.ESCALATION.value, escalation_node)
    workflow.add_node(CallStep.CANCELLATION.value, cancellation_node)
    workflow.add_node(CallStep.CLOSING.value, closing_node)
    
    workflow.add_edge(START, "router")
    
    workflow.add_conditional_edges(
        "router",  
        execution_router,  # This function determines which node to execute
        {
            CallStep.INTRODUCTION.value: CallStep.INTRODUCTION.value,
            CallStep.NAME_VERIFICATION.value: CallStep.NAME_VERIFICATION.value,
            CallStep.DETAILS_VERIFICATION.value: CallStep.DETAILS_VERIFICATION.value,
            CallStep.REASON_FOR_CALL.value: CallStep.REASON_FOR_CALL.value,
            CallStep.NEGOTIATION.value: CallStep.NEGOTIATION.value,
            CallStep.PROMISE_TO_PAY.value: CallStep.PROMISE_TO_PAY.value,
            CallStep.DEBICHECK_SETUP.value: CallStep.DEBICHECK_SETUP.value,
            CallStep.PAYMENT_PORTAL.value: CallStep.PAYMENT_PORTAL.value,
            CallStep.SUBSCRIPTION_REMINDER.value: CallStep.SUBSCRIPTION_REMINDER.value,
            CallStep.CLIENT_DETAILS_UPDATE.value: CallStep.CLIENT_DETAILS_UPDATE.value,
            CallStep.REFERRALS.value: CallStep.REFERRALS.value,
            CallStep.FURTHER_ASSISTANCE.value: CallStep.FURTHER_ASSISTANCE.value,
            CallStep.QUERY_RESOLUTION.value: CallStep.QUERY_RESOLUTION.value,
            CallStep.ESCALATION.value: CallStep.ESCALATION.value,
            CallStep.CANCELLATION.value: CallStep.CANCELLATION.value,
            CallStep.CLOSING.value: CallStep.CLOSING.value,
        }
    )
    
    # Compile with optional memory
    compile_kwargs = {}
    if config.get('configurable', {}).get('use_memory'):
        compile_kwargs["checkpointer"] = MemorySaver()
    
    return workflow.compile(**compile_kwargs)
"""
Fixed Call Center Agent Workflow with Current Step Validation for Handoffs
Only handoff to next step by checking current_step value and ensuring current step is complete
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
    """Create the complete call center agent workflow with current step validation."""
    
    config = config or {}
    max_name_attempts = CONFIG.get("verification", {}).get("max_name_verification_attempts", 5)
    max_details_attempts = CONFIG.get("verification", {}).get("max_details_verification_attempts", 5)

    # Create all specialized sub-agents
    agents = {
        "introduction": create_introduction_agent(model, client_data, script_type, agent_name, config=config),
        "name_verification": create_name_verification_agent(model, client_data, script_type, agent_name, config=config),
        "details_verification": create_details_verification_agent(model, client_data, script_type, agent_name, config=config),
        "reason_for_call": create_reason_for_call_agent(model, client_data, script_type, agent_name, config=config),
        "negotiation": create_negotiation_agent(model, client_data, script_type, agent_name, config=config),
        "promise_to_pay": create_promise_to_pay_agent(model, client_data, script_type, agent_name, config=config),
        "debicheck_setup": create_debicheck_setup_agent(model, client_data, script_type, agent_name, config=config),
        "payment_portal": create_payment_portal_agent(model, client_data, script_type, agent_name, config=config),
        "subscription_reminder": create_subscription_reminder_agent(model, client_data, script_type, agent_name, config=config),
        "client_details_update": create_client_details_update_agent(model, client_data, script_type, agent_name, config=config),
        "referrals": create_referrals_agent(model, client_data, script_type, agent_name, config=config),
        "further_assistance": create_further_assistance_agent(model, client_data, script_type, agent_name, config=config),
        "query_resolution": create_query_resolution_agent(model, client_data, script_type, agent_name, config=config),
        "escalation": create_escalation_agent(model, client_data, script_type, agent_name, config=config),
        "cancellation": create_cancellation_agent(model, client_data, script_type, agent_name, config=config),
        "closing": create_closing_agent(model, client_data, script_type, agent_name, config=config)
    }

    # Helper functions
    def has_emergency_keywords(state: CallCenterAgentState) -> bool:
        """Check for emergency keywords that bypass normal flow."""
        if not state.get("messages"):
            return False
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
            content_lower = last_message.content.lower()
            emergency_words = ["supervisor", "manager", "cancel", "complaint"]
            return any(word in content_lower for word in emergency_words)
        return False
    
    def has_hard_state_override(state: CallCenterAgentState) -> bool:
        """Check if state requires hard override (verification failures, etc.)"""
        if (state.get("name_verification_attempts", 0) >= max_name_attempts and 
            state.get("name_verification_status") != VerificationStatus.VERIFIED.value):
            return True
        if (state.get("details_verification_attempts", 0) >= max_details_attempts and 
            state.get("details_verification_status") != VerificationStatus.VERIFIED.value):
            return True
        return state.get("route_override") or state.get("is_call_ended")
    
    def classify_message_intent(state: CallCenterAgentState) -> str:
        """Use small LLM to classify message intent."""
        router_llm = ChatOllama(model="qwen2.5:3b-instruct", temperature=0)
        prompt_content = get_router_prompt(state.to_dict() if hasattr(state, 'to_dict') else state)
        prompt = [SystemMessage(content=prompt_content)]
        
        try:
            response = router_llm.invoke(prompt)
            classification = parse_router_decision(response, state)
            logger.info(f"Router classified message as: {classification}")
            return classification
        except Exception as e:
            logger.warning(f"Router LLM failed: {e}, defaulting to STEP_RELATED")
            return "STEP_RELATED"
    
    def determine_next_step(state: CallCenterAgentState) -> str:
        """SINGLE SOURCE OF TRUTH: All routing decisions happen here."""
        
        # 1. Hard state overrides (business rules)
        if has_hard_state_override(state):
            logger.info("Hard state override - routing to closing")
            return CallStep.CLOSING.value
        
        # 2. Emergency keywords (bypass LLM)
        if has_emergency_keywords(state):
            logger.info("Emergency keyword detected - routing to escalation")
            return CallStep.ESCALATION.value
        
        # 3. LLM classification for off-topic detection
        if state.get("messages") and len(state["messages"]) > 1:
            classification = classify_message_intent(state)
            if classification == "ESCALATION":
                return CallStep.ESCALATION.value
            elif classification == "QUERY_UNRELATED":
                return CallStep.QUERY_RESOLUTION.value
        
        # 4. Default: stay in current step (let individual nodes decide progression)
        return state.get("current_step", CallStep.INTRODUCTION.value)

    # Router node - ONLY updates state
    def router_node(state: CallCenterAgentState) -> Dict[str, Any]:
        """Router: ONLY updates state with next step. No execution control."""
        
        # Skip router for first message (introduction)
        if len(state.get("messages", [])) < 2: 
            return {"current_step": CallStep.INTRODUCTION.value}
        
        next_step = determine_next_step(state)
        
        updates = {"current_step": next_step}
        
        # Set return step for query resolution
        if next_step == CallStep.QUERY_RESOLUTION.value:
            updates["return_to_step"] = state.get("current_step", CallStep.INTRODUCTION.value)
            logger.info(f"Setting return_to_step: {updates['return_to_step']}")
        
        logger.info(f"Router: Setting current_step to {next_step}")
        return updates

    def execution_router(state: CallCenterAgentState) -> str:
        """Simple execution router: reads current_step and routes to that node."""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        if state.get("is_call_ended"):
            return END
            
        logger.info(f"Execution router: Routing to {current_step}")
        return current_step

    # Helper function to analyze AI messages for payment agreements
    def get_last_ai_message(messages: list) -> str:
        """Get the last AI message content."""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == "ai":
                return msg.content.lower()
        return ""

    # STEP COMPLETION VALIDATION FUNCTIONS
    def is_step_complete(step: str, state: CallCenterAgentState, result: Dict[str, Any]) -> bool:
        """Check if current step is complete and ready for handoff."""
        
        if step == CallStep.NAME_VERIFICATION.value:
            # Complete if VERIFIED
            return result.get("name_verification_status") == VerificationStatus.VERIFIED.value
            
        elif step == CallStep.DETAILS_VERIFICATION.value:
            # Complete if VERIFIED
            return result.get("details_verification_status") == VerificationStatus.VERIFIED.value
            
        elif step == CallStep.REASON_FOR_CALL.value:
            # Always complete after one turn - handoff to negotiation
            return True
            
        elif step == CallStep.NEGOTIATION.value:
            # Always complete after one turn - handoff to promise to pay
            return True
            
        elif step == CallStep.PROMISE_TO_PAY.value:
            # Complete if payment was secured
            messages = result.get("messages", [])
            last_ai_message = get_last_ai_message(messages)
            return any(word in last_ai_message for word in ["perfect", "excellent", "great", "i'm setting up", "arranging"])
            
        elif step in [CallStep.DEBICHECK_SETUP.value, CallStep.PAYMENT_PORTAL.value]:
            # Always complete after explaining process - handoff to subscription reminder
            return True
            
        elif step in [
            CallStep.SUBSCRIPTION_REMINDER.value, 
            CallStep.CLIENT_DETAILS_UPDATE.value, 
            CallStep.REFERRALS.value
        ]:
            # Always complete after one turn - continue to next step
            return True
            
        elif step == CallStep.FURTHER_ASSISTANCE.value:
            # Always complete - handoff to closing
            return True
            
        # Default: step not complete, wait for more interaction
        return False

    def get_next_step_after_completion(step: str, state: CallCenterAgentState, result: Dict[str, Any]) -> str:
        """Get next step after current step is complete."""
        
        if step == CallStep.NAME_VERIFICATION.value:
            return CallStep.DETAILS_VERIFICATION.value
            
        elif step == CallStep.DETAILS_VERIFICATION.value:
            return CallStep.REASON_FOR_CALL.value
            
        elif step == CallStep.REASON_FOR_CALL.value:
            return CallStep.NEGOTIATION.value
            
        elif step == CallStep.NEGOTIATION.value:
            return CallStep.PROMISE_TO_PAY.value
            
        elif step == CallStep.PROMISE_TO_PAY.value:
            # Route based on payment method if payment secured
            messages = result.get("messages", [])
            last_ai_message = get_last_ai_message(messages)
            
            if "debicheck" in last_ai_message or "bank" in last_ai_message:
                return CallStep.DEBICHECK_SETUP.value
            elif "portal" in last_ai_message or "link" in last_ai_message:
                return CallStep.PAYMENT_PORTAL.value
            else:
                return CallStep.SUBSCRIPTION_REMINDER.value
                
        elif step in [CallStep.DEBICHECK_SETUP.value, CallStep.PAYMENT_PORTAL.value]:
            return CallStep.SUBSCRIPTION_REMINDER.value
            
        elif step == CallStep.SUBSCRIPTION_REMINDER.value:
            return CallStep.CLIENT_DETAILS_UPDATE.value
            
        elif step == CallStep.CLIENT_DETAILS_UPDATE.value:
            return CallStep.REFERRALS.value
            
        elif step == CallStep.REFERRALS.value:
            return CallStep.FURTHER_ASSISTANCE.value
            
        elif step == CallStep.FURTHER_ASSISTANCE.value:
            return CallStep.CLOSING.value
            
        # Default: go to closing
        return CallStep.CLOSING.value

    # OPTIMIZED NODE FUNCTIONS WITH STEP COMPLETION VALIDATION

    def introduction_node(state: CallCenterAgentState) -> Command[Literal["name_verification", "__end__"]]:
        """Introduction step - always handoff to name verification."""
        result = agents["introduction"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Introduction always hands off to name verification
        return Command(
            update={
                "messages": messages, 
                "current_step": CallStep.NAME_VERIFICATION.value
            },
            goto=CallStep.NAME_VERIFICATION.value
        )

    def name_verification_node(state: CallCenterAgentState) -> Command[Literal["details_verification", "__end__"]]:
        """Name verification step - handoff only if VERIFIED."""
        
        # Only proceed if we're actually in name verification step
        if state.get("current_step") != CallStep.NAME_VERIFICATION.value:
            logger.warning(f"Name verification called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["name_verification"].invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        name_verification_status = result.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        name_verification_attempts = result.get("name_verification_attempts", 0)

        update = {
            "messages": messages,
            "name_verification_status": name_verification_status,
            "name_verification_attempts": name_verification_attempts,
            "current_step": CallStep.NAME_VERIFICATION.value
        }
        
        # CRITICAL: Only handoff if step is complete (VERIFIED)
        if is_step_complete(CallStep.NAME_VERIFICATION.value, state, result):
            next_step = get_next_step_after_completion(CallStep.NAME_VERIFICATION.value, state, result)
            update["current_step"] = next_step
            logger.info(f"Name verification COMPLETE - handing off to {next_step}")
            return Command(update=update, goto=next_step)
        
        # Step not complete - wait for more interaction
        logger.info("Name verification INCOMPLETE - waiting for more interaction")
        return Command(update=update, goto="__end__")

    def details_verification_node(state: CallCenterAgentState) -> Command[Literal["reason_for_call", "__end__"]]:
        """Details verification step - handoff only if VERIFIED."""
        
        # Only proceed if we're actually in details verification step
        if state.get("current_step") != CallStep.DETAILS_VERIFICATION.value:
            logger.warning(f"Details verification called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["details_verification"].invoke(state)
        
        messages = result.get("messages", state.get("messages", []))
        details_verification_status = result.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        details_verification_attempts = result.get("details_verification_attempts", 0)
        matched_fields = result.get("matched_fields", [])

        update = {
            "messages": messages,
            "details_verification_status": details_verification_status,
            "details_verification_attempts": details_verification_attempts,
            "matched_fields": matched_fields,
            "current_step": CallStep.DETAILS_VERIFICATION.value
        }
        
        # CRITICAL: Only handoff if step is complete (VERIFIED)
        if is_step_complete(CallStep.DETAILS_VERIFICATION.value, state, result):
            next_step = get_next_step_after_completion(CallStep.DETAILS_VERIFICATION.value, state, result)
            update["current_step"] = next_step
            logger.info(f"Details verification COMPLETE - handing off to {next_step}")
            return Command(update=update, goto=next_step)
        
        # Step not complete - wait for more interaction
        logger.info("Details verification INCOMPLETE - waiting for more interaction")
        return Command(update=update, goto="__end__")

    def reason_for_call_node(state: CallCenterAgentState) -> Command[Literal["negotiation"]]:
        """Reason for call step - always handoff to negotiation after one turn."""
        
        # Only proceed if we're actually in reason for call step
        if state.get("current_step") != CallStep.REASON_FOR_CALL.value:
            logger.warning(f"Reason for call called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["reason_for_call"].invoke(state)
        messages = result.get("messages", state.get("messages", []))

        # Always handoff to negotiation after explaining reason
        next_step = CallStep.NEGOTIATION.value
        logger.info(f"Reason for call COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def negotiation_node(state: CallCenterAgentState) -> Command[Literal["promise_to_pay"]]:
        """Negotiation step - always handoff to promise to pay after one turn."""
        
        # Only proceed if we're actually in negotiation step
        if state.get("current_step") != CallStep.NEGOTIATION.value:
            logger.warning(f"Negotiation called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["negotiation"].invoke(state)
        messages = result.get("messages", state.get("messages", []))

        # Always handoff to promise to pay after negotiation
        next_step = CallStep.PROMISE_TO_PAY.value
        logger.info(f"Negotiation COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def promise_to_pay_node(state: CallCenterAgentState) -> Command[Literal["debicheck_setup", "payment_portal", "subscription_reminder", "__end__"]]:
        """Promise to pay step - handoff based on payment agreement analysis."""
        
        # Only proceed if we're actually in promise to pay step
        if state.get("current_step") != CallStep.PROMISE_TO_PAY.value:
            logger.warning(f"Promise to pay called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["promise_to_pay"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Analyze AI response for payment agreement
        last_ai_message = get_last_ai_message(messages)
        
        # Detect payment secured
        payment_secured = any(word in last_ai_message for word in 
                             ["perfect", "excellent", "great", "i'm setting up", "arranging"])
        
        # Detect payment method
        payment_method = "none"
        if payment_secured:
            if "debicheck" in last_ai_message or "bank" in last_ai_message:
                payment_method = "debicheck"
            elif "portal" in last_ai_message or "link" in last_ai_message:
                payment_method = "payment_portal"

        update = {
            "messages": messages,
            "payment_secured": payment_secured,
            "payment_method": payment_method,
            "current_step": CallStep.PROMISE_TO_PAY.value
        }
        
        # CRITICAL: Only handoff if step is complete (payment secured)
        if is_step_complete(CallStep.PROMISE_TO_PAY.value, state, result):
            next_step = get_next_step_after_completion(CallStep.PROMISE_TO_PAY.value, state, result)
            update["current_step"] = next_step
            logger.info(f"Promise to pay COMPLETE - payment secured, handing off to {next_step}")
            return Command(update=update, goto=next_step)
        
        # Step not complete - no payment secured, wait for more interaction
        logger.info("Promise to pay INCOMPLETE - no payment secured, waiting for more interaction")
        return Command(update=update, goto="__end__")

    def debicheck_setup_node(state: CallCenterAgentState) -> Command[Literal["subscription_reminder"]]:
        """DebiCheck setup step - always handoff to subscription reminder after explanation."""
        
        # Only proceed if we're actually in debicheck setup step
        if state.get("current_step") != CallStep.DEBICHECK_SETUP.value:
            logger.warning(f"DebiCheck setup called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["debicheck_setup"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to subscription reminder after explaining DebiCheck
        next_step = CallStep.SUBSCRIPTION_REMINDER.value
        logger.info(f"DebiCheck setup COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def payment_portal_node(state: CallCenterAgentState) -> Command[Literal["subscription_reminder"]]:
        """Payment portal step - always handoff to subscription reminder after guidance."""
        
        # Only proceed if we're actually in payment portal step
        if state.get("current_step") != CallStep.PAYMENT_PORTAL.value:
            logger.warning(f"Payment portal called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["payment_portal"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to subscription reminder after portal guidance
        next_step = CallStep.SUBSCRIPTION_REMINDER.value
        logger.info(f"Payment portal COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def subscription_reminder_node(state: CallCenterAgentState) -> Command[Literal["client_details_update"]]:
        """Subscription reminder step - always handoff to client details update."""
        
        # Only proceed if we're actually in subscription reminder step
        if state.get("current_step") != CallStep.SUBSCRIPTION_REMINDER.value:
            logger.warning(f"Subscription reminder called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["subscription_reminder"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to client details update
        next_step = CallStep.CLIENT_DETAILS_UPDATE.value
        logger.info(f"Subscription reminder COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def client_details_update_node(state: CallCenterAgentState) -> Command[Literal["referrals"]]:
        """Client details update step - always handoff to referrals."""
        
        # Only proceed if we're actually in client details update step
        if state.get("current_step") != CallStep.CLIENT_DETAILS_UPDATE.value:
            logger.warning(f"Client details update called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["client_details_update"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to referrals
        next_step = CallStep.REFERRALS.value
        logger.info(f"Client details update COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def referrals_node(state: CallCenterAgentState) -> Command[Literal["further_assistance"]]:
        """Referrals step - always handoff to further assistance."""
        
        # Only proceed if we're actually in referrals step
        if state.get("current_step") != CallStep.REFERRALS.value:
            logger.warning(f"Referrals called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["referrals"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to further assistance
        next_step = CallStep.FURTHER_ASSISTANCE.value
        logger.info(f"Referrals COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def further_assistance_node(state: CallCenterAgentState) -> Command[Literal["closing"]]:
        """Further assistance step - always handoff to closing."""
        
        # Only proceed if we're actually in further assistance step
        if state.get("current_step") != CallStep.FURTHER_ASSISTANCE.value:
            logger.warning(f"Further assistance called but current_step is {state.get('current_step')}")
            return Command(update={}, goto="__end__")
        
        result = agents["further_assistance"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to closing
        next_step = CallStep.CLOSING.value
        logger.info(f"Further assistance COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def query_resolution_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Query resolution - answer briefly then return to main goal."""
        result = agents["query_resolution"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        return_to_step = state.get("return_to_step", CallStep.CLOSING.value)
        
        # Always return to previous step after query resolution
        logger.info(f"Query resolution COMPLETE - returning to {return_to_step}")
        
        return Command(
            update={
                "messages": messages,
                "current_step": return_to_step,
                "return_to_step": None
            },
            goto="__end__"
        )

    def escalation_node(state: CallCenterAgentState) -> Command[Literal["closing"]]:
        """Escalation step - always handoff to closing."""
        result = agents["escalation"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to closing after escalation
        next_step = CallStep.CLOSING.value
        logger.info(f"Escalation COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def cancellation_node(state: CallCenterAgentState) -> Command[Literal["closing"]]:
        """Cancellation step - always handoff to closing."""
        result = agents["cancellation"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Always handoff to closing after cancellation
        next_step = CallStep.CLOSING.value
        logger.info(f"Cancellation COMPLETE - handing off to {next_step}")
        
        return Command(
            update={
                "messages": messages, 
                "current_step": next_step
            },
            goto=next_step
        )

    def closing_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Closing step - ends the call."""
        result = agents["closing"].invoke(state)
        messages = result.get("messages", state.get("messages", []))
        
        # Call is complete - set ended flag
        logger.info("Closing COMPLETE - call ended")
        
        return Command(
            update={
                "messages": messages,
                "is_call_ended": True,
                "current_step": CallStep.CLOSING.value
            },
            goto="__end__"
        )
    
    # Build workflow
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
    
    # Router is entry point
    workflow.add_edge(START, "router")
    
    # Router routes to appropriate step based on current_step
    workflow.add_conditional_edges(
        "router",  
        execution_router,
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
            END: END
        }
    )
    
    # Compile with optional memory
    compile_kwargs = {}
    if config.get('configurable', {}).get('use_memory'):
        compile_kwargs["checkpointer"] = MemorySaver()
    
    return workflow.compile(**compile_kwargs)
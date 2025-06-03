# src/Agents/graph_call_center_agent.py
"""
Complete Enhanced Call Center Agent Workflow - One Turn Per Agent Pattern
Each agent handles one conversation turn then waits for debtor response
"""
import logging
from typing import Literal, Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# Import enhanced components
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.call_scripts import determine_script_type_from_aging

# Import all enhanced specialized sub-agents
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
    script_type: str = None,
    agent_name: str = "AI Agent",
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> CompiledGraph:
    """
    Create complete enhanced call center agent workflow with one-turn-per-agent pattern.
    
    Each agent handles exactly one conversation turn then waits for debtor response.
    Router only activates when new human message arrives.
    """
    
    # Auto-determine script type if not provided
    if not script_type:
        account_aging = client_data.get("account_aging", {})
        script_type = determine_script_type_from_aging(account_aging, client_data)
        logger.info(f"Auto-determined script type: {script_type}")

    # Optimized model usage - different sizes for different complexity
    model_3b = ChatOllama(model="qwen2.5:3b-instruct", temperature=0, num_ctx=4096)
    model_7b = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=8192)
    model_14b = ChatOllama(model="qwen2.5:14b-instruct", temperature=0, num_ctx=32000)

    config = config or {}

    # ========================================================================
    # ENHANCED AGENT CREATION - Optimized model assignment
    # ========================================================================
    
    # Simple conversation agents use 3B model for efficiency
    introduction_agent = create_introduction_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    name_verification_agent = create_name_verification_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    details_verification_agent = create_details_verification_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    reason_for_call_agent = create_reason_for_call_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Negotiation uses 7B for better objection handling
    negotiation_agent = create_negotiation_agent(
        model=model_7b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Complex tool-using agents use 14B model for better tool usage
    promise_to_pay_agent = create_promise_to_pay_agent(
        model=model_14b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Payment setup agents use 7B/14B based on complexity
    debicheck_setup_agent = create_debicheck_setup_agent(
        model=model_7b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    payment_portal_agent = create_payment_portal_agent(
        model=model_14b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Post-payment agents use 3B for efficiency
    subscription_reminder_agent = create_subscription_reminder_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Account management agents use 7B for tool usage
    client_details_update_agent = create_client_details_update_agent(
        model=model_7b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Simple closing agents use 3B
    referrals_agent = create_referrals_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    further_assistance_agent = create_further_assistance_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Special handling agents
    query_resolution_agent = create_query_resolution_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    escalation_agent = create_escalation_agent(
        model=model_3b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    cancellation_agent = create_cancellation_agent(
        model=model_7b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )
    
    # Closing agent with comprehensive tools uses 7B
    closing_agent = create_closing_agent(
        model=model_7b, client_data=client_data, script_type=script_type,
        agent_name=agent_name, verbose=verbose, config=config
    )

    logger.info("✅ All enhanced agents created with optimized model assignment")

    # ========================================================================
    # ROUTER FUNCTIONS
    # ========================================================================

    def router_node(state: CallCenterAgentState) -> Dict[str, str]:
        """Router that directs flow based on current step and business rules"""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        # Business rule checks for terminal states
        if state.get("is_call_ended"):
            logger.info("Call ended - routing to closing")
            return {"current_step": CallStep.CLOSING.value}

        # Special routing for urgent requests
        if state.get("escalation_requested") and current_step != CallStep.ESCALATION.value:
            logger.info("Escalation requested - routing to escalation")
            return {"current_step": CallStep.ESCALATION.value}
        
        if state.get("cancellation_requested") and current_step != CallStep.CANCELLATION.value:
            logger.info("Cancellation requested - routing to cancellation")
            return {"current_step": CallStep.CANCELLATION.value}

        logger.info(f"Router: Current step is {current_step}")
        return {"current_step": current_step}

    def execution_router(state: CallCenterAgentState) -> str:
        """Determine which step node to execute based on current_step"""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        # Business rule overrides
        if state.get("is_call_ended"):
            return CallStep.CLOSING.value
        
        if state.get("escalation_requested") and current_step != CallStep.ESCALATION.value:
            return CallStep.ESCALATION.value
        
        if state.get("cancellation_requested") and current_step != CallStep.CANCELLATION.value:
            return CallStep.CANCELLATION.value
            
        return current_step

    # ========================================================================
    # NODE IMPLEMENTATIONS - One Turn Per Agent Pattern
    # ========================================================================

    def introduction_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Introduction step - one turn then wait for debtor response."""
        result = introduction_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", CallStep.NAME_VERIFICATION.value)
            }
        
        logger.info("Introduction complete - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def name_verification_node(state: CallCenterAgentState) -> Command[Literal["details_verification", "__end__"]]:
        """Name verification - one turn then wait, except for direct verification success."""
        result = name_verification_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step")),
                "name_verification_status": getattr(result, "name_verification_status", state.get("name_verification_status")),
                "name_verification_attempts": getattr(result, "name_verification_attempts", state.get("name_verification_attempts")),
                "is_call_ended": getattr(result, "is_call_ended", state.get("is_call_ended", False))
            }
        
        # Direct handoff ONLY if verified successfully
        if state_updates.get("name_verification_status") == VerificationStatus.VERIFIED.value:
            logger.info("Name verification SUCCESS - direct handoff to details verification")
            state_updates["current_step"] = CallStep.DETAILS_VERIFICATION.value
            return Command(update=state_updates, goto="details_verification")
        
        # All other cases: wait for debtor response
        logger.info(f"Name verification result: {state_updates.get('name_verification_status')} - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def details_verification_node(state: CallCenterAgentState) -> Command[Literal["reason_for_call", "__end__"]]:
        """Details verification - one turn then wait, except for direct verification success."""
        result = details_verification_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step")),
                "details_verification_status": getattr(result, "details_verification_status", state.get("details_verification_status")),
                "details_verification_attempts": getattr(result, "details_verification_attempts", state.get("details_verification_attempts")),
                "matched_fields": getattr(result, "matched_fields", state.get("matched_fields", [])),
                "field_to_verify": getattr(result, "field_to_verify", state.get("field_to_verify")),
                "is_call_ended": getattr(result, "is_call_ended", state.get("is_call_ended", False))
            }
        
        # Direct handoff ONLY if fully verified
        if state_updates.get("details_verification_status") == VerificationStatus.VERIFIED.value:
            logger.info("Details verification SUCCESS - direct handoff to reason for call")
            state_updates["current_step"] = CallStep.REASON_FOR_CALL.value
            return Command(update=state_updates, goto="reason_for_call")
        
        # All other cases: wait for debtor response
        logger.info(f"Details verification result: {state_updates.get('details_verification_status')} - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")
    
    def reason_for_call_node(state: CallCenterAgentState) -> Command[Literal["negotiation"]]:
        """Reason for call - one turn then wait for debtor response."""
        result = reason_for_call_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", CallStep.NEGOTIATION.value)
            }
        
        logger.info("Account explanation provided - waiting for debtor response")
        return Command(update=state_updates, goto="negotiation")
    
    def negotiation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Negotiation - one turn then wait for debtor response."""
        result = negotiation_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step")),
                "escalation_requested": getattr(result, "escalation_requested", state.get("escalation_requested", False))
            }
        
        logger.info("Negotiation turn completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def promise_to_pay_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Promise to pay - one turn then wait for debtor response."""
        result = promise_to_pay_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step")),
                "payment_secured": getattr(result, "payment_secured", state.get("payment_secured", False)),
                "payment_method_preference": getattr(result, "payment_method_preference", state.get("payment_method_preference", ""))
            }
        
        logger.info("Promise to pay processing completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def debicheck_setup_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """DebiCheck setup - one turn then wait for debtor response."""
        result = debicheck_setup_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step"))
            }
        
        logger.info("DebiCheck setup completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def payment_portal_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Payment portal - one turn then wait for debtor response."""
        result = payment_portal_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step")),
                "payment_method_preference": getattr(result, "payment_method_preference", state.get("payment_method_preference", ""))
            }
        
        logger.info("Payment portal processing completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def subscription_reminder_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Subscription reminder - one turn then wait for debtor response."""
        result = subscription_reminder_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step"))
            }
        
        logger.info("Subscription reminder completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def client_details_update_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Client details update - one turn then wait for debtor response."""
        result = client_details_update_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step"))
            }
        
        logger.info("Client details update completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def referrals_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Referrals - one turn then wait for debtor response."""
        result = referrals_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step"))
            }
        
        logger.info("Referrals completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def further_assistance_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Further assistance - one turn then wait for debtor response."""
        result = further_assistance_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step")),
                "escalation_requested": getattr(result, "escalation_requested", state.get("escalation_requested", False)),
                "cancellation_requested": getattr(result, "cancellation_requested", state.get("cancellation_requested", False)),
                "return_to_step": getattr(result, "return_to_step", None)
            }
        
        logger.info("Further assistance completed - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def query_resolution_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Query resolution - one turn then wait for debtor response."""
        result = query_resolution_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": getattr(result, "current_step", state.get("current_step")),
                "return_to_step": getattr(result, "return_to_step", None),
                "last_client_question": getattr(result, "last_client_question", "")
            }
        
        logger.info("Query resolved - waiting for debtor response")
        return Command(update=state_updates, goto="__end__")

    def escalation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Escalation - terminal state, processes then ends call."""
        result = escalation_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": CallStep.CLOSING.value,
                "escalation_requested": True,
                "call_outcome": "escalated",
                "is_call_ended": True
            }
        
        logger.info("Escalation processed - call ending")
        return Command(update=state_updates, goto="__end__")

    def cancellation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Cancellation - terminal state, processes then ends call."""
        result = cancellation_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "current_step": CallStep.CLOSING.value,
                "cancellation_requested": True,
                "call_outcome": "cancellation_requested",
                "is_call_ended": True
            }
        
        logger.info("Cancellation processed - call ending")
        return Command(update=state_updates, goto="__end__")

    def closing_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Closing - final state with comprehensive logging then END."""
        result = closing_agent.invoke(state)
        
        state_updates = {}
        if isinstance(result, dict):
            state_updates = result
        else:
            state_updates = {
                "messages": getattr(result, "messages", []),
                "is_call_ended": True,
                "call_outcome": getattr(result, "call_outcome", "completed"),
                "current_step": CallStep.CLOSING.value
            }
        
        logger.info("Call ended with comprehensive logging")
        return Command(update=state_updates, goto="__end__")

    # ========================================================================
    # WORKFLOW CONSTRUCTION - One Turn Per Agent Pattern
    # ========================================================================
    
    workflow = StateGraph(CallCenterAgentState)
    
    # Add router node (entry point for each conversation turn)
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
    
    # ========================================================================
    
    # START -> router (entry point for each conversation)
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
            CallStep.CLOSING.value: CallStep.CLOSING.value
        }
    )
    
    compile_kwargs = {}
    if config.get('configurable', {}).get('use_memory'):
        compile_kwargs["checkpointer"] = MemorySaver()

    return workflow.compile(**compile_kwargs)
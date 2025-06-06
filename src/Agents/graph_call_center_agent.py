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

    account_aging = client_data.get("account_aging", {})
    script_type = determine_script_type_from_aging(account_aging, client_data)
    logger.info(f"Auto-determined script type: {script_type}")

    # ========================================================================
    # MODEL ASSIGNMENT CONFIGURATION
    # ========================================================================
    
    # Available models with different capabilities
    model_3b = ChatOllama(model="qwen2.5:3b-instruct", temperature=0, num_ctx=40000)
    model_7b = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=40000)
    # model_7b = ChatOllama(model="qwen3:4b-q4_K_M", temperature=0, num_ctx=40000, enable_thinking=False)
    model_14b = ChatOllama(model="qwen3:8b-q4_K_M", temperature=0, num_ctx=40000, enable_thinking=False)

    # Model assignment strategy by step complexity
    STEP_MODEL_ASSIGNMENT = {
        # Simple conversation - 3B models for efficiency
        "introduction": model_7b,           # Simple greeting
        "reason_for_call": model_3b,        # Account explanation
        "subscription_reminder": model_3b,   # Billing clarification
        "referrals": model_3b,              # Referral offer
        "further_assistance": model_3b,      # Final assistance check
        "query_resolution": model_3b,        # Quick Q&A
        "escalation": model_3b,             # Escalation handling
        "cancellation": model_3b,           # Cancellation processing
        "closing": model_3b,                # Call conclusion
        
        # Verification & negotiation - 7B models for better reasoning
        "name_verification": model_7b,       # Identity confirmation
        "details_verification": model_7b,    # Security verification
        "negotiation": model_7b,            # Objection handling (changed to 7b)

        # Complex tool usage - 14B models for tool proficiency
        "promise_to_pay": model_14b,        # Payment arrangement tools
        "payment_portal": model_14b,        # Payment link generation
        
        # Simple tool usage - 3B models sufficient
        "debicheck_setup": model_3b,        # Basic mandate setup
        "client_details_update": model_3b,  # Contact updates
    }

    # Agent creation factory functions
    AGENT_CREATORS = {
        "introduction": create_introduction_agent,
        "name_verification": create_name_verification_agent,
        "details_verification": create_details_verification_agent,
        "reason_for_call": create_reason_for_call_agent,
        "negotiation": create_negotiation_agent,
        "promise_to_pay": create_promise_to_pay_agent,
        "debicheck_setup": create_debicheck_setup_agent,
        "payment_portal": create_payment_portal_agent,
        "subscription_reminder": create_subscription_reminder_agent,
        "client_details_update": create_client_details_update_agent,
        "referrals": create_referrals_agent,
        "further_assistance": create_further_assistance_agent,
        "query_resolution": create_query_resolution_agent,
        "escalation": create_escalation_agent,
        "cancellation": create_cancellation_agent,
        "closing": create_closing_agent,
    }

    config = config or {}

    # ========================================================================
    # ENHANCED AGENT CREATION - Optimized and DRY
    # ========================================================================
    
    # Create all agents using dictionary configuration
    agents = {}
    for step_name, creator_func in AGENT_CREATORS.items():
        assigned_model = STEP_MODEL_ASSIGNMENT[step_name]
        agents[step_name] = creator_func(
            model=assigned_model,
            client_data=client_data,
            script_type=script_type,
            agent_name=agent_name,
            verbose=verbose,
            config=config
        )
        if verbose:
            model_name = assigned_model.model if hasattr(assigned_model, 'model') else 'unknown'
            logger.debug(f"Created {step_name} agent with {model_name}")

    # Extract individual agents for backward compatibility
    introduction_agent : CompiledGraph = agents["introduction"]
    name_verification_agent : CompiledGraph = agents["name_verification"]
    details_verification_agent : CompiledGraph = agents["details_verification"]
    reason_for_call_agent : CompiledGraph = agents["reason_for_call"]
    negotiation_agent : CompiledGraph = agents["negotiation"]
    promise_to_pay_agent : CompiledGraph = agents["promise_to_pay"]
    debicheck_setup_agent : CompiledGraph = agents["debicheck_setup"]
    payment_portal_agent : CompiledGraph = agents["payment_portal"]
    subscription_reminder_agent : CompiledGraph = agents["subscription_reminder"]
    client_details_update_agent : CompiledGraph = agents["client_details_update"]
    referrals_agent : CompiledGraph = agents["referrals"]
    further_assistance_agent : CompiledGraph = agents["further_assistance"]
    query_resolution_agent : CompiledGraph = agents["query_resolution"]
    escalation_agent : CompiledGraph = agents["escalation"]
    cancellation_agent : CompiledGraph = agents["cancellation"]
    closing_agent : CompiledGraph = agents["closing"]

    logger.info(f"✅ All {len(agents)} call step agents created ")

    # ========================================================================
    # ROUTER FUNCTIONS
    # ========================================================================

    def router_node(state: CallCenterAgentState) -> Dict[str, str]:
        """Router that directs flow based on current step and business rules"""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        logger.info(f"Router: Current step is {current_step}")
        return {"current_step": current_step}

    def execution_router(state: CallCenterAgentState) -> str:
        """Determine which step node to execute based on current_step"""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        # Business rule overrides
        if state.get("is_call_ended"):
            return CallStep.CLOSING.value
        
        # if state.get("escalation_requested") and current_step != CallStep.ESCALATION.value:
        #     return CallStep.ESCALATION.value
        
        # if state.get("cancellation_requested") and current_step != CallStep.CANCELLATION.value:
        #     return CallStep.CANCELLATION.value
            
        return current_step

    # ========================================================================
    # NODE IMPLEMENTATIONS - One Turn Per Agent Pattern
    # ========================================================================

    def introduction_node(state: CallCenterAgentState) -> Command[Literal[END]]:
        """Introduction step - one turn then wait for debtor response."""
        result = introduction_agent.invoke(state)
        logger.info("Introduction complete - waiting for debtor response")
        return Command(update=result, goto=END)

    def name_verification_node(state: CallCenterAgentState) -> Command[Literal[CallStep.DETAILS_VERIFICATION.value, END]]:
        """Simplified name verification - use current_step from agent result"""
        result = name_verification_agent.invoke(state)

        # Use the current_step that the agent set in its result
        next_step = result.get("current_step", CallStep.NAME_VERIFICATION.value)

        # Route based on what the agent decided
        if next_step == CallStep.DETAILS_VERIFICATION.value:
            logger.info("Agent decided: move to details verification")
            return Command(update=result, goto=CallStep.DETAILS_VERIFICATION.value)

        else:
            # Continue name verification - wait for next human response
            logger.info("Agent decided: continue name verification")
            return Command(update=result, goto=END)

    def details_verification_node(state: CallCenterAgentState) -> Command[Literal[CallStep.REASON_FOR_CALL.value, END]]:
        """Details verification - one turn then wait, except for direct verification success."""
        result = details_verification_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.REASON_FOR_CALL.value:
            logger.info("Agent decided: move to reason for call")
            return Command(update=result, goto=CallStep.REASON_FOR_CALL.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)

        else:
            # Continue details verification - wait for next human response
            logger.info("Agent decided: continue details verification")
            return Command(update=result, goto=END)

    def reason_for_call_node(state: CallCenterAgentState) -> Command[Literal[CallStep.NEGOTIATION.value]]:
        """Reason for call - explain reason then go to negotiation."""
        result = reason_for_call_agent.invoke(state)
        logger.info("Account explanation provided - going to negotiation")
        return Command(update=result, goto=CallStep.NEGOTIATION.value)

    def negotiation_node(state: CallCenterAgentState) -> Command[Literal[CallStep.PROMISE_TO_PAY.value, END]]:
        """Negotiation - one turn then wait for debtor response."""
        result = negotiation_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.PROMISE_TO_PAY.value:
            logger.info("Agent decided: move to promise to pay")
            return Command(update=result, goto=CallStep.PROMISE_TO_PAY.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)

        else:
            # Continue negotiation - wait for next human response
            logger.info("Agent decided: continue negotiation")
            return Command(update=result, goto=END)

    def promise_to_pay_node(state: CallCenterAgentState) -> Command[Literal[CallStep.DEBICHECK_SETUP.value, CallStep.PAYMENT_PORTAL.value, CallStep.SUBSCRIPTION_REMINDER.value, END]]:
        """Promise to pay - one turn then wait for debtor response."""
        result = promise_to_pay_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.DEBICHECK_SETUP.value:
            logger.info("Agent decided: move to debicheck setup")
            return Command(update=result, goto=CallStep.DEBICHECK_SETUP.value)

        elif next_step == CallStep.PAYMENT_PORTAL.value:
            logger.info("Agent decided: move to payment portal")
            return Command(update=result, goto=CallStep.PAYMENT_PORTAL.value)

        elif next_step == CallStep.SUBSCRIPTION_REMINDER.value:
            logger.info("Agent decided: move to subscription reminder")
            return Command(update=result, goto=CallStep.SUBSCRIPTION_REMINDER.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)
        
        elif next_step == CallStep.ESCALATION.value:
            logger.info("Agent decided: move to escalation")
            return Command(update=result, goto=CallStep.ESCALATION.value)
        
        elif next_step == CallStep.CANCELLATION.value:
            logger.info("Agent decided: move to cancellation")
            return Command(update=result, goto=CallStep.CANCELLATION.value)

        else:
            # Continue promise to pay - wait for next human response
            logger.info("Agent decided: continue promise to pay")
            return Command(update=result, goto=END)

    def debicheck_setup_node(state: CallCenterAgentState) -> Command[Literal[CallStep.SUBSCRIPTION_REMINDER.value, END]]:
        """DebiCheck setup - one turn then wait for debtor response."""
        result = debicheck_setup_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.SUBSCRIPTION_REMINDER.value:
            logger.info("Agent decided: move to subscription reminder")
            return Command(update=result, goto=CallStep.SUBSCRIPTION_REMINDER.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)
        
        elif next_step == CallStep.ESCALATION.value:
            logger.info("Agent decided: move to escalation")
            return Command(update=result, goto=CallStep.ESCALATION.value)
        
        elif next_step == CallStep.CANCELLATION.value:
            logger.info("Agent decided: move to cancellation")
            return Command(update=result, goto=CallStep.CANCELLATION.value)

        else:
            # Continue debicheck setup - wait for next human response
            logger.info("Agent decided: continue debicheck setup")
            return Command(update=result, goto=END)

    def payment_portal_node(state: CallCenterAgentState) -> Command[Literal[CallStep.SUBSCRIPTION_REMINDER.value, END]]:
        """Payment portal - one turn then wait for debtor response."""
        result = payment_portal_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.SUBSCRIPTION_REMINDER.value:
            logger.info("Agent decided: move to subscription reminder")
            return Command(update=result, goto=CallStep.SUBSCRIPTION_REMINDER.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)
        
        elif next_step == CallStep.ESCALATION.value:
            logger.info("Agent decided: move to escalation")
            return Command(update=result, goto=CallStep.ESCALATION.value)
        
        elif next_step == CallStep.CANCELLATION.value:
            logger.info("Agent decided: move to cancellation")
            return Command(update=result, goto=CallStep.CANCELLATION.value)

        else:
            # Continue payment portal - wait for next human response
            logger.info("Agent decided: continue payment portal")
            return Command(update=result, goto=END)

    def subscription_reminder_node(state: CallCenterAgentState) -> Command[Literal[CallStep.CLIENT_DETAILS_UPDATE.value, END]]:
        """Subscription reminder - one turn then wait for debtor response."""
        result = subscription_reminder_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.CLIENT_DETAILS_UPDATE.value:
            logger.info("Agent decided: move to client details update")
            return Command(update=result, goto=CallStep.CLIENT_DETAILS_UPDATE.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)

        else:
            # Continue subscription reminder - wait for next human response
            logger.info("Agent decided: continue subscription reminder")
            return Command(update=result, goto=END)

    def client_details_update_node(state: CallCenterAgentState) -> Command[Literal[CallStep.REFERRALS.value, END]]:
        """Client details update - one turn then wait for debtor response."""
        result = client_details_update_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.REFERRALS.value:
            logger.info("Agent decided: move to referrals")
            return Command(update=result, goto=CallStep.REFERRALS.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)
        
        elif next_step == CallStep.ESCALATION.value:
            logger.info("Agent decided: move to escalation")
            return Command(update=result, goto=CallStep.ESCALATION.value)
        
        elif next_step == CallStep.CANCELLATION.value:
            logger.info("Agent decided: move to cancellation")
            return Command(update=result, goto=CallStep.CANCELLATION.value)

        else:
            # Continue client details update - wait for next human response
            logger.info("Agent decided: continue client details update")
            return Command(update=result, goto=END)

    def referrals_node(state: CallCenterAgentState) -> Command[Literal[CallStep.FURTHER_ASSISTANCE.value, END]]:
        """Referrals - one turn then wait for debtor response."""
        result = referrals_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.FURTHER_ASSISTANCE.value:
            logger.info("Agent decided: move to further assistance")
            return Command(update=result, goto=CallStep.FURTHER_ASSISTANCE.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)

        else:
            # Continue referrals - wait for next human response
            logger.info("Agent decided: continue referrals")
            return Command(update=result, goto=END)

    def further_assistance_node(state: CallCenterAgentState) -> Command[Literal[CallStep.CLOSING.value, END]]:
        """Further assistance - one turn then wait for debtor response."""
        result = further_assistance_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided
        if next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: move to closing")
            return Command(update=result, goto=CallStep.CLOSING.value)

        elif next_step == CallStep.CANCELLATION.value:
            logger.info("Agent decided: move to cancellation")
            return Command(update=result, goto=CallStep.CANCELLATION.value)

        elif next_step == CallStep.QUERY_RESOLUTION.value:
            logger.info("Agent decided: move to query resolution")
            return Command(update=result, goto=CallStep.QUERY_RESOLUTION.value)

        else:
            # Continue further assistance - wait for next human response
            logger.info("Agent decided: continue further assistance")
            return Command(update=result, goto=END)
    
    def query_resolution_node(state: CallCenterAgentState) -> Command[Literal[END]]:
        """Query resolution - one turn then wait for debtor response."""
        result = query_resolution_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step")

        # Route based on what the agent decided - query resolution can return to various steps
        if next_step == CallStep.NAME_VERIFICATION.value:
            logger.info("Agent decided: return to name verification")
            return Command(update=result, goto=CallStep.NAME_VERIFICATION.value)

        elif next_step == CallStep.DETAILS_VERIFICATION.value:
            logger.info("Agent decided: return to details verification")
            return Command(update=result, goto=CallStep.DETAILS_VERIFICATION.value)

        elif next_step == CallStep.PROMISE_TO_PAY.value:
            logger.info("Agent decided: return to promise to pay")
            return Command(update=result, goto=CallStep.PROMISE_TO_PAY.value)

        elif next_step == CallStep.FURTHER_ASSISTANCE.value:
            logger.info("Agent decided: return to further assistance")
            return Command(update=result, goto=CallStep.FURTHER_ASSISTANCE.value)

        elif next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: end call")
            return Command(update=result, goto=CallStep.CLOSING.value)

        else:
            # Continue query resolution - wait for next human response
            logger.info("Agent decided: continue query resolution")
            return Command(update=result, goto=END)

    def escalation_node(state: CallCenterAgentState) -> Command[Literal[END]]:
        """Escalation - terminal state, processes then ends call."""
        result = escalation_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step", CallStep.CLOSING.value)

        # Escalation typically goes to closing
        if next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: move to closing after escalation")
            return Command(update=result, goto=CallStep.CLOSING.value)

        else:
            # Wait for human response if not ready to close
            logger.info("Agent decided: continue escalation process")
            return Command(update=result, goto=END)

    def cancellation_node(state: CallCenterAgentState) -> Command[Literal[END]]:
        """Cancellation - terminal state, processes then ends call."""
        result = cancellation_agent.invoke(state)
        
        # Use the current_step that the agent set in its result
        next_step = result.get("current_step", CallStep.CLOSING.value)

        # Cancellation typically goes to closing
        if next_step == CallStep.CLOSING.value:
            logger.info("Agent decided: move to closing after cancellation")
            return Command(update=result, goto=CallStep.CLOSING.value)

        else:
            # Wait for human response if not ready to close
            logger.info("Agent decided: continue cancellation process")
            return Command(update=result, goto=END)

    def closing_node(state: CallCenterAgentState) -> Command[Literal[END]]:
        """Closing - final state with comprehensive logging then END."""
        result = closing_agent.invoke(state)
        
        logger.info("Call ended with comprehensive logging")
        return Command(update=result, goto=END)
        

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
    # workflow.add_node(CallStep.QUERY_RESOLUTION.value, query_resolution_node)
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
            # CallStep.QUERY_RESOLUTION.value: CallStep.QUERY_RESOLUTION.value,
            CallStep.ESCALATION.value: CallStep.ESCALATION.value,
            CallStep.CANCELLATION.value: CallStep.CANCELLATION.value,
            CallStep.CLOSING.value: CallStep.CLOSING.value
        }
    )
    
    compile_kwargs = {}
    if config.get('configurable', {}).get('use_memory'):
        compile_kwargs["checkpointer"] = MemorySaver()

    return workflow.compile(**compile_kwargs)
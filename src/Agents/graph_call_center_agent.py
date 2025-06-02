"""
Optimized Call Center Agent Workflow - Updated for refactored architecture
Router intelligently detects step relevance, completion, and handles routing decisions
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

# Import existing components
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.call_scripts import determine_script_type_from_aging

# Import all specialized sub-agents (updated imports)
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

# Import all required database tools
from src.Database.CartrackSQLDatabase import (
    date_helper,
    # Account and Payment Information
    get_client_account_overview,
    get_client_account_aging,
    get_client_payment_history,
    get_client_failed_payments,
    get_client_banking_details,
    get_client_subscription_amount,
    get_client_debit_mandates,
    
    # Payment Creation and Processing
    create_payment_arrangement,
    create_debicheck_payment,
    create_payment_arrangement_payment_portal,
    generate_sms_payment_url,
    
    # Client Information Updates
    update_client_contact_number,
    update_client_email,
    add_client_note,
    
    # Call Management
    save_call_disposition,
    get_disposition_types,
    update_payment_arrangements
)
# ============================================================================
# TOOL DEFINITIONS - Clean and Organized
# ============================================================================

# Core Call Flow Tools
introduction_tools = []  # Introduction needs no tools - just greeting

name_verification_tools = []  # Uses built-in verification logic

details_verification_tools = []  # Uses built-in verification tools

reason_for_call_tools = [
    # get_client_account_overview,
    # get_client_account_aging,
]

negotiation_tools = [
    # get_client_payment_history,
    # get_client_failed_payments,
]

# Payment Processing Tools
promise_to_pay_tools = [
    # get_client_account_overview,
    date_helper,
    get_client_banking_details,
    create_payment_arrangement,
    create_debicheck_payment
]

debicheck_tools = [
    get_client_debit_mandates,
    create_debicheck_payment,
]

payment_portal_tools = [
    create_payment_arrangement_payment_portal,
    generate_sms_payment_url,
]

# Account Management Tools
subscription_reminder_tools = [
    get_client_subscription_amount,
]

client_details_update_tools = [
    update_client_contact_number,
    update_client_email,
    add_client_note
]

referrals_tools = [
    add_client_note  # For logging referral interest
]

further_assistance_tools = [
    add_client_note  # For logging additional assistance provided
]

# Special Handling Tools
query_resolution_tools = [
    add_client_note  # For logging query resolution
]

escalation_tools = [
    add_client_note,
    save_call_disposition
]

cancellation_tools = [
    get_client_account_aging,
    add_client_note,
    save_call_disposition
]

closing_tools = [
    save_call_disposition,
    get_disposition_types,
    add_client_note,
    update_payment_arrangements
]

########################################################################
logger = logging.getLogger(__name__)


def create_call_center_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,
    agent_name: str = "AI Agent",
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> CompiledGraph:
    """Create the complete call center agent workflow with optimized router."""
    # Auto-determine script type if not provided
    if not script_type:
        account_aging = client_data.get("account_aging", {})
        script_type = determine_script_type_from_aging(account_aging, client_data)
        logger.info(f"Auto-determined script type: {script_type}")

    model_3b = ChatOllama(model="qwen2.5:3b-instruct", temperature=0, num_ctx=4096)
    model_7b = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=4096)
    llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0, num_ctx=32000)

    config = config or {}
    max_name_attempts = CONFIG.get("verification", {}).get("max_name_verification_attempts", 5)
    max_details_attempts = CONFIG.get("verification", {}).get("max_details_verification_attempts", 5)

    # ========================================================================
    # CORE CALL FLOW AGENTS
    # ========================================================================
    
    introduction_agent = create_introduction_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=introduction_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Introduction agent created")
    
    name_verification_agent = create_name_verification_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=name_verification_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Name verification agent created")
    
    details_verification_agent = create_details_verification_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=details_verification_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Details verification agent created")
    
    reason_for_call_agent = create_reason_for_call_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=reason_for_call_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Reason for call agent created")
    
    negotiation_agent = create_negotiation_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=negotiation_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Negotiation agent created")
    
    # ========================================================================
    # PAYMENT PROCESSING AGENTS
    # ========================================================================
    
    promise_to_pay_agent = create_promise_to_pay_agent(
        model=model,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=promise_to_pay_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Promise to pay agent created")
    
    debicheck_setup_agent = create_debicheck_setup_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=debicheck_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ DebiCheck setup agent created")
    
    payment_portal_agent = create_payment_portal_agent(
        model=model,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=payment_portal_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Payment portal agent created")
    
    # ========================================================================
    # ACCOUNT MANAGEMENT AGENTS
    # ========================================================================
    
    subscription_reminder_agent = create_subscription_reminder_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=subscription_reminder_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Subscription reminder agent created")
    
    client_details_update_agent = create_client_details_update_agent(
        model=model,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=client_details_update_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Client details update agent created")
    
    referrals_agent = create_referrals_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=referrals_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Referrals agent created")
    
    further_assistance_agent = create_further_assistance_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=further_assistance_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Further assistance agent created")
    
    # ========================================================================
    # SPECIAL HANDLING AGENTS
    # ========================================================================
    
    query_resolution_agent = create_query_resolution_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=query_resolution_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Query resolution agent created")
    
    escalation_agent = create_escalation_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=escalation_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Escalation agent created")
    
    cancellation_agent = create_cancellation_agent(
        model=model,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=cancellation_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Cancellation agent created")
    
    closing_agent = create_closing_agent(
        model=model_3b,
        client_data=client_data,
        script_type=script_type,
        agent_name=agent_name,
        tools=closing_tools,
        verbose=verbose,
        config=config
    )
    logger.info("✅ Closing agent created")

    # OPTIMIZED ROUTER LLM PROMPT - Step-Aware Classification
    def _get_optimized_router_prompt(state):
        """Generate highly optimized step-aware router classification prompt."""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        # Get last client message
        messages = state.get("messages", [])
        last_client_message = ""
        last_ai_message = ""
        
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and not last_client_message:
                last_client_message = msg.content
            elif isinstance(msg, AIMessage) and not last_ai_message:
                last_ai_message = msg.content

        # Pre-analyze message for keyword patterns
        msg_lower = last_client_message.lower()
        
        # Determine likely classification based on patterns
        has_escalation_words = any(word in msg_lower for word in [
            "supervisor", "manager", "complaint", "harassment", "legal", "lawyer"
        ])
        
        has_cancellation_words = any(word in msg_lower for word in [
            "cancel", "terminate", "stop service", "end service", "disconnect"
        ])
        
        has_question_words = any(pattern in msg_lower for pattern in [
            "how does", "what happens", "why wasn", "what's included", "how do you"
        ])
        
        has_agreement_words = any(word in msg_lower for word in [
            "okay", "yes", "fine", "alright", "let's do", "sure"
        ])
        
        has_objection_words = any(pattern in msg_lower for pattern in [
            "can't afford", "don't have", "no money", "won't pay", "refuse", "wrong", "dispute"
        ])

        return f"""<role>Debt Collection Call Router</role>

    <message_to_classify>"{last_client_message}"</message_to_classify>

    <context>
    Current Step: {current_step}
    AI Said: "{last_ai_message}"
    </context>

    <classification_rules>
    1. ESCALATION: Contains "supervisor", "manager", "complaint", "harassment", "legal"
    Examples: "I want a supervisor", "This is harassment", "Filing a complaint"

    2. CANCELLATION: Contains "cancel", "terminate", "stop service", "disconnect"  
    Examples: "Cancel my account", "Stop all services", "Terminate this"

    3. QUERY_UNRELATED: Questions about service features/technical issues unrelated to payment
    Examples: "How does Cartrack work?", "What happens if stolen?", "Why wasn't payment taken?"

    4. STEP_RELATED: Directly answers current step's purpose
    Examples: Name confirmations, providing ID, payment discussions, verification responses

    5. AGREEMENT: Clear acceptance of AI's request
    Examples: "Okay let's arrange payment", "Yes I understand", "Fine I'll pay"

    6. OBJECTION: Refusal or resistance to current step  
    Examples: "Can't afford", "Don't have money", "That's wrong", "Won't pay"
    </classification_rules>

    <keyword_analysis>
    Escalation keywords detected: {has_escalation_words}
    Cancellation keywords detected: {has_cancellation_words}  
    Question keywords detected: {has_question_words}
    Agreement keywords detected: {has_agreement_words}
    Objection keywords detected: {has_objection_words}
    </keyword_analysis>

    <step_context>
    For {current_step}:
    - Identity confirmations (name, ID) = STEP_RELATED
    - Payment amount questions = STEP_RELATED
    - Service feature questions = QUERY_UNRELATED
    - Financial refusals = OBJECTION
    - Positive responses to requests = AGREEMENT
    </step_context>

    <critical_examples>
    "I want to speak to a supervisor" → ESCALATION (supervisor keyword)
    "Stop all services immediately" → CANCELLATION (stop + services keywords)  
    "How does Cartrack work?" → QUERY_UNRELATED (service question)
    "Yes, this is John Smith" → STEP_RELATED (identity confirmation)
    "I can't afford R199" → OBJECTION (financial refusal)
    "Okay, let's arrange payment" → AGREEMENT (accepting request)
    "Can I pay half now?" → STEP_RELATED (payment discussion)
    "Why wasn't my payment taken?" → QUERY_UNRELATED (technical question)
    </critical_examples>

    <instructions>
    1. Check for escalation keywords FIRST
    2. Check for cancellation keywords SECOND  
    3. Distinguish questions (QUERY_UNRELATED) from responses (STEP_RELATED)
    4. Financial refusals are always OBJECTION
    5. Clear acceptance is AGREEMENT
    6. Direct responses to current step are STEP_RELATED
    </instructions>

    <output_format>
    Respond with EXACTLY ONE WORD: ESCALATION, CANCELLATION, STEP_RELATED, QUERY_UNRELATED, AGREEMENT, or OBJECTION
    </output_format>"""

    # STEP COMPLETION DETECTION
    def is_step_complete(step: str, state: CallCenterAgentState) -> bool:
        """Enhanced step completion detection using state values and conversation analysis."""
        
        if step == CallStep.NAME_VERIFICATION.value:
            return state.get("name_verification_status") == VerificationStatus.VERIFIED.value
            
        elif step == CallStep.DETAILS_VERIFICATION.value:
            return state.get("details_verification_status") == VerificationStatus.VERIFIED.value
            
        elif step == CallStep.REASON_FOR_CALL.value:
            # Complete after one AI explanation of reason
            messages = state.get("messages", [])
            if len(messages) >= 2:
                # Check if AI explained the overdue amount
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        content = msg.content.lower()
                        if any(indicator in content for indicator in ["overdue", "payment", "owe", "balance"]):
                            return True
            return False
            
        elif step == CallStep.NEGOTIATION.value:
            # Complete if client shows agreement or objections are addressed
            messages = state.get("messages", [])
            if len(messages) >= 2:
                # Check last client response for agreement
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        content = msg.content.lower()
                        agreement_indicators = ["okay", "fine", "yes", "understand", "let's do it"]
                        if any(indicator in content for indicator in agreement_indicators):
                            return True
                        break
            return False
            
        elif step == CallStep.PROMISE_TO_PAY.value:
            # Complete if payment arrangement was secured (AI confirms arrangement)
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content.lower()
                    secured_indicators = ["perfect", "excellent", "great", "setting up", "arranging", "processing"]
                    if any(indicator in content for indicator in secured_indicators):
                        return True
                    break
            return False
            
        elif step in [
            CallStep.DEBICHECK_SETUP.value, 
            CallStep.PAYMENT_PORTAL.value,
            CallStep.SUBSCRIPTION_REMINDER.value, 
            CallStep.CLIENT_DETAILS_UPDATE.value, 
            CallStep.REFERRALS.value,
            CallStep.FURTHER_ASSISTANCE.value
        ]:
            # These steps complete after one AI turn
            messages = state.get("messages", [])
            return len(messages) >= 2  # At least one AI response after entry
            
        return False

    def get_next_step_after_completion(step: str, state: CallCenterAgentState) -> str:
        """Determine next step after current step is complete."""
        
        step_progression = {
            CallStep.NAME_VERIFICATION.value: CallStep.DETAILS_VERIFICATION.value,
            CallStep.DETAILS_VERIFICATION.value: CallStep.REASON_FOR_CALL.value,
            CallStep.REASON_FOR_CALL.value: CallStep.NEGOTIATION.value,
            CallStep.NEGOTIATION.value: CallStep.PROMISE_TO_PAY.value,
            CallStep.SUBSCRIPTION_REMINDER.value: CallStep.CLIENT_DETAILS_UPDATE.value,
            CallStep.CLIENT_DETAILS_UPDATE.value: CallStep.REFERRALS.value,
            CallStep.REFERRALS.value: CallStep.FURTHER_ASSISTANCE.value,
            CallStep.FURTHER_ASSISTANCE.value: CallStep.CLOSING.value
        }
        
        # Special logic for promise to pay
        if step == CallStep.PROMISE_TO_PAY.value:
            # Detect payment method from conversation
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content.lower()
                    if "debicheck" in content or "bank" in content:
                        return CallStep.DEBICHECK_SETUP.value
                    elif "portal" in content or "link" in content:
                        return CallStep.PAYMENT_PORTAL.value
                    break
            return CallStep.SUBSCRIPTION_REMINDER.value
        
        # Special logic for payment method steps
        if step in [CallStep.DEBICHECK_SETUP.value, CallStep.PAYMENT_PORTAL.value]:
            return CallStep.SUBSCRIPTION_REMINDER.value
        
        return step_progression.get(step, CallStep.CLOSING.value)

    # OPTIMIZED ROUTER NODE WITH ENHANCED LOGIC
    def router_node(state: CallCenterAgentState) -> Dict[str, Any]:
        """Enhanced router with step-aware classification and robust completion detection."""
        
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        # Skip for first message
        if len(state.get("messages", [])) < 2:
            return {"current_step": CallStep.INTRODUCTION.value}
        
        # 1. Business rule overrides (highest priority)
        name_failed = (state.get("name_verification_attempts", 0) >= max_name_attempts and 
                      state.get("name_verification_status") != VerificationStatus.VERIFIED.value)
        details_failed = (state.get("details_verification_attempts", 0) >= max_details_attempts and 
                         state.get("details_verification_status") != VerificationStatus.VERIFIED.value)
        
        if name_failed or details_failed or state.get("is_call_ended"):
            logger.info("Business rule override - routing to closing")
            return {"current_step": CallStep.CLOSING.value}
        
        if current_step == CallStep.NAME_VERIFICATION.value and state.get("name_verification_status") != VerificationStatus.VERIFIED.value:
            return {"current_step": CallStep.NAME_VERIFICATION.value}
        
        if current_step == CallStep.DETAILS_VERIFICATION.value and state.get("details_verification_status") != VerificationStatus.VERIFIED.value:
            return {"current_step": CallStep.DETAILS_VERIFICATION.value}
        
        # 2. Use optimized router LLM for classification
        router_llm = ChatOllama(model="qwen2.5:3b-instruct", temperature=0, num_ctx=4096)
        prompt_content = _get_optimized_router_prompt(state)
        prompt = [SystemMessage(content=prompt_content)]
        print(f"Router prompt: {prompt_content}")
        try:
            response = router_llm.invoke(prompt)
            classification = response.content.strip().upper()
            print(f"Router LLM classified: {classification} for step: {current_step}")
            
            # 3. Handle classification results
            if classification == "ESCALATION":
                logger.info("Escalation detected - routing to escalation")
                return {"current_step": CallStep.ESCALATION.value}
            
            elif classification == "CANCELLATION":
                logger.info("Cancellation detected - routing to cancellation")
                return {"current_step": CallStep.CANCELLATION.value}
            
            elif classification == "QUERY_UNRELATED":
                logger.info(f"Off-topic query detected - routing to query resolution, will return to {current_step}")
                return {
                    "current_step": CallStep.QUERY_RESOLUTION.value,
                    "return_to_step": current_step
                }
            
            # 4. For STEP_RELATED, AGREEMENT, OBJECTION - check if step is complete
            if classification in ["STEP_RELATED", "AGREEMENT", "OBJECTION"]:
                if is_step_complete(current_step, state):
                    next_step = get_next_step_after_completion(current_step, state)
                    logger.info(f"Step {current_step} complete - progressing to {next_step}")
                    return {"current_step": next_step}
                else:
                    logger.info(f"Step {current_step} not complete - staying in current step")
                    return {"current_step": current_step}
        
        except Exception as e:
            logger.warning(f"Router LLM failed: {e}, using fallback logic")
        
        # 5. Fallback: Check step completion without LLM classification
        if is_step_complete(current_step, state):
            next_step = get_next_step_after_completion(current_step, state)
            logger.info(f"Fallback: Step {current_step} complete - progressing to {next_step}")
            return {"current_step": next_step}
        
        # 6. Default: Stay in current step
        logger.info(f"Default: Staying in current step: {current_step}")
        return {"current_step": current_step}

    def execution_router(state: CallCenterAgentState) -> str:
        """Simple execution router: reads current_step and routes to that node."""
        current_step = state.get("current_step", CallStep.INTRODUCTION.value)
        
        if state.get("is_call_ended"):
            return CallStep.CLOSING.value
            
        logger.info(f"Execution router: Routing to {current_step}")
        return current_step

    # NODE IMPLEMENTATIONS - Following Architecture Guide

    # RARE HANDOFF: Introduction → Always handoff to name verification
    def introduction_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Introduction step - always handoff to name verification."""
 
        result = introduction_agent.invoke(state)
        logger.info("Introduction complete - handing off to name verification")
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.NAME_VERIFICATION.value
            },
            goto="__end__"
        )

    # RARE HANDOFF: Name Verification → Handoff only if VERIFIED
    def name_verification_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Name verification step - handoff only if VERIFIED, end call for terminal states."""

        result = name_verification_agent.invoke(state)
        
        update = {
            "messages": result.get("messages", []),
            "name_verification_status": result.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
            "name_verification_attempts": result.get("name_verification_attempts", 0),
            "current_step": CallStep.NAME_VERIFICATION.value
        }
        
        verification_status = result.get("name_verification_status")
        
        # RARE HANDOFF: Only if VERIFIED - proceed to details verification
        if verification_status == VerificationStatus.VERIFIED.value:
            update["current_step"] = CallStep.DETAILS_VERIFICATION.value
            logger.info("Name verification VERIFIED - handing off to details verification")
            return Command(update=update, goto=CallStep.DETAILS_VERIFICATION.value)
        
        # END CALL: Terminal states that should end the call
        elif verification_status in [
            VerificationStatus.THIRD_PARTY.value,
            VerificationStatus.UNAVAILABLE.value,
            VerificationStatus.WRONG_PERSON.value,
            VerificationStatus.VERIFICATION_FAILED.value
        ]:
            update['is_call_ended'] = True
            update["current_step"] = CallStep.CLOSING.value
            logger.info(f"Name verification terminal state: {verification_status} - ending call")
            return Command(update=update, goto="__end__")
        
        # CONTINUE CALL: Only INSUFFICIENT_INFO should continue
        elif verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            # Stay in name verification step for router to continue trying
            logger.info(f"Name verification insufficient info - returning to router")
            return Command(update=update, goto="__end__")
        
        # FALLBACK: Unknown status - stay in verification
        else:
            logger.warning(f"Unknown name verification status: {verification_status} - staying in verification")
            return Command(update=update, goto="__end__")

    # RARE HANDOFF: Details Verification → Handoff only if VERIFIED  
    def details_verification_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Details verification step - handoff only if VERIFIED, end call for terminal states."""
        
        result = details_verification_agent.invoke(state)
        
        update = {
            "messages": result.get("messages", []),
            "details_verification_status": result.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
            "details_verification_attempts": result.get("details_verification_attempts", 0),
            "matched_fields": result.get("matched_fields", []),
            "current_step": CallStep.DETAILS_VERIFICATION.value
        }
        
        verification_status = result.get("details_verification_status")
        
        # RARE HANDOFF: Only if VERIFIED - proceed to reason for call
        if verification_status == VerificationStatus.VERIFIED.value:
            update["current_step"] = CallStep.REASON_FOR_CALL.value
            logger.info("Details verification VERIFIED - handing off to reason for call")
            return Command(update=update, goto=CallStep.REASON_FOR_CALL.value)
        
        # END CALL: Terminal states that should end the call
        elif verification_status in [
            VerificationStatus.THIRD_PARTY.value,
            VerificationStatus.UNAVAILABLE.value,
            VerificationStatus.WRONG_PERSON.value,
            VerificationStatus.VERIFICATION_FAILED.value
        ]:
            update['is_call_ended'] = True
            update["current_step"] = CallStep.CLOSING.value
            logger.info(f"Details verification terminal state: {verification_status} - ending call")
            return Command(update=update, goto="__end__")
        
        # CONTINUE CALL: Only INSUFFICIENT_INFO should continue
        elif verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            # Stay in details verification step for router to continue trying
            logger.info(f"Details verification insufficient info - returning to router")
            return Command(update=update, goto="__end__")
        
        # FALLBACK: Unknown status - stay in verification
        else:
            logger.warning(f"Unknown details verification status: {verification_status} - staying in verification")
            return Command(update=update, goto="__end__")

    
    def reason_for_call_node(state: CallCenterAgentState) -> Command[Literal["negotiation"]]:
        """Reason for call step """
        
        result = reason_for_call_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.NEGOTIATION.value
            },
            goto="negotiation"
        )
    
    # ALL OTHER STEPS → Return to __end__
    def negotiation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Negotiation step - return to __end__ after one turn."""
           
        result = negotiation_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.NEGOTIATION.value
            },
            goto="__end__"
        )

    def promise_to_pay_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Promise to pay step - return to __end__ after one turn."""
        
        result = promise_to_pay_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.PROMISE_TO_PAY.value
            },
            goto="__end__"
        )

    def debicheck_setup_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """DebiCheck setup step - return to __end__ after one turn."""
        if state.get("current_step") != CallStep.DEBICHECK_SETUP.value:
            return Command(update={}, goto="__end__")
        
        result = debicheck_setup_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.DEBICHECK_SETUP.value
            },
            goto="__end__"
        )

    def payment_portal_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Payment portal step - return to __end__ after one turn."""
        if state.get("current_step") != CallStep.PAYMENT_PORTAL.value:
            return Command(update={}, goto="__end__")
        
        result = payment_portal_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.PAYMENT_PORTAL.value
            },
            goto="__end__"
        )

    def subscription_reminder_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Subscription reminder step - return to __end__ after one turn."""
        if state.get("current_step") != CallStep.SUBSCRIPTION_REMINDER.value:
            return Command(update={}, goto="__end__")
        
        result = subscription_reminder_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.SUBSCRIPTION_REMINDER.value
            },
            goto="__end__"
        )

    def client_details_update_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Client details update step - return to __end__ after one turn."""
        if state.get("current_step") != CallStep.CLIENT_DETAILS_UPDATE.value:
            return Command(update={}, goto="__end__")
        
        result = client_details_update_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.CLIENT_DETAILS_UPDATE.value
            },
            goto="__end__"
        )

    def referrals_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Referrals step - return to __end__ after one turn."""
        if state.get("current_step") != CallStep.REFERRALS.value:
            return Command(update={}, goto="__end__")
        
        result = referrals_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.REFERRALS.value
            },
            goto="__end__"
        )

    def further_assistance_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Further assistance step - return to __end__ after one turn."""
        if state.get("current_step") != CallStep.FURTHER_ASSISTANCE.value:
            return Command(update={}, goto="__end__")
        
        result = further_assistance_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.FURTHER_ASSISTANCE.value
            },
            goto="__end__"
        )

    def query_resolution_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Query resolution - answer briefly then return to main goal."""
        result = query_resolution_agent.invoke(state)
        return_to_step = state.get("return_to_step", CallStep.CLOSING.value)
        
        logger.info(f"Query resolution complete - returning to {return_to_step}")
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": return_to_step,
                "return_to_step": None
            },
            goto="__end__"
        )

    def escalation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Escalation step - return to __end__ after handling."""
        result = escalation_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.CLOSING.value  # Route to closing after escalation
            },
            goto="__end__"
        )

    def cancellation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Cancellation step - return to __end__ after handling."""
        result = cancellation_agent.invoke(state)
        return Command(
            update={
                "messages": result.get("messages", []),
                "current_step": CallStep.CLOSING.value  # Route to closing after cancellation
            },
            goto="__end__"
        )

    def closing_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
        """Closing step - ends the call."""
        result = closing_agent.invoke(state)
        logger.info("Closing complete - call ended")
        return Command(
            update={
                "messages": result.get("messages", []),
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
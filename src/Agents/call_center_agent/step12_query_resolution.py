# ===============================================================================
# STEP 12: QUERY RESOLUTION AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step12_query_resolution.py
"""
Query Resolution Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value, calculate_outstanding_amount, format_currency
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

from src.Database.CartrackSQLDatabase import add_client_note

def get_query_resolution_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate aging-aware query resolution prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    
    # Calculate outstanding amount
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Get redirect info
    return_to_step = state.get("return_to_step", "closing")
    redirect_message = state.get("redirect_message", "Anything else?")
    
    # Build aging-specific redirect strategies
    redirect_strategies_by_category = {
        "First Missed Payment": {
            "payment_redirect": f"Now, can we arrange {outstanding_amount} today?",
            "service_question": f"That's how it works. Can we debit {outstanding_amount} to restore services?",
            "tone": "helpful and solution-focused"
        },
        "Failed Promise to Pay": {
            "payment_redirect": f"Let's honor your commitment with {outstanding_amount} now",
            "service_question": f"Services restore once you pay the {outstanding_amount} as agreed",
            "tone": "accountability-focused"
        },
        "2-3 Months Overdue": {
            "payment_redirect": f"We need {outstanding_amount} immediately to prevent escalation",
            "service_question": f"Services suspended until {outstanding_amount} is paid",
            "tone": "urgent and direct"
        },
        "Pre-Legal 120+ Days": {
            "payment_redirect": f"{outstanding_amount} required now to prevent court action",
            "service_question": f"Services and legal issues resolved with {outstanding_amount} payment",
            "tone": "serious and final opportunity"
        },
        "Legal 150+ Days": {
            "payment_redirect": f"Legal demand for {outstanding_amount} - immediate payment required",
            "service_question": f"Court proceedings stop only with {outstanding_amount} payment",
            "tone": "legal authority and urgency"
        }
    }
    
    category = aging_context['category']
    redirect_strategy = redirect_strategies_by_category.get(category, redirect_strategies_by_category["First Missed Payment"])
    
    # Build urgency-appropriate response lengths and examples
    response_examples_by_urgency = {
        "Medium": {
            "cartrack_question": f"Vehicle tracking and security. Now, can we arrange {outstanding_amount} today?",
            "payment_question": f"Bank declined it. Can we try a different method for {outstanding_amount}?",
            "max_words": "15 words maximum"
        },
        "High": {
            "cartrack_question": f"Tracking system. {outstanding_amount} needed urgently to restore services.",
            "payment_question": f"Payment failed. {outstanding_amount} required immediately.",
            "max_words": "12 words maximum"
        },
        "Very High": {
            "cartrack_question": f"Vehicle security. {outstanding_amount} needed now to prevent escalation.",
            "payment_question": f"Payment issue. {outstanding_amount} required to stop legal action.",
            "max_words": "10 words maximum"
        },
        "Critical": {
            "cartrack_question": f"Security system. Pay {outstanding_amount} now to stop court proceedings.",
            "payment_question": f"Payment failed. {outstanding_amount} required immediately for legal compliance.",
            "max_words": "8 words maximum"
        }
    }
    
    urgency_level = aging_context['urgency']
    response_examples = response_examples_by_urgency.get(urgency_level, response_examples_by_urgency["Medium"])

    # Base prompt
    base_prompt_un_verified = f"""<role>
You are a professional debt collection specialist from Cartrack Account Department.
</role>

<context>
- Client: {client_full_name}
</context>

<task>
Answer question BRIEFLY then redirect to verification step.
</task>

<style>
- {response_examples['max_words']}
- {aging_context['tone']}
- Stay focused on payment goal with appropriate urgency
- Natural, conversational tone
- Use exact redirect message
</style>"""
    
    base_prompt_verified = f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<context>
- Client: {client_full_name}
- Outstanding: {outstanding_amount}
- Aging Category: {category}
- Urgency Level: {urgency_level}
- Return to: {return_to_step}
</context>

<task>
Answer question BRIEFLY then redirect to payment using aging-appropriate urgency.
</task>

<aging_specific_examples>
Q: "How does Cartrack work?"
A: "{response_examples['cartrack_question']}"

Q: "Why wasn't my payment taken?"
A: "{response_examples['payment_question']}"
</aging_specific_examples>

<redirect_strategy>
**Payment Focus**: "{redirect_strategy['payment_redirect']}"
**Service Questions**: "{redirect_strategy['service_question']}"
**Tone**: "{redirect_strategy['tone']}"
</redirect_strategy>

<redirect_target>
Return to: {return_to_step}
</redirect_target>

<urgency_adaptation>
{aging_context['approach']}
</urgency_adaptation>

<style>
- {response_examples['max_words']}
- {aging_context['tone']}
- Stay focused on payment goal with appropriate urgency
- Natural, conversational tone
- Use exact redirect message
</style>"""
    # Context
    name_verification_status = state.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
    details_verification_status = state.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
    if name_verification_status != VerificationStatus.VERIFIED or details_verification_status != VerificationStatus.VERIFIED:
        base_prompt = base_prompt_un_verified
        return base_prompt
    else:
        base_prompt = base_prompt_verified

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.QUERY_RESOLUTION,
        client_data=client_data,
        state=state
    )

def create_query_resolution_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a query resolution agent with aging-aware scripts."""
    
    # agent_tools = [add_client_note] + (tools or [])
    agent_tools = (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Get the return step
        return_to_step = state.get("return_to_step", CallStep.CLOSING.value)
        
        # Analyze the client's question to categorize it
        conversation_messages = state.get("messages", [])
        last_client_message = ""
        
        if conversation_messages:
            for msg in reversed(conversation_messages):
                if hasattr(msg, 'type') and msg.type == "human":
                    last_client_message = msg.content.lower()
                    break
                elif isinstance(msg, dict) and msg.get("role") in ["user", "human"]:
                    last_client_message = msg.get("content", "").lower()
                    break
        
        # Categorize the query type
        query_type = "general"
        query_keywords = {
            "service_question": ["cartrack", "tracking", "work", "service", "how does", "what does"],
            "payment_question": ["payment", "pay", "bank", "debit", "card", "money"],
            "account_question": ["balance", "owe", "amount", "cost", "charge", "billing"],
            "technical_question": ["app", "device", "installation", "setup", "not working"],
            "policy_question": ["cancel", "contract", "terms", "conditions", "policy"]
        }
        
        for category, keywords in query_keywords.items():
            if any(keyword in last_client_message for keyword in keywords):
                query_type = category
                break
        
        # Check verification status
        name_verified = state.get("name_verification_status") == VerificationStatus.VERIFIED.value
        details_verified = state.get("details_verification_status") == VerificationStatus.VERIFIED.value
        is_fully_verified = name_verified and details_verified
        
        # Determine redirect strategy based on verification and urgency
        account_aging = client_data.get("account_aging", {})
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        if not name_verified:
            redirect_strategy = "verify_name"
            redirect_target = CallStep.NAME_VERIFICATION.value
            redirect_message = f"Are you {get_safe_value(client_data, 'profile.client_info.client_full_name', 'Client')}?"
        elif not details_verified:
            redirect_strategy = "verify_details"
            redirect_target = CallStep.DETAILS_VERIFICATION.value
            redirect_message = "Your ID number please?"
        elif return_to_step in [CallStep.PROMISE_TO_PAY.value, CallStep.NEGOTIATION.value]:
            redirect_strategy = "secure_payment"
            redirect_target = return_to_step
            outstanding_amount = state.get("outstanding_amount", "the amount")
            redirect_message = f"Now, can we arrange {outstanding_amount} today?"
        else:
            redirect_strategy = "continue_call"
            redirect_target = return_to_step
            redirect_message = "Anything else I can help with?"
        
        return Command(
            update={
                "query_type": query_type,
                "redirect_strategy": redirect_strategy,
                "redirect_target": redirect_target,
                "redirect_message": redirect_message,
                "is_fully_verified": is_fully_verified,
                "last_client_question": last_client_message,
                "return_to_step": return_to_step,
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
                "current_step": CallStep.QUERY_RESOLUTION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_query_resolution_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(prompt_content)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="QueryResolutionAgent"
    )

# src/Agents/call_center_agent/step12_query_resolution.py
"""
Query Resolution Agent - Self-contained with own prompt
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

# Import relevant database tools
from src.Database.CartrackSQLDatabase import add_client_note

def get_query_resolution_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate query resolution specific prompt."""
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    
    # Calculate outstanding amount
    account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Get redirect info
    return_to_step = state.get("return_to_step", "closing")
    redirect_message = state.get("redirect_message", "Anything else?")
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Answer question BRIEFLY (under 15 words) then redirect to payment.
</task>

<format>
Brief answer + redirect to payment goal
</format>

<examples>
Q: "How does Cartrack work?"
A: "Vehicle tracking and security. Now, can we arrange {outstanding_amount} today?"

Q: "What happens if I don't pay?"
A: "Services stop working. Let's arrange payment now to avoid that."

Q: "Why wasn't my payment taken?"
A: "Bank declined it. Can we try a different method for {outstanding_amount}?"

Q: "Who are you again?"
A: "Agent from Cartrack. Are you {client_full_name}?"
</examples>

<redirect_target>
Return to: {return_to_step}
Redirect message: "{redirect_message}"
</redirect_target>

<style>
- MAXIMUM 15 words total
- Stay focused on payment goal
- Natural, conversational tone
- Use exact redirect message
</style>"""

def create_query_resolution_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a query resolution agent with brief answers and smart redirects."""
    
    agent_tools = [add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to analyze query and determine appropriate redirect strategy."""
        
        # Get the return step (where we should redirect to)
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
        
        # Categorize the query type for appropriate response
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
        
        # Check verification status to determine redirect approach
        name_verified = state.get("name_verification_status") == VerificationStatus.VERIFIED.value
        details_verified = state.get("details_verification_status") == VerificationStatus.VERIFIED.value
        is_fully_verified = name_verified and details_verified
        
        # Determine redirect strategy based on verification and return step
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
        
        # Create response template based on query type
        response_templates = {
            "service_question": "Vehicle tracking and security.",
            "payment_question": "Payment methods include debit and online portal.",
            "account_question": f"Your balance is {state.get('outstanding_amount', 'R 0.00')}.",
            "technical_question": "Technical support: 011 250 3000.",
            "policy_question": "Policy details available on request.",
            "general": "I understand."
        }
        
        brief_answer = response_templates.get(query_type, "I understand.")
        
        return Command(
            update={
                "query_type": query_type,
                "redirect_strategy": redirect_strategy,
                "redirect_target": redirect_target,
                "redirect_message": redirect_message,
                "brief_answer": brief_answer,
                "is_fully_verified": is_fully_verified,
                "last_client_question": last_client_message,
                "return_to_step": return_to_step,
                "current_step": CallStep.QUERY_RESOLUTION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for query resolution with smart redirect."""
        prompt_content = get_query_resolution_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
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
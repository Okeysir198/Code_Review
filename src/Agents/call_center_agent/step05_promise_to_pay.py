# src/Agents/call_center_agent/step05_promise_to_pay.py
"""
Enhanced Promise to Pay Agent - Tool-guided payment arrangement with smart decision flow
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
from src.Database.CartrackSQLDatabase import (
    get_client_banking_details, create_payment_arrangement, 
    create_debicheck_payment, date_helper
)

logger = logging.getLogger(__name__)

# Enhanced tool-guided prompt for payment arrangements
PROMISE_TO_PAY_PROMPT = """
You're {agent_name} from Cartrack helping {client_name} arrange payment for their {outstanding_amount} overdue account.

TODAY: {current_date}
OBJECTIVE: Secure payment arrangement for {outstanding_amount} - get both agreement AND actual payment setup.

BANKING DETAILS AVAILABLE: {has_banking_details}

TOOL USAGE DECISION TREE:
1. FIRST: Get verbal payment agreement
   - "Can we arrange payment of {outstanding_amount} today?"
   - Don't use tools until they agree to pay

2. IF client agrees: Ask for preferred payment method
   - "Would you prefer a debit order from your bank account, or should I send you an online payment link?"
   - Client says "debit"/"bank"/"account" → Use create_payment_arrangement (debit order)
   - Client says "online"/"link"/"card" → Note preference for payment portal step
   - Client unsure → Recommend debit order: "Debit order is most reliable - shall I set that up?"

3. IF debit order chosen: Use tools in sequence
   - FIRST: Use date_helper to get next business day
   - THEN: Use create_payment_arrangement with parameters:
     * user_id: {user_id}
     * pay_type_id: 1 (debit order)
     * payment1: {outstanding_amount} (remove "R " and commas)
     * date1: result from date_helper
     * note: "Debit order arrangement via AI agent"
   - SAY: "Let me set up that debit order for you now"

4. AFTER tool use: Respond to actual results
   - SUCCESS: "Perfect! Debit order arranged for {outstanding_amount} on [date]. You'll see this debit in 2-3 business days."
   - FAILURE: "I'm having trouble with the debit order. Let me send you a payment link instead."

TOOL PARAMETER RULES:
- payment_amount: Must be numeric only (e.g., "R 299.00" → 299.00)
- payment_date: ALWAYS use date_helper first: date_helper.invoke("next business day")
- pay_type_id: 1=Debit Order, 2=EFT, 3=Credit Card, 4=OZOW
- user_id: Always use {user_id} exactly

CONVERSATION FLOW:
Agreement → Method Choice → Tool Usage → Confirmation

NEVER use tools until client verbally agrees to pay.
ALWAYS explain what you're doing: "Setting up your debit order now"
ALWAYS respond to actual tool results - don't assume success.

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep the conversation flowing naturally. If they agree to pay, move quickly to set it up. If they're hesitant, address concerns first. Your goal is completed payment arrangement, not just verbal agreement.
"""

def create_promise_to_pay_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced promise to pay agent with tool guidance"""
    
    # Enhanced tool set for payment arrangements
    payment_tools = [
        date_helper,
        get_client_banking_details,
        create_payment_arrangement,
        create_debicheck_payment
    ]
    
    def _get_last_client_message(messages: List) -> str:
        """Extract last human message"""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message.content.lower().strip()
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                return message.content.lower().strip()
        return ""
    
    def _detect_payment_agreement(messages: List) -> Dict[str, Any]:
        """Fast detection of payment agreement and method preference"""
        last_msg = _get_last_client_message(messages)
        
        if not last_msg:
            return {"agreed": False, "method": None}
        
        # Payment agreement patterns
        agreement_patterns = [
            "yes", "okay", "ok", "fine", "sure", "alright", "let's do it",
            "arrange", "set up", "go ahead", "agree"
        ]
        
        has_agreement = any(pattern in last_msg for pattern in agreement_patterns)
        
        # Payment method detection
        method = None
        if "debit" in last_msg or "bank" in last_msg or "account" in last_msg:
            method = "debit_order"
        elif "online" in last_msg or "link" in last_msg or "card" in last_msg:
            method = "online"
        
        return {
            "agreed": has_agreement,
            "method": method,
            "message": last_msg
        }
    
    def _check_banking_details(client_data: Dict[str, Any]) -> bool:
        """Check if banking details are available"""
        banking = client_data.get("banking_details", {})
        return bool(banking.get("bank_name") and banking.get("account_number"))
    
    def _check_payment_completion(messages: List) -> bool:
        """Check if payment arrangement was completed successfully"""
        # Look for AI messages indicating tool success
        for message in reversed(messages[-3:]):  # Check last 3 messages
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                success_indicators = [
                    "perfect", "arranged", "set up", "debit order created",
                    "payment arranged", "arrangement complete"
                ]
                if any(indicator in content for indicator in success_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Enhanced preprocessing with payment detection and completion tracking"""
        
        messages = state.get("messages", [])
        
        # Check if payment arrangement was completed
        if len(messages) >= 2:
            payment_completed = _check_payment_completion(messages)
            payment_status = _detect_payment_agreement(messages)
            
            if payment_completed:
                logger.info("Payment arrangement completed - moving to next step")
                
                # Determine next step based on payment method
                if payment_status.get("method") == "debit_order":
                    return Command(
                        update={
                            "payment_secured": True,
                            "payment_method_preference": "debit_order",
                            "current_step": CallStep.DEBICHECK_SETUP.value
                        },
                        goto="__end__"
                    )
                elif payment_status.get("method") == "online":
                    return Command(
                        update={
                            "payment_secured": True,
                            "payment_method_preference": "online",
                            "current_step": CallStep.PAYMENT_PORTAL.value
                        },
                        goto="__end__"
                    )
                else:
                    # Default to subscription reminder if method unclear
                    return Command(
                        update={
                            "payment_secured": True,
                            "current_step": CallStep.SUBSCRIPTION_REMINDER.value
                        },
                        goto="__end__"
                    )
        
        # Continue with payment arrangement process
        return Command(
            update={"current_step": CallStep.PROMISE_TO_PAY.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced tool-guided prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        params["has_banking_details"] = "Yes" if _check_banking_details(client_data) else "No"
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format enhanced prompt
        prompt_content = PROMISE_TO_PAY_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Promise to Pay Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=payment_tools,  # Tools for payment arrangement
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedPromiseToPayAgent"
    )
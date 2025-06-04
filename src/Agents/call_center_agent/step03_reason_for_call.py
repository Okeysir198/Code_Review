# src/Agents/call_center_agent/step03_reason_for_call.py
"""
Enhanced Reason for Call Agent - Natural account explanation with context awareness
"""
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

import logging
logger = logging.getLogger(__name__)

# Enhanced conversational prompt for account explanation
REASON_FOR_CALL_PROMPT = """
You're {agent_name} from Cartrack explaining to {client_name} why you're calling about their {outstanding_amount} overdue account.

TODAY: {current_date}
OBJECTIVE: Clearly explain the account status and create urgency for payment resolution.

ACCOUNT SITUATION:
- Outstanding Amount: {outstanding_amount}
- Account Status: {account_status}
- Urgency Level: {urgency_level}

EXPLANATION BY URGENCY LEVEL:
- Standard: "We didn't receive your {outstanding_amount} subscription payment. Can we arrange payment today?"
- High: "Your account is overdue 2+ months. {outstanding_amount} is required immediately to avoid service suspension."
- Critical: "Your account is with our legal department. Arrears of {outstanding_amount} must be settled. Do you acknowledge this debt?"

COMMUNICATION STRATEGY:
1. State the facts clearly and directly
2. Specify the exact amount owed: {outstanding_amount}
3. Create appropriate urgency for the situation
4. Request immediate action/response

CONSEQUENCES TO MENTION:
- Service interruption (app stops working, no vehicle positioning)
- Recovery fees (up to R25,000 if vehicle is stolen while unprotected)
- Legal action and credit listing (for serious overdue accounts)

URGENCY LEVEL: {urgency_level} - {aging_approach}

Be direct but professional. The client is verified, so you can discuss their account openly. Match your tone to the urgency level - be firm for serious overdue accounts, understanding for first missed payments.

Keep responses under 25 words unless detailed explanation is needed for legal/pre-legal accounts.
"""

def create_reason_for_call_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced reason for call agent with urgency-based messaging"""
    
    def _check_explanation_completion(messages: List) -> bool:
        """Check if account explanation was provided"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                explanation_indicators = [
                    "overdue", "payment", "owe", "balance", "subscription",
                    "didn't receive", "account", "arrears"
                ]
                if any(indicator in content for indicator in explanation_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Check if reason for call explanation is complete"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            explanation_given = _check_explanation_completion(messages)
            
            if explanation_given:
                logger.info("Account explanation provided - moving to negotiation")
                return Command(
                    update={
                        "current_step": CallStep.NEGOTIATION.value
                    },
                    goto="__end__"
                )
        
        # Continue with explanation
        return Command(
            update={"current_step": CallStep.REASON_FOR_CALL.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced reason for call prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, script_type, agent_name)
        
        
        
        # Format prompt
        prompt_content = REASON_FOR_CALL_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Reason for Call Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed for account explanation
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedReasonForCallAgent"
    )
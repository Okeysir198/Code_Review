# src/Agents/call_center_agent/step08_subscription_reminder.py
"""
Enhanced Subscription Reminder Agent - Clear distinction between arrears and ongoing billing
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

SUBSCRIPTION_REMINDER_PROMPT = """
You're {agent_name} from Cartrack clarifying billing for {client_name} after securing payment for {outstanding_amount}.

TODAY: {current_date}
OBJECTIVE: Prevent confusion between today's payment and ongoing subscription billing.

PAYMENT JUST SECURED: {outstanding_amount} (arrears/overdue amount)
ONGOING SUBSCRIPTION: {subscription_amount} monthly on {subscription_date}

KEY MESSAGE BY URGENCY:
- Standard: "Today's {outstanding_amount} covers missed payments. Your regular {subscription_amount} continues monthly."
- High: "Today's {outstanding_amount} clears arrears. Monthly {subscription_amount} resumes normally."
- Critical: "Today's {outstanding_amount} settles debt. Regular billing resumes after this payment."

CLARIFICATION SCRIPT:
"Perfect! Just to clarify - today's payment of {outstanding_amount} covers what was overdue. Your regular subscription of {subscription_amount} will continue as normal on the {subscription_date}. So you're not paying double."

COMMON CONFUSIONS TO ADDRESS:
- "Am I paying twice?" → "No, today covers arrears. Monthly billing is separate."
- "When is my next payment?" → "Your regular {subscription_amount} on {subscription_date}."
- "Is this a once-off?" → "Today clears overdue. Monthly subscription continues."

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep explanation under 20 words. Focus on clarity: arrears vs ongoing billing. Prevent double-payment confusion.
"""

def create_subscription_reminder_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced subscription reminder agent with billing clarification"""
    
    def _check_clarification_completion(messages: List) -> bool:
        """Check if subscription clarification was provided"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                clarification_indicators = [
                    "regular", "monthly", "subscription", "continues",
                    "separate", "not paying double", "clarify"
                ]
                if any(indicator in content for indicator in clarification_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Check if subscription clarification is complete"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            clarification_given = _check_clarification_completion(messages)
            
            if clarification_given:
                logger.info("Subscription clarification provided - moving to client details update")
                return Command(
                    update={
                        "current_step": CallStep.CLIENT_DETAILS_UPDATE.value
                    },
                    goto="__end__"
                )
        
        # Continue with clarification
        return Command(
            update={"current_step": CallStep.SUBSCRIPTION_REMINDER.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced subscription reminder prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format prompt
        prompt_content = SUBSCRIPTION_REMINDER_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Subscription Reminder Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed for clarification
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedSubscriptionReminderAgent"
    )
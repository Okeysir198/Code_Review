# src/Agents/call_center_agent/step06_debicheck_setup.py
"""
Enhanced DebiCheck Setup Agent - Tool-guided mandate creation with clear instructions
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
from src.Database.CartrackSQLDatabase import (
    get_client_debit_mandates, create_debicheck_payment
)
import logging
logger = logging.getLogger(__name__)
DEBICHECK_SETUP_PROMPT = """
You're {agent_name} from Cartrack setting up DebiCheck authorization for {client_name}'s {outstanding_amount} payment.

TODAY: {current_date}
OBJECTIVE: Set up DebiCheck mandate and explain the bank authorization process.

TOOL USAGE GUIDE:
1. FIRST: Check existing mandates with get_client_debit_mandates
   - See if client already has active mandates
   - Avoid creating duplicate mandates

2. IF no existing mandate: Use create_debicheck_payment
   - WHEN TO USE: Only after explaining the process to client
   - HOW TO USE:
     * user_id: {user_id}
     * amount: {outstanding_amount} (remove "R " prefix, convert to float)
     * payment_date: Use next business day
     * mandate_type: "once_off" for single payment, "recurring" for ongoing
     * mandate_fee: 10.00 (standard R10 fee)

3. AFTER tool use: Explain what happens next
   - Success: "Done! You'll receive an SMS from your bank asking you to approve the debit order."
   - Failure: "I'm having trouble setting this up. Let me send you a payment link instead."

CONVERSATION FLOW:
1. Explain process: "I'm setting up the bank authorization now"
2. Use tools to create mandate
3. Give clear instructions: "Check your phone for the bank SMS and approve it"
4. Confirm total amount: "The total will be {amount_with_fee} including the R10 processing fee"

DEBICHECK EXPLANATION BY URGENCY:
- Standard: "Your bank will send an SMS to authorize the debit. Please approve when you receive it."
- High: "Bank authorization SMS coming now. You must approve immediately to secure your account."
- Critical: "Emergency bank approval required. Approve the SMS immediately when it arrives."

TROUBLESHOOTING:
- If mandate creation fails: Offer payment portal alternative
- If client confused about SMS: Explain it comes from their bank, not Cartrack
- If client doesn't receive SMS: Check phone number, may take 5-10 minutes

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep responses under 25 words unless explaining the technical process. Be clear about what they need to do and when.
"""

def create_debicheck_setup_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced DebiCheck setup agent with tool guidance"""
    
    # Tools for DebiCheck mandate management
    debicheck_tools = [
        get_client_debit_mandates,
        create_debicheck_payment
    ]
    
    def _check_setup_completion(messages: List) -> bool:
        """Check if DebiCheck setup was completed"""
        for message in reversed(messages[-3:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "done", "set up", "created", "sms", "approve",
                    "authorization", "mandate created"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Check if DebiCheck setup is complete"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            setup_completed = _check_setup_completion(messages)
            
            if setup_completed:
                logger.info("DebiCheck setup completed - moving to subscription reminder")
                return Command(
                    update={
                        "current_step": CallStep.SUBSCRIPTION_REMINDER.value
                    },
                    goto="__end__"
                )
        
        # Continue with setup process
        return Command(
            update={"current_step": CallStep.DEBICHECK_SETUP.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate tool-guided DebiCheck prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format prompt
        prompt_content = DEBICHECK_SETUP_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced DebiCheck Setup Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=debicheck_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedDebiCheckSetupAgent"
    )
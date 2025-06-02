# src/Agents/call_center_agent/step07_payment_portal.py
"""
Enhanced Payment Portal Agent - Tool-guided online payment link generation
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
    create_payment_arrangement_payment_portal, generate_sms_payment_url
)
import logging
logger = logging.getLogger(__name__)
PAYMENT_PORTAL_PROMPT = """
You're {agent_name} from Cartrack generating a secure payment link for {client_name}'s {outstanding_amount} payment.

TODAY: {current_date}
OBJECTIVE: Create secure payment link and guide client through online payment process.

TOOL SEQUENCE FOR PAYMENT LINK:
1. FIRST: create_payment_arrangement_payment_portal
   - Creates the payment arrangement record
   - Parameters:
     * user_id: {user_id}
     * payment_type_id: 4 (OZOW online payment)
     * payment_date: today's date
     * amount: {outstanding_amount} (remove "R ", convert to float)
     * online_payment_reference_id: auto-generated
   - Say: "Creating your secure payment link now"

2. THEN: generate_sms_payment_url
   - Creates secure payment URL
   - Sends SMS with payment link
   - Parameters:
     * user_id: {user_id}
     * amount: {outstanding_amount} (same as above)
     * optional_reference: "AI_AGENT_" + today's date
   - Say: "Sending payment link to your phone"

3. AFTER both tools: Give clear instructions
   - Success: "Done! Check your phone for the payment link. Click 'PAY HERE', confirm {outstanding_amount}, and choose your payment method."
   - Failure: "Having trouble generating the link. Let me try the debit order option instead."

CONVERSATION FLOW:
Explain → Create Link → Send SMS → Give Instructions → Stay Connected

PAYMENT LINK INSTRUCTIONS BY URGENCY:
- Standard: "Payment link sent to your phone. Click the link, confirm {outstanding_amount}, and choose your method."
- High: "Urgent payment link sent. Complete immediately. Amount is {outstanding_amount}."
- Critical: "Emergency payment link sent. Pay {outstanding_amount} now to avoid consequences."

TOOL ERROR HANDLING:
- Step 1 fails: "System issue with payment setup. Let me try debit order instead."
- Step 2 fails: "Link creation failed. I'll call you back with payment details."
- SMS delivery issues: "Link may take 2-3 minutes to arrive. Check junk/spam folder."

STAY CONNECTED: "I'll stay on the line while you complete the payment. Let me know when you see the link."

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep responses under 25 words unless giving detailed payment instructions. Guide them through each step clearly.
"""

def create_payment_portal_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced payment portal agent with tool sequencing"""
    
    # Tools for payment link generation
    portal_tools = [
        create_payment_arrangement_payment_portal,
        generate_sms_payment_url
    ]
    
    def _check_link_generation_completion(messages: List) -> bool:
        """Check if payment link was generated successfully"""
        for message in reversed(messages[-3:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "link sent", "check your phone", "payment link", 
                    "click", "sms sent", "done"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Check if payment portal setup is complete"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            link_generated = _check_link_generation_completion(messages)
            
            if link_generated:
                logger.info("Payment link generated - moving to subscription reminder")
                return Command(
                    update={
                        "payment_method_preference": "online",
                        "current_step": CallStep.SUBSCRIPTION_REMINDER.value
                    },
                    goto="__end__"
                )
        
        # Continue with link generation
        return Command(
            update={"current_step": CallStep.PAYMENT_PORTAL.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate tool-guided payment portal prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format prompt
        prompt_content = PAYMENT_PORTAL_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Payment Portal Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=portal_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedPaymentPortalAgent"
    )
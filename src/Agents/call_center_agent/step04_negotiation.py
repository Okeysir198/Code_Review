# src/Agents/call_center_agent/step04_negotiation.py
"""
Enhanced Negotiation Agent - Natural objection handling with context awareness
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
from src.Agents.call_center_agent.parameter_helper import prepare_parameters, detect_client_mood_from_messages
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
import logging
logger = logging.getLogger(__name__)

# Enhanced negotiation prompt with objection handling
NEGOTIATION_PROMPT = """
You're {agent_name} from Cartrack negotiating with {client_name} about their {outstanding_amount} overdue account.

TODAY: {current_date}
OBJECTIVE: Handle objections and move {client_name} toward payment commitment.

CLIENT MOOD: {detected_mood}
URGENCY LEVEL: {urgency_level}

OBJECTION HANDLING STRATEGIES:
- "No money/Can't afford": "I understand finances are tight. What amount could you manage today to maintain your vehicle security?"
- "Already paid": "Let me check our records. When and how was this payment made? I want to ensure it's properly applied."
- "Dispute amount": "I can verify the breakdown while we arrange payment to avoid service interruption. What specifically concerns you?"
- "Will pay later": "I appreciate that intention. However, to prevent service suspension, we need to secure payment today. What's preventing immediate payment?"
- "Not my responsibility": "I understand your concern. Let's verify the details while ensuring your services remain active."

CONSEQUENCES BY URGENCY:
- Standard: "Services stop working. Vehicle tracking ends. App becomes inactive."
- High: "Services suspended. Potential R25,000 recovery fee if vehicle stolen. Credit listing risk."
- Critical: "Legal action commenced. Court costs. Sheriff service. Asset attachment possible."

BENEFITS OF IMMEDIATE PAYMENT:
- Standard: "Pay now, everything works immediately. Account cleared."
- High: "Immediate payment prevents legal escalation and additional fees."
- Critical: "Payment today stops court proceedings and additional legal costs."

ESCALATION TRIGGERS (route to escalation if detected):
- Repeated aggressive language or threats
- Demands for supervisor multiple times
- Claims of harassment or legal action against Cartrack
- Absolute refusal with hostile tone

MOOD-BASED APPROACH:
- Cooperative: Move quickly to payment options
- Resistant: Emphasize consequences and benefits
- Confused: Clarify situation patiently  
- Angry: Acknowledge frustration, focus on resolution
- Financial stress: Show empathy, explore options

Keep responses under 25 words unless detailed explanation is needed. Push for commitment while handling objections professionally.
"""

def create_negotiation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced negotiation agent with mood-aware objection handling"""
    
    def _get_last_client_message(messages: List) -> str:
        """Extract last human message"""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message.content.lower().strip()
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                return message.content.lower().strip()
        return ""
    
    def _detect_escalation_needed(messages: List) -> bool:
        """Detect if escalation is needed based on client responses"""
        last_msg = _get_last_client_message(messages)
        
        escalation_triggers = [
            "supervisor", "manager", "harassment", "lawyer", "attorney",
            "sue you", "take you to court", "complaint", "ombudsman"
        ]
        
        return any(trigger in last_msg for trigger in escalation_triggers)
    
    def _detect_payment_readiness(messages: List) -> bool:
        """Detect if client is ready to discuss payment"""
        last_msg = _get_last_client_message(messages)
        
        readiness_indicators = [
            "okay", "fine", "understand", "what are my options",
            "how much", "when", "arrange", "set up"
        ]
        
        return any(indicator in last_msg for indicator in readiness_indicators)
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Enhanced preprocessing with escalation and payment readiness detection"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            # Check for escalation triggers
            if _detect_escalation_needed(messages):
                logger.info("Escalation needed - client requesting supervisor/legal")
                return Command(
                    update={
                        "escalation_requested": True,
                        "current_step": CallStep.ESCALATION.value
                    },
                    goto="__end__"
                )
            
            # Check for payment readiness
            if _detect_payment_readiness(messages):
                logger.info("Client showing payment readiness - moving to promise to pay")
                return Command(
                    update={
                        "current_step": CallStep.PROMISE_TO_PAY.value
                    },
                    goto="__end__"
                )
        
        # Continue negotiation
        return Command(
            update={"current_step": CallStep.NEGOTIATION.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced negotiation prompt with mood detection"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, script_type, agent_name)
        
        # Detect client mood from conversation
        messages = state.get("messages", [])
        params["detected_mood"] = detect_client_mood_from_messages(messages)
        
       
        # Format prompt
        prompt_content = NEGOTIATION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Negotiation Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed for negotiation
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedNegotiationAgent"
    )
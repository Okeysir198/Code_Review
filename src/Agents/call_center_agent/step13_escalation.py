# src/Agents/call_center_agent/step13_escalation.py
"""
Enhanced Escalation Agent - Professional escalation handling with appropriate authority levels
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
from src.Database.CartrackSQLDatabase import add_client_note

import logging
logger = logging.getLogger(__name__)

ESCALATION_PROMPT = """
You're {agent_name} from Cartrack handling escalation request from {client_name} regarding their {outstanding_amount} account.

TODAY: {current_date}
OBJECTIVE: Professional escalation with appropriate authority level and clear timeline.

CLIENT CONTEXT:
- Mood: {detected_mood}
- Aging Category: {aging_category}
- Outstanding: {outstanding_amount}
- Reason: {escalation_reason}

ESCALATION MAPPING BY AGING:
- First Payment/Standard: Customer Service Supervisor, 24-48 hours
- Failed PTP: Accounts Supervisor, 24 hours  
- 2-3 Months: Senior Collections Manager, 24 hours
- Pre-Legal: Pre-Legal Manager, 12-24 hours
- Legal: Legal Department, 12 hours

CURRENT ESCALATION:
- Department: {target_department}
- Authority: {authority_level}
- Response Time: {response_timeline}

TOOL USAGE:
- add_client_note: Document escalation request with details
- Parameters: user_id={user_id}, note_text="Escalation requested: [reason] - Routed to {target_department}"

ESCALATION PROCESS:
1. Acknowledge concern professionally
2. Create escalation ticket with details
3. Set clear expectations for response
4. Provide timeline and next steps

RESPONSE BY CLIENT MOOD:
- Angry: "I understand your frustration. Let me escalate this immediately to {authority_level}."
- Concerned: "I hear your concerns. {authority_level} will review this personally."
- Demanding: "I'm creating an urgent escalation to {authority_level} for immediate review."

PROFESSIONAL APPROACH:
✓ Acknowledge their concern
✓ Take ownership of escalation
✓ Provide specific timeline
✓ Explain next steps clearly
✓ Document thoroughly

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep response under 20 words. Professional acknowledgment, clear timeline, immediate action.
"""

def create_escalation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced escalation agent with authority mapping"""
    
    # Tools for escalation documentation
    escalation_tools = [add_client_note]
    
    def _determine_escalation_target(script_type: str, aging_category: str) -> Dict[str, str]:
        """Determine appropriate escalation target based on account aging"""
        
        escalation_mapping = {
            "First Missed Payment": {
                "department": "Customer Service",
                "authority": "Customer Service Supervisor",
                "timeline": "24-48 hours"
            },
            "Failed Promise to Pay": {
                "department": "Accounts",
                "authority": "Accounts Supervisor", 
                "timeline": "24 hours"
            },
            "2-3 Months Overdue": {
                "department": "Senior Collections",
                "authority": "Senior Collections Manager",
                "timeline": "24 hours"
            },
            "Pre-Legal 120+ Days": {
                "department": "Pre-Legal",
                "authority": "Pre-Legal Manager",
                "timeline": "12-24 hours"
            },
            "Legal 150+ Days": {
                "department": "Legal",
                "authority": "Legal Department",
                "timeline": "12 hours"
            }
        }
        
        return escalation_mapping.get(
            aging_category,
            escalation_mapping["First Missed Payment"]  # Default fallback
        )
    
    def _get_last_client_message(messages: List) -> str:
        """Extract last human message for escalation reason"""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message.content.strip()
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                return message.content.strip()
        return ""
    
    def _extract_escalation_reason(messages: List) -> str:
        """Extract reason for escalation from conversation"""
        recent_messages = []
        for message in reversed(messages[-4:]):
            if hasattr(message, 'type') and message.type == 'human':
                recent_messages.append(message.content.lower())
                if len(recent_messages) >= 2:
                    break
        
        combined_text = " ".join(recent_messages)
        
        # Common escalation reasons
        if "supervisor" in combined_text or "manager" in combined_text:
            return "Supervisor request"
        elif "complaint" in combined_text:
            return "Formal complaint"
        elif "harassment" in combined_text:
            return "Harassment claim"
        elif "legal" in combined_text or "lawyer" in combined_text:
            return "Legal threat"
        elif "dispute" in combined_text:
            return "Account dispute"
        else:
            return "General escalation request"
    
    def _check_escalation_completion(messages: List) -> bool:
        """Check if escalation was processed"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "escalated", "supervisor", "manager", "review", 
                    "within", "hours", "ticket created"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Set up escalation context and check completion"""
        
        messages = state.get("messages", [])
        
        # Determine escalation target
        aging_context = ScriptManager.get_aging_context(script_type)
        escalation_target = _determine_escalation_target(script_type, aging_context['category'])
        
        # Extract escalation reason
        escalation_reason = _extract_escalation_reason(messages)
        
        if len(messages) >= 2:
            escalation_completed = _check_escalation_completion(messages)
            
            if escalation_completed:
                logger.info("Escalation processed - moving to closing")
                return Command(
                    update={
                        "escalation_requested": True,
                        "current_step": CallStep.CLOSING.value,
                        "call_outcome": "escalated"
                    },
                    goto="__end__"
                )
        
        # Continue with escalation process
        return Command(
            update={
                "escalation_requested": True,
                "escalation_reason": escalation_reason,
                "target_department": escalation_target["department"],
                "authority_level": escalation_target["authority"],
                "response_timeline": escalation_target["timeline"],
                "current_step": CallStep.ESCALATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate authority-appropriate escalation prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, script_type, agent_name)
        
        # Add escalation-specific context
        messages = state.get("messages", [])
        params["detected_mood"] = detect_client_mood_from_messages(messages)
        params["escalation_reason"] = state.get("escalation_reason", "General request")
        params["target_department"] = state.get("target_department", "Supervisor")
        params["authority_level"] = state.get("authority_level", "Supervisor")
        params["response_timeline"] = state.get("response_timeline", "24-48 hours")
        
        # Get aging-specific approach
        
        
        # Format prompt
        prompt_content = ESCALATION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Escalation Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=escalation_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedEscalationAgent"
    )
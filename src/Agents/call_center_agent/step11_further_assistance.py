# src/Agents/call_center_agent/step11_further_assistance.py
"""
Enhanced Further Assistance Agent - Context-aware final assistance check with urgency management
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

FURTHER_ASSISTANCE_PROMPT = """
You're {agent_name} from Cartrack offering final assistance to {client_name} before concluding the call.

TODAY: {current_date}
OBJECTIVE: Check for additional concerns while managing call conclusion appropriately.

CALL CONTEXT:
- Payment Secured: {payment_secured}
- Client Mood: {detected_mood}
- Urgency Level: {urgency_level}
- Call Progress: {call_progress}

ASSISTANCE SCOPE BY URGENCY:
- Low/Medium: "Is there anything else regarding your Cartrack account I can help you with today?"
- High: "Any other urgent account matters I can assist with quickly?"
- Critical: "Anything else critical I need to address before we conclude?"

TIME MANAGEMENT BY CONTEXT:
- Payment Secured + Cooperative: Allow time for additional questions
- Payment Secured + Urgent: Brief check, prepare to close
- No Payment + Any Mood: Keep focused, handle core concerns only
- Escalation/Cancellation Pending: Minimal additional assistance

COMMON ADDITIONAL REQUESTS:
- Service questions: "How does the tracking work?"
- Billing questions: "When is my next payment?"
- Technical issues: "App not working properly"
- Account changes: "Update my contact details"
- Cancellation: Route to cancellation process

RESPONSE APPROACH:
- Genuine concern for their needs
- Time-appropriate for urgency level
- Professional but warm closing approach
- Set expectation for call conclusion

{assistance_instruction}

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep responses under 15 words unless addressing specific questions. Show genuine concern while managing time appropriately.
"""

def create_further_assistance_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced further assistance agent with context management"""
    
    def _get_last_client_message(messages: List) -> str:
        """Extract last human message"""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message.content.lower().strip()
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                return message.content.lower().strip()
        return ""
    
    def _detect_additional_requests(messages: List) -> Dict[str, Any]:
        """Detect if client has additional requests or wants to conclude"""
        last_msg = _get_last_client_message(messages)
        
        if not last_msg:
            return {"has_request": False, "request_type": None, "ready_to_close": False}
        
        # Closure indicators
        closure_patterns = [
            "no", "nothing", "that's all", "that's it", "all good", "thanks",
            "no thanks", "i'm good", "we're done", "goodbye", "bye"
        ]
        
        ready_to_close = any(pattern in last_msg for pattern in closure_patterns)
        
        # Additional request patterns
        request_patterns = {
            "cancellation": ["cancel", "stop", "terminate", "end service"],
            "technical": ["app", "not working", "problem", "issue", "broken"],
            "billing": ["bill", "payment", "when", "cost", "charge"],
            "service": ["how does", "what is", "explain", "help me understand"],
            "contact_update": ["change", "update", "new phone", "new email"]
        }
        
        request_type = None
        for req_type, patterns in request_patterns.items():
            if any(pattern in last_msg for pattern in patterns):
                request_type = req_type
                break
        
        has_request = request_type is not None and not ready_to_close
        
        return {
            "has_request": has_request,
            "request_type": request_type,
            "ready_to_close": ready_to_close
        }
    
    def _determine_assistance_approach(state: CallCenterAgentState, messages: List) -> str:
        """Determine appropriate assistance approach"""
        
        payment_secured = state.get("payment_secured", False)
        detected_mood = detect_client_mood_from_messages(messages)
        aging_context = ScriptManager.get_aging_context(script_type)
        urgency = aging_context['urgency']
        
        if urgency == 'Critical':
            return "Brief final check. Prepare for immediate closure."
        elif urgency == 'Very High':
            return "Quick assistance check. Keep focused on resolution."
        elif payment_secured and detected_mood == 'cooperative':
            return "Allow time for questions. Client in good standing."
        elif payment_secured:
            return "Standard assistance check. Professional closure."
        else:
            return "Focused assistance. Core concerns only."
    
    def _check_assistance_completion(messages: List) -> bool:
        """Check if further assistance check was completed"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "anything else", "help you with", "assist", "further",
                    "that's all", "concluded", "thank you"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Handle additional requests or prepare for closure"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            assistance_completed = _check_assistance_completion(messages)
            additional_request = _detect_additional_requests(messages)
            
            # Route based on client response
            if additional_request["ready_to_close"] and assistance_completed:
                logger.info("Client ready to close - moving to closing")
                return Command(
                    update={
                        "current_step": CallStep.CLOSING.value
                    },
                    goto="__end__"
                )
            
            elif additional_request["request_type"] == "cancellation":
                logger.info("Cancellation request detected - routing to cancellation")
                return Command(
                    update={
                        "cancellation_requested": True,
                        "current_step": CallStep.CANCELLATION.value
                    },
                    goto="__end__"
                )
            
            elif additional_request["has_request"]:
                logger.info(f"Additional request: {additional_request['request_type']} - routing to query resolution")
                return Command(
                    update={
                        "return_to_step": CallStep.FURTHER_ASSISTANCE.value,
                        "current_step": CallStep.QUERY_RESOLUTION.value
                    },
                    goto="__end__"
                )
            
            elif assistance_completed:
                logger.info("Further assistance completed - moving to closing")
                return Command(
                    update={
                        "current_step": CallStep.CLOSING.value
                    },
                    goto="__end__"
                )
        
        # Continue with assistance check
        return Command(
            update={"current_step": CallStep.FURTHER_ASSISTANCE.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate context-aware further assistance prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Get conversation context
        messages = state.get("messages", [])
        params["detected_mood"] = detect_client_mood_from_messages(messages)
        params["payment_secured"] = "Yes" if state.get("payment_secured") else "No"
        params["call_progress"] = "Good progress" if state.get("payment_secured") else "Needs resolution"
        
        # Get assistance approach
        params["assistance_instruction"] = _determine_assistance_approach(state, messages)
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format prompt
        prompt_content = FURTHER_ASSISTANCE_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Further Assistance Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed for assistance check
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedFurtherAssistanceAgent"
    )
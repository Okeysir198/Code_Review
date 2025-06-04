# src/Agents/call_center_agent/step12_query_resolution.py
"""
Enhanced Query Resolution Agent - Smart question handling with verification-aware responses
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

import logging
logger = logging.getLogger(__name__)

QUERY_RESOLUTION_PROMPT = """
You're {agent_name} from Cartrack answering {client_name}'s question: "{last_client_question}"

TODAY: {current_date}
OBJECTIVE: Answer briefly then redirect based on verification stage and return step.

VERIFICATION STATUS:
- Name: {name_verification_status}
- Details: {details_verification_status}
- Return To: {return_to_step}

RESPONSE STRATEGY BY VERIFICATION:
🔴 Name NOT verified:
- Answer max 8 words + "Are you {client_full_name}?"
- Example: "Cartrack tracks vehicles. Are you {client_full_name}?"

🟡 Details NOT verified: 
- Answer max 10 words + "Your ID number please?"
- Example: "Vehicle tracking service. Your ID number please?"

🟢 Fully verified:
- Answer + redirect to main objective
- Example: "Vehicle security system. Can we arrange {outstanding_amount}?"

COMMON QUESTIONS & BRIEF ANSWERS:
- "What is Cartrack?" → "Vehicle tracking security"
- "How much?" → "Outstanding amount {outstanding_amount}"
- "What service?" → "Vehicle tracking device"
- "Why calling?" → "Overdue payment {outstanding_amount}"
- "Who are you?" → "Cartrack Accounts Department"
- "Is this real?" → "Yes, Cartrack official call"

REDIRECT TARGETS:
- No verification → Name verification
- Partial verification → Details verification  
- Fully verified → Continue with {return_to_step} or payment focus

URGENCY ADAPTATION:
- Critical: Very brief answers, immediate redirect
- High: Short answers, quick redirect  
- Medium: Helpful but focused answers

URGENCY LEVEL: {urgency_level} - {aging_approach}

CRITICAL RULES:
- Keep under 15 words total (answer + redirect)
- Always redirect to appropriate verification/step
- Don't get sidetracked by complex questions
- Focus on main call objective
"""

def create_query_resolution_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced query resolution agent with verification-aware responses"""
    
    def _get_last_client_message(messages: List) -> str:
        """Extract last human message"""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content.strip()
            elif hasattr(message, 'type') and message.type == 'human':
                return message.content.strip()
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                return message.content.strip()
        return ""
    
    def _determine_redirect_target(state: CallCenterAgentState) -> str:
        """Determine where to redirect based on verification status"""
        
        name_status = state.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        details_status = state.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        return_to_step = state.get("return_to_step", "")
        
        # Verification-based routing
        if name_status != VerificationStatus.VERIFIED.value:
            return CallStep.NAME_VERIFICATION.value
        elif details_status != VerificationStatus.VERIFIED.value:
            return CallStep.DETAILS_VERIFICATION.value
        elif return_to_step:
            return return_to_step
        else:
            # Default to payment focus if fully verified
            return CallStep.PROMISE_TO_PAY.value
    
    def _categorize_question(question: str) -> str:
        """Categorize the type of question for appropriate handling"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what", "cartrack", "service", "company"]):
            return "about_service"
        elif any(word in question_lower for word in ["how much", "cost", "amount", "owe"]):
            return "about_amount"
        elif any(word in question_lower for word in ["who", "calling", "you", "name"]):
            return "about_caller"
        elif any(word in question_lower for word in ["why", "reason", "purpose"]):
            return "about_purpose"
        elif any(word in question_lower for word in ["real", "scam", "legitimate", "fraud"]):
            return "about_legitimacy"
        else:
            return "general"
    
    def _check_resolution_completion(messages: List) -> bool:
        """Check if query was resolved and redirect given"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "are you", "your id", "can we arrange", "may i speak",
                    "let's continue", "back to"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Extract question and determine redirect target"""
        
        messages = state.get("messages", [])
        last_question = _get_last_client_message(messages)
        redirect_target = _determine_redirect_target(state)
        
        if len(messages) >= 2:
            resolution_completed = _check_resolution_completion(messages)
            
            if resolution_completed:
                logger.info(f"Query resolved - redirecting to {redirect_target}")
                return Command(
                    update={
                        "current_step": redirect_target,
                        "return_to_step": None  # Clear return step
                    },
                    goto="__end__"
                )
        
        # Continue with query resolution
        return Command(
            update={
                "last_client_question": last_question,
                "current_step": CallStep.QUERY_RESOLUTION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate verification-aware query resolution prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, script_type, agent_name)
        
        # Add query-specific context
        params["last_client_question"] = state.get("last_client_question", "")
        
        # Get aging-specific approach
        
        
        # Format prompt
        prompt_content = QUERY_RESOLUTION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Query Resolution Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed for query resolution
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedQueryResolutionAgent"
    )
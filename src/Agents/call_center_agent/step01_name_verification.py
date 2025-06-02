# src/Agents/call_center_agent/step01_name_verification.py
"""
Enhanced Name Verification Agent - Natural conversation with fast completion detection
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

import logging
logger = logging.getLogger(__name__)

# Enhanced conversational prompt - clean and focused
NAME_VERIFICATION_PROMPT = """
You're {agent_name}, a professional debt collection specialist at Cartrack's Accounts Department.

TODAY: {current_date}
OBJECTIVE: Confirm you're speaking with {client_full_name} before discussing their {outstanding_amount} overdue account.

RESPOND NATURALLY BASED ON THEIR RESPONSE:
- If they confirm identity: "Thank you {client_name}. For security, I need to verify some details with you."
- If they're suspicious: "I understand your caution. I'm calling about your Cartrack vehicle tracking account to resolve an overdue payment."
- If they're busy: "I appreciate you're busy. This will take just 2 minutes to resolve your account and avoid any service interruption."
- If wrong person: "I apologize for the confusion. I'll update our records. Have a good day."

URGENCY LEVEL: {urgency_level} - {aging_approach}

You're having a real conversation with a real person who might be cautious about unknown callers. Be professional but human. Build trust while staying focused on verification. Match their energy - if they're cooperative, be efficient. If they're concerned, take time to reassure them.

Keep responses natural and under 25 words unless explanation is needed.
"""

def create_name_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced name verification agent with fast completion detection"""
    
    def _get_last_client_message(messages: List) -> str:
        """Extract last human message"""
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'human':
                return message.content.lower().strip()
            elif hasattr(message, 'content') and not hasattr(message, 'type'):
                # Assume it's human if no type specified and not first message
                return message.content.lower().strip()
        return ""
    
    def _quick_verification_check(messages: List, client_name: str) -> str:
        """Fast pattern matching for name verification completion"""
        last_msg = _get_last_client_message(messages)
        
        if not last_msg:
            return VerificationStatus.INSUFFICIENT_INFO.value
        
        # Positive confirmation patterns
        positive_patterns = [
            "yes", "yeah", "yep", "speaking", "this is", "that's me", 
            "correct", "right", client_name.lower().split()[0]  # First name
        ]
        
        if any(pattern in last_msg for pattern in positive_patterns):
            # Extra confidence if they mention their name
            if any(name_part in last_msg for name_part in client_name.lower().split()):
                return VerificationStatus.VERIFIED.value
            # General positive response
            elif any(confirm in last_msg for confirm in ["yes", "speaking", "this is", "that's me"]):
                return VerificationStatus.VERIFIED.value
        
        # Wrong person patterns
        wrong_person_patterns = [
            "wrong person", "wrong number", "not me", "not him", "not her",
            "don't know", "never heard", "no one here"
        ]
        
        if any(pattern in last_msg for pattern in wrong_person_patterns):
            return VerificationStatus.WRONG_PERSON.value
        
        # Third party patterns
        third_party_patterns = [
            "not here", "not available", "he's not", "she's not", 
            "at work", "not home", "can i take a message"
        ]
        
        if any(pattern in last_msg for pattern in third_party_patterns):
            return VerificationStatus.THIRD_PARTY.value
        
        # Still need more info
        return VerificationStatus.INSUFFICIENT_INFO.value
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Enhanced preprocessing with fast completion detection"""
        
        client_full_name = client_data.get("profile", {}).get("client_info", {}).get("client_full_name", "Client")
        attempts = state.get("name_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_name_verification_attempts", 5)
        messages = state.get("messages", [])
        
        # Fast verification check - no LLM needed
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        
        if len(messages) >= 2:  # Only check after client has responded
            verification_status = _quick_verification_check(messages, client_full_name)
            
            if verbose:
                logger.info(f"Fast verification check result: {verification_status}")
        
        # Auto-fail if max attempts reached without verification
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        # Determine next action based on verification result
        if verification_status == VerificationStatus.VERIFIED.value:
            logger.info("Name verification VERIFIED - jumping to details verification")
            return Command(
                update={
                    "name_verification_status": verification_status,
                    "name_verification_attempts": attempts,
                    "current_step": CallStep.DETAILS_VERIFICATION.value
                },
                goto="__end__" 
            )
        
        elif verification_status in [
            VerificationStatus.WRONG_PERSON.value,
            VerificationStatus.VERIFICATION_FAILED.value,
            VerificationStatus.THIRD_PARTY.value
        ]:
            logger.info(f"Name verification terminal: {verification_status} - ending call")
            return Command(
                update={
                    "name_verification_status": verification_status,
                    "name_verification_attempts": attempts,
                    "is_call_ended": True,
                    "current_step": CallStep.CLOSING.value
                },
                goto="agent"  
            )
        
        
        else:
            # Continue verification process
            logger.info(f"Name verification continuing - attempt {attempts}")
            return Command(
                update={
                    "name_verification_status": verification_status,
                    "name_verification_attempts": attempts,
                    "current_step": CallStep.NAME_VERIFICATION.value
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced conversational prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format enhanced prompt
        prompt_content = NAME_VERIFICATION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Name Verification Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed for name verification
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedNameVerificationAgent"
    )
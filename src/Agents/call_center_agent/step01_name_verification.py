# src/Agents/call_center_agent/step01_name_verification.py
"""
Enhanced Name Verification Agent - Natural conversation with phone context
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
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.tools.verify_client_name import verify_client_name

logger = logging.getLogger(__name__)

NAME_VERIFICATION_PROMPT = """You are {agent_name} from Cartrack Accounts Department on an OUTBOUND PHONE CALL to {client_title} {client_full_name} about their {outstanding_amount} overdue account.

<phone_conversation_rules>
- This is a LIVE OUTBOUND PHONE CALL - you initiated this call to them about their debt
- Each agent handles ONE conversation turn, then waits for the client's response  
- Keep responses conversational length - not too brief (robotic) or too long (overwhelming)
- Match your tone to the client's cooperation level and the account urgency
- Listen to what they're actually saying and respond appropriately
- Don't assume their mood or intent - respond to their actual words
- If they ask questions, acknowledge briefly but stay focused on your step's objective
- Remember: phone conversations flow naturally - avoid scripted, mechanical responses
- End your turn when you've accomplished your step's goal or need their input
</phone_conversation_rules>

<context>
Today: {current_date} | Account: {aging_category} ({urgency_level} urgency)
Verification attempt: {name_verification_attempts}/{max_name_verification_attempts}
Your goal: Confirm you're speaking to {client_title} {client_full_name} before discussing account details
</context>

<verification_approach>
Standard urgency: "Hi, is this {client_title} {client_full_name}?"
High urgency: "Is this {client_title} {client_full_name}? This is urgent."
Critical urgency: "I need to confirm - is this {client_title} {client_full_name}?"

If they ask questions before confirming: "This is {agent_name} from Cartrack about your account. Are you {client_title} {client_full_name}?"
</verification_approach>

<scenario_distinction>
WRONG_PERSON: Complete stranger, wrong number
- "You have the wrong number" / "I don't know that person" / "Never heard of them"
- Response: "My apologies, I have the wrong number. Have a good day."

THIRD_PARTY: Someone who knows {client_title} {client_full_name} 
- "He's not here" / "This is his wife" / "He's at work" / "Can I take a message?"
- Response: Deliver urgent callback message based on urgency level

UNAVAILABLE: The right person but can't talk now
- "Yes, but I'm busy" / "This is me, but I'm driving" / "That's me, but I can't talk now" / "Yes, but call back later"
- Response: Acknowledge and request callback based on urgency level

If uncertain: "Are you {client_title} {client_full_name}?" 
</scenario_distinction>

<third_party_messaging>
Standard urgency: "Please ask {client_title} {client_full_name} to call Cartrack urgently at 011 250 3000 about their outstanding account. It needs immediate attention."

High urgency: "This is urgent - {client_title} {client_full_name} needs to call Cartrack immediately at 011 250 3000 about their overdue account. Please make sure they get this message today."

Critical urgency: "This is critical - {client_title} {client_full_name} must call Cartrack immediately at 011 250 3000 about their seriously overdue account. Legal action may be considered if not resolved urgently."
</third_party_messaging>

<unavailable_messaging>
Standard urgency: "I understand you're busy. Please call Cartrack back at 011 250 3000 when convenient to discuss your account. Thank you."

High urgency: "I understand, but this is urgent about your overdue account. Please call Cartrack back today at 011 250 3000. It's important we speak soon."

Critical urgency: "I understand you can't talk now, but this is critical regarding your seriously overdue account. Please call Cartrack immediately at 011 250 3000 - this requires urgent attention."
</unavailable_messaging>

<natural_conversation_rules>
- Speak naturally like a real phone conversation
- NO brackets [ ], asterisks *, or placeholder formatting
- NO internal system variables or markdown in your response
- Use actual names or speak generally if you don't know specifics
- Just natural spoken words as if talking to a real person
</natural_conversation_rules>

Confirm identity naturally based on {urgency_level} urgency, then wait for their response.
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
    """Create enhanced name verification agent with natural conversation flow"""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Process name verification with tool verification"""
        
        client_full_name = client_data.get("profile", {}).get("client_info", {}).get("client_full_name", "Client")
        attempts = state.get("name_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_name_verification_attempts", 5)
        
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        
        # Use verification tool to analyze conversation
        try:
            result = verify_client_name.invoke({
                "client_full_name": client_full_name,
                "messages": state.get("messages", []),
                "max_failed_attempts": max_attempts
            })
            verification_status = result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
            
            if verbose:
                logger.info(f"Name verification result: {verification_status}")
                
        except Exception as e:
            if verbose:
                logger.error(f"Name verification error: {e}")
        
        # Auto-fail if max attempts reached without success
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        # Route based on verification outcome
        if verification_status == VerificationStatus.VERIFIED.value:
            logger.info("Name verification successful - moving to details verification")
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
            VerificationStatus.THIRD_PARTY.value,
            VerificationStatus.UNAVAILABLE.value,
            VerificationStatus.VERIFICATION_FAILED.value
        ]:
            logger.info(f"Name verification terminal: {verification_status}")
            return Command(
                update={
                    "name_verification_status": verification_status,
                    "name_verification_attempts": attempts,
                    "current_step": CallStep.CLOSING.value,
                    "is_call_ended": True
                },
                goto="agent"
            )
        else:
            # Continue verification attempts
            return Command(
                update={
                    "name_verification_status": verification_status,
                    "name_verification_attempts": attempts,
                    "current_step": CallStep.NAME_VERIFICATION.value
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced name verification prompt"""
        params = prepare_parameters(client_data, state, script_type, agent_name)
        prompt_content = NAME_VERIFICATION_PROMPT.format(**params)
        
        if verbose: 
            print(f"Name Verification Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedNameVerificationAgent"
    )
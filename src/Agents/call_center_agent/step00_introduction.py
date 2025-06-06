# src/Agents/call_center_agent/step00_introduction.py
"""
Enhanced Introduction Agent - Context-aware professional greeting with engagement hooks
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters

import logging
logger = logging.getLogger(__name__)

INTRODUCTION_PROMPT = """You are {agent_name} from Cartrack Accounts Department making an OUTBOUND PHONE CALL to {client_title} {client_full_name} about their overdue account.

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
Call type: OUTBOUND debt collection - you called them about money they owe
Your goal: Deliver engaging opening greeting that locks them into the conversation
</context>

<greeting_by_urgency>
Medium urgency: "Good day, this is {agent_name} from Cartrack Accounts Department. May I speak to {client_title} {client_full_name} about your Cartrack account, please?"

High urgency: "Good day, this is {agent_name} from Cartrack Accounts calling {client_title} {client_full_name} about an urgent account matter."

Critical urgency: "Good day, this is {agent_name} from Cartrack Accounts Department. I need to speak to {client_title} {client_full_name} about your account immediately."
</greeting_by_urgency>

<delivery_approach>
Be confident and authoritative - YOU called THEM about legitimate business requiring their attention. Use engagement hooks like "about your account" or "urgent matter" to create curiosity and prevent easy dismissal. Match your tone to the urgency: warmer for standard accounts, more direct for critical situations.
</delivery_approach>

Deliver the appropriate greeting for {urgency_level} urgency, then wait for their response.
"""

def create_introduction_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced introduction agent with engagement-focused greetings"""
    
    def _determine_urgency_category(urgency_level: str) -> str:
        """Determine urgency category for appropriate greeting selection"""
        if urgency_level in ['Critical', 'Very High']:
            return "Pre-Legal/Critical"
        elif urgency_level == 'High':
            return "Higher Urgency/Failed PTP"
        else:
            return "Standard/First Payment"
    
    def _create_greeting(params: Dict[str, str], urgency_category: str) -> str:
        """Create the appropriate greeting based on urgency level"""
        greetings = {
            "Standard/First Payment": 
                f"Good day, this is {params['agent_name']} from Cartrack Accounts Department. May I speak to {params['client_title']} {params['client_full_name']} about your Cartrack account, please?",
            
            "Higher Urgency/Failed PTP":
                f"Good day, this is {params['agent_name']} from Cartrack Accounts calling {params['client_title']} {params['client_full_name']} about an urgent account matter.",
            
            "Pre-Legal/Critical":
                f"Good day, this is {params['agent_name']} from Cartrack Accounts Department. I need to speak to {params['client_title']} {params['client_full_name']} about your account immediately."
        }
        
        return greetings.get(urgency_category, greetings["Standard/First Payment"])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Deliver engagement-focused greeting and move to name verification"""

        BYPASS_LLM = False  # Set to True to bypass LLM for immediate greeting
        # Determine urgency and create greeting
        params = prepare_parameters(client_data, state, script_type, agent_name)
        urgency_category = _determine_urgency_category(params["urgency_level"])

        greeting = _create_greeting(params, urgency_category)
        
        # Deliver greeting immediately
        greeting_message = AIMessage(content=greeting)
        
        logger.info(f"Greeting delivered: {urgency_category}")
        if BYPASS_LLM:
            return Command(
                update={
                    "messages": [greeting_message],
                    "current_step": CallStep.NAME_VERIFICATION.value,
                    "urgency_category": urgency_category
                },
                goto="__end__"
            )
        else:
            # If not bypassing LLM, return to agent for further processing
            return Command(
                update={
                    "current_step": CallStep.NAME_VERIFICATION.value,
                    "urgency_category": urgency_category
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate introduction prompt (fallback - not used with immediate greeting)"""
        params = prepare_parameters(client_data, state, script_type, agent_name)
        prompt_content = INTRODUCTION_PROMPT.format(**params)
        
        if verbose:
            print(f"Introduction Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedIntroductionAgent"
    )
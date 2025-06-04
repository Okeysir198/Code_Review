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
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

import logging
logger = logging.getLogger(__name__)

# Enhanced engagement-focused prompt
INTRODUCTION_PROMPT = """You are {agent_name}, a professional debt collection specialist from Cartrack's Accounts Department making an outbound call about a {outstanding_amount} overdue account.

<context>
Today: {current_date} | Account: {aging_category} | Urgency: {urgency_level}
Purpose: Lock debtor into conversation while requesting specific person
</context>

<engagement_greetings>
Standard/First Payment: 
"Good day, this is {agent_name} from Cartrack Accounts Department. May I speak to {client_title} {client_full_name} about their Cartrack account, please?"

Higher Urgency/Failed PTP:
"Good day, {agent_name} from Cartrack Accounts calling {client_title} {client_full_name} about an urgent account matter."

Pre-Legal/Critical:
"Good day, this is {agent_name} from Cartrack Accounts Department. I need to speak to {client_title} {client_full_name} about their account immediately."
</engagement_greetings>

<engagement_strategy>
Each greeting includes an engagement hook that makes it harder to dismiss:

- "about their Cartrack account" - creates curiosity about what's wrong
- "urgent account matter" - implies time-sensitivity requiring attention  
- "need to speak...immediately" - creates urgency and importance

The word "account" signals this is serious business they need to address, not a sales call they can easily reject.
</engagement_strategy>

<authority_building>
Use confident, business-like delivery that implies:
- This is official business requiring attention
- You have legitimate authority to make this call
- The matter is important enough to warrant immediate discussion
- Dismissing the call would be inappropriate

Higher urgency accounts get more direct, authoritative language that's harder to brush off.
</authority_building>

<outcome>
Create immediate engagement where they feel compelled to either confirm identity or ask what this is about - both responses allow the next agent to begin verification and explanation.
</outcome>

Lock them in with professional urgency while maintaining courtesy. Aging approach: {aging_approach}
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
    
    def _determine_urgency_level(script_type: str) -> str:
        """Determine urgency level for appropriate greeting selection"""
        aging_context = ScriptManager.get_aging_context(script_type)
        urgency = aging_context.get('urgency', 'Medium')
        
        # Map urgency to greeting categories
        if urgency in ['Critical', 'Very High']:
            return "Pre-Legal/Critical"
        elif urgency == 'High':
            return "Higher Urgency/Failed PTP"
        else:
            return "Standard/First Payment"
    
    def _select_appropriate_greeting(params: Dict[str, str], urgency_category: str) -> str:
        """Select the appropriate greeting based on urgency level"""
        greetings = {
            "Standard/First Payment": 
                f"Good day, this is {params['agent_name']} from Cartrack Accounts Department. May I speak to {params['client_title']} {params['client_full_name']} about their Cartrack account, please?",
            
            "Higher Urgency/Failed PTP":
                f"Good day, {params['agent_name']} from Cartrack Accounts calling {params['client_title']} {params['client_full_name']} about an urgent account matter.",
            
            "Pre-Legal/Critical":
                f"Good day, this is {params['agent_name']} from Cartrack Accounts Department. I need to speak to {params['client_title']} {params['client_full_name']} about their account immediately."
        }
        
        return greetings.get(urgency_category, greetings["Standard/First Payment"])
    

    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Enhanced preprocessing with engagement-focused greeting delivery"""
        
        messages = state.get("messages", [])
        
        # Prepare for greeting delivery
        urgency_category = _determine_urgency_level(script_type)
        params = prepare_parameters(client_data, state, script_type, agent_name)
        selected_greeting = _select_appropriate_greeting(params, urgency_category)
        
        # Option 1: Deliver greeting immediately via preprocessing (faster)
        if config.get("immediate_greeting", True):
            greeting_message = AIMessage(content=selected_greeting)
            
            logger.info(f"Immediate greeting delivered: {urgency_category}")
            return Command(
                update={
                    "messages": [greeting_message],
                    "current_step": CallStep.NAME_VERIFICATION.value,
                    "urgency_category": urgency_category
                },
                goto="__end__"
            )
        
        # Option 2: Let agent generate greeting (more flexible)
        else:
            return Command(
                update={
                    "current_step": CallStep.NAME_VERIFICATION.value,
                    "urgency_category": urgency_category
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate enhanced engagement-focused introduction prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, script_type, agent_name)
        
        # Format prompt
        prompt_content = INTRODUCTION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Introduction Prompt: {prompt_content}")
        
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
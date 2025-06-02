# ===============================================================================
# STEP 11: FURTHER ASSISTANCE AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step11_further_assistance.py
"""
Further Assistance Agent - Enhanced with aging-aware script integration
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

FURTHER_ASSISTANCE_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                             
<context>
Client: {client_full_name} | Urgency: {urgency_level} | Aging Category: {aging_category} 
Verification Status: VERIFIED
user_id: {user_id}
</context>

<script>{formatted_script}</script>

<task>Check for additional concerns appropriate to account urgency.</task>

<scope_by_urgency>
Medium: Account questions, service issues, payment queries, general assistance
High: Payment concerns, service restoration, urgent matters only
Critical: Legal clarifications, immediate concerns only
</scope_by_urgency>

<time_management>
Medium: Allow time for questions
High: Keep focused on priorities  
Critical: Conclude efficiently
</time_management>

<response_style>
Under 15 words. Genuine concern. Time-appropriate for urgency.
</response_style>
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
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        return Command(update={"current_step": CallStep.FURTHER_ASSISTANCE.value}, goto="agent")

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.FURTHER_ASSISTANCE)
        formatted_script = script_template.format(**params) if script_template else "Anything else I can help with today?"
        params["formatted_script"] = formatted_script
        
        prompt_content = FURTHER_ASSISTANCE_PROMPT.format(**params)
        if verbose: print(f"Further Assistance Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="FurtherAssistanceAgent"
    )

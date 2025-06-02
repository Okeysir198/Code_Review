# ===============================================================================
# STEP 14: CANCELLATION AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step14_cancellation.py
"""
Cancellation Agent - Enhanced with aging-aware script integration
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


CANCELLATION_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today time: {current_date}
</role>
                                                       
<context>
Client: {client_full_name} | Cancellation Fee: {cancellation_fee} | Total: {total_balance}
Category: {aging_category} | Urgency: {urgency_level} 
Verification Status: VERIFIED
user_id: {user_id}
</context>

<script>{formatted_script}</script>

<task>Process cancellation with appropriate settlement requirements.</task>

<approach_by_category>
First Payment: Standard cancellation process and fees
Failed PTP: All commitments must be settled first
2-3 Months: All overdue amounts cleared before cancellation
Pre-Legal: All legal amounts settled to prevent court action
Legal: Legal proceedings continue until debt satisfied
</approach_by_category>

<settlement_requirements>
Fee: {cancellation_fee} | Outstanding: {total_balance} | Total due before cancellation
</settlement_requirements>

<response_style>
Under 20 words. Professional acceptance. Clear fee explanation.
</response_style>
"""

def create_cancellation_agent(
    model: BaseChatModel, client_data: Dict[str, Any], script_type: str,
    agent_name: str = "AI Agent", tools: Optional[List[BaseTool]] = None,
    verbose: bool = False, config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        return Command(
            update={
                "cancellation_requested": True,
                "current_step": CallStep.CANCELLATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.CANCELLATION)
        formatted_script = script_template.format(**params) if script_template else f"Cancellation fee {params['cancellation_fee']}. Total balance {params['total_balance']}."
        params["formatted_script"] = formatted_script
        
        prompt_content = CANCELLATION_PROMPT.format(**params)
        if verbose: print(f"Cancellation Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(model, dynamic_prompt, tools or [], pre_processing_node, 
                            CallCenterAgentState, verbose, config, "CancellationAgent")
# ===============================================================================
# STEP 10: REFERRALS AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step10_referrals.py
"""
Referrals Agent - Enhanced with aging-aware script integration
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

REFERRALS_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                    
<context>
Client name: {client_full_name} | Urgency: {urgency_level} | AgCategory: {aging_category}
Should Offer: {should_offer_referrals} 
Verification Status: VERIFIED
user_id: {user_id}
</context>

<script>{formatted_script}</script>

<task>
{{'Mention referral program if appropriate' if '{should_offer_referrals}' == 'True' else 'Skip referrals due to urgency'}}
</task>

<appropriateness>
Medium/Low: "Know anyone interested in Cartrack? 2 months free for referrals."
High: Brief mention only after payment focus
Critical: Skip entirely - focus on resolution
</appropriateness>

<response_style>
Under 15 words. Brief mention only. No pressure if not interested.
</response_style>
"""

def create_referrals_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        aging_context = ScriptManager.get_aging_context(script_type)
        should_offer = aging_context['urgency'] not in ['Very High', 'Critical']
        
        return Command(
            update={
                "should_offer_referrals": should_offer,
                "current_step": CallStep.REFERRALS.value
            }, 
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        params["should_offer_referrals"] = state.get("should_offer_referrals", True)
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.REFERRALS)
        formatted_script = script_template.format(**params) if script_template else ("Know anyone interested in Cartrack? 2 months free." if params["should_offer_referrals"] else "Focus on account resolution.")
        params["formatted_script"] = formatted_script
        
        prompt_content = REFERRALS_PROMPT.format(**params)
        if verbose: print(f"Referrals Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ReferralsAgent"
    )

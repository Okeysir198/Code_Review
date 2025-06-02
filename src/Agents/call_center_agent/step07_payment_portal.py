# src/Agents/call_center_agent/step07_payment_portal.py
"""
Payment Portal Agent - Enhanced with aging-aware script integration
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

PAYMENT_PORTAL_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                         
<context>                                                          
Client name: {client_full_name} | Outstanding amount: {outstanding_amount} | Total amount with fee: {amount_with_fee} | Verification Status: VERIFIED
Account Status: {account_status} | Aging Category: {aging_category} | Urgency: {urgency_level} 
Verification Status: VERIFIED
user_id: {user_id}
Banking information: {banking_details}
</context>

<script>{formatted_script}</script>

<task>Guide through payment portal. Stay connected until complete.</task>

<instructions_by_urgency>
Medium: "Sending payment link. Click, confirm {outstanding_amount}, choose method."
High: "Payment link sent. Complete immediately. Amount {outstanding_amount}."
Critical: "Emergency payment link. Pay {outstanding_amount} now or face consequences."
</instructions_by_urgency>

<response_style>
Under 15 words. Step-by-step. Stay connected.
</response_style>
"""

def create_payment_portal_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        return Command(update={"current_step": CallStep.PAYMENT_PORTAL.value}, goto="agent")

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.PAYMENT_PORTAL)
        formatted_script = script_template.format(**params) if script_template else f"Sending payment link for {params['outstanding_amount']}"
        params["formatted_script"] = formatted_script
        
        prompt_content = PAYMENT_PORTAL_PROMPT.format(**params)
        if verbose: print(f"Payment Portal Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])

    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="PaymentPortalAgent"
    )
   
# src/Agents/call_center_agent/step08_subscription_reminder.py
"""
Subscription Reminder Agent - Enhanced with aging-aware script integration
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


SUBSCRIPTION_REMINDER_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                                
<context>                                                          
Client name: {client_full_name} | Arrears: {outstanding_amount} | Subscription: {subscription_amount} | Verification Status: VERIFIED
Aging Category: {aging_category} | Urgency: {urgency_level} 
user_id: {user_id}
</context>
                                                                
<script>{formatted_script}</script>

<task>Clarify arrears vs ongoing subscription. Prevent confusion.</task>

<clarification_by_urgency>
Medium: "Today's {outstanding_amount} covers missed payment. Regular {subscription_amount} continues monthly."
High: "Today's {outstanding_amount} clears arrears. Monthly {subscription_amount} resumes."
Critical: "Today's {outstanding_amount} settles debt. Regular billing resumes after."
</clarification_by_urgency>

<response_style>
Under 15 words. Clear separation. Prevent double-payment confusion.
</response_style>
"""

def create_subscription_reminder_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        return Command(update={"current_step": CallStep.SUBSCRIPTION_REMINDER.value}, goto="agent")

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.SUBSCRIPTION_REMINDER)
        formatted_script = script_template.format(**params) if script_template else f"Today's {params['outstanding_amount']} covers arrears. Monthly {params['subscription_amount']} continues."
        params["formatted_script"] = formatted_script
        
        prompt_content = SUBSCRIPTION_REMINDER_PROMPT.format(**params)
        if verbose: print(f"Subscription Reminder Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="SubscriptionReminderAgent"
    )

# ===============================================================================
# STEP 13: ESCALATION AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step13_escalation.py
"""
Escalation Agent - Enhanced with aging-aware script integration
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

ESCALATION_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                     
<context>
Client: {client_full_name} | Aging Category: {aging_category} | Urgency: {urgency_level}
Department: {department} | Response: {response_time} 
Verification Status: VERIFIED
user_id: {user_id}
</context>

<script>{formatted_script}</script>

<task>Handle escalation with appropriate authority level and timeline.</task>

<escalation_by_category>
First Payment: Customer Service Supervisor, 24-48 hours
Failed PTP: Accounts Supervisor, 24 hours  
2-3 Months: Senior Collections Manager, 24 hours
Pre-Legal: Pre-Legal Manager, 12-24 hours
Legal: Legal Department, 12 hours
</escalation_by_category>

<approach>
Acknowledge concern → Create ticket → Set expectations → Professional handoff
</approach>

<response_style>
Under 20 words. Professional acknowledgment. Clear next steps.
</response_style>
"""

def create_escalation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        account_aging = client_data.get("account_aging", {})
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        # Department mapping by category
        escalation_mapping = {
            "First Missed Payment": ("Customer Service Supervisor", "24-48 hours"),
            "Failed Promise to Pay": ("Accounts Supervisor", "24 hours"),
            "2-3 Months Overdue": ("Senior Collections Manager", "24 hours"),
            "Pre-Legal 120+ Days": ("Pre-Legal Manager", "12-24 hours"),
            "Legal 150+ Days": ("Legal Department", "12 hours")
        }
        
        category = aging_context['category']
        department, response_time = escalation_mapping.get(category, ("Supervisor", "24-48 hours"))
        
        return Command(
            update={
                "department": department,
                "response_time": response_time,
                "escalation_requested": True,
                "current_step": CallStep.ESCALATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        params.update({
            "department": state.get("department", "Supervisor"),
            "response_time": state.get("response_time", "24-48 hours")
        })
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.ESCALATION)
        formatted_script = script_template.format(**params) if script_template else f"Escalating to {params['department']}. Response within {params['response_time']}."
        params["formatted_script"] = formatted_script
        
        prompt_content = ESCALATION_PROMPT.format(**params)
        if verbose: print(f"Escalation Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EscalationAgent"
    )

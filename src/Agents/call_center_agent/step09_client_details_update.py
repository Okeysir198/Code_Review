# src/Agents/call_center_agent/step09_client_details_update.py
"""
Client Details Update Agent - Enhanced with aging-aware script integration
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

CLIENT_DETAILS_UPDATE_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today time: {current_date}
</role>
                                                                
<context>
Client: {client_full_name} | Current Mobile: {current_mobile} | Current Email: {current_email}
Urgency: {urgency_level} | Category: {aging_category} 
Verification Status: VERIFIED
user_id: {user_id}
</context>

<script>{formatted_script}</script>

<task>Update contact details. Quick verification process.</task>

<justification_by_urgency>
Medium: "Verifying details for account notifications."
High: "Updating details for urgent communications."
Critical: "Required for legal correspondence."
</justification_by_urgency>

<process>
1. Confirm mobile number
2. Confirm email address  
3. Update complete
</process>

<response_style>
Under 15 words. Quick verification. Professional efficiency.
</response_style>
"""

def create_client_details_update_agent(
    model: BaseChatModel, client_data: Dict[str, Any], script_type: str,
    agent_name: str = "AI Agent", tools: Optional[List[BaseTool]] = None,
    verbose: bool = False, config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        return Command(update={"current_step": CallStep.CLIENT_DETAILS_UPDATE.value}, goto="agent")

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        params["current_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.CLIENT_DETAILS_UPDATE)
        formatted_script = script_template.format(**params) if script_template else "Verifying your contact details for notifications."
        params["formatted_script"] = formatted_script
        
        prompt_content = CLIENT_DETAILS_UPDATE_PROMPT.format(**params)
        if verbose: print(f"Client Details Update Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(model, dynamic_prompt, tools or [], pre_processing_node, 
                            CallCenterAgentState, verbose, config, "ClientDetailsUpdateAgent")
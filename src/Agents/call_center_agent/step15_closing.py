# src/Agents/call_center_agent/step15_closing.py
"""
Closing Agent - Self-contained with own prompt
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

CLOSING_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today time: {current_date}
</role>
                                                  
<context>
Client: {client_full_name} | Call Outcome: {call_outcome} | Outstanding amount: {outstanding_amount}
user_id: {user_id}
</context>

<script>{formatted_script}</script>

<task>End call professionally with clear outcome summary.</task>

<summary_by_outcome>
Payment Secured: "Perfect. Payment of {outstanding_amount} arranged."
Escalation: "Escalated with reference. You'll hear back soon."
Cancellation: "Cancellation logged. Settlement required first."
Incomplete: "Thanks for your time. Call us at 011 250 3000."
</summary_by_outcome>

<response_style>
MAXIMUM 20 words. Thank client. Clear outcome. Professional closure.
</response_style>
"""

def create_closing_agent(
    model: BaseChatModel, client_data: Dict[str, Any], script_type: str,
    agent_name: str = "AI Agent", tools: Optional[List[BaseTool]] = None,
    verbose: bool = False, config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Determine call outcome
        if state.get("payment_secured"):
            call_outcome = "payment_secured"
        elif state.get("cancellation_requested"):
            call_outcome = "cancelled"
        elif state.get("escalation_requested"):
            call_outcome = "escalated"
        else:
            call_outcome = "incomplete"
        
        return Command(
            update={
                "call_outcome": call_outcome,
                "is_call_ended": True,
                "current_step": CallStep.CLOSING.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        params["current_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        params["call_outcome"] = state.get("call_outcome", "incomplete")
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.CLOSING)
        formatted_script = script_template.format(**params) if script_template else f"Thank you {params['client_name']}. Call completed."
        params["formatted_script"] = formatted_script
        
        prompt_content = CLOSING_PROMPT.format(**params)
        if verbose: print(f"Closing Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(model, dynamic_prompt, tools or [], pre_processing_node, 
                            CallCenterAgentState, verbose, config, "ClosingAgent")
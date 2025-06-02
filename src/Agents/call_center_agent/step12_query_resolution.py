# ===============================================================================
# STEP 12: QUERY RESOLUTION AGENT - Enhanced with Call Scripts & 2-Step Verification
# ===============================================================================

# src/Agents/call_center_agent/step12_query_resolution.py
"""
Query Resolution Agent - Enhanced with call scripts and 2-step verification using basic_agent
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

QUERY_RESOLUTION_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                           
<context>
Target client: {client_full_name} | Outstanding amount: {outstanding_amount} | Question: "{last_client_question}"
Verification: Name={name_verification_status}, Details={details_verification_status} 
Return to current step: {return_to_step} | Urgency: {urgency_level} 
Verification Status: VERIFIED
user_id: {user_id}
</context>

<script>
{formatted_script}
</script>

<task>
Answer briefly then redirect based on verification stage.
</task>

<approach based on verification>
Name is not VERIFIED: Answer max 8 words + "Are you {client_full_name}?"
Details is not VERIFIED: Answer max 10 words + "Your ID number please?"
Fully Verified: Answer + redirect to payment focus
</approach based on verification>

<examples>
"How does Cartrack work?" 
- Name is not VERIFIED: "Vehicle tracking. Are you {client_full_name}?"
- Details is not VERIFIED: "Tracking system. Your ID number please?"
- Fully Verified: "Vehicle security. Can we arrange {outstanding_amount}?"
</examples>

<response_style>
CRITICAL: Keep under 15 words total. Answer + redirect to current step . No explanations.
</response_style>
"""

def create_query_resolution_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Get last client message
        messages = state.get("messages", [])
        last_client_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                last_client_message = msg.content.lower()
                break
        
        return Command(
            update={
                "last_client_question": last_client_message,
                "current_step": CallStep.QUERY_RESOLUTION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        params = prepare_parameters(client_data, state, agent_name)
        params.update({
            "last_client_question": state.get("last_client_question", ""),
            "return_to_step": state.get("return_to_step", "")
        })
        
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.QUERY_RESOLUTION)
        formatted_script = script_template.format(**params) if script_template else "Brief answer then redirect to verification/payment."
        params["formatted_script"] = formatted_script
        
        prompt_content = QUERY_RESOLUTION_PROMPT.format(**params)
        if verbose: print(f"Query Resolution Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="QueryResolutionAgent"
    )

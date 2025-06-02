# src/Agents/call_center_agent/step03_reason_for_call.py
"""
Reason for Call Agent - Lean version with concise responses
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

# Optimized prompt for concise reason delivery
REASON_FOR_CALL_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                          
<context>
Client name: {client_full_name} | Outstanding amount: {outstanding_amount} | Verification Status: VERIFIED
Account Status: {account_status} | Aging Category: {aging_category} | Urgency: {urgency_level} 
user_id: {user_id}
</context>

<script>
{formatted_script}
</script>

<task>
State account status and required payment. Match urgency to account severity.
</task>

<aging_approach>
First Payment: "We didn't receive your {outstanding_amount} subscription payment. Can we debit today?"
Failed PTP: "We didn't receive your {outstanding_amount} as arranged. Account overdue."
2-3 Months: "Account overdue 2+ months. {outstanding_amount} required immediately."
Pre-Legal: "Account 4+ months overdue, pre-legal. {outstanding_amount} required now."
Legal: "Account with attorneys. Arrears {outstanding_amount}. Do you acknowledge debt?"
</aging_approach>

<communication_strategy>
1. State status directly
2. Specify exact amount: {outstanding_amount}
3. Create appropriate urgency
4. Request immediate action
</communication_strategy>

<consequences>
{aging_consequences}
</consequences>

<response_style>
CRITICAL: Keep under 15 words. State amount clearly. Direct action request.
Examples:
✓ "Payment overdue {outstanding_amount}. Can we debit today?"
✓ "Account 2 months overdue. Need {outstanding_amount} immediately."
✗ Long explanations about billing cycles or service details
</response_style>
"""

def create_reason_for_call_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create reason for call agent with concise messaging"""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Minimal preprocessing"""
        return Command(
            update={"current_step": CallStep.REASON_FOR_CALL.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate concise reason for call prompt"""
        # Step 1: Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Step 2: Format script
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.REASON_FOR_CALL)
        formatted_script = script_template.format(**params) if script_template else f"Payment overdue {params['outstanding_amount']}. Action required."
        params["formatted_script"] = formatted_script
        
        # Step 3: Format prompt
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_consequences"] = aging_context['consequences']
        
        prompt_content = REASON_FOR_CALL_PROMPT.format(**params)
        
        if verbose: print(f"Reason for Call Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ReasonForCallAgent"
    )
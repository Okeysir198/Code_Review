# src/Agents/call_center_agent/step05_promise_to_pay.py
"""
Promise to Pay Agent - Lean version with concise payment securing
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
from src.Database.CartrackSQLDatabase import (
    get_client_banking_details, get_client_account_overview, 
    create_payment_arrangement, date_helper
)

# Optimized prompt for payment commitment
PROMISE_TO_PAY_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                                                                       
<context>
Client name: {client_full_name} | Outstanding amount: {outstanding_amount} | Verification Status: VERIFIED
Account Status: {account_status} | Aging Category: {aging_category} | Urgency: {urgency_level} 
user_id: {user_id}
Banking information: {banking_details}
</context>
                                                         
<script>
{formatted_script}
</script>

<task>
Secure payment arrangement. No exit without commitment.
</task>

<payment_hierarchy>
1. Immediate debit: "Can we debit {outstanding_amount} today?"
2. DebiCheck: "Setting up secure bank payment for {amount_with_fee}"
3. Payment portal: "Sending payment link for {outstanding_amount}"
</payment_hierarchy>

<urgency_approach>
Medium: Offer options, flexible timing
High: Limited options, today preferred  
Critical: Payment demanded now, no delay
</urgency_approach>

<no_exit_rule>
Must secure SOME arrangement. Urgency determines flexibility:
- Medium: Multiple options, flexible timing
- High: Immediate preferred, minimal alternatives
- Critical: Payment now, no alternatives
</no_exit_rule>

<persistence_by_urgency>
Medium: "What payment method works for you?"
High: "Which option can we do immediately?"
Critical: "Payment required now. Choose your method."
</persistence_by_urgency>

<response_style>
CRITICAL: Under 15 words. Assume they'll pay. Direct yes/no questions.
Examples:
✓ "Can we debit {outstanding_amount} today?"
✓ "Which works: bank debit or payment link?"
✗ Long explanations about payment benefits or options
</response_style>
"""

def create_promise_to_pay_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create promise to pay agent with concise payment securing"""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Check banking details availability"""
        
        return Command(
            update={
                "current_step": CallStep.PROMISE_TO_PAY.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate concise payment commitment prompt"""
        # Step 1: Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)

        
        # Step 2: Format script
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.PROMISE_TO_PAY)
        formatted_script = script_template.format(**params) if script_template else f"Can we debit {params['outstanding_amount']} today?"
        params["formatted_script"] = formatted_script
        
        # Step 3: Format prompt
        prompt_content = PROMISE_TO_PAY_PROMPT.format(**params)
        
        if verbose: print(f"Promise to Pay Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="PromiseToPayAgent"
    )
# src/Agents/call_center_agent/step06_debicheck_setup.py
"""
DebiCheck Setup Agent - Self-contained with own prompt
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import calculate_outstanding_amount, format_currency

from src.Database.CartrackSQLDatabase import get_client_debit_mandates, add_client_note

def get_debicheck_setup_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate DebiCheck setup specific prompt."""
    # Calculate amounts
    account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    total_with_fee = outstanding_float + 10
    amount_with_fee = format_currency(total_with_fee)
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Explain DebiCheck process and next steps. MAXIMUM 20 words per response.
</task>

<process_explanation>
1. "Your bank will send an authentication request"
2. "You'll receive this via your banking app or SMS"  
3. "You must approve this request to authorize payment"
4. "Total amount will be {amount_with_fee} including R10 processing fee"
</process_explanation>

<style>
- MAXIMUM 20 words per response
- Clear, step-by-step guidance
- Professional confidence
- Ensure client understands process
</style>"""

def create_debicheck_setup_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a DebiCheck setup agent."""
    
    agent_tools = [get_client_debit_mandates, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare DebiCheck setup context."""
        
        # Get outstanding amount and calculate total with fee
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        mandate_fee = 10.0
        total_amount = outstanding_amount + mandate_fee
        
        return Command(
            update={
                "amount_with_fee": format_currency(total_amount),
                "mandate_fee": mandate_fee,
                "outstanding_float": outstanding_amount,
                "process_explanation": "Your bank will send an authentication request",
                "current_step": CallStep.DEBICHECK_SETUP.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_debicheck_setup_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="DebiCheckSetupAgent"
    )
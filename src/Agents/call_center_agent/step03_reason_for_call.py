# ./src/Agents/call_center_agent/step03_reason_for_call.py
"""
Reason for Call Agent - Explains account status and outstanding amounts.
SIMPLIFIED: No query detection - router handles all routing decisions.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    get_client_account_aging,
    get_client_account_overview,
    add_client_note
)


def create_reason_for_call_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a reason for call agent for debt collection calls."""
    
    # Add relevant database tools
    agent_tools = [
        get_client_account_aging,
        get_client_account_overview,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to gather account information only."""
        
        # Extract account information from client data
        account_aging = client_data.get("account_aging", {})
        account_overview = client_data.get("account_overview", {})
        
        outstanding_balance = "R 0.00"
        if account_aging:
            balance = account_aging.get("xbalance", "0")
            try:
                outstanding_balance = f"R {float(balance):.2f}" if balance else "R 0.00"
            except (ValueError, TypeError):
                outstanding_balance = "R 0.00"
        
        account_status = account_overview.get("account_status", "Overdue") if account_overview else "Overdue"
        
        return Command(
            update={
                "outstanding_amount": outstanding_balance,
                "account_status": account_status
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for reason for call step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.REASON_FOR_CALL.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.REASON_FOR_CALL.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="ReasonForCallAgent"
    )
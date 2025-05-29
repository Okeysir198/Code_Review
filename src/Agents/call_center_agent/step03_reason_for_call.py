# ./src/Agents/call_center_agent/step03_reason_for_call.py
"""
Reason for Call Agent - Optimized with pre-processing only.
Explains account status and outstanding amounts directly.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters, calculate_outstanding_amount
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
    
    agent_tools = [
        get_client_account_aging,
        get_client_account_overview,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare account status information."""
        
        # Extract account information from client data
        account_aging = client_data.get("account_aging", {})
        account_overview = client_data.get("account_overview", {})
        
        # Calculate outstanding amount (overdue amount, not total balance)
        outstanding_amount = calculate_outstanding_amount(account_aging)
        outstanding_formatted = f"R {outstanding_amount:.2f}" if outstanding_amount > 0 else "R 0.00"
        
        # Determine account status
        account_status = account_overview.get("account_status", "Overdue") if account_overview else "Overdue"
        
        # Create urgency level based on amount and days overdue
        urgency_level = "standard"
        if outstanding_amount > 1000:
            urgency_level = "high"
        elif outstanding_amount > 500:
            urgency_level = "medium"
        
        # Prepare reason message components
        reason_components = {
            "status_statement": f"Your account is {account_status.lower()}",
            "amount_statement": f"{outstanding_formatted} is required",
            "urgency_statement": "Immediate payment needed" if urgency_level == "high" else "Payment needed today"
        }
        
        return Command(
            update={
                "outstanding_amount": outstanding_formatted,
                "account_status": account_status,
                "urgency_level": urgency_level,
                "reason_components": reason_components,
                "outstanding_float": outstanding_amount,
                "current_step": CallStep.REASON_FOR_CALL.value
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
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ReasonForCallAgent"
    )
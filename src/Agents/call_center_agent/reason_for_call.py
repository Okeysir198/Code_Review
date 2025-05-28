# ./src/Agents/call_center_agents/reason_for_call.py
"""
Reason for Call Agent - Explains account status and outstanding amounts.
"""
from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus


# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    get_client_account_aging,
    get_client_account_overview,
    get_client_billing_analysis
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
    """
    Create a reason for call agent for debt collection calls.
    
    Args:
        model: Language model to use
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled reason for call agent workflow
    """
    
    # Add relevant database tools
    agent_tools = [
        get_client_account_aging,
        get_client_account_overview,
        get_client_billing_analysis
    ]
    if tools:
        agent_tools.extend(tools)
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process to gather account information for explanation."""
        
        try:
            # Get account information
            account_aging = client_data['account_aging']
            account_overview = client_data['account_overview']
            
            # Extract key information
            outstanding_balance = "R 0.00"
            account_status = "Unknown"
            
            if account_aging:
                balance = account_aging[0].get("xbalance", "0")
                outstanding_balance = f"R {float(balance):.2f}" if balance else "R 0.00"
            
            if account_overview:
                account_status = account_overview.get("account_status", "Overdue")
            
            return {
                "outstanding_amount": outstanding_balance,
                "account_status": account_status,
                "call_info": {
                    "account_aging": account_aging[0] if account_aging else {},
                    "account_overview": account_overview or {}
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error gathering account info: {e}")
            
            return {
                "outstanding_amount": "the outstanding amount",
                "account_status": "overdue",
                "call_info": {}
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for reason for call step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.REASON_FOR_CALL.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.REASON_FOR_CALL.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="ReasonForCallAgent"
    )
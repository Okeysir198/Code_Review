# ./src/Agents/call_center_agent/step06_debicheck_setup.py
"""
DebiCheck Setup Agent - Optimized with pre-processing only.
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

from src.Database.CartrackSQLDatabase import get_client_debit_mandates, add_client_note


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
                "amount_with_fee": f"R {total_amount:.2f}",
                "mandate_fee": mandate_fee,
                "outstanding_float": outstanding_amount,
                "process_explanation": "Your bank will send an authentication request",
                "current_step": CallStep.DEBICHECK_SETUP.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.DEBICHECK_SETUP.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        prompt_content = get_step_prompt(CallStep.DEBICHECK_SETUP.value, parameters)
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

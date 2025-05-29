# ./src/Agents/call_center_agent/step06_debicheck_setup.py
"""
DebiCheck Setup Agent - Optimized with only pre-processing.
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
    get_client_debit_mandates,
    add_client_note
)


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
        """Prepare DebiCheck setup context."""
        
        try:
            # Get existing mandates
            existing_mandates = client_data.get('existing_mandates', [])
            
            # Check for active mandates
            active_mandates = []
            if existing_mandates:
                active_mandates = [
                    m for m in existing_mandates 
                    if m.get("debicheck_mandate_state") in ["Created", "Authenticated"]
                ]
            
            # Get outstanding amount
            account_aging = client_data.get("account_aging", {})
            outstanding_amount = calculate_outstanding_amount(account_aging)
            
            # Calculate amount with DebiCheck fee
            mandate_fee = 10.0
            total_amount = outstanding_amount + mandate_fee
            
            # Determine if new mandate needed
            needs_new_mandate = len(active_mandates) == 0
            
            # Get banking details
            banking_details = client_data.get('banking_details', {})
            mandate_ready = needs_new_mandate and banking_details and outstanding_amount > 0
            
            return Command(
                update={
                    "existing_mandates_count": len(existing_mandates) if existing_mandates else 0,
                    "active_mandates_count": len(active_mandates),
                    "needs_new_mandate": needs_new_mandate,
                    "mandate_ready": mandate_ready,
                    "amount_with_fee": f"R {total_amount:.2f}",
                    "mandate_fee": mandate_fee,
                    "outstanding_float": outstanding_amount
                },
                goto="agent"
            )
            
        except Exception as e:
            if verbose:
                print(f"Error in DebiCheck pre-processing: {e}")
            
            return Command(
                update={
                    "needs_new_mandate": True,
                    "mandate_ready": False,
                    "amount_with_fee": "R 10.00",
                    "mandate_fee": 10.0
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for DebiCheck setup step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.DEBICHECK_SETUP.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.DEBICHECK_SETUP.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        # NO post_processing_node - removed as per instructions
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="DebiCheckSetupAgent"
    )
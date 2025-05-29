# ./src/Agents/call_center_agent/step07_payment_portal.py
"""
Payment Portal Agent - Optimized with only pre-processing.
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
    generate_sms_payment_url,
    add_client_note
)


def create_payment_portal_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a payment portal agent."""
    
    agent_tools = [generate_sms_payment_url, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Generate payment URL and prepare portal guidance."""
        
        try:
            # Get outstanding amount
            account_aging = client_data.get("account_aging", {})
            outstanding_amount = calculate_outstanding_amount(account_aging)
            user_id = client_data.get('user_id')
            
            # Generate payment URL if amount is available
            payment_url = None
            reference_id = None
            url_generated = False
            
            if outstanding_amount > 0 and user_id:
                try:
                    url_result = generate_sms_payment_url.invoke({
                        "user_id": int(user_id),
                        "amount": outstanding_amount,
                        "optional_reference": f"PTP_{user_id}"
                    })
                    
                    if url_result.get("success"):
                        payment_url = url_result.get("payment_url")
                        reference_id = url_result.get("reference_id")
                        url_generated = True
                        
                except Exception as url_error:
                    if verbose:
                        print(f"Error generating payment URL: {url_error}")
            
            return Command(
                update={
                    "payment_url": payment_url,
                    "reference_id": reference_id,
                    "url_generated": url_generated,
                    "payment_amount": outstanding_amount,
                    "portal_payment_complete": False
                },
                goto="agent"
            )
            
        except Exception as e:
            if verbose:
                print(f"Error in payment portal pre-processing: {e}")
            
            return Command(
                update={
                    "payment_url": None,
                    "url_generated": False,
                    "portal_payment_complete": False
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for payment portal step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.PAYMENT_PORTAL.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.PAYMENT_PORTAL.value, parameters)
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
        name="PaymentPortalAgent"
    )
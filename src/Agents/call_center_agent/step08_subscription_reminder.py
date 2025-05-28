# ./src/Agents/call_center_agent/step08_subscription_reminder.py
"""
Subscription Reminder Agent - Clarifies arrears vs ongoing subscription.
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
    get_client_subscription_amount,
    add_client_note
)


def create_subscription_reminder_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a subscription reminder agent for debt collection calls."""
    
    agent_tools = [
        get_client_subscription_amount,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare subscription information only."""
        
        # Get subscription information
        subscription = client_data.get("subscription", {})
        subscription_amount = subscription.get("subscription_amount", "199.00")
        
        try:
            subscription_str = f"R {float(subscription_amount):.2f}"
        except (ValueError, TypeError):
            subscription_str = "R 199.00"
        
        return Command(
            update={
                "subscription_amount": subscription_str,
                "subscription_date": "5th of each month"
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for subscription reminder step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.SUBSCRIPTION_REMINDER.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.SUBSCRIPTION_REMINDER.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="SubscriptionReminderAgent"
    )
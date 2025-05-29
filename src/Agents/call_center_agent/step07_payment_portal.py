# src/Agents/call_center_agent/step07_payment_portal.py
"""
Payment Portal Agent - Self-contained with own prompt
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

from src.Database.CartrackSQLDatabase import generate_sms_payment_url, add_client_note

def get_payment_portal_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate payment portal specific prompt."""
    # Calculate outstanding amount
    account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Guide client through payment portal. MAXIMUM 20 words per response.
</task>

<guidance>
"I'll send you a secure payment link. You can pay {outstanding_amount} while we're on the call"
</guidance>

<instructions>
1. "Click the link I'm sending"
2. "Confirm the amount as {outstanding_amount}"
3. "Choose your payment method"
4. "I'll stay on the line to help"
</instructions>

<style>
- MAXIMUM 20 words per response
- Stay connected during process
- Immediate problem solving
- Professional guidance
</style>"""

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
        """Pre-process to generate payment URL and prepare portal guidance."""
        
        # Get outstanding amount
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        user_id = client_data.get('user_id')
        
        # Generate payment URL if possible
        payment_url = None
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
                    url_generated = True
                    
            except Exception as e:
                if verbose:
                    print(f"Error generating payment URL: {e}")
        
        return Command(
            update={
                "payment_url": payment_url,
                "url_generated": url_generated,
                "portal_guidance": "I'll send you a secure payment link",
                "current_step": CallStep.PAYMENT_PORTAL.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_payment_portal_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="PaymentPortalAgent"
    )
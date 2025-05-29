# ===============================================================================
# STEP 14: CANCELLATION AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step14_cancellation.py
"""
Cancellation Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
import uuid
from datetime import datetime

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import calculate_outstanding_amount, format_currency
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

from src.Database.CartrackSQLDatabase import (
    add_client_note,
    get_client_account_aging
)

def get_cancellation_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware cancellation prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Get cancellation details
    cancellation_fee = state.get("cancellation_fee", "R 0.00")
    total_balance = state.get("total_balance", "R 0.00")
    ticket_number = state.get("ticket_number", "CAN12345")
    
    # Build aging-specific cancellation approaches
    cancellation_approaches_by_category = {
        "First Missed Payment": {
            "tone": "understanding and professional",
            "approach": "I understand you want to cancel. Let me explain the process and fees involved.",
            "emphasis": "cancellation process and standard fees"
        },
        "Failed Promise to Pay": {
            "tone": "professional acknowledgment",
            "approach": "I understand you want to cancel. All outstanding amounts must be settled first.",
            "emphasis": "settlement of commitments before cancellation"
        },
        "2-3 Months Overdue": {
            "tone": "professional and direct",
            "approach": "I understand you want to cancel. All overdue amounts must be cleared before cancellation.",
            "emphasis": "full settlement required before cancellation"
        },
        "Pre-Legal 120+ Days": {
            "tone": "serious and factual",
            "approach": "I understand you want to cancel. All legal amounts must be settled to prevent court action.",
            "emphasis": "legal settlement requirements before cancellation"
        },
        "Legal 150+ Days": {
            "tone": "formal legal authority",
            "approach": "Cancellation request noted. However, legal proceedings continue until debt is satisfied.",
            "emphasis": "legal obligations continue regardless of cancellation"
        }
    }
    
    category = aging_context['category']
    cancellation_approach = cancellation_approaches_by_category.get(category, cancellation_approaches_by_category["First Missed Payment"])
    
    # Base prompt
    base_prompt = f"""<role>
You are {agent_name}, a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<context>
- Cancellation Fee: {cancellation_fee}
- Total Balance: {total_balance}
- Ticket Number: {ticket_number}
- Aging Category: {category}
- Urgency Level: {aging_context['urgency']}
</context>

<task>
Process cancellation professionally using aging-appropriate approach and requirements.
</task>

<aging_specific_approach>
**Initial Response**: "{cancellation_approach['approach']}"
**Tone**: "{cancellation_approach['tone']}"
**Emphasis**: "{cancellation_approach['emphasis']}"
</aging_specific_approach>

<cancellation_process>
1. Acknowledge request with appropriate tone
2. Explain fees and total balance requirements
3. Emphasize settlement requirements for aging category
4. Create ticket reference: {ticket_number}
5. Set appropriate expectations
</cancellation_process>

<follow_up_messaging>
"I'll escalate this to our cancellations team. Reference: {ticket_number}"
"They'll contact you regarding {cancellation_approach['emphasis']}"
</follow_up_messaging>

<urgency_context>
{aging_context['approach']}
</urgency_context>

<style>
- {aging_context['tone']} with {cancellation_approach['tone']}
- Professional acceptance of request
- Clear fee and settlement explanation
- No retention attempts (focus on process)
- Authority level appropriate to account status
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.CANCELLATION,
        client_data=client_data,
        state=state
    )

def create_cancellation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a cancellation agent."""
    
    agent_tools = [add_client_note, get_client_account_aging] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to calculate cancellation fees and create ticket."""
        
        # Calculate cancellation fee and total balance
        account_aging = client_data.get("account_aging", {})
        outstanding_balance = calculate_outstanding_amount(account_aging)
        
        # Standard cancellation fee (adjust based on business rules)
        cancellation_fee = 0.0  # Set based on your business rules
        total_balance = outstanding_balance + cancellation_fee
        
        # Generate cancellation ticket
        ticket_number = f"CAN{datetime.now().strftime('%Y%m%d%H%M')}{str(uuid.uuid4())[:4].upper()}"
        
        # Add cancellation note
        user_id = client_data.get("user_id")
        if user_id:
            try:
                add_client_note.invoke({
                    "user_id": user_id,
                    "note_text": f"Cancellation requested. Ticket: {ticket_number}, Total balance: {format_currency(total_balance)}"
                })
            except Exception as e:
                if verbose:
                    print(f"Error adding cancellation note: {e}")
        
        return Command(
            update={
                "cancellation_fee": format_currency(cancellation_fee),
                "total_balance": format_currency(total_balance),
                "ticket_number": ticket_number,
                "cancellation_requested": True,
                "current_step": CallStep.CANCELLATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_cancellation_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(f"Prompt: {prompt_content}")
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="CancellationAgent"
    )
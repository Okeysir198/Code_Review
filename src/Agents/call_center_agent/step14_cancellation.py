# src/Agents/call_center_agent/step14_cancellation.py
"""
Enhanced Cancellation Agent - Professional cancellation handling with settlement requirements
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters, detect_client_mood_from_messages
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
from src.Database.CartrackSQLDatabase import add_client_note

import logging
logger = logging.getLogger(__name__)

CANCELLATION_PROMPT = """
You're {agent_name} from Cartrack processing cancellation request from {client_name} with {outstanding_amount} outstanding.

TODAY: {current_date}
OBJECTIVE: Handle cancellation professionally with clear settlement requirements.

ACCOUNT STATUS:
- Outstanding: {outstanding_amount}
- Total Balance: {total_balance}
- Cancellation Fee: {cancellation_fee}
- Client Mood: {detected_mood}

SETTLEMENT REQUIREMENTS BY AGING:
- First Payment: Outstanding + cancellation fee = {total_settlement}
- Failed PTP: All commitments + cancellation fee must be settled
- 2-3 Months: All overdue amounts + fee before cancellation
- Pre-Legal: All legal amounts + fee to prevent court action
- Legal: Legal proceedings continue until debt satisfied

TOOL USAGE:
- add_client_note: Document cancellation request and requirements
- Parameters: user_id={user_id}, note_text="Cancellation requested - Settlement required: {total_settlement}"

CANCELLATION PROCESS:
1. Acknowledge cancellation request professionally
2. Explain settlement requirements clearly
3. Calculate total amount due
4. Offer payment arrangements for settlement
5. Escalate to cancellations team

RESPONSE BY MOOD:
- Frustrated: "I understand you want to cancel. Let me explain the process and requirements."
- Angry: "I'll help with cancellation. First, we need to settle the outstanding balance of {total_settlement}."
- Cooperative: "I can arrange cancellation. The total settlement amount is {total_settlement}."

SETTLEMENT EXPLANATION:
"To complete cancellation, the total settlement is {total_settlement}. This includes {outstanding_amount} outstanding plus {cancellation_fee} cancellation fee. Would you like to arrange payment today?"

NEXT STEPS:
- If willing to settle: Arrange payment then process cancellation
- If cannot settle: Explain account remains active until settled
- Either way: Escalate to cancellations team for formal processing

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep response under 25 words. Professional acceptance, clear fees, settlement focus.
"""

def create_cancellation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced cancellation agent with settlement processing"""
    
    # Tools for cancellation documentation
    cancellation_tools = [add_client_note]
    
    def _calculate_settlement_amount(client_data: Dict[str, Any]) -> Dict[str, str]:
        """Calculate total settlement amount including fees"""
        
        account_aging = client_data.get("account_aging", {})
        account_overview = client_data.get("account_overview", {})
        
        try:
            # Outstanding amount (overdue)
            total = float(account_aging.get("xbalance", 0))
            current = float(account_aging.get("x0", 0))
            outstanding = max(total - current, 0.0)
            
            # Cancellation fee
            cancellation_fee = float(account_overview.get("cancellation_fee", 250))
            
            # Total settlement
            total_settlement = outstanding + cancellation_fee
            
            return {
                "outstanding_amount": f"R {outstanding:.2f}",
                "cancellation_fee": f"R {cancellation_fee:.2f}",
                "total_settlement": f"R {total_settlement:.2f}",
                "total_balance": f"R {total:.2f}"
            }
            
        except (ValueError, TypeError):
            return {
                "outstanding_amount": "R 0.00",
                "cancellation_fee": "R 250.00",
                "total_settlement": "R 250.00",
                "total_balance": "R 0.00"
            }
    
    def _extract_cancellation_reason(messages: List) -> str:
        """Extract reason for cancellation"""
        recent_messages = []
        for message in reversed(messages[-3:]):
            if hasattr(message, 'type') and message.type == 'human':
                recent_messages.append(message.content.lower())
                if len(recent_messages) >= 2:
                    break
        
        combined_text = " ".join(recent_messages)
        
        if "not using" in combined_text or "don't use" in combined_text:
            return "Not using service"
        elif "too expensive" in combined_text or "can't afford" in combined_text:
            return "Cost concerns"
        elif "service" in combined_text and "bad" in combined_text:
            return "Service issues"
        elif "moving" in combined_text or "relocating" in combined_text:
            return "Relocation"
        elif "sold" in combined_text and "car" in combined_text:
            return "Vehicle sold"
        else:
            return "General cancellation"
    
    def _check_cancellation_completion(messages: List) -> bool:
        """Check if cancellation process was explained"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "settlement", "cancellation fee", "total", "escalate",
                    "cancellations team", "arrange payment"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Process cancellation request and calculate settlement"""
        
        messages = state.get("messages", [])
        
        # Calculate settlement amounts
        settlement_amounts = _calculate_settlement_amount(client_data)
        
        # Extract cancellation reason
        cancellation_reason = _extract_cancellation_reason(messages)
        
        if len(messages) >= 2:
            cancellation_completed = _check_cancellation_completion(messages)
            
            if cancellation_completed:
                logger.info("Cancellation process explained - moving to closing")
                return Command(
                    update={
                        "cancellation_requested": True,
                        "current_step": CallStep.CLOSING.value,
                        "call_outcome": "cancellation_requested"
                    },
                    goto="__end__"
                )
        
        # Continue with cancellation process
        update_data = {
            "cancellation_requested": True,
            "cancellation_reason": cancellation_reason,
            "current_step": CallStep.CANCELLATION.value
        }
        update_data.update(settlement_amounts)
        
        return Command(
            update=update_data,
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate settlement-focused cancellation prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Add cancellation-specific context
        messages = state.get("messages", [])
        params["detected_mood"] = detect_client_mood_from_messages(messages)
        
        # Add settlement amounts from state
        params["cancellation_fee"] = state.get("cancellation_fee", "R 250.00")
        params["total_settlement"] = state.get("total_settlement", "R 250.00")
        params["total_balance"] = state.get("total_balance", "R 0.00")
        
        # Override outstanding_amount if we have it in state
        if state.get("outstanding_amount"):
            params["outstanding_amount"] = state.get("outstanding_amount")
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format prompt
        prompt_content = CANCELLATION_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Cancellation Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=cancellation_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedCancellationAgent"
    )
# ./src/Agents/call_center_agent/step04_negotiation.py
"""
Negotiation Agent - Handles objections and explains consequences/benefits.
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
    get_client_payment_history,
    get_client_failed_payments,
    add_client_note
)


def create_negotiation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a negotiation agent for debt collection calls."""
    
    # Add relevant database tools
    agent_tools = [
        get_client_payment_history,
        get_client_failed_payments,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to analyze client behavior and prepare negotiation strategy."""
        
        # Analyze client behavior for negotiation strategy
        payment_history = client_data.get("payment_history", [])
        failed_payments = client_data.get("failed_payments", [])
        
        # Determine payment reliability
        payment_reliability = "unknown"
        if payment_history:
            total_payments = len(payment_history)
            failed_count = len(failed_payments) if failed_payments else 0
            
            if total_payments > 0:
                success_rate = (total_payments - failed_count) / total_payments
                if success_rate >= 0.8:
                    payment_reliability = "high"
                elif success_rate >= 0.5:
                    payment_reliability = "medium"
                else:
                    payment_reliability = "low"
        
        # Detect objections from recent conversation
        recent_messages = state.get("messages", [])[-2:] if state.get("messages") else []
        objection_keywords = {
            "no_money": ["no money", "can't afford", "broke", "tight", "financial"],
            "dispute_amount": ["wrong", "incorrect", "dispute", "not right", "don't owe"],
            "already_paid": ["already paid", "paid already", "made payment"],
            "will_pay_later": ["later", "next week", "soon", "when I get paid"]
        }
        
        detected_objections = []
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                content_lower = msg.content.lower()
                for objection, keywords in objection_keywords.items():
                    if any(keyword in content_lower for keyword in keywords):
                        detected_objections.append(objection)
        
        # Determine emotional state and payment willingness
        emotional_state = "defensive" if detected_objections else "neutral"
        payment_willingness = "low" if "no_money" in detected_objections else "medium"
        
        return Command(
            update={
                "payment_reliability": payment_reliability,
                "detected_objections": detected_objections,
                "emotional_state": emotional_state,
                "payment_willingness": payment_willingness
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for negotiation step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.NEGOTIATION.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.NEGOTIATION.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="NegotiationAgent"
    )
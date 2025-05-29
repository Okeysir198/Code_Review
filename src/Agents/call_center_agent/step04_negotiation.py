# ./src/Agents/call_center_agent/step04_negotiation.py
"""
Negotiation Agent - Optimized with pre-processing only.
Handles objections and explains consequences with emotional intelligence.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters, ConversationAnalyzer
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
    """Create a negotiation agent with emotional intelligence and objection handling."""
    
    agent_tools = [
        get_client_payment_history,
        get_client_failed_payments,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to analyze conversation and prepare negotiation strategy."""
        
        # Get conversation messages for analysis
        conversation_messages = state.get("messages", [])
        
        # Analyze client behavior using conversation intelligence
        emotional_state = ConversationAnalyzer.detect_emotional_state(conversation_messages)
        real_objections = ConversationAnalyzer.detect_real_objections(conversation_messages)
        
        # Get outstanding amount for payment analysis
        account_aging = client_data.get("account_aging", {})
        try:
            outstanding_amount = float(account_aging.get("xbalance", 0))
        except (ValueError, TypeError):
            outstanding_amount = 0.0
        
        payment_conversation = ConversationAnalyzer.analyze_payment_conversation(
            conversation_messages, outstanding_amount
        )
        
        # Analyze payment history for reliability assessment
        payment_history = client_data.get("payment_history", [])
        failed_payments = client_data.get("failed_payments", [])
        
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
        
        # Determine negotiation approach based on conversation intelligence
        negotiation_approach = "standard"
        if emotional_state == "angry":
            negotiation_approach = "de_escalation"
        elif emotional_state == "worried":
            negotiation_approach = "reassuring"
        elif emotional_state == "cooperative":
            negotiation_approach = "direct"
        elif payment_conversation.get("payment_commitment") == "willing":
            negotiation_approach = "immediate_closure"
        elif payment_conversation.get("payment_commitment") == "unwilling":
            negotiation_approach = "flexible_options"
        
        # Determine consequences to emphasize based on objections
        consequences_focus = "service_disruption"  # default
        if "no_money" in real_objections or "cant_afford" in real_objections:
            consequences_focus = "gradual_escalation"
        elif "dispute_amount" in real_objections:
            consequences_focus = "verification_protection"
        elif "already_paid" in real_objections:
            consequences_focus = "immediate_resolution"
        
        # Determine payment willingness
        payment_willingness = payment_conversation.get("payment_commitment", "unknown")
        if payment_willingness == "unknown":
            if "no_money" in real_objections or "cant_afford" in real_objections:
                payment_willingness = "unwilling"
            elif "will_pay_later" in real_objections:
                payment_willingness = "considering"
        
        return Command(
            update={
                "emotional_state": emotional_state,
                "detected_objections": real_objections,
                "payment_willingness": payment_willingness,
                "negotiation_approach": negotiation_approach,
                "payment_reliability": payment_reliability,
                "consequences_focus": consequences_focus,
                "conversation_intelligence": {
                    "emotional_state": emotional_state,
                    "real_objections": real_objections,
                    "payment_conversation": payment_conversation
                },
                "current_step": CallStep.NEGOTIATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for negotiation step with conversation intelligence."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.NEGOTIATION.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.NEGOTIATION.value, parameters)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
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
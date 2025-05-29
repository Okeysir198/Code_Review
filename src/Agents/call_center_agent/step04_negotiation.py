# ./src/Agents/call_center_agent/step04_negotiation.py
"""
Negotiation Agent - Collaborative approach with emotional intelligence.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters, ConversationAnalyzer
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.call_scripts import ScriptType

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
    """Create a negotiation agent with collaborative approach and emotional intelligence."""
    
    # Add relevant database tools
    agent_tools = [
        get_client_payment_history,
        get_client_failed_payments,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Command[Literal["agent"]]:
        """Enhanced pre-processing with conversation analysis and emotional intelligence."""
        
        # Get conversation messages for analysis
        conversation_messages = state.get("messages", [])
        
        # Analyze client behavior using conversation intelligence
        payment_history = client_data.get("payment_history", [])
        failed_payments = client_data.get("failed_payments", [])
        
        # Traditional payment reliability analysis
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
        
        # NEW: Conversation intelligence analysis
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
        
        # Determine payment willingness based on conversation
        payment_willingness = payment_conversation.get("payment_commitment", "unknown")
        if payment_willingness == "unknown":
            # Fallback to objection analysis
            if "no_money" in real_objections or "cant_afford" in real_objections:
                payment_willingness = "unwilling"
            elif "will_pay_later" in real_objections:
                payment_willingness = "considering"
        
        # Update state with conversation intelligence
        return Command(
            update={
                "payment_reliability": payment_reliability,
                "detected_objections": real_objections,
                "emotional_state": emotional_state,
                "payment_willingness": payment_willingness,
                "negotiation_approach": negotiation_approach,
                "conversation_intelligence": {
                    "emotional_state": emotional_state,
                    "real_objections": real_objections,
                    "payment_conversation": payment_conversation
                }
            },
            goto="agent"
        )

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Track negotiation outcomes and emotional state changes."""
        
        try:
            # Analyze the agent's response to understand negotiation outcome
            recent_messages = state.get("messages", [])[-2:] if state.get("messages") else []
            
            negotiation_outcome = "ongoing"
            objection_resolved = False
            client_response_type = "neutral"
            
            # Look for client response patterns
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    content = msg.content.lower()
                    
                    # Client messages
                    if msg.type == "human":
                        # Positive responses
                        if any(phrase in content for phrase in ["okay", "fine", "yes", "i understand"]):
                            client_response_type = "acceptance"
                            objection_resolved = True
                        # New objections
                        elif any(phrase in content for phrase in ["but", "however", "can't", "won't"]):
                            client_response_type = "objection"
                        # Questions or confusion
                        elif any(phrase in content for phrase in ["what", "how", "why", "when"]):
                            client_response_type = "questioning"
                    
                    # Agent messages - check for negotiation tactics used
                    elif msg.type == "ai":
                        if any(phrase in content for phrase in ["understand", "i get it", "that's fair"]):
                            negotiation_outcome = "empathetic_approach"
                        elif any(phrase in content for phrase in ["however", "but", "need to"]):
                            negotiation_outcome = "firm_redirect"
                        elif any(phrase in content for phrase in ["what about", "could you", "would you"]):
                            negotiation_outcome = "alternative_offered"
            
            # Add note about negotiation progress
            user_id = client_data.get("user_id")
            if user_id and objection_resolved:
                try:
                    add_client_note.invoke({
                        "user_id": user_id,
                        "note_text": f"Negotiation progress: {client_response_type} response, approach: {state.get('negotiation_approach', 'standard')}"
                    })
                except Exception as e:
                    if verbose:
                        print(f"Error adding negotiation note: {e}")
            
            return {
                "negotiation_outcome": negotiation_outcome,
                "objection_resolved": objection_resolved,
                "client_response_type": client_response_type,
                "negotiation_in_progress": not objection_resolved
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in negotiation post-processing: {e}")
            
            return {
                "negotiation_outcome": "ongoing",
                "objection_resolved": False,
                "client_response_type": "unknown"
            }

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
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        post_processing_node=post_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="NegotiationAgent"
    )
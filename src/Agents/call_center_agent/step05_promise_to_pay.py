# ./src/Agents/call_center_agent/step05_promise_to_pay.py
"""
Promise to Pay Agent - Progressive payment options with emotional intelligence.
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
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters, ConversationAnalyzer, PaymentFlexibilityAnalyzer
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, PaymentMethod

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    get_client_banking_details,
    create_debicheck_payment,
    create_payment_arrangement,
    create_mandate,
    generate_sms_payment_url,
    get_payment_arrangement_types,
    add_client_note
)


def create_promise_to_pay_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a promise to pay agent with progressive payment options."""
    
    # Add relevant database tools
    agent_tools = [
        get_client_banking_details,
        create_debicheck_payment,
        create_payment_arrangement,
        create_mandate,
        generate_sms_payment_url,
        get_payment_arrangement_types,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Command[Literal["agent"]]:
        """Enhanced pre-processing with payment flexibility and conversation analysis."""
        
        try:
            # Get client banking details
            banking_details = client_data.get('banking_details', {})
            has_banking_details = bool(banking_details and len(banking_details) > 0)
            
            # Get payment arrangement types
            payment_types = get_payment_arrangement_types.invoke("all")
            
            # Get outstanding amount for calculations
            account_aging = client_data.get("account_aging", {})
            try:
                outstanding_amount = float(account_aging.get("xbalance", 0))
            except (ValueError, TypeError):
                outstanding_amount = 0.0
            
            # NEW: Enhanced conversation analysis
            conversation_messages = state.get("messages", [])
            
            # Analyze payment conversation intelligence
            payment_conversation = ConversationAnalyzer.analyze_payment_conversation(
                conversation_messages, outstanding_amount
            )
            
            # Assess payment flexibility based on client data
            payment_flexibility = PaymentFlexibilityAnalyzer.assess_payment_capacity(client_data)
            
            # Determine current negotiation stage
            offers_made = state.get("payment_offers_made", [])
            current_offer_level = len(offers_made) + 1
            
            # Select appropriate payment option based on conversation and capacity
            available_options = payment_flexibility["payment_options"]
            
            # Override option selection based on conversation intelligence
            if payment_conversation.get("mentioned_amount"):
                # Client mentioned a specific amount - be flexible
                mentioned = payment_conversation["mentioned_amount"]
                if mentioned >= outstanding_amount * 0.3:  # At least 30%
                    current_option = {
                        "type": "client_suggested",
                        "amount": mentioned,
                        "description": f"Client suggested amount: R {mentioned:.2f}",
                        "priority": 1
                    }
                else:
                    # Too low - counter with minimum
                    current_option = {
                        "type": "minimum_counter",
                        "amount": outstanding_amount * 0.3,
                        "description": f"Minimum payment: R {outstanding_amount * 0.3:.2f}",
                        "priority": 1
                    }
            else:
                # Use progressive options based on attempt number
                current_option = available_options[min(current_offer_level - 1, len(available_options) - 1)]
            
            # Determine negotiation strategy based on emotional state and payment willingness
            emotional_state = ConversationAnalyzer.detect_emotional_state(conversation_messages)
            payment_commitment = payment_conversation.get("payment_commitment", "unknown")
            
            if emotional_state == "cooperative" and payment_commitment == "willing":
                negotiation_strategy = "direct_closure"
            elif emotional_state in ["worried", "embarrassed"] or payment_commitment == "unwilling":
                negotiation_strategy = "supportive_flexible"
            elif emotional_state in ["angry", "frustrated"]:
                negotiation_strategy = "de_escalate_then_solve"
            else:
                negotiation_strategy = "progressive_standard"
            
            # Recommended payment approach
            if payment_commitment == "willing" and has_banking_details:
                recommended_approach = "immediate_debit"
            elif payment_commitment == "willing":
                recommended_approach = "payment_portal"
            elif payment_conversation.get("payment_method_preference"):
                recommended_approach = payment_conversation["payment_method_preference"]
            else:
                recommended_approach = "ask_preference"

            return Command(
                update={
                    "has_banking_details": has_banking_details,
                    "payment_conversation": payment_conversation,
                    "payment_flexibility": payment_flexibility,
                    "current_payment_option": current_option,
                    "negotiation_stage": current_offer_level,
                    "negotiation_strategy": negotiation_strategy,
                    "recommended_approach": recommended_approach,
                    "available_payment_types": payment_types or [],
                    "outstanding_float": outstanding_amount
                },
                goto="agent"
            )
            
        except Exception as e:
            if verbose:
                print(f"Error in PTP pre-processing: {e}")
            
            return Command(
                update={
                    "has_banking_details": False,
                    "recommended_approach": "ask_immediate_debit",
                    "negotiation_strategy": "progressive_standard",
                    "outstanding_float": 0.0
                },
                goto="agent"
            )

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Enhanced post-processing to create actual payment arrangements."""
        
        try:
            # Check if payment arrangement was agreed upon in the conversation
            recent_messages = state.get("messages", [])[-3:] if state.get("messages") else []
            
            arrangement_created = False
            arrangement_details = {}
            client_agreed = False
            agent_confirmed = False
            
            # Analyze conversation for payment agreement
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    content = msg.content.lower()
                    
                    # Client agreement indicators
                    if (msg.type == "human" and 
                        any(phrase in content for phrase in [
                            "yes", "okay", "fine", "sure", "go ahead", "do it", "agreed"
                        ]) and
                        not any(phrase in content for phrase in [
                            "no", "can't", "won't", "refuse", "decline"
                        ])):
                        client_agreed = True
                    
                    # Agent confirmation of arrangement
                    if (msg.type == "ai" and 
                        any(phrase in content for phrase in [
                            "perfect", "excellent", "great", "i'm setting up", 
                            "arrangement created", "payment scheduled"
                        ])):
                        agent_confirmed = True
                        
                        # Extract arrangement details from agent message
                        import re
                        
                        # Extract amount
                        amount_match = re.search(r'r\s*(\d+(?:\.\d{2})?)', content)
                        amount = float(amount_match.group(1)) if amount_match else None
                        
                        # Extract method
                        if "debit" in content or "bank" in content:
                            method = PaymentMethod.DEBICHECK.value
                        elif "portal" in content or "link" in content:
                            method = PaymentMethod.PAYMENT_PORTAL.value
                        else:
                            method = PaymentMethod.IMMEDIATE_DEBIT.value
                        
                        arrangement_details = {
                            "payment_method": method,
                            "amount": amount,
                            "date": "today",
                            "arrangement_created": True
                        }
            
            # Create arrangement if both agreed and confirmed
            if client_agreed and agent_confirmed and arrangement_details:
                arrangement_created = True
                
                # Add detailed note about payment arrangement
                user_id = client_data.get("user_id")
                if user_id:
                    note_text = (f"Payment arrangement: {arrangement_details.get('payment_method', 'Unknown')} "
                               f"for R{arrangement_details.get('amount', 'Unknown')} "
                               f"- Strategy: {state.get('negotiation_strategy', 'standard')}")
                    
                    add_client_note.invoke({
                        "user_id": user_id,
                        "note_text": note_text
                    })
            
            # Track payment offer made
            current_option = state.get("current_payment_option", {})
            if current_option:
                payment_offer = {
                    "amount": current_option.get("amount", 0),
                    "type": current_option.get("type", "unknown"),
                    "accepted": arrangement_created,
                    "negotiation_stage": state.get("negotiation_stage", 1)
                }
                
                # Add to offers made list
                offers_made = state.get("payment_offers_made", [])
                offers_made.append(payment_offer)
            else:
                offers_made = state.get("payment_offers_made", [])
            
            return {
                "payment_secured": arrangement_created,
                "payment_arrangement": arrangement_details,
                "client_agreed": client_agreed,
                "agent_confirmed": agent_confirmed,
                "payment_offers_made": offers_made
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in PTP post-processing: {e}")
            
            return {
                "payment_secured": False,
                "payment_arrangement": {},
                "client_agreed": False,
                "payment_offers_made": state.get("payment_offers_made", [])
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for promise to pay step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.PROMISE_TO_PAY.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.PROMISE_TO_PAY.value, parameters)
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
        name="PromiseToPayAgent"
    )
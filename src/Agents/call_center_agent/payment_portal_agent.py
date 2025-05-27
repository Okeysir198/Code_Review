# ./src/Agents/call_center_agent/payment_portal.py
"""
Payment Portal Agent - Guides clients through online payment process.
"""
from typing import Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    generate_sms_payment_url,
    create_payment_arrangement_payment_portal,
    add_client_note
)


def create_payment_portal_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False
) -> CompiledGraph:
    """
    Create a payment portal agent for debt collection calls.
    
    Args:
        model: Language model to use
        client_data: client information
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled payment portal agent workflow
    """
    
    # Add relevant database tools
    agent_tools = [
        generate_sms_payment_url,
        create_payment_arrangement_payment_portal,
        add_client_note
    ]
    if tools:
        agent_tools.extend(tools)
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process to generate payment URL and prepare portal guidance."""
        
        try:
            # Get payment amount from state
            payment_arrangement = getattr(state, 'payment_arrangement', {})
            amount = payment_arrangement.get('amount', 0)
            user_id = client_data['user_id']
            # Generate payment URL if amount is available
            payment_url = None
            reference_id = None
            url_generated = False
            
            if amount > 0:
                try:
                    url_result = generate_sms_payment_url.invoke({
                        "user_id": int(user_id),
                        "amount": amount,
                        "optional_reference": f"PTP_{user_id}"
                    })
                    
                    if url_result.get("success"):
                        payment_url = url_result.get("payment_url")
                        reference_id = url_result.get("reference_id")
                        url_generated = True
                        
                except Exception as url_error:
                    if verbose:
                        print(f"Error generating payment URL: {url_error}")
            
            return {
                "payment_url": payment_url,
                "reference_id": reference_id,
                "url_generated": url_generated,
                "payment_amount": amount,
                "portal_payment_complete": False,
                "call_info": {
                    "payment_url_status": "generated" if url_generated else "failed",
                    "amount": amount
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in payment portal pre-processing: {e}")
            
            return {
                "payment_url": None,
                "reference_id": None,
                "url_generated": False,
                "payment_amount": 0,
                "portal_payment_complete": False,
                "call_info": {"payment_url_status": "error"}
            }

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Post-process to track payment completion and create arrangement record."""
        
        try:
            # Check if client confirmed payment completion
            recent_messages = state['messages'][-3:] if len(state['messages']) >= 3 else state['messages']
            user_id = client_data['user_id']
            payment_completed = False
            payment_confirmed = False
            
            # Look for payment completion indicators
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    content = msg.content.lower()
                    
                    # Client confirmation of payment
                    if (msg.type == "human" and 
                        any(phrase in content for phrase in [
                            "paid", "payment done", "completed", "finished", "successful"
                        ])):
                        payment_confirmed = True
                    
                    # Agent confirmation of payment
                    elif (msg.type == "ai" and 
                          any(phrase in content for phrase in [
                              "payment completed", "payment successful", "received confirmation"
                          ])):
                        payment_completed = True
            
            # Create payment arrangement record if payment completed
            arrangement_id = None
            if payment_completed or payment_confirmed:
                try:
                    # Get payment details from state
                    amount = state.get('payment_amount', 0)
                    reference_id = state.get('reference_id')
                    
                    if amount > 0:
                        # Create payment arrangement record
                        arrangement_result = create_payment_arrangement_payment_portal.invoke({
                            "user_id": int(user_id),
                            "payment_type_id": 4,  # OZOW/Portal payment
                            "payment_date": "2024-01-01",  # Should be actual date
                            "amount": amount,
                            "online_payment_reference_id": reference_id
                        })
                        
                        if arrangement_result.get("success"):
                            arrangement_id = arrangement_result.get("arrangement_id")
                            
                            # Add note about payment completion
                            add_client_note.invoke({
                                "user_id": user_id,
                                "note_text": f"Payment portal payment completed. Amount: R{amount}, Reference: {reference_id}"
                            })
                        
                except Exception as arrangement_error:
                    if verbose:
                        print(f"Error creating payment arrangement: {arrangement_error}")
            
            return {
                "payment_confirmed": payment_confirmed,
                "portal_payment_complete": payment_completed or payment_confirmed,
                "arrangement_id": arrangement_id,
                "call_info": {
                    "portal_payment_status": "completed" if (payment_completed or payment_confirmed) else "pending"
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in payment portal post-processing: {e}")
            
            return {
                "payment_confirmed": False,
                "portal_payment_complete": False,
                "arrangement_id": None,
                "call_info": {"portal_payment_status": "error"}
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for payment portal step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.PAYMENT_PORTAL.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.PAYMENT_PORTAL.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        post_processing_node=post_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="PaymentPortalAgent"
    )
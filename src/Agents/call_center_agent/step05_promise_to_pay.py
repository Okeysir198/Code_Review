# ./src/Agents/call_center_agent/promise_to_pay.py
"""
Promise to Pay Agent - Secures payment arrangements and creates commitments.
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
    verbose: bool = False
) -> CompiledGraph:
    """
    Create a promise to pay agent for debt collection calls.
    
    Args:
        model: Language model to use
        client_data: client information
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled promise to pay agent workflow
    """
    
    # Add relevant database tools
    agent_tools = [
        get_client_banking_details,
        create_debicheck_payment,
        create_payment_arrangement,
        create_mandate,
        generate_sms_payment_url,
        get_payment_arrangement_types,
        add_client_note
    ]
    if tools:
        agent_tools.extend(tools)
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process to prepare payment options and analyze client response."""
        
        try:
            # Get client banking details
            banking_details = client_data['banking_details']
            has_banking_details = bool(banking_details and len(banking_details) > 0)
            
            # Get payment arrangement types
            payment_types = get_payment_arrangement_types.invoke("all")
            
            # Analyze recent messages for payment commitment
            recent_messages = state['messages'][-5:] if len(state['messages']) >= 5 else state['messages']
            
            payment_commitment = None
            payment_amount = None
            payment_date = None
            payment_method_preference = None
            
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    content = msg.content.lower()
                    
                    # Look for payment commitments
                    if any(phrase in content for phrase in ["can pay", "will pay", "able to pay", "yes"]):
                        payment_commitment = "willing"
                    elif any(phrase in content for phrase in ["can't pay", "cannot pay", "no money", "refuse"]):
                        payment_commitment = "unwilling"
                    
                    # Look for amounts (R100, 100, hundred)
                    import re
                    amount_patterns = [
                        r'r\s*(\d+)',  # R100
                        r'(\d+)\s*rand',  # 100 rand
                        r'(\d+)\s*r',  # 100R
                        r'(\d{2,4})'  # Just numbers 100-9999
                    ]
                    
                    for pattern in amount_patterns:
                        match = re.search(pattern, content)
                        if match:
                            try:
                                payment_amount = float(match.group(1))
                                break
                            except:
                                continue
                    
                    # Look for dates
                    date_keywords = ["friday", "monday", "tuesday", "wednesday", "thursday", "saturday", "sunday", 
                                   "tomorrow", "today", "next week", "end of month", "payday"]
                    for keyword in date_keywords:
                        if keyword in content:
                            payment_date = keyword
                            break
                    
                    # Look for payment method preferences
                    if "debit" in content or "bank" in content:
                        payment_method_preference = "debicheck"
                    elif "portal" in content or "online" in content or "link" in content:
                        payment_method_preference = "payment_portal"
            
            # Determine recommended approach
            if payment_commitment == "willing" and has_banking_details:
                recommended_approach = "immediate_debit"
            elif payment_commitment == "willing":
                recommended_approach = "payment_portal"
            elif payment_commitment == "unwilling":
                recommended_approach = "partial_payment"
            else:
                recommended_approach = "ask_immediate_debit"
            
            return {
                "has_banking_details": has_banking_details,
                "payment_commitment": payment_commitment or "unknown",
                "payment_amount": payment_amount,
                "payment_date": payment_date,
                "payment_method_preference": payment_method_preference,
                "recommended_approach": recommended_approach,
                "available_payment_types": payment_types or [],
                "call_info": {
                    "banking_details_count": len(banking_details) if banking_details else 0
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in PTP pre-processing: {e}")
            
            return {
                "has_banking_details": False,
                "payment_commitment": "unknown",
                "payment_amount": None,
                "payment_date": None,
                "payment_method_preference": None,
                "recommended_approach": "ask_immediate_debit",
                "available_payment_types": [],
                "call_info": {}
            }

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Post-process to create actual payment arrangements if commitment secured."""
        
        try:
            # Check if payment arrangement was agreed upon in the conversation
            recent_messages = state['messages'][-3:] if len(state['messages']) >= 3 else state['messages']
            
            arrangement_created = False
            arrangement_details = {}
            
            # Look for confirmation of payment arrangement
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    content = msg.content.lower()
                    
                    # If agent confirmed a payment arrangement
                    if (msg.type == "ai" and 
                        any(phrase in content for phrase in [
                            "arrangement created", "payment scheduled", "debit order set up",
                            "payment link sent", "arrangement confirmed"
                        ])):
                        
                        # Extract arrangement details from agent message
                        import re
                        
                        # Extract amount
                        amount_match = re.search(r'r\s*(\d+(?:\.\d{2})?)', content)
                        amount = float(amount_match.group(1)) if amount_match else None
                        
                        # Extract method
                        if "debit" in content:
                            method = PaymentMethod.DEBICHECK.value
                        elif "portal" in content or "link" in content:
                            method = PaymentMethod.PAYMENT_PORTAL.value
                        else:
                            method = PaymentMethod.IMMEDIATE_DEBIT.value
                        
                        arrangement_details = {
                            "payment_method": method,
                            "amount": amount,
                            "arrangement_created": True
                        }
                        arrangement_created = True
                        break
            
            # Add note about payment arrangement
            if arrangement_created:
                user_id = client_data['user_id']
                note_text = f"Payment arrangement created: {arrangement_details.get('payment_method', 'Unknown')} for R{arrangement_details.get('amount', 'Unknown')}"
                add_client_note.invoke({
                    "user_id": user_id,
                    "note_text": note_text
                })
            
            return {
                "payment_secured": arrangement_created,
                "payment_arrangement": arrangement_details,
                "call_info": {
                    "arrangement_status": "created" if arrangement_created else "pending"
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in PTP post-processing: {e}")
            
            return {
                "payment_secured": False,
                "payment_arrangement": {},
                "call_info": {"arrangement_status": "failed"}
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for promise to pay step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.PROMISE_TO_PAY.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.PROMISE_TO_PAY.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        post_processing_node=post_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="PromiseToPayAgent"
    )
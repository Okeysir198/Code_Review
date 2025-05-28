# ./src/Agents/call_center_agent/debicheck_setup.py
"""
DebiCheck Setup Agent - Explains DebiCheck process and creates mandates.
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
    get_client_debit_mandates,
    create_mandate,
    get_client_banking_details,
    add_client_note
)


def create_debicheck_setup_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False
) -> CompiledGraph:
    """
    Create a DebiCheck setup agent for debt collection calls.
    
    Args:
        model: Language model to use
        client_data: client information
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled DebiCheck setup agent workflow
    """
    
    # Add relevant database tools
    agent_tools = [
        get_client_debit_mandates,
        create_mandate,
        get_client_banking_details,
        add_client_note
    ]
    if tools:
        agent_tools.extend(tools)
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process to check existing mandates and prepare DebiCheck setup."""
        
        try:
            # Get existing debit mandates
            existing_mandates = client_data['existing_mandates']
            
            # Get banking details
            banking_details = client_data['banking_details']
            
            # Check for active mandates
            active_mandates = []
            if existing_mandates:
                active_mandates = [
                    m for m in existing_mandates 
                    if m.get("debicheck_mandate_state") in ["Created", "Authenticated"]
                ]
            
            # Determine if new mandate needed
            needs_new_mandate = len(active_mandates) == 0
            
            # Get payment arrangement details from state
            payment_arrangement = getattr(state, 'payment_arrangement', {})
            amount = payment_arrangement.get('amount', 0)
            
            # Calculate amount with fee
            mandate_fee = 10.0
            total_amount = amount + mandate_fee if amount else mandate_fee
            
            # Prepare mandate creation if needed
            mandate_ready = needs_new_mandate and banking_details and amount > 0
            
            return {
                "existing_mandates_count": len(existing_mandates) if existing_mandates else 0,
                "active_mandates_count": len(active_mandates),
                "needs_new_mandate": needs_new_mandate,
                "mandate_ready": mandate_ready,
                "amount_with_fee": f"R {total_amount:.2f}",
                "mandate_fee": mandate_fee,
                "debicheck_setup_complete": False,
                "call_info": {
                    "banking_details_available": bool(banking_details),
                    "payment_amount": amount
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in DebiCheck pre-processing: {e}")
            
            return {
                "existing_mandates_count": 0,
                "active_mandates_count": 0,
                "needs_new_mandate": True,
                "mandate_ready": False,
                "amount_with_fee": "R 10.00",
                "mandate_fee": 10.0,
                "debicheck_setup_complete": False,
                "call_info": {}
            }

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Post-process to create mandate if client agreed to DebiCheck."""
        
        try:
            # Check if client agreed to DebiCheck from conversation
            recent_messages = state['messages'][-3:] if len(state['messages']) >= 3 else state['messages']
            
            client_agreed = False
            mandate_created = False
            mandate_id = None
            user_id = client_data['user_id']
            
            # Look for client agreement
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    content = msg.content.lower()
                    
                    # Client agreement indicators
                    if (msg.type == "human" and 
                        any(phrase in content for phrase in [
                            "yes", "okay", "ok", "agree", "approve", "go ahead", "fine"
                        ]) and
                        not any(phrase in content for phrase in [
                            "no", "don't", "refuse", "decline"
                        ])):
                        client_agreed = True
                        break
            
            # Create mandate if client agreed and we have the necessary info
            if client_agreed and state.get('mandate_ready', False):
                try:
                    # Get payment amount from state
                    payment_arrangement = getattr(state, 'payment_arrangement', {})
                    amount = payment_arrangement.get('amount', 0)
                    
                    if amount > 0:
                        # Create the mandate
                        mandate_result = create_mandate.invoke({
                            "user_id": int(user_id),
                            "service": "PTP",
                            "amount": amount,
                            "collection_date": None,  # For recurring mandate
                            "authentication_code": None
                        })
                        
                        if mandate_result.get("success"):
                            mandate_created = True
                            mandate_id = mandate_result.get("mandate_id")
                            
                            # Add note about mandate creation
                            add_client_note.invoke({
                                "user_id": user_id,
                                "note_text": f"DebiCheck mandate created. ID: {mandate_id}, Amount: R{amount}"
                            })
                            
                except Exception as mandate_error:
                    if verbose:
                        print(f"Error creating mandate: {mandate_error}")
            
            return {
                "client_agreed_debicheck": client_agreed,
                "mandate_created": mandate_created,
                "mandate_id": mandate_id,
                "debicheck_setup_complete": mandate_created,
                "call_info": {
                    "mandate_creation_status": "success" if mandate_created else "pending"
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in DebiCheck post-processing: {e}")
            
            return {
                "client_agreed_debicheck": False,
                "mandate_created": False,
                "mandate_id": None,
                "debicheck_setup_complete": False,
                "call_info": {"mandate_creation_status": "failed"}
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for DebiCheck setup step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.DEBICHECK_SETUP.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.DEBICHECK_SETUP.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        post_processing_node=post_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="DebiCheckSetupAgent"
    )
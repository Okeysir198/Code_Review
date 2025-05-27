# ./src/Agents/call_center_agent/negotiation.py
"""
Negotiation Agent - Handles objections and explains consequences/benefits.
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
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.call_scripts import ScriptType

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    get_client_contracts,
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
    verbose: bool = False
) -> CompiledGraph:
    """
    Create a negotiation agent for debt collection calls.
    
    Args:
        model: Language model to use
        client_data: client information
        script_type: Script type (e.g., "ratio_1_inflow")
        agent_name: Name of the agent
        tools: Optional tools for the agent
        verbose: Enable verbose logging
        
    Returns:
        Compiled negotiation agent workflow
    """
    
    # Add relevant database tools
    agent_tools = [
        get_client_contracts,
        get_client_payment_history,
        get_client_failed_payments,
        add_client_note
    ]
    if tools:
        agent_tools.extend(tools)
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Pre-process to analyze client behavior and prepare negotiation strategy."""
        
        try:
            # Get client payment behavior data
            payment_history = client_data['payment_history']
            failed_payments = client_data['failed_payments']
            contracts = client_data['contracts']
            
            # Analyze client behavior for negotiation strategy
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
            
            # Determine service status
            active_contracts = [c for c in contracts if c.get("contract_state") == "Active"] if contracts else []
            service_status = "active" if active_contracts else "cancelled"
            
            # Detect objections from conversation
            recent_messages = state['messages'][-3:] if len(state['messages']) >= 3 else state['messages']
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
            
            return {
                "payment_reliability": payment_reliability,
                "service_status": service_status,
                "detected_objections": detected_objections,
                "payment_willingness": "low" if "no_money" in detected_objections else "medium",
                "emotional_state": "defensive" if detected_objections else "neutral",
                "call_info": {
                    "payment_history_count": len(payment_history) if payment_history else 0,
                    "failed_payments_count": len(failed_payments) if failed_payments else 0,
                    "active_contracts": len(active_contracts)
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in negotiation pre-processing: {e}")
            
            return {
                "payment_reliability": "unknown",
                "service_status": "unknown",
                "detected_objections": [],
                "payment_willingness": "unknown",
                "emotional_state": "neutral",
                "call_info": {}
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for negotiation step."""
        # Build parameters using real client data
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.NEGOTIATION.value,
            state=state.to_dict(),
            script_type=script_type,
            agent_name=agent_name
        )
        
        # Generate step-specific prompt
        prompt_content = get_step_prompt(CallStep.NEGOTIATION.value, parameters)
        
        return SystemMessage(content=prompt_content)
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="NegotiationAgent"
    )
# ./src/Agents/call_center_agent/step05_promise_to_pay.py
"""
Promise to Pay Agent - Optimized with only pre-processing.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters, ConversationAnalyzer, calculate_outstanding_amount
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    get_client_banking_details,
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
    
    agent_tools = [get_client_banking_details, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Analyze conversation and prepare payment context."""
        
        try:
            # Get banking details availability
            banking_details = client_data.get('banking_details', {})
            has_banking_details = bool(banking_details and len(banking_details) > 0)
            
            # Get outstanding amount for calculations
            account_aging = client_data.get("account_aging", {})
            outstanding_amount = calculate_outstanding_amount(account_aging)
            
            # Analyze conversation for payment intelligence
            conversation_messages = state.get("messages", [])
            emotional_state = ConversationAnalyzer.detect_emotional_state(conversation_messages)
            objections_raised = ConversationAnalyzer.detect_objections(conversation_messages)
            
            # Analyze payment conversation for mentioned amounts
            payment_willingness = "unknown"
            mentioned_amount = None
            
            for msg in conversation_messages[-5:]:
                if hasattr(msg, 'type') and msg.type == "human":
                    content = msg.content.lower()
                    
                    # Detect willingness
                    if any(phrase in content for phrase in ["can pay", "will pay", "yes", "okay"]):
                        payment_willingness = "willing"
                    elif any(phrase in content for phrase in ["can't pay", "no money", "refuse"]):
                        payment_willingness = "unwilling"
                    
                    # Extract mentioned amounts
                    import re
                    amount_match = re.search(r'r\s*(\d+)', content)
                    if amount_match:
                        try:
                            amount = float(amount_match.group(1))
                            if 50 <= amount <= outstanding_amount * 2:
                                mentioned_amount = amount
                        except:
                            pass
            
            # Determine negotiation strategy
            if emotional_state == "cooperative" and payment_willingness == "willing":
                negotiation_strategy = "direct_closure"
                recommended_approach = "immediate_debit"
            elif emotional_state in ["worried", "frustrated"]:
                negotiation_strategy = "supportive_flexible"
                recommended_approach = "payment_portal"
            else:
                negotiation_strategy = "progressive_standard"
                recommended_approach = "immediate_debit" if has_banking_details else "payment_portal"
            
            # Calculate payment options
            payment_options = [
                {"type": "full_payment", "amount": outstanding_amount, "priority": 1},
                {"type": "partial_80", "amount": outstanding_amount * 0.8, "priority": 2},
                {"type": "partial_50", "amount": outstanding_amount * 0.5, "priority": 3}
            ]
            
            # Select current option based on conversation
            if mentioned_amount and mentioned_amount >= outstanding_amount * 0.3:
                current_option = {"amount": mentioned_amount, "type": "client_suggested"}
            else:
                current_option = payment_options[0]  # Start with full payment
            
            return Command(
                update={
                    "has_banking_details": has_banking_details,
                    "outstanding_float": outstanding_amount,
                    "payment_willingness": payment_willingness,
                    "emotional_state": emotional_state,
                    "objections_raised": objections_raised,
                    "negotiation_strategy": negotiation_strategy,
                    "recommended_approach": recommended_approach,
                    "current_payment_option": current_option,
                    "payment_options": payment_options,
                    "mentioned_amount": mentioned_amount
                },
                goto="agent"
            )
            
        except Exception as e:
            if verbose:
                print(f"Error in PTP pre-processing: {e}")
            
            return Command(
                update={
                    "has_banking_details": False,
                    "recommended_approach": "immediate_debit",
                    "negotiation_strategy": "progressive_standard",
                    "outstanding_float": 0.0
                },
                goto="agent"
            )

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
        # NO post_processing_node - removed as per instructions
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="PromiseToPayAgent"
    )
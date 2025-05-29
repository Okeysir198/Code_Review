# ./src/Agents/call_center_agent/step12_query_resolution.py
"""
Query Resolution Agent - Brief answers with smart redirect to payment goal.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters

# Import relevant database tools
from src.Database.CartrackSQLDatabase import add_client_note


def create_query_resolution_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a query resolution agent with brief answers and smart redirects."""
    
    agent_tools = [add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Command[Literal["agent"]]:
        """Analyze query and determine appropriate redirect strategy."""
        
        try:
            # Get the return step (where we should redirect to)
            return_to_step = state.get("return_to_step", CallStep.CLOSING.value)
            
            # Analyze the client's question to categorize it
            conversation_messages = state.get("messages", [])
            last_client_message = ""
            
            if conversation_messages:
                for msg in reversed(conversation_messages):
                    if hasattr(msg, 'type') and msg.type == "human":
                        last_client_message = msg.content.lower()
                        break
                    elif isinstance(msg, dict) and msg.get("role") in ["user", "human"]:
                        last_client_message = msg.get("content", "").lower()
                        break
            
            # Categorize the query type
            query_type = "general"
            if any(word in last_client_message for word in ["how", "what", "why", "when", "where"]):
                if any(word in last_client_message for word in ["cartrack", "tracking", "work", "service"]):
                    query_type = "service_question"
                elif any(word in last_client_message for word in ["payment", "pay", "bank", "debit"]):
                    query_type = "payment_question"
                elif any(word in last_client_message for word in ["balance", "owe", "amount", "cost"]):
                    query_type = "account_question"
                else:
                    query_type = "general_question"
            
            # Check verification status to determine redirect approach
            name_verified = state.get("name_verification_status") == VerificationStatus.VERIFIED.value
            details_verified = state.get("details_verification_status") == VerificationStatus.VERIFIED.value
            is_fully_verified = name_verified and details_verified
            
            # Determine redirect strategy
            if not name_verified:
                redirect_strategy = "verify_name"
                redirect_target = CallStep.NAME_VERIFICATION.value
            elif not details_verified:
                redirect_strategy = "verify_details"
                redirect_target = CallStep.DETAILS_VERIFICATION.value
            elif return_to_step in [CallStep.PROMISE_TO_PAY.value, CallStep.NEGOTIATION.value]:
                redirect_strategy = "secure_payment"
                redirect_target = return_to_step
            else:
                redirect_strategy = "continue_call"
                redirect_target = return_to_step
            
            return Command(
                update={
                    "query_type": query_type,
                    "redirect_strategy": redirect_strategy,
                    "redirect_target": redirect_target,
                    "is_fully_verified": is_fully_verified,
                    "last_client_question": last_client_message
                },
                goto="agent"
            )
            
        except Exception as e:
            if verbose:
                print(f"Error in query resolution pre-processing: {e}")
            
            return Command(
                update={
                    "query_type": "general",
                    "redirect_strategy": "continue_call",
                    "redirect_target": state.get("return_to_step", CallStep.CLOSING.value)
                },
                goto="agent"
            )

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Track query resolution and redirect success."""
        
        try:
            # Add note about query handled
            user_id = client_data.get("user_id")
            query_type = state.get("query_type", "unknown")
            redirect_strategy = state.get("redirect_strategy", "unknown")
            
            if user_id:
                try:
                    add_client_note.invoke({
                        "user_id": user_id,
                        "note_text": f"Query resolved: {query_type}, redirect: {redirect_strategy}"
                    })
                except Exception as e:
                    if verbose:
                        print(f"Error adding query resolution note: {e}")
            
            return {
                "query_resolved": True,
                "redirect_completed": True,
                "successful_pivot": True
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in query resolution post-processing: {e}")
            
            return {
                "query_resolved": False,
                "redirect_completed": False
            }

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for query resolution with smart redirect."""
        
        # Extract key information
        query_type = state.get("query_type", "general")
        redirect_strategy = state.get("redirect_strategy", "continue_call")
        is_fully_verified = state.get("is_fully_verified", False)
        outstanding_amount = state.get("outstanding_amount", "the amount")
        client_name = client_data.get("profile", {}).get("client_info", {}).get("client_full_name", "Client")
        
        # Build custom prompt based on verification status and query type
        if redirect_strategy == "verify_name":
            custom_prompt = f"""<role>You're {agent_name} from Cartrack.</role>

<task>Answer their question briefly (under 10 words) then redirect to name verification.</task>

<examples>
Q: "Who are you?" 
A: "I'm {agent_name} from Cartrack. Are you {client_name}?"

Q: "What's this about?"
A: "About your Cartrack account. First, are you {client_name}?"

Q: "What company?"
A: "Cartrack vehicle tracking. Is this {client_name}?"
</examples>

<pattern>Brief answer + "Are you {client_name}?"</pattern>
<style>Maximum 15 words total. Sound natural and friendly.</style>"""

        elif redirect_strategy == "verify_details":
            field_to_verify = state.get("field_to_verify", "ID number")
            custom_prompt = f"""<role>You're {agent_name} from Cartrack.</role>

<task>Answer their question briefly (under 10 words) then redirect to details verification.</task>

<examples>
Q: "Who are you?"
A: "I'm {agent_name} from Cartrack. What's your {field_to_verify}?"

Q: "What's this about?"
A: "Your Cartrack account. Please confirm your {field_to_verify}."

Q: "What company?"
A: "Cartrack vehicle tracking. Your {field_to_verify} please?"
</examples>

<pattern>Brief answer + "Your {field_to_verify} please?"</pattern>
<style>Maximum 15 words total. Sound professional but friendly.</style>"""

        elif redirect_strategy == "secure_payment":
            custom_prompt = f"""<role>You're {agent_name} from Cartrack.</role>

<task>Answer their question briefly (under 10 words) then redirect to payment.</task>

<examples>
Q: "Who are you?"
A: "I'm {agent_name} from Cartrack. About that {outstanding_amount} though..."

Q: "What happens if I don't pay?"
A: "Services stop working. Can we arrange {outstanding_amount} today?"

Q: "How does Cartrack work?"
A: "Vehicle tracking and security. Let's sort this {outstanding_amount} payment."

Q: "Why wasn't my payment taken?"
A: "Bank declined it. Can we try a different method for {outstanding_amount}?"
</examples>

<pattern>Brief answer + redirect to payment</pattern>
<style>Maximum 15 words total. Stay focused on payment goal.</style>"""

        else:
            # Default: continue with call flow
            custom_prompt = f"""<role>You're {agent_name} from Cartrack.</role>

<task>Answer their question briefly (under 10 words) then continue the call.</task>

<examples>
Q: "How does this work?"
A: "Vehicle tracking and security. Is there anything else I can help with?"

Q: "What's my balance?"
A: "It's {outstanding_amount}. Anything else about your account?"

Q: "When is this due?"
A: "It's overdue now. Any other questions?"
</examples>

<pattern>Brief answer + "Anything else?"</pattern>
<style>Maximum 15 words total. Sound helpful and professional.</style>"""
        
        return [SystemMessage(content=custom_prompt)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        post_processing_node=post_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="QueryResolutionAgent"
    )
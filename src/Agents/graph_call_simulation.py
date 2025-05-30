# simulation_graph.py
"""
Lean Debt Collection Call Simulation Graph - FIXED
"""
import logging
from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.types import Command
from langgraph.graph.graph import CompiledGraph

# Import existing components
from src.Agents.graph_debtor_simulator import create_debtor_simulator, swap_roles
from src.Agents.graph_call_center_agent import create_call_center_agent

logger = logging.getLogger(__name__)


class SimulationState(MessagesState):
    """Simulation state that includes agent state persistence and tool tracking."""
    turn_count: int = 0
    max_turns: int = 20
    ended: bool = False
    
    # Agent state persistence
    agent_current_step: str = "introduction"
    name_verification_status: str = "INSUFFICIENT_INFO"
    details_verification_status: str = "INSUFFICIENT_INFO"
    name_verification_attempts: int = 0
    details_verification_attempts: int = 0
    matched_fields: list = []
    payment_secured: bool = False
    outstanding_amount: str = "R 0.00"
    is_call_ended: bool = False
    escalation_requested: bool = False
    cancellation_requested: bool = False
    
    # Tool call tracking
    tools_used: list = []
    last_tool_results: dict = {}


def create_call_simulation(
    agent_llm: BaseChatModel,
    debtor_llm: BaseChatModel,
    client_data: dict,
    debtor_personality: str = "cooperative",
    max_turns: int = 20,
    config: dict = None
) -> CompiledGraph:
    """Create lean call simulation between agent and debtor."""
    
    # Create agents once
    agent = create_call_center_agent(agent_llm, client_data, config=config)
    debtor = create_debtor_simulator(debtor_llm, client_data, debtor_personality)
    
    def agent_turn(state: SimulationState) -> Command[Literal["debtor_turn", "__end__"]]:
        """Agent speaks."""
        turn_count = state.get("turn_count", 0)
        state_max_turns = state.get("max_turns", max_turns)
        ended = state.get("ended", False)
        
        if turn_count >= state_max_turns or ended:
            return Command(update={"ended": True}, goto="__end__")
        
        # Build persistent agent state from simulation state
        agent_state = {
            "messages": state["messages"],
            "current_step": state.get("agent_current_step", "introduction"),
            "name_verification_status": state.get("name_verification_status", "INSUFFICIENT_INFO"),
            "details_verification_status": state.get("details_verification_status", "INSUFFICIENT_INFO"),
            "name_verification_attempts": state.get("name_verification_attempts", 0),
            "details_verification_attempts": state.get("details_verification_attempts", 0),
            "matched_fields": state.get("matched_fields", []),
            "payment_secured": state.get("payment_secured", False),
            "outstanding_amount": state.get("outstanding_amount", "R 0.00"),
            "is_call_ended": state.get("is_call_ended", False),
            "escalation_requested": state.get("escalation_requested", False),
            "cancellation_requested": state.get("cancellation_requested", False)
        }
        
        # Run agent with persistent state
        try:
            result = agent.invoke(agent_state)
            
            # Get ALL new messages from the agent (including tool calls and results)
            current_msg_count = len(state["messages"])
            all_new_messages = []
            tools_used = state.get("tools_used", [])
            last_tool_results = {}
            
            if "messages" in result and len(result["messages"]) > current_msg_count:
                all_new_messages = result["messages"][current_msg_count:]
                
                # Track tool calls and results from all messages
                for msg in all_new_messages:
                    # Track tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_info = {
                                "tool_name": tool_call.get("name", "unknown"),
                                "tool_args": tool_call.get("args", {}),
                                "tool_id": tool_call.get("id", ""),
                                "turn": turn_count + 1
                            }
                            tools_used.append(tool_info)
                    
                    # Track tool results
                    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                        last_tool_results[msg.tool_call_id] = {
                            "content": msg.content,
                            "tool_call_id": msg.tool_call_id
                        }
            
            # Find the final AI response (the one that should go to the debtor)
            final_ai_message = None
            for msg in reversed(all_new_messages):
                if hasattr(msg, 'type') and msg.type == "ai" and not getattr(msg, 'tool_calls', None):
                    final_ai_message = msg
                    break
            
            # If no final AI message found, use the last message
            if not final_ai_message and all_new_messages:
                final_ai_message = all_new_messages[-1]
            
            # Extract updated agent state
            updated_agent_state = {
                "agent_current_step": result.get("current_step", state.get("agent_current_step", "introduction")),
                "name_verification_status": result.get("name_verification_status", state.get("name_verification_status", "INSUFFICIENT_INFO")),
                "details_verification_status": result.get("details_verification_status", state.get("details_verification_status", "INSUFFICIENT_INFO")),
                "name_verification_attempts": result.get("name_verification_attempts", state.get("name_verification_attempts", 0)),
                "details_verification_attempts": result.get("details_verification_attempts", state.get("details_verification_attempts", 0)),
                "matched_fields": result.get("matched_fields", state.get("matched_fields", [])),
                "payment_secured": result.get("payment_secured", state.get("payment_secured", False)),
                "outstanding_amount": result.get("outstanding_amount", state.get("outstanding_amount", "R 0.00")),
                "is_call_ended": result.get("is_call_ended", state.get("is_call_ended", False)),
                "escalation_requested": result.get("escalation_requested", state.get("escalation_requested", False)),
                "cancellation_requested": result.get("cancellation_requested", state.get("cancellation_requested", False)),
                "tools_used": tools_used,
                "last_tool_results": last_tool_results
            }
            
            # Check if call ended
            if result.get("is_call_ended") or (final_ai_message and "goodbye" in final_ai_message.content.lower()):
                return Command(
                    update={
                        "messages": all_new_messages,  # Include ALL messages (tool calls, results, AI response)
                        "turn_count": turn_count + 1,
                        "max_turns": state_max_turns,
                        "ended": True,
                        **updated_agent_state
                    },
                    goto="__end__"
                )
            
            return Command(
                update={
                    "messages": all_new_messages,  # Include ALL messages (tool calls, results, AI response)
                    "turn_count": turn_count + 1,
                    "max_turns": state_max_turns,
                    **updated_agent_state
                },
                goto="debtor_turn"
            )
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return Command(update={"ended": True}, goto="__end__")
    
    def debtor_turn(state: SimulationState) -> Command[Literal["agent_turn", "__end__"]]:
        """Debtor responds."""
        turn_count = state.get("turn_count", 0)
        state_max_turns = state.get("max_turns", max_turns)  # Use the function parameter as default
        ended = state.get("ended", False)
        
        if turn_count >= state_max_turns or ended:
            return Command(update={"ended": True}, goto="__end__")
        
        try:
            # Swap roles for debtor simulator
            swapped_messages = swap_roles(state["messages"])
            result = debtor.invoke({"messages": swapped_messages})
            
            # Get debtor's AI response and convert to Human message
            debtor_response = result["messages"][-1]
            human_msg = HumanMessage(content=debtor_response.content)
            
            # Check if debtor wants to end
            if any(phrase in debtor_response.content.lower() 
                   for phrase in ["goodbye", "hang up", "end call"]):
                return Command(
                    update={
                        "messages": [human_msg],  # MessagesState will auto-add this
                        "turn_count": turn_count + 1,
                        "max_turns": state_max_turns,  # Ensure it's set in state
                        "ended": True
                    },
                    goto="__end__"
                )
            
            return Command(
                update={
                    "messages": [human_msg],  # MessagesState will auto-add this
                    "turn_count": turn_count + 1,
                    "max_turns": state_max_turns  # Ensure it's set in state
                },
                goto="agent_turn"
            )
        except Exception as e:
            logger.error(f"Debtor error: {e}")
            return Command(update={"ended": True}, goto="__end__")
    
    # Build workflow
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_turn", agent_turn)
    workflow.add_node("debtor_turn", debtor_turn)
    
    workflow.add_edge(START, "agent_turn")  # Agent starts
    
    return workflow.compile()


# Factory functions for all personality types
def create_cooperative_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Cooperative debtor - willing to pay and verify identity."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "cooperative", 50, config)

def create_difficult_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Difficult debtor - resistant and requires persuasion."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "difficult", 25, config)

def create_confused_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Confused debtor - needs clarification and guidance."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "confused", 20, config)

def create_busy_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Busy debtor - wants quick resolution."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "busy", 12, config)

def create_suspicious_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Suspicious debtor - hesitant to provide information."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "suspicious", 18, config)

def create_wrong_person_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Wrong person - not the debtor, different person entirely."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "wrong_person", 8, config)

def create_third_party_spouse_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Third party spouse - debtor's partner who can take messages."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "third_party_spouse", 10, config)

def create_third_party_parent_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Third party parent - debtor's parent, concerned but helpful."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "third_party_parent", 10, config)

def create_third_party_assistant_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Third party assistant - work colleague, professional boundaries."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "third_party_assistant", 8, config)

def create_third_party_employee_simulation(agent_llm, debtor_llm, client_data, config=None):
    """Third party employee - workplace receptionist, follows protocols."""
    return create_call_simulation(agent_llm, debtor_llm, client_data, "third_party_employee", 8, config)

# Generic third party function for backward compatibility
def create_third_party_simulation(agent_llm, debtor_llm, client_data, third_party_type="spouse", config=None):
    """Create third party simulation - defaults to spouse."""
    if third_party_type == "spouse":
        return create_third_party_spouse_simulation(agent_llm, debtor_llm, client_data, config)
    elif third_party_type == "parent":
        return create_third_party_parent_simulation(agent_llm, debtor_llm, client_data, config)
    elif third_party_type == "assistant":
        return create_third_party_assistant_simulation(agent_llm, debtor_llm, client_data, config)
    elif third_party_type == "employee":
        return create_third_party_employee_simulation(agent_llm, debtor_llm, client_data, config)
    else:
        return create_third_party_spouse_simulation(agent_llm, debtor_llm, client_data, config)

# Convenience function to create all simulations at once
def create_all_simulations(agent_llm, debtor_llm, client_data, config=None):
    """Create all simulation types for comprehensive testing."""
    return {
        "cooperative": create_cooperative_simulation(agent_llm, debtor_llm, client_data, config),
        "difficult": create_difficult_simulation(agent_llm, debtor_llm, client_data, config),
        "confused": create_confused_simulation(agent_llm, debtor_llm, client_data, config),
        "busy": create_busy_simulation(agent_llm, debtor_llm, client_data, config),
        "suspicious": create_suspicious_simulation(agent_llm, debtor_llm, client_data, config),
        "wrong_person": create_wrong_person_simulation(agent_llm, debtor_llm, client_data, config),
        "third_party_spouse": create_third_party_spouse_simulation(agent_llm, debtor_llm, client_data, config),
        "third_party_parent": create_third_party_parent_simulation(agent_llm, debtor_llm, client_data, config),
        "third_party_assistant": create_third_party_assistant_simulation(agent_llm, debtor_llm, client_data, config),
        "third_party_employee": create_third_party_employee_simulation(agent_llm, debtor_llm, client_data, config)
    }
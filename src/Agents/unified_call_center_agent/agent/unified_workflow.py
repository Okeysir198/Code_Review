# src/Agents/unified_call_center_agent/agent/unified_workflow.py
# Enhanced with better error handling and compatibility

"""
Unified Workflow Factory - Creates simple LangGraph workflow with single intelligent node
"""
import logging
from typing import Dict, Any, Literal, Optional, List
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.graph.graph import CompiledGraph

from ..core.unified_agent_state import UnifiedAgentState, ConversationObjective, VerificationStatus
from .unified_agent import UnifiedCallCenterAgent

logger = logging.getLogger(__name__)

def create_unified_call_center_workflow(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    config: Dict[str, Any],
    agent_name: str = "Sarah"
) -> CompiledGraph:
    """
    Create streamlined workflow with single intelligent conversation node.
    
    This replaces the complex 15+ node system with one smart node that:
    - Uses LangGraph's MessagesState for automatic message handling
    - Leverages built-in memory and persistence
    - Handles entire conversation flow with context awareness
    - Calls tools contextually based on conversation needs
    
    Args:
        model: Language model for the agent
        client_data: Client information and account data
        config: Configuration settings
        agent_name: Name of the agent (default: Sarah)
        
    Returns:
        Compiled LangGraph workflow ready for conversation
    """
    
    # Create the unified agent
    unified_agent = UnifiedCallCenterAgent(
        model=model,
        client_data=client_data,
        config=config,
        agent_name=agent_name
    )
    
    def conversation_node(state: UnifiedAgentState) -> Command[Literal["__end__"]]:
        """
        Single conversation node that handles entire call flow.
        
        This node:
        1. Processes each conversation turn with full context
        2. Uses the unified agent for intelligent responses
        3. Manages conversation state automatically
        4. Calls database tools when appropriate
        5. Handles verification, payment, and call completion
        """
        
        try:
            # Check if this is the first turn (no messages yet)
            messages = state.get("messages", [])
            
            if not messages:
                # Initialize state with client data using factory method
                initial_state_data = UnifiedAgentState.create_initial_state(client_data)
                
                # Create greeting message
                greeting = f"Good day, this is {agent_name} from Cartrack Accounts Department. May I speak with {initial_state_data.client_name}, please?"
                
                logger.info(f"Starting new conversation for client: {initial_state_data.client_name}")
                
                return Command(
                    update={
                        "user_id": initial_state_data.user_id,
                        "client_name": initial_state_data.client_name,
                        "outstanding_amount": initial_state_data.outstanding_amount,
                        "current_objective": initial_state_data.current_objective,
                        "name_verification": initial_state_data.name_verification,
                        "details_verification": initial_state_data.details_verification,
                        "verification_attempts": initial_state_data.verification_attempts,
                        "turn_count": 1,
                        "completed_objectives": [],
                        "client_concerns": [],
                        "mentioned_topics": [],
                        "payment_secured": False,
                        "call_ended": False,
                        "messages": [AIMessage(content=greeting)],
                        "last_action": "greeting"
                    },
                    goto="__end__"
                )
            
            # Check termination conditions
            if state.call_ended or state.turn_count > 50:
                logger.info(f"Call ended - reason: {'call_ended flag' if state.call_ended else 'max turns reached'}")
                return Command(
                    update={"call_ended": True},
                    goto="__end__"
                )
            
            # Process conversation turn with unified agent
            logger.info(f"Processing turn {state.turn_count + 1} for {state.client_name}")
            
            # Get turn results from unified agent
            turn_results = unified_agent.process_conversation_turn(state)
            
            logger.info(f"Turn processed - Action: {turn_results.get('last_action')}, Intent: {turn_results.get('last_intent')}")
            
            # Return state updates
            return Command(
                update=turn_results,
                goto="__end__"
            )
            
        except Exception as e:
            logger.error(f"Error in conversation node: {e}")
            
            # Graceful error handling
            error_message = "I apologize, I'm having technical difficulties. Let me connect you with a supervisor."
            
            return Command(
                update={
                    "messages": [AIMessage(content=error_message)],
                    "call_ended": True,
                    "last_action": "technical_error",
                    "escalation_requested": True
                },
                goto="__end__"
            )
    
    # Build the workflow
    workflow = StateGraph(UnifiedAgentState)
    
    # Add single conversation node
    workflow.add_node("conversation", conversation_node)
    
    # Simple flow: START -> conversation -> END
    workflow.add_edge(START, "conversation")
    
    # Compile with optional memory
    compile_kwargs = {}
    
    # Add memory/persistence if configured
    if config.get('configurable', {}).get('use_memory', True):
        try:
            compile_kwargs["checkpointer"] = MemorySaver()
            logger.info("Workflow compiled with memory persistence")
        except Exception as e:
            logger.warning(f"Memory persistence failed, compiling without: {e}")
    else:
        logger.info("Workflow compiled without persistence")
    
    try:
        compiled_workflow = workflow.compile(**compile_kwargs)
        logger.info("Unified call center workflow created successfully")
        return compiled_workflow
    except Exception as e:
        logger.error(f"Failed to compile workflow: {e}")
        # Fallback: compile without memory
        compiled_workflow = workflow.compile()
        logger.info("Workflow compiled without memory as fallback")
        return compiled_workflow

def create_conversation_config(thread_id: str = None) -> Dict[str, Any]:
    """
    Create configuration for conversation with thread ID.
    
    Args:
        thread_id: Unique identifier for conversation thread
        
    Returns:
        Configuration dictionary for LangGraph
    """
    
    if not thread_id:
        thread_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }

def get_conversation_state(workflow: CompiledGraph, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get current conversation state from workflow.
    
    Args:
        workflow: Compiled LangGraph workflow
        config: Configuration with thread_id
        
    Returns:
        Current state dictionary or None if not found
    """
    
    try:
        state = workflow.get_state(config)
        if state and hasattr(state, 'values'):
            return state.values
    except Exception as e:
        logger.error(f"Error getting conversation state: {e}")
    
    return None

def update_conversation_state(
    workflow: CompiledGraph, 
    config: Dict[str, Any], 
    updates: Dict[str, Any]
) -> bool:
    """
    Update conversation state.
    
    Args:
        workflow: Compiled LangGraph workflow
        config: Configuration with thread_id
        updates: State updates to apply
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        workflow.update_state(config, updates)
        logger.info(f"State updated with: {list(updates.keys())}")
        return True
    except Exception as e:
        logger.error(f"Error updating conversation state: {e}")
        return False

def get_conversation_history(workflow: CompiledGraph, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get conversation message history.
    
    Args:
        workflow: Compiled LangGraph workflow
        config: Configuration with thread_id
        
    Returns:
        List of message dictionaries
    """
    
    try:
        state = workflow.get_state(config)
        if state and hasattr(state, 'values') and 'messages' in state.values:
            messages = state.values['messages']
            return [
                {
                    "role": "assistant" if hasattr(msg, 'type') and msg.type == 'ai' else "human",
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in messages
            ]
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
    
    return []

def clear_conversation_history(workflow: CompiledGraph, config: Dict[str, Any]) -> bool:
    """
    Clear conversation history while preserving state.
    
    Args:
        workflow: Compiled LangGraph workflow
        config: Configuration with thread_id
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Update state to clear messages but keep other state
        current_state = workflow.get_state(config)
        if current_state and hasattr(current_state, 'values'):
            # Keep all state except messages
            preserved_state = {k: v for k, v in current_state.values.items() if k != 'messages'}
            preserved_state['messages'] = []
            preserved_state['turn_count'] = 0
            
            workflow.update_state(config, preserved_state)
            logger.info("Conversation history cleared")
            return True
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
    
    return False

# Integration helper for existing voice chat frontend
def create_voice_compatible_workflow(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    config: Dict[str, Any],
    agent_name: str = "Sarah"
) -> CompiledGraph:
    """
    Create workflow compatible with existing voice chat frontend.
    
    This function provides the same interface as the current system
    but uses the new unified agent internally.
    """
    
    logger.info(f"Creating voice-compatible workflow for client: {client_data.get('profile', {}).get('client_info', {}).get('client_full_name', 'Unknown')}")
    
    # Create unified workflow
    workflow = create_unified_call_center_workflow(
        model=model,
        client_data=client_data,
        config=config,
        agent_name=agent_name
    )
    
    # Add compatibility logging
    logger.info("Voice-compatible workflow created - ready for integration")
    
    return workflow

# Example usage and testing
def test_unified_workflow():
    """Test function for the unified workflow"""
    
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        print("langchain_ollama not available for testing")
        return None
    
    # Sample client data
    test_client_data = {
        'user_id': '12345',
        'profile': {
            'user_id': '12345',
            'client_info': {
                'client_full_name': 'John Smith',
                'first_name': 'John',
                'title': 'Mr'
            }
        },
        'account_aging': {
            'xbalance': '299.00',
            'x0': '0.00',
            'x30': '299.00'
        }
    }
    
    # Test configuration
    test_config = {
        'configurable': {
            'use_memory': False  # Disable memory for testing
        }
    }
    
    try:
        # Create model and workflow
        model = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
        workflow = create_unified_call_center_workflow(
            model=model,
            client_data=test_client_data,
            config=test_config
        )
        
        # Test conversation
        config = create_conversation_config("test_thread_001")
        
        print("=== Testing Unified Workflow ===")
        
        # Initial invocation (should generate greeting)
        result1 = workflow.invoke({}, config)
        if result1 and 'messages' in result1 and result1['messages']:
            print(f"Agent: {result1['messages'][-1].content}")
        
        # Simulate client response
        result2 = workflow.invoke({
            "messages": [HumanMessage(content="Yes, this is John speaking")]
        }, config)
        if result2 and 'messages' in result2 and result2['messages']:
            print(f"Agent: {result2['messages'][-1].content}")
        
        # Simulate payment agreement
        result3 = workflow.invoke({
            "messages": [HumanMessage(content="Okay, let's arrange payment")]
        }, config)
        if result3 and 'messages' in result3 and result3['messages']:
            print(f"Agent: {result3['messages'][-1].content}")
        
        print(f"Final state - Call ended: {result3.get('call_ended', False)}")
        print(f"Objectives completed: {len(result3.get('completed_objectives', []))}")
        
        return workflow
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None

if __name__ == "__main__":
    test_unified_workflow()
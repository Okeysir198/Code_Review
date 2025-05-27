# negotiation.py
"""
Negotiation Agent for Call Center.

Implements a specialized agent for the negotiation phase of debt collection calls,
explaining consequences and benefits of payment.
"""
from typing import Dict, Any, Optional
import logging

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from src.Agents.agent00_basic_agent import BasicAgent, BasicAgentState
from src.Agents.call_center.state import CallStep
from src.Agents.call_center.utils import create_system_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NegotiationAgentState(BasicAgentState):
    """State for the Negotiation Agent, extending BasicAgentState."""
    current_call_step: str = CallStep.NEGOTIATION.value
    query_detected: bool = False

def query_detection_processing(state: NegotiationAgentState) -> Dict[str, Any]:
    """
    Pre-processing function to detect queries in client messages.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with query detection flag
    """
    messages = state.get("messages", [])
    if not messages:
        return state
    
    # Get last message from client (human messages)
    client_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not client_messages:
        return state
    
    last_client_message = client_messages[-1]
    content = last_client_message.content.lower() if hasattr(last_client_message, 'content') else ""
    
    # Simple query detection - look for question marks or question phrases
    query_indicators = [
        "?", "what", "why", "how", "when", "where", "who", "which", 
        "can you", "could you", "explain", "tell me"
    ]
    
    query_detected = any(indicator in content for indicator in query_indicators)
    
    # Return updated state
    return {"query_detected": query_detected}

class NegotiationAgent(BasicAgent):
    """
    Specialized agent for the negotiation phase of a call.
    Inherits from BasicAgent and customizes behavior with query detection.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        client_info: Dict[str, Any],
        script_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """
        Initialize NegotiationAgent with specialized configuration.
        
        Args:
            model: Language model to use
            client_info: Information about the client
            script_type: Type of script to use
            config: Additional configuration options
            verbose: Whether to display verbose logs
        """
        # Extract client information and configuration
        self.client_info = client_info
        self.config = config or {}
        self.script_type = script_type

        # Generate specialized system prompt
        system_prompt = create_system_prompt(
            state={'current_call_step': CallStep.NEGOTIATION.value},
            script_type=self.script_type,
            client_info=self.client_info,
        )
        
        # Initialize parent class (BasicAgent) with the specialized prompt and query detection
        super().__init__(
            model=model,
            prompt=system_prompt,
            tools=[],  # No tools needed for this agent
            verbose=verbose,
            pre_processing_node=query_detection_processing,  # Add query detection
            config=self.config
        )
        
        logger.info("NegotiationAgent initialized for explaining consequences and benefits")
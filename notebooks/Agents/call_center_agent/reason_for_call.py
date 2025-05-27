# reason_for_call.py
"""
Reason For Call Agent for Call Center.

Implements a specialized agent for clearly explaining to the client
why they are being contacted, specifically regarding overdue amounts.
"""
from typing import Dict, Any, Optional
import logging

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from src.Agents.agent00_basic_agent import BasicAgent, BasicAgentState
from src.Agents.call_center.state import CallStep, VerificationStatus
from src.Agents.call_center.utils import create_system_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReasonForCallAgentState(BasicAgentState):
    """State for the Reason For Call Agent, extending BasicAgentState."""
    current_call_step: str = CallStep.REASON_FOR_CALL.value

class ReasonForCallAgent(BasicAgent):
    """
    Specialized agent for the reason for call phase.
    Inherits from BasicAgent and customizes behavior for this specific step.
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
        Initialize ReasonForCallAgent with specialized configuration.
        
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
            state={'current_call_step': CallStep.REASON_FOR_CALL.value},
            script_type=self.script_type,
            client_info=self.client_info,
        )
        
        # Initialize parent class (BasicAgent) with the specialized prompt
        super().__init__(
            model=model,
            prompt=system_prompt,
            tools=[],  # No tools needed for this agent
            verbose=verbose,
            config=self.config
        )
        
        logger.info("ReasonForCallAgent initialized for explaining account status")
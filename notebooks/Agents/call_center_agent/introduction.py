"""
Introduction Agent for Call Center.

Implements a specialized agent for the introduction phase
of debt collection calls, establishing the initial context with the client.
Uses optimized prompts for natural conversation flow.
"""
from typing import Dict, Any, Optional, List
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.Agents.agent00_basic_agent import BasicAgent, BasicAgentState
from src.Agents.call_center.state import CallStep, VerificationStatus
from src.Agents.call_center.utils import create_system_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntroductionAgentState(BasicAgentState):
    """State for the Introduction Agent, extending BasicAgentState."""
    # Add any introduction-specific state attributes here
    current_call_step: str = CallStep.INTRODUCTION.value
    name_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value

class IntroductionAgent(BasicAgent):
    """
    Specialized agent for the introduction phase of a call.
    Inherits from BasicAgent and customizes behavior for introduction step.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        client_info: Dict[str, Any],
        script_type: str,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = False
    ):
        """
        Initialize IntroductionAgent with specialized introduction configuration.
        
        Args:
            model: Language model to use
            client_info: Information about the client
            config: Additional configuration options
            tools: Optional list of tools (typically empty for introduction)
            verbose: Whether to display verbose logs
        """
        # Extract client information and configuration
        self.client_info = client_info
        self.config = config or {}
        self.script_type = script_type

        # Generate specialized system prompt
        system_prompt = create_system_prompt(
            state={ 'current_call_step': CallStep.INTRODUCTION.value },
            client_info=client_info,
            script_type=self.script_type,
            
        )
        def pre_processing_node(state:BasicAgentState):
            return 
        
        # Initialize parent class (BasicAgent)
        super().__init__(
            model=model,
            prompt=system_prompt,
            tools=tools or [],
            verbose=verbose,
            pre_processing_node=pre_processing_node,
            config=self.config
        )
        
        logger.info("IntroductionAgent initialized for call introduction")



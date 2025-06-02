# src/Agents/unified_call_center_agent/__init__.py
"""
Unified Call Center Agent - Complete replacement for complex multi-agent system

This package provides a single intelligent agent that handles entire debt collection
conversations with superior flow, context awareness, and tool usage.

Key Components:
- UnifiedCallCenterAgent: Single adaptive agent
- FastIntentDetector: Fast intent detection without LLM
- ConversationManager: Objective-based flow management
- UnifiedAgentState: LangGraph-optimized state management
- VoiceIntegration: Seamless integration with existing frontend
"""

from .agent.unified_workflow import (
    create_unified_call_center_workflow,
    create_voice_compatible_workflow,
    create_conversation_config
)

from .integration.voice_integration import (
    UnifiedVoiceHandler,
    VoiceInteractionHandler,  # Compatibility wrapper
    create_unified_voice_handler
)

from .core.unified_agent_state import (
    UnifiedAgentState,
    ConversationObjective,
    ClientMood,
    VerificationStatus
)

from .agent.unified_agent import UnifiedCallCenterAgent
from .routing.intent_detector import FastIntentDetector
from .core.conversation_manager import ConversationManager

__version__ = "1.0.0"
__author__ = "Claude AI Assistant"

# Main factory function for easy integration
def create_call_center_agent(
    model, 
    client_data, 
    config, 
    agent_name="Sarah"
):
    """
    Main factory function to create unified call center agent.
    
    Args:
        model: Language model
        client_data: Client information
        config: Configuration dictionary
        agent_name: Agent name (default: Sarah)
        
    Returns:
        Compiled LangGraph workflow ready for conversation
    """
    return create_unified_call_center_workflow(
        model=model,
        client_data=client_data,
        config=config,
        agent_name=agent_name
    )

# For backward compatibility with existing imports
def create_call_center_agent_workflow(*args, **kwargs):
    """Backward compatibility alias"""
    return create_call_center_agent(*args, **kwargs)

__all__ = [
    # Main factory functions
    'create_call_center_agent',
    'create_unified_call_center_workflow',
    'create_voice_compatible_workflow',
    
    # Core classes
    'UnifiedCallCenterAgent',
    'UnifiedVoiceHandler',
    'VoiceInteractionHandler',
    'FastIntentDetector',
    'ConversationManager',
    
    # State management
    'UnifiedAgentState',
    'ConversationObjective',
    'ClientMood',
    'VerificationStatus',
    
    # Utilities
    'create_unified_voice_handler',
    'create_conversation_config'
]

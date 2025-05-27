"""
Optimized Name Verification Agent for Secure Client Identification.

Implements a streamlined workflow for verifying client identity during outbound calls.
"""
import logging
from typing import Dict, Any, Optional, Literal, Union
from enum import Enum

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import Checkpointer
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, END, START  
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

from src.Agents.call_center.state import CallStep, VerificationStatus
from src.Agents.call_center.tools.verify_client_name import verify_client_name
from src.Agents.call_center.utils import create_system_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NameVerificationAgentState(MessagesState):
    """State management for name verification process."""
    current_call_step: str = CallStep.NAME_VERIFICATION.value
    is_call_ended: bool = False
    
    name_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    name_verification_attempts: int = 0
    
    system_prompt: Optional[str] = None

class NodeName(Enum):
    """Node names for the verification workflow."""
    VERIFICATION_TOOL = "name_verification_tool"
    GENERATE_RESPONSE = "name_verification_response"
    GENERATE_VERIFIED_RESPONSE = "generate_verified_reponse"

class NameVerificationAgent:
    """Specialized agent for secure client name verification."""
    CONTEXT_WINDOW_SIZE = 20
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        client_info: Dict[str, Any],
        script_type: str, 
        checkpointer: Optional[Checkpointer] = None,
        store: Optional[BaseStore] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Name Verification Agent."""
        self.llm = llm
        self.client_info = client_info
        self.config = config or {}
        self.script_type = script_type

        # Configuration
        self.client_full_name = client_info.get('full_name', 'Client')
        self.max_name_verification_attempts = self.config.get('verification',{}).get('max_name_verification_attempts', 3)
        self.show_logs = self.config.get('app',{}).get('show_logs', False)
        
        
        # Memory components
        self.checkpointer = checkpointer
        if not self.checkpointer and config and config.get('configurable', {}).get('use_memory', False):
            self.checkpointer = MemorySaver()
        self.store = store

        # Build workflow
        self.workflow = self._build_workflow()

    def _run_verification_tool(self, state: NameVerificationAgentState) -> Dict[str, Any]:
        """Run name verification tool and update state."""
        # Increment attempt count
        attempts = state.get("name_verification_attempts", 0) + 1
        
        # Run verification tool
        verification_result = verify_client_name.invoke({
            "client_full_name": self.client_full_name,
            "messages": state.get("messages", []),
            "max_failed_attempts": self.max_name_verification_attempts
        })
        
        # Extract verification status
        name_verification_status = verification_result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
        
        # Log result
        if self.show_logs:
            logger.info(f"Name Verification: status={name_verification_status}, attempt={attempts}/{self.max_name_verification_attempts}")
        
        # Determine if call should end based on max attempts
        is_call_ended = (
            name_verification_status in [VerificationStatus.THIRD_PARTY.value, 
                      VerificationStatus.UNAVAILABLE.value, 
                      VerificationStatus.WRONG_PERSON.value] or
            (attempts >= self.max_name_verification_attempts and 
             name_verification_status != VerificationStatus.VERIFIED.value)
        )
        
        # Prepare state update
        update = {
            "name_verification_attempts": attempts,
            "name_verification_status": name_verification_status,
            "suggested_response": verification_result.get("suggested_response"),
            "is_call_ended": is_call_ended,
            "current_call_step": CallStep.NAME_VERIFICATION.value
        }
        
        return update

    def _generate_response(self, state: NameVerificationAgentState) -> Dict[str, Any]:
        """Generate appropriate response based on verification status."""
        if self.show_logs:
            logger.info(f"Generating response for step: {state.get('current_call_step')}")
        
        # Generate system prompt
        system_prompt = create_system_prompt(
            state=state,
            script_type=self.script_type,
            client_info=self.client_info,
        )
        
        # Create messages list with context window
        all_messages = state.get("messages", [])[-self.CONTEXT_WINDOW_SIZE:]
        messages = [SystemMessage(content=system_prompt)] + all_messages
        
        # Generate response
        try:
            response = self.llm.invoke(messages)
            response_message = AIMessage(content=response.content)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response_message = AIMessage(content="I apologize, but I'm having trouble with our system. Could we try again?")
        
        return {
            "messages": [response_message],
            "system_prompt": system_prompt
        }
    def _generate_verified_response(self, state: NameVerificationAgentState) -> Dict[str, Any]:
        """Generate appropriate response based on verification status."""
        response = "Thanks for confirming. Please note this call is recorded for quality and security purposes. \
And to ensure I'm speaking to right person, I would need to verify a couple of quick details on your account before we continue."
        # return {
        #     "messages": [AIMessage(content=response)],
        # }
    
    def _verification_router(self, state: NameVerificationAgentState) -> str:
        """Route to next node based on verification result."""
        verification_status = state.get("name_verification_status")
        
        # End verification if client is verified
        if verification_status != VerificationStatus.VERIFIED.value:
            return NodeName.GENERATE_RESPONSE.value

        return NodeName.GENERATE_VERIFIED_RESPONSE.value
  
    def _build_workflow(self) -> StateGraph:
        """Construct the verification workflow."""
        workflow = StateGraph(NameVerificationAgentState)
        
        # Add nodes
        workflow.add_node(NodeName.VERIFICATION_TOOL.value, self._run_verification_tool)
        workflow.add_node(NodeName.GENERATE_RESPONSE.value, self._generate_response)
        workflow.add_node(NodeName.GENERATE_VERIFIED_RESPONSE.value, self._generate_verified_response)
        
        # Entry point
        # workflow.set_entry_point(NodeName.VERIFICATION_TOOL.value)
        workflow.add_edge(START, NodeName.VERIFICATION_TOOL.value)
        
        # Add routing
        workflow.add_conditional_edges(
            NodeName.VERIFICATION_TOOL.value,
            self._verification_router,
            {
                NodeName.GENERATE_RESPONSE.value: NodeName.GENERATE_RESPONSE.value,
                NodeName.GENERATE_VERIFIED_RESPONSE.value: NodeName.GENERATE_VERIFIED_RESPONSE.value
            }
        )
        
        # Terminal edge
        workflow.add_edge(NodeName.GENERATE_RESPONSE.value, END)
        workflow.add_edge(NodeName.GENERATE_VERIFIED_RESPONSE.value, END)
        
        # Compile with proper parameters
        compile_kwargs = {}
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer
        if self.store:
            compile_kwargs["store"] = self.store
            
        return workflow.compile(**compile_kwargs)

    
      
    
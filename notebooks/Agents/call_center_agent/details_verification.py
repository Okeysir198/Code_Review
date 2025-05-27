"""
Optimized Details Verification Agent for Secure Client Identification.

Implements a streamlined workflow for verifying client identity during outbound calls,
with random verification item selection for enhanced security.
"""
import logging
import random
from typing import Dict, Any, Optional, Literal, Union, List
from enum import Enum

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import Checkpointer
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, END, START  
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

from src.Agents.call_center.state import CallStep, VerificationStatus
from src.Agents.call_center.tools.verify_client_details import verify_client_details, verification_fields
from src.Agents.call_center.utils import create_system_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailsVerificationAgentState(MessagesState):
    """State management for details verification process."""
    current_call_step: str = CallStep.DETAILS_VERIFICATION.value
    is_call_ended: bool = False
    previous_step: Optional[str] = None

    details_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    details_verification_attempts: int = 0
    matched_fields: Optional[List[str]] = None
    field_to_verify: Optional[str] = None
    
    call_info: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None

class NodeName(Enum):
    """Node names for the verification workflow."""
    VERIFICATION_TOOL = "details_verification_tool"
    GENERATE_RESPONSE = "details_verification_response"

class DetailsVerificationAgent:
    """Specialized agent for secure client details verification with random field selection."""
    CONTEXT_WINDOW_SIZE = 20
    
    # Priority order for verification fields
    VERIFICATION_PRIORITY = ["id_number", "passport_number", "vehicle_registration", 
                             "username", "vehicle_make", "vehicle_model", 
                             "vehicle_color", "email"]
    
    # Define fields that should be available in the client database
    HIGH_VALUE_FIELDS = ["vehicle_registration", "vehicle_model","id_number", "passport_number" ]
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        client_info: Dict[str, Any],
        script_type: str, 
        checkpointer: Optional[Checkpointer] = None,
        store: Optional[BaseStore] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Details Verification Agent."""
        self.llm = llm
        self.client_info = client_info
        self.config = config or {}
        self.script_type = script_type

        # Configuration
        self.client_full_name = client_info.get('full_name', 'Client')
        self.max_details_verification_attempts = self.config.get('verification', {}).get('max_details_verification_attempts', 3)
        self.show_logs = self.config.get('app', {}).get('show_logs', False)
        
        # Identify available verification fields in client data
        self.available_fields = self._identify_available_fields()
        if self.show_logs:
            logger.info(f"Available verification fields: {self.available_fields}")
        
        # Memory components
        self.checkpointer = checkpointer
        if not self.checkpointer and config and config.get('configurable', {}).get('use_memory', False):
            self.checkpointer = MemorySaver()
        self.store = store

        # Build workflow
        self.workflow = self._build_workflow()

    def _identify_available_fields(self) -> List[str]:
        """Identify verification fields available in client data."""
        available_fields = []
        
        # Check for direct attributes
        for field in self.VERIFICATION_PRIORITY:
            if field in self.client_info and self.client_info[field]:
                available_fields.append(field)
            # Special handling for vehicle fields
            elif field.startswith("vehicle_") and "vehicles" in self.client_info and self.client_info["vehicles"]:
                vehicle_field = field.replace("vehicle_", "")
                if vehicle_field in self.client_info["vehicles"][0] and self.client_info["vehicles"][0][vehicle_field]:
                    available_fields.append(field)
        
        # Ensure at least some fields are available
        if not available_fields:
            logger.warning("No verification fields found in client data. Using default fields.")
            # Default to basic fields if nothing found
            available_fields = ["id_number", "vehicle_registration", "email"]
        
        return available_fields

    def _select_next_verification_field(self, state: DetailsVerificationAgentState) -> str:
        """
        Select the next verification field to ask for.
        
        For first attempt, prioritize high-value fields or randomly select.
        For subsequent attempts, select from remaining fields not yet matched.
        """
        matched_fields = state.get("matched_fields", [])
        
        # First, get remaining fields to check (ignore already matched fields)
        remaining_fields = [f for f in self.available_fields if f not in matched_fields]
        
        # First attempt: prioritize high-value fields if available, otherwise random
        if state.get("details_verification_attempts", 0) <= 1:
            high_value_available = [f for f in self.HIGH_VALUE_FIELDS if f in remaining_fields]
            if high_value_available:
                # Randomly select from high-value fields
                return random.choice(high_value_available)
        
        # If no remaining fields, use any available field (shouldn't happen normally)
        if not remaining_fields:
            return random.choice(self.available_fields)
            
        # Random selection from remaining fields for unpredictability
        return random.choice(remaining_fields)

    def _run_verification_tool(self, state: DetailsVerificationAgentState) -> Dict[str, Any]:
        """Run details verification tool and update state."""
        # Increment attempt count
        attempts = state.get("details_verification_attempts", 0) + 1
        
        # Get previously matched fields
        matched_fields = state.get("matched_fields", [])
        
        # Get friendly display name for the field
        field_to_verify = self._select_next_verification_field(state)
        
        if self.show_logs:
            logger.info(f"Verification attempt {attempts}: Requesting field '{field_to_verify}'")
            logger.info(f"Previously matched fields: {matched_fields}")
        
        # Run verification tool
        verification_result = verify_client_details.invoke({
            "client_details": self.client_info,
            "messages": state.get("messages", []),
            'required_match_count': 3,
            "max_failed_attempts": self.max_details_verification_attempts
        })
        
        if self.show_logs:
            logger.info(f"Verification result: {verification_result}")
        
        # Extract verification status
        details_verification_status = verification_result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
        
        # Update matched fields with newly matched fields
        new_matched_fields = verification_result.get('matched_fields', [])
        all_matched_fields = list(set(matched_fields + new_matched_fields))
        
        # Determine if call should end based on max attempts or verification
        is_call_ended = (attempts >= self.max_details_verification_attempts and 
                        details_verification_status != VerificationStatus.VERIFIED.value)
        
        # Prepare state update
        update = {
            "current_call_step": CallStep.DETAILS_VERIFICATION.value,
            "is_call_ended": is_call_ended,

            "details_verification_attempts": attempts,
            "details_verification_status": details_verification_status,
            "matched_fields": all_matched_fields,
            "field_to_verify": field_to_verify,  # Add the friendly field name to state
            
            "call_info": {
                "extracted_fields": verification_result.get("extracted_fields", {}),
            }
        }
        
        return update

    def _generate_response(self, state: DetailsVerificationAgentState) -> Dict[str, Any]:
        """Generate appropriate response based on verification status."""
        if self.show_logs:
            logger.info(f"Generating response for step: {state.get('current_call_step')}")
            logger.info(f"Field to verify: {state.get('field_to_verify')}")
        
        # Generate system prompt
        system_prompt = create_system_prompt(
            state=state,
            script_type=self.script_type,
            client_info=self.client_info,
        )
        
        # Create messages list with context window
        all_messages =[msg for msg in state.get("messages", []) if not isinstance(msg, SystemMessage)]
        all_messages =all_messages[-self.CONTEXT_WINDOW_SIZE:]

        logger.info(f"Messages for LLM: {all_messages}")
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
    
    def _verification_router(self, state: DetailsVerificationAgentState) -> str:
        """Route to next node based on verification result."""
        verification_status = state.get("details_verification_status")
        matched_count = len(state.get("matched_fields", []))
        
        # If matched 3 or more fields, or explicitly verified, end verification
        if verification_status == VerificationStatus.VERIFIED.value or matched_count >= 3:
            return END

        return NodeName.GENERATE_RESPONSE.value
  
    def _build_workflow(self) -> StateGraph:
        """Construct the verification workflow."""
        workflow = StateGraph(DetailsVerificationAgentState)
        
        # Add nodes
        workflow.add_node(NodeName.VERIFICATION_TOOL.value, self._run_verification_tool)
        workflow.add_node(NodeName.GENERATE_RESPONSE.value, self._generate_response)
        
        # Entry point
        workflow.add_edge(START, NodeName.VERIFICATION_TOOL.value)
        
        # Add routing
        workflow.add_conditional_edges(
            NodeName.VERIFICATION_TOOL.value,
            self._verification_router,
            {
                NodeName.GENERATE_RESPONSE.value: NodeName.GENERATE_RESPONSE.value,
                END: END
            }
        )
        
        # Terminal edge
        workflow.add_edge(NodeName.GENERATE_RESPONSE.value, END)
        
        # Compile with proper parameters
        compile_kwargs = {}
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer
        if self.store:
            compile_kwargs["store"] = self.store
            
        return workflow.compile(**compile_kwargs)
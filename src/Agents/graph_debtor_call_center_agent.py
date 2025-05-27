# """
# Complete Call Center Agent

# Implements a comprehensive debt collection agent that follows a structured call flow,
# using specialized sub-agents for each phase of the conversation within a LangGraph workflow.
# Includes optional simulated debtor capability for testing.
# """

# import logging
# from typing import Dict, Any, Optional, Callable, Annotated, List, Union, cast
# from enum import Enum

# from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
# from langchain_core.language_models import BaseChatModel
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import MessagesState
# from langgraph.graph.graph import CompiledGraph
# from langgraph.types import Command, Checkpointer
# from langgraph.checkpoint.memory import MemorySaver

# from src.Agents.call_center_agent.call_scripts import ScriptManager, ScriptType
# from src.Agents.call_center_agent.state import CallStep, CallCenterAgentState, PaymentMethod, VerificationStatus
# from src.Agents.call_center_agent.introduction import IntroductionAgent
# from src.Agents.call_center_agent.name_verification import NameVerificationAgent
# from src.Agents.call_center_agent.details_verification import DetailsVerificationAgent
# from src.Agents.call_center_agent.reason_for_call import ReasonForCallAgent
# from src.Agents.call_center_agent.negotiation import NegotiationAgent

# # Import debtor simulation only if needed
# try:
#     from src.Agents.graph_debtor_call_center_agent import create_simulated_debtor, _swap_roles
#     DEBTOR_SIMULATION_AVAILABLE = True
# except ImportError:
#     DEBTOR_SIMULATION_AVAILABLE = False

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# class CallCenterAgent():
#     """
#     Comprehensive call center agent that manages the entire debt collection call flow.
    
#     Uses specialized sub-agents for each phase of the conversation and maintains
#     a deterministic, rule-based workflow with structured state transitions.
    
#     Optionally includes a simulated debtor for testing and demonstration purposes.
#     """
    
#     # Node names as constants for clarity
#     NODE_ROUTER = "router"
#     NODE_INTRODUCTION = "00_introduction"
#     NODE_NAME_VERIFICATION = "01_name_verification"
#     NODE_DETAILS_VERIFICATION = "02_details_verification"
#     NODE_REASON_FOR_CALL = "03_reason_for_call"
#     NODE_NEGOTIATION = "04_negotiation"
#     NODE_PROMISE_TO_PAY = "05_promise_to_pay"
#     NODE_DEBICHECK_SETUP = "06_debicheck_setup"
#     NODE_SUBSCRIPTION_REMINDER = "07_subscription_reminder"
#     NODE_PAYMENT_PORTAL = "08_payment_portal"
#     NODE_CLIENT_DETAILS_UPDATE = "09_client_details_update"
#     NODE_REFERRALS = "10_referrals"
#     NODE_FURTHER_ASSISTANCE = "11_further_assistance"
#     NODE_CANCELLATION = "12_cancellation"
#     NODE_ESCALATION = "13_escalation"
#     NODE_CLOSING = "14_closing"
#     NODE_QUERY_HANDLER = "query_handler"
#     NODE_END_CALL = "end_call"
#     NODE_DEBTOR_SIMULATION = "simulated_debtor"
    
#     def __init__(self, app_config: Dict[str, Any], simulate_debtor: bool = False, debtor_profile: str = "cooperative"):
#         """
#         Initialize the call center agent with configuration and sub-agents.
        
#         Args:
#             app_config: Application configuration dictionary
#             simulate_debtor: Whether to enable debtor simulation for testing
#             debtor_profile: Profile type for simulated debtor (cooperative, resistant, confused, etc.)
#         """
#         # Initialize base components
#         super().__init__(app_config)
        
#         # Setup persistence
#         self.checkpointer = MemorySaver()

#         # Client information
#         self.client_info = app_config.get("client_details", {})
#         self.client_full_name = self.client_info.get("full_name", "Client")
        
#         # Verification configuration
#         verification_config = app_config.get("verification", {})
#         self.max_name_verification_attempts = verification_config.get("max_name_verification_attempts", 3)
#         self.max_details_verification_attempts = verification_config.get("max_details_verification_attempts", 3)
        
#         # Script settings
#         script_config = app_config.get("script", {})
#         script_type_str = script_config.get("type", "ratio_1_inflow")
        
#         # Set script type with fallback
#         try:
#             self.script_type = ScriptType(script_type_str)
#         except ValueError:
#             if self.show_logs:
#                 logger.warning(f"Invalid script type: {script_type_str}. Using default.")
#             self.script_type = ScriptType.RATIO_1_INFLOW
        
#         # Debtor simulation settings
#         self.simulate_debtor = simulate_debtor and DEBTOR_SIMULATION_AVAILABLE
#         self.debtor_profile = debtor_profile
        
#         # Initialize specialized sub-agents
#         self._init_sub_agents()
        
#         # Build workflow
#         self.workflow = self._build_workflow()
        
#         if self.show_logs:
#             logger.info(f"CallCenterAgent initialized with script: {self.script_type.value}")
#             if self.simulate_debtor:
#                 logger.info(f"Debtor simulation enabled with profile: {self.debtor_profile}")
    
#     def _init_sub_agents(self):
#         """Initialize all specialized sub-agents for each call step."""
#         # Existing agents
#         self.introduction_agent = IntroductionAgent(
#             model=self.llm,
#             client_info=self.client_info,
#             script_type=self.script_type,
#             config=self.config,
#         ).workflow
        
#         self.name_verification_agent = NameVerificationAgent(
#             llm=self.llm,
#             client_info=self.client_info,
#             script_type=self.script_type,
#             config=self.config
#         ).workflow

#         self.details_verification_agent = DetailsVerificationAgent(
#             llm=self.llm,
#             client_info=self.client_info,
#             script_type=self.script_type,
#             config=self.config
#         ).workflow
        
#         # New BasicAgent-based agents
#         self.reason_for_call_agent = ReasonForCallAgent(
#             model=self.llm,
#             client_info=self.client_info,
#             script_type=self.script_type,
#             config=self.config,
#             verbose=self.show_logs
#         ).workflow
        
#         self.negotiation_agent = NegotiationAgent(
#             model=self.llm,
#             client_info=self.client_info,
#             script_type=self.script_type,
#             config=self.config,
#             verbose=self.show_logs
#         ).workflow

#         # Initialize debtor simulation if enabled
#         if self.simulate_debtor and DEBTOR_SIMULATION_AVAILABLE:
#             try:
#                 self.debtor_agent = create_simulated_debtor(self.debtor_profile).workflow
#                 if self.show_logs:
#                     logger.info(f"Debtor simulation agent initialized with profile: {self.debtor_profile}")
#             except Exception as e:
#                 self.simulate_debtor = False
#                 logger.error(f"Failed to initialize debtor simulation: {e}")
#         else:
#             self.debtor_agent = None
#             if self.simulate_debtor and not DEBTOR_SIMULATION_AVAILABLE:
#                 logger.warning("Debtor simulation requested but not available. Continuing without simulation.")
#                 self.simulate_debtor = False

#     #########################################################################################
#     # Helper Methods
#     #########################################################################################
#     def _prepare_messages(self, state: CallCenterAgentState) -> List[Union[AIMessage, HumanMessage]]:
#         """
#         Get conversation messages from the state, filtering out system messages.
        
#         Args:
#             state: Current conversation state
            
#         Returns:
#             List of conversation messages
#         """
#         messages = state.get("messages", [])
#         return [msg for msg in messages if not isinstance(msg, SystemMessage)]
    
#     def _get_thread_config(self) -> Dict[str, Any]:
#         """
#         Get configuration for thread persistence.
        
#         Returns:
#             Configuration dictionary with thread_id if available
#         """
#         thread_id = self.config.get('configurable', {}).get('thread_id')
#         if thread_id:
#             return {"configurable": {"thread_id": thread_id}}
#         return {}

#     #########################################################################################
#     # Node Factory Methods - Create delegate nodes for sub-agents
#     #########################################################################################
#     def _create_introduction_node(self, agent: CompiledGraph) -> Callable[[CallCenterAgentState], Command]:
#         """
#         Create a node that delegates to the introduction agent.
        
#         Args:
#             agent: Compiled introduction agent graph
            
#         Returns:
#             Function that processes the introduction step
#         """
#         def introduction_node(
#             state: CallCenterAgentState,
#             graph: Annotated[CompiledGraph, agent] = None
#         ) -> Command:
#             # Get messages from state
#             messages = self._prepare_messages(state)
            
#             thread_config = self._get_thread_config()

#             # Invoke the introduction agent
#             result = agent.invoke({"messages": messages}, config=thread_config)
            
#             # Extract the agent's response
#             result_messages = result.get("messages", [])
#             response_message = result_messages[-1] if result_messages else None
#             system_prompt = result.get("system_prompt","")
            
#             # Log step transition if in debug mode
#             if self.debug_mode:
#                 logger.debug("Introduction step completed")
            
#             # Return command with updates
#             return Command(
#                 update={
#                     "messages": [AIMessage(content=response_message.content)] if isinstance(response_message, AIMessage) else [],
#                     "current_call_step": CallStep.NAME_VERIFICATION.value,
#                     "previous_step": CallStep.INTRODUCTION.value,
#                     "system_prompt":system_prompt
#                 },
#                 goto=self.NODE_DEBTOR_SIMULATION if self.simulate_debtor else END,
#             )
        
#         return introduction_node
    
#     def _create_name_verification_node(self, agent: CompiledGraph) -> Callable[[CallCenterAgentState], Command]:
#         """
#         Create a node that delegates to the name verification agent.
        
#         Args:
#             agent: Compiled name verification agent graph
            
#         Returns:
#             Function that processes the name verification step
#         """
#         def name_verification_node(
#             state: CallCenterAgentState,
#             graph: Annotated[CompiledGraph, agent] = None
#         ) -> Command:
#             # Get messages from state
#             messages = self._prepare_messages(state)
            
#             # Get thread configuration
#             thread_config = self._get_thread_config()
            
#             # Run the name verification workflow
#             result = agent.invoke({"messages": messages}, config=thread_config)
            
#             # Extract results
#             name_verification_status = result.get("name_verification_status")
#             name_verification_attempts = result.get("name_verification_attempts", 0)
#             is_call_ended = result.get("is_call_ended", False)
            
#             # Extract the latest message
#             result_messages = result.get("messages", [])
#             response_message = result_messages[-1] if result_messages else None
#             system_prompt = result.get("system_prompt", "")

#             # Log verification result if in debug mode
#             if self.debug_mode:
#                 logger.debug(f"Name verification: status={name_verification_status}, attempts={name_verification_attempts}, ended={is_call_ended}")
            
#             # Determine next step based on verification status
#             next_step = CallStep.END_CONVERSATION.value if is_call_ended else CallStep.NAME_VERIFICATION.value
#             goto_node = END
            
#             # If verified, move to details verification
#             if name_verification_status == VerificationStatus.VERIFIED.value:
#                 next_step = CallStep.DETAILS_VERIFICATION.value
#                 goto_node = self.NODE_DETAILS_VERIFICATION
            
#             # Determine where to go next based on simulation settings
#             final_goto = self.NODE_DEBTOR_SIMULATION if self.simulate_debtor and not is_call_ended else goto_node
            
#             # Return command with updates
#             return Command(
#                 update={
#                     "messages": [AIMessage(content=response_message.content)] if isinstance(response_message, AIMessage) else [],
#                     "name_verification_status": name_verification_status,
#                     "name_verification_attempts": name_verification_attempts,
#                     "is_call_ended": is_call_ended,
#                     "current_call_step": next_step,
#                     "previous_step": CallStep.NAME_VERIFICATION.value,
#                     "system_prompt": system_prompt
#                 },
#                 goto=final_goto,
#             )
      
#         return name_verification_node
    
#     def _create_details_verification_node(self, agent: CompiledGraph) -> Callable[[CallCenterAgentState], Command]:
#         """
#         Create a node that delegates to the details verification agent.
        
#         Args:
#             agent: Compiled details verification agent graph
            
#         Returns:
#             Function that processes the details verification step
#         """
#         def details_verification_node(
#             state: CallCenterAgentState,
#             graph: Annotated[CompiledGraph, agent] = None
#         ) -> Command:
#             # Get messages from state
#             messages = self._prepare_messages(state)
            
#             # Get thread configuration
#             thread_config = self._get_thread_config()
            
#             # Run the details verification workflow
#             result = agent.invoke({"messages": messages}, config=thread_config)
            
#             # Extract results
#             details_verification_status = result.get("details_verification_status")
#             details_verification_attempts = result.get("details_verification_attempts", 0)
#             is_call_ended = result.get("is_call_ended", False)
#             matched_fields = result.get("matched_fields", [])
            
#             # Extract the latest message
#             result_messages = result.get("messages", [])
#             response_message = result_messages[-1] if result_messages else None
#             system_prompt = result.get("system_prompt","")
            
#             # Log verification result if in debug mode
#             if self.debug_mode:
#                 logger.debug(f"Details verification: status={details_verification_status}, attempts={details_verification_attempts}, matched={matched_fields}, ended={is_call_ended}")
            
#             # Determine next step based on verification status
#             next_step = CallStep.END_CONVERSATION.value if is_call_ended else CallStep.DETAILS_VERIFICATION.value
#             goto_node = END
            
#             # If verified, move to reason for call
#             if details_verification_status == VerificationStatus.VERIFIED.value:
#                 next_step = CallStep.REASON_FOR_CALL.value
#                 goto_node = self.NODE_REASON_FOR_CALL
            
#             # Determine where to go next based on simulation settings
#             final_goto = self.NODE_DEBTOR_SIMULATION if self.simulate_debtor and not is_call_ended else goto_node
            
#             # Return command with updates
#             return Command(
#                 update={
#                     "messages": [AIMessage(content=response_message.content)] if isinstance(response_message, AIMessage) else [],
#                     "details_verification_status": details_verification_status,
#                     "details_verification_attempts": details_verification_attempts,
#                     "matched_fields": matched_fields,
#                     "is_call_ended": is_call_ended,
#                     "current_call_step": next_step,
#                     "previous_step": CallStep.DETAILS_VERIFICATION.value,
#                     "system_prompt": system_prompt
#                 },
#                 goto=final_goto,
#             )
      
#         return details_verification_node
    
#     def _create_basic_agent_node(self, agent: CompiledGraph, current_step: str, next_step: str) -> Callable[[CallCenterAgentState], Command]:
#         """
#         Create a node that delegates to a BasicAgent-based workflow.
        
#         Args:
#             agent: Compiled agent graph
#             current_step: Current step being processed
#             next_step: Next step to move to after processing
            
#         Returns:
#             Function that processes the step
#         """
#         def basic_agent_node(
#             state: CallCenterAgentState,
#             graph: Annotated[CompiledGraph, agent] = None
#         ) -> Command:
#             # Get messages from state
#             messages = self._prepare_messages(state)
            
#             # Get thread configuration
#             thread_config = self._get_thread_config()
            
#             # Run the agent workflow
#             result = agent.invoke({"messages": messages}, config=thread_config)
            
#             # Extract results
#             query_detected = result.get("query_detected", False)
            
#             # Extract the latest message
#             result_messages = result.get("messages", [])
#             response_message = result_messages[-1] if result_messages else None
#             system_prompt = result.get("system_prompt","")
            
#             # Log step if in debug mode
#             if self.debug_mode:
#                 logger.debug(f"{current_step} step completed")
            
#             # Determine actual next step
#             actual_next_step = next_step
#             if current_step == CallStep.NEGOTIATION.value and query_detected:
#                 actual_next_step = CallStep.QUERY_RESOLUTION.value
            
#             # Determine where to go next based on simulation settings
#             final_goto = self.NODE_DEBTOR_SIMULATION if self.simulate_debtor else END
            
#             # Return command with updates
#             return Command(
#                 update={
#                     "messages": [AIMessage(content=response_message.content)] if isinstance(response_message, AIMessage) else [],
#                     "query_detected": query_detected if current_step == CallStep.NEGOTIATION.value else False,
#                     "current_call_step": actual_next_step,
#                     "previous_step": current_step,
#                     "system_prompt": system_prompt
#                 },
#                 goto=final_goto,
#             )
    
#         return basic_agent_node
    
#     def _create_debtor_simulation_node(self, agent: Optional[CompiledGraph]) -> Callable[[CallCenterAgentState], Dict[str, Any]]:
#         """
#         Create a node for simulated debtor responses.
        
#         Args:
#             agent: Compiled debtor agent graph, or None if simulation disabled
            
#         Returns:
#             Function that processes debtor simulation
#         """
#         def simulated_debtor_node(
#             state: CallCenterAgentState,
#             graph: Annotated[Optional[CompiledGraph], agent] = None
#         ) -> Dict[str, Any]:
            
#             # Check if call has ended
#             if state.get("is_call_ended", False):
#                 logger.info("Call already ended - skipping debtor simulation")
#                 return {"is_call_ended": True}
                
#             # Get messages from state
#             messages = self._prepare_messages(state)

#             # Swap roles for debtor agent
#             swapped_messages = _swap_roles(messages)
#             thread_config = self._get_thread_config()
            
#             # Invoke debtor agent
#             result = agent.invoke(
#                 {"messages": swapped_messages},
#                 config=thread_config
#             )
            
#             # Extract response
#             result_messages = result.get("messages", [])
#             response_message = result_messages[-1] if result_messages else None
            
#             return {"messages": [HumanMessage(content=response_message.content)]}
       
#         return simulated_debtor_node
    
#     #########################################################################################
#     # Routing Logic
#     #########################################################################################
#     def _router(self, state: CallCenterAgentState) -> Dict[str, Any]:
#         """
#         Rule-based router that determines the next step in the call flow.
#         Optimized for fast response times without LLM dependency.
        
#         Args:
#             state: Current conversation state
            
#         Returns:
#             Dictionary with routing decision
#         """
#         # Get current state info
#         current_step = state.get('current_call_step', CallStep.INTRODUCTION.value)
#         is_call_ended = state.get('is_call_ended', False)
#         name_verification_status = state.get('name_verification_status')
#         details_verification_status = state.get('details_verification_status')
#         payment_method = state.get('payment_method')
        
#         # Check if call ending conditions are met
#         if is_call_ended:
#             next_node = self.NODE_END_CALL
#             reason = "Call has ended"
#             if self.show_logs:
#                 logger.info(f"Call ended, routing to {self.NODE_END_CALL}")
#             return {
#                 'current_call_step': current_step,
#                 'recommended_next_node': next_node,
#                 'router_reasoning': reason
#             }
        
#         # Check for special flags that override normal flow
#         if state.get('query_detected', False):
#             next_node = self.NODE_QUERY_HANDLER
#             reason = "Query detected"
#             if self.show_logs:
#                 logger.info(f"Query detected, routing to query handler")
#             return {
#                 'current_call_step': current_step,
#                 'recommended_next_node': next_node,
#                 'router_reasoning': reason
#             }
            
#         if state.get('cancellation_requested', False):
#             next_node = self.NODE_CANCELLATION
#             reason = "Cancellation requested"
#             if self.show_logs:
#                 logger.info(f"Cancellation requested, routing to cancellation handler")
#             return {
#                 'current_call_step': current_step,
#                 'recommended_next_node': next_node,
#                 'router_reasoning': reason
#             }
            
#         if state.get('escalation_requested', False):
#             next_node = self.NODE_ESCALATION
#             reason = "Escalation requested"
#             if self.show_logs:
#                 logger.info(f"Escalation requested, routing to escalation handler")
#             return {
#                 'current_call_step': current_step,
#                 'recommended_next_node': next_node,
#                 'router_reasoning': reason
#             }
        
#         # Step to node mapping for standard flow
#         step_to_node = {
#             # Core flow
#             CallStep.INTRODUCTION.value: self.NODE_INTRODUCTION,
#             CallStep.NAME_VERIFICATION.value: self.NODE_NAME_VERIFICATION,
#             CallStep.DETAILS_VERIFICATION.value: self.NODE_DETAILS_VERIFICATION,
#             CallStep.REASON_FOR_CALL.value: self.NODE_REASON_FOR_CALL,
#             CallStep.NEGOTIATION.value: self.NODE_NEGOTIATION,
#             CallStep.PROMISE_TO_PAY.value: self.NODE_PROMISE_TO_PAY,
            
#             # Payment method specific paths
#             CallStep.DEBICHECK_SETUP.value: self.NODE_DEBICHECK_SETUP,
#             CallStep.PAYMENT_PORTAL.value: self.NODE_PAYMENT_PORTAL,
            
#             # Post-payment steps
#             CallStep.SUBSCRIPTION_REMINDER.value: self.NODE_SUBSCRIPTION_REMINDER,
#             CallStep.CLIENT_DETAILS_UPDATE.value: self.NODE_CLIENT_DETAILS_UPDATE,
#             CallStep.REFERRALS.value: self.NODE_REFERRALS,
#             CallStep.FURTHER_ASSISTANCE.value: self.NODE_FURTHER_ASSISTANCE,
            
#             # Special cases
#             CallStep.QUERY_RESOLUTION.value: self.NODE_QUERY_HANDLER,
#             CallStep.CANCELLATION.value: self.NODE_CANCELLATION,
#             CallStep.ESCALATION.value: self.NODE_ESCALATION,
#             CallStep.CLOSING.value: self.NODE_CLOSING,
#             CallStep.END_CONVERSATION.value: self.NODE_END_CALL
#         }
        
#         # Enforce verification order and conditional branching
#         if current_step == CallStep.INTRODUCTION.value:
#             next_node = self.NODE_NAME_VERIFICATION
#             reason = "Progress from introduction to name verification"
#         elif current_step == CallStep.NAME_VERIFICATION.value:
#             if name_verification_status == VerificationStatus.VERIFIED.value:
#                 next_node = self.NODE_DETAILS_VERIFICATION
#                 reason = "Name verified, moving to details verification"
#             else:
#                 next_node = self.NODE_NAME_VERIFICATION
#                 reason = "Continue name verification"
#         elif current_step == CallStep.DETAILS_VERIFICATION.value:
#             if details_verification_status == VerificationStatus.VERIFIED.value:
#                 next_node = self.NODE_REASON_FOR_CALL
#                 reason = "Details verified, proceeding to reason for call"
#             else:
#                 next_node = self.NODE_DETAILS_VERIFICATION
#                 reason = "Continue details verification"
#         else:
#             # Standard progression for post-verification steps
#             next_node = step_to_node.get(current_step, self.NODE_NEGOTIATION)
#             reason = f"Standard progression from {current_step}"
            
#             # Special case for payment method selection
#             if current_step == CallStep.PROMISE_TO_PAY.value and payment_method:
#                 if payment_method == PaymentMethod.DEBICHECK.value:
#                     next_node = self.NODE_DEBICHECK_SETUP
#                     reason = "DebiCheck payment method selected"
#                 elif payment_method == PaymentMethod.PAYMENT_PORTAL.value:
#                     next_node = self.NODE_PAYMENT_PORTAL
#                     reason = "Payment portal method selected"
#                 else:
#                     # Direct debit methods go straight to subscription reminder
#                     next_node = self.NODE_SUBSCRIPTION_REMINDER
#                     reason = f"{payment_method} payment method selected"
        
#         if self.show_logs:
#             logger.info(f"Router: {current_step} -> {next_node} ({reason})")
            
#         return {
#             'current_call_step': current_step,
#             'recommended_next_node': next_node,
#             'router_reasoning': reason
#         }
    
#     def _router_condition(self, state: CallCenterAgentState) -> str:
#         """
#         Determine next node based on router's recommendation.
        
#         Args:
#             state: Current agent state
                
#         Returns:
#             Name of the next node to execute
#         """
#         # Check if conversation has ended
#         if state.get('is_call_ended', False):
#             return self.NODE_END_CALL
        
#         # Get recommended node from router
#         recommended_node = state.get('recommended_next_node')
        
#         # Valid node names
#         valid_nodes = [
#             self.NODE_INTRODUCTION,
#             self.NODE_NAME_VERIFICATION,
#             self.NODE_DETAILS_VERIFICATION,
#             self.NODE_REASON_FOR_CALL,
#             self.NODE_NEGOTIATION,
#             self.NODE_PROMISE_TO_PAY,
#             self.NODE_DEBICHECK_SETUP,
#             self.NODE_SUBSCRIPTION_REMINDER,
#             self.NODE_PAYMENT_PORTAL,
#             self.NODE_CLIENT_DETAILS_UPDATE,
#             self.NODE_REFERRALS,
#             self.NODE_FURTHER_ASSISTANCE,
#             self.NODE_CANCELLATION,
#             self.NODE_ESCALATION,
#             self.NODE_CLOSING,
#             self.NODE_QUERY_HANDLER,
#             self.NODE_END_CALL
#         ]
        
#         # If recommended node is valid, use it
#         if recommended_node and recommended_node in valid_nodes:
#             return recommended_node
        
#         # Otherwise, try to map current step to a node
#         current_step = state.get('current_call_step', CallStep.INTRODUCTION.value)
#         step_to_node = {
#             CallStep.INTRODUCTION.value: self.NODE_INTRODUCTION,
#             CallStep.NAME_VERIFICATION.value: self.NODE_NAME_VERIFICATION,
#             CallStep.DETAILS_VERIFICATION.value: self.NODE_DETAILS_VERIFICATION,
#             CallStep.REASON_FOR_CALL.value: self.NODE_REASON_FOR_CALL,
#             CallStep.NEGOTIATION.value: self.NODE_NEGOTIATION,
#             CallStep.PROMISE_TO_PAY.value: self.NODE_PROMISE_TO_PAY,
#             CallStep.DEBICHECK_SETUP.value: self.NODE_DEBICHECK_SETUP,
#             CallStep.PAYMENT_PORTAL.value: self.NODE_PAYMENT_PORTAL,
#             CallStep.SUBSCRIPTION_REMINDER.value: self.NODE_SUBSCRIPTION_REMINDER,
#             CallStep.CLIENT_DETAILS_UPDATE.value: self.NODE_CLIENT_DETAILS_UPDATE,
#             CallStep.REFERRALS.value: self.NODE_REFERRALS,
#             CallStep.FURTHER_ASSISTANCE.value: self.NODE_FURTHER_ASSISTANCE,
#             CallStep.CANCELLATION.value: self.NODE_CANCELLATION,
#             CallStep.ESCALATION.value: self.NODE_ESCALATION,
#             CallStep.CLOSING.value: self.NODE_CLOSING,
#             CallStep.QUERY_RESOLUTION.value: self.NODE_QUERY_HANDLER,
#             CallStep.END_CONVERSATION.value: self.NODE_END_CALL
#         }
        
#         if current_step in step_to_node:
#             return step_to_node[current_step]
        
#         # Default to introduction if all else fails
#         return self.NODE_INTRODUCTION
    
#     def _should_continue_simulation(self, state: CallCenterAgentState) -> str:
#         """
#         Determine if simulation should continue or revert to router.
        
#         Args:
#             state: Current agent state
                
#         Returns:
#             Next node to execute
#         """
#         # Check if call has ended
#         if state.get("is_call_ended", False):
#             logger.info("Call ended - stopping simulation")
#             return END
        
#         # Continue the conversation
#         return self.NODE_ROUTER
    
#     #########################################################################################
#     # Workflow Construction
#     #########################################################################################
#     def _build_workflow(self) -> StateGraph:
#         """
#         Build the complete LangGraph workflow with all call steps.
        
#         Returns:
#             Compiled StateGraph workflow
#         """
#         # Initialize state graph
#         workflow = StateGraph(CallCenterAgentState)
        
#         # Add central router node
#         workflow.add_node(self.NODE_ROUTER, self._router)
        
#         # Create process nodes for each call step
#         intro_node = self._create_introduction_node(self.introduction_agent)
#         verification_node = self._create_name_verification_node(self.name_verification_agent)
#         details_verification_node = self._create_details_verification_node(self.details_verification_agent)
        
#         # Create process nodes for BasicAgent-based steps
#         reason_for_call_node = self._create_basic_agent_node(
#             self.reason_for_call_agent, 
#             CallStep.REASON_FOR_CALL.value, 
#             CallStep.NEGOTIATION.value
#         )
        
#         negotiation_node = self._create_basic_agent_node(
#             self.negotiation_agent, 
#             CallStep.NEGOTIATION.value, 
#             CallStep.PROMISE_TO_PAY.value
#         )

#         # Add all nodes to the graph
#         workflow.add_node(self.NODE_INTRODUCTION, intro_node)
#         workflow.add_node(self.NODE_NAME_VERIFICATION, verification_node)
#         workflow.add_node(self.NODE_DETAILS_VERIFICATION, details_verification_node)
#         workflow.add_node(self.NODE_REASON_FOR_CALL, reason_for_call_node)
#         workflow.add_node(self.NODE_NEGOTIATION, negotiation_node)
        
#         # Set entry point to router
#         workflow.set_entry_point(self.NODE_ROUTER)
        
#         # Add router edges to all possible destinations
#         workflow.add_conditional_edges(
#             self.NODE_ROUTER,
#             self._router_condition,
#             {
#                 self.NODE_INTRODUCTION: self.NODE_INTRODUCTION,
#                 self.NODE_NAME_VERIFICATION: self.NODE_NAME_VERIFICATION,
#                 self.NODE_DETAILS_VERIFICATION: self.NODE_DETAILS_VERIFICATION,
#                 self.NODE_REASON_FOR_CALL: self.NODE_REASON_FOR_CALL,
#                 self.NODE_NEGOTIATION: self.NODE_NEGOTIATION,
#                 # Additional nodes would be added here in a complete implementation
#             }
#         )

#         # Build the workflow differently based on simulation mode
#         if self.simulate_debtor and self.debtor_agent:
#             # Add debtor simulation node
#             workflow.add_node(self.NODE_DEBTOR_SIMULATION, self._create_debtor_simulation_node(self.debtor_agent))
            
#             # Connect debtor simulation back to router
#             workflow.add_conditional_edges(
#                 self.NODE_DEBTOR_SIMULATION,
#                 self._should_continue_simulation,
#                 {
#                     END: END,
#                     self.NODE_ROUTER: self.NODE_ROUTER,
#                 }
#             )
            
#             if self.show_logs:
#                 logger.info("Workflow built with debtor simulation enabled")
#         else:
#             if self.show_logs:
#                 logger.info("Workflow built without debtor simulation")

#         # Compile with memory if configured
#         if self.config.get('configurable', {}).get('use_memory', False):
#             return workflow.compile(checkpointer=self.checkpointer)
#         else:
#             return workflow.compile()
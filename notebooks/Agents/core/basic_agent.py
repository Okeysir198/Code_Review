"""
Agent with tools and memory capabilities.

This module provides an agent that can use tools to solve problems and
maintain conversation context through memory persistence.
"""
from typing import List, Dict, Callable, Any, Literal, Optional, Union, cast, Sequence
import logging
from enum import Enum
from dataclasses import dataclass
import inspect

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import MessagesState
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.runnables import Runnable
from langgraph.utils.runnable import RunnableCallable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#########################################################################################
# Tools
#########################################################################################
@tool
def perform_calculation(expression: str) -> str:
    """Useful for when you need to do simple calculations."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"

# Initialize search tool with limited results for faster responses
search = TavilySearchResults(max_results=2)

# Default tools available to the agent
DEFAULT_TOOLS = [perform_calculation, search]
#########################################################################################
# Agent State
#########################################################################################
class NodeName(Enum):
    """Node names for the workflow graph."""
    PRE_PROCESSING = "pre_processing"
    AGENT = "agent"
    TOOLS = "tools"
    POST_PROCESSING = "post_processing"

@dataclass
class BasicAgentState(MessagesState):
    """State for the agent, extending MessagesState with system prompt."""
    system_prompt: Optional[str] = "You are helpful AI assistant"

# Define Prompt type
Prompt = Union[
    str,
    SystemMessage,
    ChatPromptTemplate,
    Callable[[BasicAgentState], List[BaseMessage]],
    Runnable
]

#########################################################################################
# Agent Implementation
#########################################################################################
class BasicAgent():
    """
    An agent that utilizes tools based on language model decisions within a LangGraph workflow.
    
    Features:
    - Process user inputs and generate coherent responses
    - Use tools to perform calculations and search for information
    - Maintain conversation context across multiple interactions
    - Support for custom pre-processing and post-processing nodes
    - Message context windowing to prevent token overflow
    - Support for runnable prompts for flexible system prompts
    - Optimized handling of messages that are already a list of BaseMessage objects
    """
    # Number of most recent messages to keep in context window
    CONTEXT_WINDOW_SIZE = 20

    def __init__(self, 
                 model: BaseChatModel,
                 tools: Optional[List[Union[Callable[..., Any], BaseTool]]] = None,
                 prompt: Optional[Prompt] = None,
                 checkpointer: Optional[Checkpointer] = None,
                 store: Optional[BaseStore] = None,
                 pre_processing_node: Optional[Callable] = None,
                 post_processing_node: Optional[Callable] = None,
                 verbose: bool = True,
                 config: Optional[Dict[str, Any]] = None,
                ) -> None:
        """
        Initialize the agent with tools and memory capabilities.
        
        Args:
            model: The language model to use for the agent
            tools: List of tools the agent can use
            prompt: Prompt to use for the agent (str, SystemMessage, ChatPromptTemplate, or Runnable)
            checkpointer: Component for persisting conversation state
            store: Component for cross-conversation memory
            pre_processing_node: Optional function to process state before the agent node
            post_processing_node: Optional function to process state after the agent node
            verbose: Whether to log detailed information
            config: Additional configuration parameters
        """
        
        # Core components
        self.llm = model
        self.tools = tools or []
        self.verbose = verbose
        self.context_window = self.CONTEXT_WINDOW_SIZE
        
        # Convert prompt to system prompt string or runnable
        self.prompt_runnable = self._get_prompt_runnable(prompt)
        
        # Bind tools to LLM if tools are provided
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else None
            
        # Setup memory components
        self.checkpointer = checkpointer
        if not self.checkpointer and config and config.get('configurable', {}).get('use_memory', False):
            self.checkpointer = MemorySaver()
            
        # Store for cross-conversation memory
        self.store = store
        
        # Custom processing nodes
        self.pre_processing_node = pre_processing_node
        self.post_processing_node = post_processing_node
        
        # Store configuration
        self.config = config or {}
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Log initialization details
        if self.verbose:
            self._log_initialization()

    def _log_initialization(self) -> None:
        """Log initialization details."""
        logger.info(f"Initializing BasicAgent:")
        logger.info(f"  - Tools: {len(self.tools) if self.tools else 0}")
        logger.info(f"  - Context window size: {self.context_window}")
        logger.info(f"  - Prompt type: {type(self.prompt_runnable).__name__}")
        
        if self.checkpointer:
            logger.info(f"  - Memory persistence: {type(self.checkpointer).__name__}")
        if self.store:
            logger.info(f"  - Cross-conversation store: {type(self.store).__name__}")
        if self.pre_processing_node:
            logger.info(f"  - Pre-processing: {self.pre_processing_node.__name__}")
        if self.post_processing_node:
            logger.info(f"  - Post-processing: {self.post_processing_node.__name__}")        
    #########################################################################################
    # Prepare prompt
    #########################################################################################
    def _get_prompt_runnable(self, prompt: Optional[Prompt]) -> Runnable:
        """Convert the provided prompt into a Runnable."""
        if prompt is None:
            # Default system prompt as a string
            return RunnableCallable(
                lambda state: self._prepare_messages_with_system_prompt(
                    state, "You are helpful AI assistant"
                ),
                name="DefaultPrompt"
            )
        elif isinstance(prompt, str):
            # String prompt - convert to system message
            return RunnableCallable(
                lambda state: self._prepare_messages_with_system_prompt(state, prompt),
                name="StringPrompt"
            )
        elif isinstance(prompt, SystemMessage):
            # System message prompt
            return RunnableCallable(
                lambda state: [prompt] + self._get_messages_from_state(state),
                name="SystemMessagePrompt"
            )
        elif isinstance(prompt, ChatPromptTemplate):
            # Chat prompt template
            return RunnableCallable(
                lambda state: prompt.format_messages(messages=self._get_messages_from_state(state)),
                name="TemplatePrompt"
            )
        elif inspect.iscoroutinefunction(prompt):
            # Async function prompt
            return RunnableCallable(
                None,
                prompt,
                name="AsyncFunctionPrompt"
            )
        elif callable(prompt):
            # Function prompt
            return RunnableCallable(
                prompt,
                name="FunctionPrompt"
            )
        elif isinstance(prompt, Runnable):
            # Already a Runnable
            return prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
    
    def _get_messages_from_state(self, state: Union[BasicAgentState, Dict[str, Any]]) -> List[BaseMessage]:
        """
        Extract messages from a state object, handling various formats.
        
        Args:
            state: Agent state or dictionary containing messages
            
        Returns:
            List of BaseMessage objects
        """
        if isinstance(state, dict):
            messages = state.get("messages", [])
            # Handle case where messages is already a list of BaseMessage
            if messages and isinstance(messages, list):
                if all(isinstance(m, BaseMessage) for m in messages):
                    return messages
                # Handle case where messages contains a single list of BaseMessages
                elif len(messages) == 1 and isinstance(messages[0], list) and all(isinstance(m, BaseMessage) for m in messages[0]):
                    return messages[0]
            return []
        else:
            # For MessagesState or BasicAgentState
            return state.messages if hasattr(state, "messages") else []
            
    def _prepare_messages_with_system_prompt(self, state: Union[BasicAgentState, Dict[str, Any]], system_prompt: str) -> List[BaseMessage]:
        """Prepare messages with a system prompt."""
        messages = self._get_messages_from_state(state)
        return [SystemMessage(content=system_prompt)] + messages
            
    #########################################################################################
    # Node functions and edge logics
    #########################################################################################
    def _process_with_llm(self, state: BasicAgentState) -> Dict[str, Any]:
        """
        Process the current conversation with the LLM and potentially call tools.
        
        Args:
            state: Current agent state containing messages and system prompt
            
        Returns:
            Updated state with new message from LLM
        """
        try:
            # Extract messages from state
            all_messages = self._get_messages_from_state(state)
            
            # Filter out system messages for context windowing
            non_system_messages = [m for m in all_messages if not isinstance(m, SystemMessage)]
            
            # Limit non-system messages to context window
            recent_messages = non_system_messages[-self.context_window:]
            
            # Use the prompt runnable to prepare messages
            messages = self.prompt_runnable.invoke({"messages": recent_messages})

            # Store the final prompt in state for debugging
            # Look for a system message in the prepared messages
            system_prompt = next((m.content for m in messages if isinstance(m, SystemMessage)), None)
            
            # Log the final prompt if verbose mode is enabled
            if self.verbose:
                logger.info("Final prompt to LLM:")
                for m in messages:
                    logger.info(f"{m.type}: {m.content[:100]}..." if len(m.content) > 100 else f"{m.type}: {m.content}")
            
            # Get response from LLM (with tools if available)
            if self.llm_with_tools:
                response = self.llm_with_tools.invoke(messages)
            else:
                response = self.llm.invoke(messages)
                
            return {
                "messages": [response],
                "system_prompt": system_prompt,
            }
        
        except Exception as e:
            if self.verbose:
                logger.error(f"Error in LLM processing: {e}")
            return {
                "messages": [AIMessage(content="I'm sorry, I encountered an error processing your request.")]
            }
    
    def _should_use_tools(self, state: Union[List[BaseMessage], Dict[str, Any]]) -> str:
        """
        Determine routing based on tool calls in the message.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of the next node to route to
        """
        # Extract messages from state
        messages = None
        if isinstance(state, list):
            messages = state
        elif isinstance(state, dict):
            if "messages" in state:
                messages_data = state["messages"]
                # Handle direct list of BaseMessage
                if isinstance(messages_data, list) and all(isinstance(m, BaseMessage) for m in messages_data):
                    messages = messages_data
                # Handle nested list of BaseMessage
                elif len(messages_data) == 1 and isinstance(messages_data[0], list) and all(isinstance(m, BaseMessage) for m in messages_data[0]):
                    messages = messages_data[0]
                # Single BaseMessage
                elif len(messages_data) == 1 and isinstance(messages_data[0], BaseMessage):
                    messages = messages_data
        
        if not messages:
            if self.verbose:
                logger.warning(f"No usable messages found in state: {state}")
            # Default to END if we can't find messages
            return END
            
        # Get the last AI message
        ai_message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        
        # If the message has tool calls, route to the tools node
        if ai_message and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            return NodeName.TOOLS.value
        
        # If there's a post-processing node, route to it; otherwise end
        if self.post_processing_node:
            return NodeName.POST_PROCESSING.value
        
        # Return the string literal "END" rather than the constant
        return END
    #########################################################################################
    # workflow
    #########################################################################################
    
    def _build_workflow(self) -> CompiledGraph:
        """
        Create the workflow graph with nodes and edges.
        
        Returns:
            Compiled graph ready for execution
        """
        if self.verbose:
            logger.debug("Building workflow graph")
            
        # Create the graph
        workflow = StateGraph(BasicAgentState)
        
        # Add the agent node (core LLM processing)
        workflow.add_node(NodeName.AGENT.value, self._process_with_llm)
        
        # Add pre-processing node if provided
        if self.pre_processing_node:
            workflow.add_node(NodeName.PRE_PROCESSING.value, self.pre_processing_node)
            workflow.add_edge(START, NodeName.PRE_PROCESSING.value)
            workflow.add_edge(NodeName.PRE_PROCESSING.value, NodeName.AGENT.value)
        else:
            workflow.add_edge(START, NodeName.AGENT.value)
        
        # Add tools node if tools are provided
        if self.tools:
            tool_executor = ToolNode(self.tools)
            workflow.add_node(NodeName.TOOLS.value, tool_executor)
            workflow.add_edge(NodeName.TOOLS.value, NodeName.AGENT.value)
        
        # Add post-processing node if provided
        if self.post_processing_node:
            workflow.add_node(NodeName.POST_PROCESSING.value, self.post_processing_node)
            workflow.add_edge(NodeName.POST_PROCESSING.value, END)
            
        # Configure conditional edges based on available nodes
        self._configure_conditional_edges(workflow)
        
        # Compile with the appropriate parameters
        compile_kwargs = {}
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer
        if self.store:
            compile_kwargs["store"] = self.store
            
        compiled_graph = workflow.compile(**compile_kwargs)
        
        if self.verbose:
            logger.info(f"Workflow graph built successfully")
            
        return compiled_graph
    
    def _configure_conditional_edges(self, workflow: StateGraph) -> None:
        """
        Configure conditional edges based on available nodes.
        
        Args:
            workflow: StateGraph to configure
        """
        # Different routing configurations based on available nodes
        if self.tools and self.post_processing_node:
            # Both tools and post-processing are present
            workflow.add_conditional_edges(
                NodeName.AGENT.value,
                self._should_use_tools,
                {
                    NodeName.TOOLS.value: NodeName.TOOLS.value,
                    NodeName.POST_PROCESSING.value: NodeName.POST_PROCESSING.value,
                }
            )
        elif self.tools:
            # Only tools are present
            workflow.add_conditional_edges(
                NodeName.AGENT.value,
                self._should_use_tools,
                {
                    NodeName.TOOLS.value: NodeName.TOOLS.value,
                    END: END
                }
            )
        elif self.post_processing_node:
            # Only post-processing is present
            workflow.add_edge(NodeName.AGENT.value, NodeName.POST_PROCESSING.value)
        else:
            # Neither tools nor post-processing are present
            workflow.add_edge(NodeName.AGENT.value, END)
    #########################################################################################
    # Invoke and stream
    #########################################################################################

    def _prepare_messages(self, input_message: Union[str, Dict, BaseMessage, List[BaseMessage]]) -> List[BaseMessage]:
        """
        Convert various input formats to a list of messages.
        
        Args:
            input_message: Input in various formats
            
        Returns:
            List of BaseMessage objects
        """
        # Already a list of BaseMessage objects
        if isinstance(input_message, list) and all(isinstance(m, BaseMessage) for m in input_message):
            return input_message
        # String input
        elif isinstance(input_message, str):
            return [HumanMessage(content=input_message)]
        # Dict with messages key
        elif isinstance(input_message, dict) and "messages" in input_message:
            messages = input_message["messages"]
            # Handle case where messages is already a list of BaseMessage
            if isinstance(messages, list):
                if all(isinstance(m, BaseMessage) for m in messages):
                    return messages
                # Handle case where messages contains a single list of BaseMessages
                elif len(messages) == 1 and isinstance(messages[0], list) and all(isinstance(m, BaseMessage) for m in messages[0]):
                    return messages[0]
            return []
        # Single BaseMessage
        elif isinstance(input_message, BaseMessage):
            return [input_message]
        else:
            raise ValueError(f"Unsupported input type: {type(input_message)}")
    
    def _prepare_config(self, thread_id: Optional[str], user_id: Optional[str]) -> Dict[str, Any]:
        """
        Prepare the configuration dictionary for workflow invocation.
        
        Args:
            thread_id: Optional thread ID for persistence
            user_id: Optional user ID for user-specific configuration
            
        Returns:
            Configuration dictionary
        """
        config = {"configurable": {}}
        if thread_id:
            config["configurable"]["thread_id"] = thread_id
        if user_id:
            config["configurable"]["user_id"] = user_id
        return config
    
    def invoke(self, 
               input_message: Union[str, Dict, BaseMessage, List[BaseMessage]], 
               thread_id: Optional[str] = None, 
               user_id: Optional[str] = None) -> Dict:
        """
        Process a user message through the agent workflow.
        
        Args:
            input_message: User input message (string, dict, BaseMessage, or list of BaseMessages)
            thread_id: Optional thread ID for persistence
            user_id: Optional user ID for user-specific configuration
            
        Returns:
            Response from the agent
        """
        try:
            messages = self._prepare_messages(input_message)
            config = self._prepare_config(thread_id, user_id)
            return self.workflow.invoke({"messages": messages}, config=config)
        except Exception as e:
            if self.verbose:
                logger.error(f"Error invoking workflow: {e}")
            messages = self._prepare_messages(input_message)
            return {"messages": messages + [AIMessage(content="I encountered an error processing your request.")]}
    
    def stream(self, 
               input_message: Union[str, Dict, BaseMessage, List[BaseMessage]], 
               thread_id: Optional[str] = None,
               user_id: Optional[str] = None,
               stream_mode: str = "values") -> Any:
        """
        Stream the agent's processing of a user message.
        
        Args:
            input_message: User input message (string, dict, BaseMessage, or list of BaseMessages)
            thread_id: Optional thread ID for persistence
            user_id: Optional user ID for user-specific configuration
            stream_mode: Mode for streaming ("values", "messages", etc.)
            
        Returns:
            Streaming iterator for agent responses
        """
        try:
            messages = self._prepare_messages(input_message)
            config = self._prepare_config(thread_id, user_id)
            return self.workflow.stream({"messages": messages}, config=config, stream_mode=stream_mode)
        except Exception as e:
            if self.verbose:
                logger.error(f"Error streaming workflow: {e}")
            messages = self._prepare_messages(input_message)
            
            def error_stream():
                yield {"messages": messages + [AIMessage(content="I encountered an error processing your request.")]}
            return error_stream()


"""
Optimized Basic Agent Function

This module provides a function that creates a compiled graph with tools and memory capabilities,
similar to create_react_agent but with enhanced functionality from the original BasicAgent class.
"""
from typing import List, Dict, Callable, Any, Optional, Union, Sequence, Type, TypeVar
import logging
from enum import Enum
from dataclasses import dataclass
import inspect

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import MessagesState
from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable, RunnableBinding, RunnableSequence
from langgraph.utils.runnable import RunnableCallable
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default tools available to the agent
@tool
def perform_calculation(expression: str) -> str:
    """Useful for when you need to do simple calculations."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"

# Initialize search tool with limited results for faster responses
search = TavilySearchResults(max_results=2)
DEFAULT_TOOLS = [perform_calculation, search]

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

# Define Prompt type - similar to create_react_agent
Prompt = Union[
    str,
    SystemMessage,
    ChatPromptTemplate,
    Callable[[BasicAgentState], Union[List[BaseMessage], str]],
    Runnable
]

StateSchema = TypeVar("StateSchema", bound=Union[BasicAgentState, Dict[str, Any]])

def _get_state_value(state: StateSchema, key: str, default: Any = None) -> Any:
    """Get value from state, handling both dict and object formats."""
    return (
        state.get(key, default)
        if isinstance(state, dict)
        else getattr(state, key, default)
    )

def _get_prompt_runnable(prompt: Optional[Prompt]) -> Runnable:
    """Convert the provided prompt into a Runnable with context windowing."""
    
    if prompt is None:
        return RunnableCallable(
            lambda state: state,
            name="DefaultPrompt"
        )
    elif isinstance(prompt, str):
        return RunnableCallable(
            lambda state: state,
            name="StringPrompt"
        )
    elif isinstance(prompt, SystemMessage):
        return RunnableCallable(
            lambda state: state,
            name="SystemMessagePrompt"
        )
    elif isinstance(prompt, ChatPromptTemplate):
        return prompt
    elif inspect.iscoroutinefunction(prompt):
        return RunnableCallable(
            None,
            prompt,
            name="AsyncFunctionPrompt"
        )
    elif callable(prompt):
        return RunnableCallable(
            prompt,
            name="FunctionPrompt"
        )
    elif isinstance(prompt, Runnable):
        return prompt
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

def _should_bind_tools(model: LanguageModelLike, tools: Sequence[BaseTool]) -> bool:
    """Check if tools should be bound to the model (adapted from create_react_agent)."""
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if not isinstance(model, RunnableBinding):
        return True

    if "tools" not in model.kwargs:
        return True

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools):
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_basic_agent must match"
        )

    tool_names = set(tool.name for tool in tools)
    bound_tool_names = set()
    for bound_tool in bound_tools:
        # OpenAI-style tool
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        # Anthropic-style tool
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            continue

        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in the model.bind_tools()")

    return False

def create_basic_agent(
    model: BaseChatModel,
    tools: Optional[List[Union[Callable[..., Any], BaseTool]]] = None,
    *,
    prompt: Optional[Prompt] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    pre_processing_node: Optional[Callable] = None,
    post_processing_node: Optional[Callable] = None,
    state_schema: Optional[Type[BasicAgentState]] = None,
    config_schema: Optional[Type[Any]] = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    debug: bool = False,
    verbose: bool = True,
    config: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> CompiledGraph:
    """
    Create a basic agent with tools and memory capabilities.

    Args:
        model: The language model to use for the agent
        tools: List of tools the agent can use
        prompt: Prompt to use for the agent (str, SystemMessage, ChatPromptTemplate, callable, or Runnable)
        checkpointer: Component for persisting conversation state
        store: Component for cross-conversation memory
        pre_processing_node: Optional function to process state before the agent node.
            Should return either:
            - Dict[str, Any]: Updates state and continues to agent (LLM) node
            - Command: Controls workflow routing and state updates
                * Command(update={...}, goto="__end__"): Skip LLM, end workflow
                * Command(update={...}, goto="tools"): Skip LLM, go to tools
                * Command(update={...}, goto="post_processing"): Skip LLM, go to post-processing
                * Command(update={...}, goto="agent"): Explicit routing to LLM (same as returning Dict)
        post_processing_node: Optional function to process state after the agent node.
            Should return Dict[str, Any] with state updates
        state_schema: Optional custom state schema (defaults to BasicAgentState)
        config_schema: Optional schema for configuration
        interrupt_before: List of node names to interrupt before
        interrupt_after: List of node names to interrupt after
        debug: Whether to enable debug mode
        verbose: Whether to log detailed information
        config: Additional configuration parameters
        name: Optional name for the compiled graph

    Returns:
        Compiled LangGraph that can be used for chat interactions

    Example:
        ```python
        # Pre-processing that continues to LLM
        def prep_node(state):
            return {"prepared_data": "some_value"}
        
        # Pre-processing that bypasses LLM
        def prep_node(state):
            if should_skip_llm():
                return Command(
                    update={"messages": [AIMessage(content="Response")]},
                    goto="__end__"
                )
            return {"prepared_data": "some_value"}
        
        agent = create_basic_agent(
            model=llm,
            pre_processing_node=prep_node,
            tools=[some_tool]
        )
        ```
    """
    # Validate inputs
    tools = tools or []
    config = config or {}
    
    # Setup state schema
    if state_schema is None:
        state_schema = BasicAgentState
    
    # Setup memory components
    if not checkpointer and config.get('configurable', {}).get('use_memory', False):
        checkpointer = MemorySaver()
    
    # Convert prompt to runnable
    prompt_runnable = _get_prompt_runnable(prompt)
    
    # Bind tools to LLM if tools are provided and should be bound
    tool_classes = [tool for tool in tools if isinstance(tool, BaseTool)]
    if _should_bind_tools(model, tool_classes) and tool_classes:
        model_with_tools = model.bind_tools(tool_classes)
    else:
        model_with_tools = model if tool_classes else None
    
    # Log initialization details
    if verbose:
        logger.info("Creating BasicAgent:")
        logger.info(f"  - Tools: {len(tools)}")
        logger.info(f"  - Prompt type: {type(prompt_runnable).__name__}")
        
        if checkpointer:
            logger.info(f"  - Memory persistence: {type(checkpointer).__name__}")
        if store:
            logger.info(f"  - Cross-conversation store: {type(store).__name__}")
        if pre_processing_node:
            logger.info(f"  - Pre-processing: {pre_processing_node.__name__}")
        if post_processing_node:
            logger.info(f"  - Post-processing: {post_processing_node.__name__}")

    def _process_with_llm(state: StateSchema) -> Dict[str, Any]:
        """
        Process the current conversation with the LLM and potentially call tools.
        
        Args:
            state: Current agent state containing messages and system prompt
            
        Returns:
            Updated state with new message from LLM
        """
        try:
            # Use the prompt runnable to get processed state or messages
            processed_result = prompt_runnable.invoke(state)
            
            # Extract and process messages after prompt_runnable.invoke
            if isinstance(processed_result, list) and all(isinstance(m, BaseMessage) for m in processed_result):
                # If prompt_runnable returned a list of messages directly
                messages = processed_result
            elif hasattr(processed_result, 'messages'):
                # If prompt_runnable returned a state-like object with messages
                raw_messages = processed_result.messages
            elif isinstance(processed_result, dict) and 'messages' in processed_result:
                # If prompt_runnable returned a dict with messages
                raw_messages = processed_result['messages']
            else:
                # Fallback: extract messages from original state
                if isinstance(state, dict):
                    raw_messages = state.get("messages", [])
                else:
                    raw_messages = getattr(state, "messages", [])
            
            # Process messages if we have raw_messages (not already processed)
            if 'raw_messages' in locals():
                # Handle nested message lists
                if raw_messages and isinstance(raw_messages, list):
                    if all(isinstance(m, BaseMessage) for m in raw_messages):
                        all_messages = raw_messages
                    elif (len(raw_messages) == 1 and isinstance(raw_messages[0], list) and 
                          all(isinstance(m, BaseMessage) for m in raw_messages[0])):
                        all_messages = raw_messages[0]
                    else:
                        all_messages = []
                else:
                    all_messages = []
                
                # Apply context windowing and add system prompt
                non_system_messages = [m for m in all_messages if not isinstance(m, SystemMessage)]
                
                # Add system prompt based on prompt type
                if isinstance(prompt, str):
                    system_prompt_content = prompt
                elif isinstance(prompt, SystemMessage):
                    system_prompt_content = prompt.content
                elif prompt is None:
                    system_prompt_content = "You are helpful AI assistant"
                else:
                    # For other prompt types, use default or extract from state
                    system_prompt_content = _get_state_value(state, "system_prompt", "You are helpful AI assistant")
                
                messages = [SystemMessage(content=system_prompt_content)] + non_system_messages
            
            # Store the final system prompt in state for debugging
            system_prompt = next((m.content for m in messages if isinstance(m, SystemMessage)), None)
            
            # Log the final prompt if verbose mode is enabled
            if verbose:
                logger.info(f"Final prompt to LLM: {system_prompt}")
                logger.info(f"Messages content as below:")
                for msg in messages:
                    logger.info(f"{msg.type.upper()}: {msg.content}")
                
                logger.info(f"Len of Messages: {len(messages)}")
            
            # Get response from LLM (with tools if available)
            if model_with_tools and tool_classes:
                response = model_with_tools.invoke(messages)
            else:
                response = model.invoke(messages)
                
            return {
                "messages": [response],
                "system_prompt": system_prompt,
            }
        
        except Exception as e:
            if verbose:
                logger.error(f"Error in LLM processing: {e}")
            return {
                "messages": [AIMessage(content="I'm sorry, I encountered an error processing your request.")]
            }

    def _should_use_tools(state: Union[BasicAgentState, Dict[str, Any]]) -> str:
        """
        Determine routing based on tool calls in the message.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of the next node to route to
        """
        # Extract messages from state
        messages = _get_state_value(state, "messages", [])
        
        if not messages:
            if verbose:
                logger.warning("No usable messages found in state")
            return END
                
        # Get the last AI message
        ai_message = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        
        # If the message has tool calls, route to the tools node
        if ai_message and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            return NodeName.TOOLS.value
        
        # If there's a post-processing node, route to it; otherwise end
        if post_processing_node:
            return NodeName.POST_PROCESSING.value
        
        return END

    def _configure_conditional_edges(workflow: StateGraph) -> None:
        """
        Configure conditional edges based on available nodes.
        
        Args:
            workflow: StateGraph to configure
        """
        if tools and post_processing_node:
            # Both tools and post-processing are present
            workflow.add_conditional_edges(
                NodeName.AGENT.value,
                _should_use_tools,
                {
                    NodeName.TOOLS.value: NodeName.TOOLS.value,
                    NodeName.POST_PROCESSING.value: NodeName.POST_PROCESSING.value,
                    END: END
                }
            )
        elif tools:
            # Only tools are present
            workflow.add_conditional_edges(
                NodeName.AGENT.value,
                _should_use_tools,
                {
                    NodeName.TOOLS.value: NodeName.TOOLS.value,
                    END: END
                }
            )
        elif post_processing_node:
            # Only post-processing is present
            workflow.add_edge(NodeName.AGENT.value, NodeName.POST_PROCESSING.value)
        else:
            # Neither tools nor post-processing are present
            workflow.add_edge(NodeName.AGENT.value, END)

    def _build_workflow() -> CompiledGraph:
        """
        Create the workflow graph with nodes and edges.
        
        Returns:
            Compiled graph ready for execution
        """
        if verbose:
            logger.debug("Building workflow graph")
            
        # Create the graph
        workflow = StateGraph(state_schema, config_schema=config_schema)
        
        # Add the agent node (core LLM processing)
        workflow.add_node(NodeName.AGENT.value, _process_with_llm)
        
        # Add pre-processing node if provided
        if pre_processing_node:
            workflow.add_node(NodeName.PRE_PROCESSING.value, pre_processing_node)
            workflow.add_edge(START, NodeName.PRE_PROCESSING.value)
        else:
            workflow.add_edge(START, NodeName.AGENT.value)
        
        # Add tools node if tools are provided
        if tools:
            tool_executor = ToolNode(tools)
            workflow.add_node(NodeName.TOOLS.value, tool_executor)
            workflow.add_edge(NodeName.TOOLS.value, NodeName.AGENT.value)
        
        # Add post-processing node if provided
        if post_processing_node:
            workflow.add_node(NodeName.POST_PROCESSING.value, post_processing_node)
            workflow.add_edge(NodeName.POST_PROCESSING.value, END)
            
        # Configure conditional edges based on available nodes
        _configure_conditional_edges(workflow)
        
        # Compile with the appropriate parameters
        compile_kwargs = {}
        if checkpointer:
            compile_kwargs["checkpointer"] = checkpointer
        if store:
            compile_kwargs["store"] = store
        if interrupt_before:
            compile_kwargs["interrupt_before"] = interrupt_before
        if interrupt_after:
            compile_kwargs["interrupt_after"] = interrupt_after
        if debug:
            compile_kwargs["debug"] = debug
        if name:
            compile_kwargs["name"] = name
            
        compiled_graph = workflow.compile(**compile_kwargs)
        
        if verbose:
            logger.info("Workflow graph built successfully")
            
        return compiled_graph

    # Build and return the compiled workflow
    return _build_workflow()

def create_agent_with_tools(
    model: BaseChatModel,
    tools: List[Union[Callable[..., Any], BaseTool]],
    **kwargs
) -> CompiledGraph:
    """
    Convenience function to create an agent with tools.
    
    Args:
        model: The language model to use
        tools: List of tools for the agent
        **kwargs: Additional arguments passed to create_basic_agent
        
    Returns:
        Compiled graph with tools enabled
    """
    return create_basic_agent(model=model, tools=tools, **kwargs)

def create_simple_agent(
    model: BaseChatModel,
    prompt: Optional[Prompt] = None,
    **kwargs
) -> CompiledGraph:
    """
    Convenience function to create a simple agent without tools.
    
    Args:
        model: The language model to use
        prompt: Optional prompt for the agent
        **kwargs: Additional arguments passed to create_basic_agent
        
    Returns:
        Compiled graph without tools
    """
    return create_basic_agent(model=model, tools=[], prompt=prompt, **kwargs)

# Backward compatibility - keep the original class for existing code
class BasicAgent:
    """
    DEPRECATED: Use create_basic_agent() function instead.
    This class is kept for backward compatibility.
    """
    
    def __init__(self, model: BaseChatModel, **kwargs):
        logger.warning("BasicAgent class is deprecated. Use create_basic_agent() function instead.")
        self.workflow = create_basic_agent(model=model, **kwargs)
    
    def invoke(self, input_message, thread_id=None, user_id=None):
        """Invoke the agent workflow."""
        messages = self._prepare_messages(input_message)
        config = self._prepare_config(thread_id, user_id)
        return self.workflow.invoke({"messages": messages}, config=config)
    
    def stream(self, input_message, thread_id=None, user_id=None, stream_mode="values"):
        """Stream the agent workflow."""
        messages = self._prepare_messages(input_message)
        config = self._prepare_config(thread_id, user_id)
        return self.workflow.stream({"messages": messages}, config=config, stream_mode=stream_mode)
    
    def _prepare_messages(self, input_message):
        """Convert various input formats to a list of messages."""
        if isinstance(input_message, list) and all(isinstance(m, BaseMessage) for m in input_message):
            return input_message
        elif isinstance(input_message, str):
            return [HumanMessage(content=input_message)]
        elif isinstance(input_message, BaseMessage):
            return [input_message]
        else:
            raise ValueError(f"Unsupported input type: {type(input_message)}")
    
    def _prepare_config(self, thread_id, user_id):
        """Prepare configuration for workflow invocation."""
        config = {"configurable": {}}
        if thread_id:
            config["configurable"]["thread_id"] = thread_id
        if user_id:
            config["configurable"]["user_id"] = user_id
        return config

# Export the main function and convenience functions
__all__ = [
    "create_basic_agent",
    "create_agent_with_tools", 
    "create_simple_agent",
    "BasicAgent",  # For backward compatibility
    "BasicAgentState",
    "DEFAULT_TOOLS"
]
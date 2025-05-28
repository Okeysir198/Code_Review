# ./src/Agents/call_center_agent/step12_query_resolution.py
"""
Query Resolution Agent - Handles client questions and redirects to main goal.
SIMPLIFIED: Just handles queries, router manages entry/exit.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import add_client_note


def create_query_resolution_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a query resolution agent for handling client questions."""
    
    agent_tools = [add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Simple pre-processing - no routing logic."""
        return Command(update={}, goto="agent")

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for query resolution step."""
        
        # Get client info
        profile = client_data.get('profile', {})
        client_info = profile.get('client_info', {})
        client_name = client_info.get('client_full_name', 'Client')
        
        # Custom prompt for query resolution with brief answers and redirection
        query_prompt = f"""<role>
You are {agent_name}, a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<context>
- Client VERIFIED: {client_name}
- Outstanding Amount: {state.get('outstanding_amount', 'amount due')}
- Current step: Query Resolution
- Return to: {state.get('return_to_step', 'payment discussion')}
</context>

<task>
Answer the client's question BRIEFLY (under 15 words) then redirect to payment resolution.
</task>

<approach>
1. Give a direct, helpful answer to their question
2. Immediately redirect: "Now, regarding your account payment..."
3. Keep responses conversational and natural
4. Stay focused on securing payment
</approach>

<examples>
Client: "Why wasn't my payment taken?"
You: "Bank declined it. Now, regarding your account payment, can we arrange immediate settlement?"

Client: "What happens if I don't pay?"
You: "Services suspend and fees apply. Let's arrange payment today to avoid that."

Client: "When is this due?"
You: "It's overdue now. Can we debit the outstanding amount from your account today?"

Client: "How does Cartrack work?"
You: "Vehicle tracking and security. Now, can we settle your {state.get('outstanding_amount', 'account balance')} today?"
</examples>

<style>
- Maximum 20 words total
- Answer briefly, then redirect immediately
- Don't get sidetracked from payment goal
- Natural, conversational tone
</style>

<objective>
Answer briefly, then secure payment arrangement immediately.
</objective>"""
        
        return [SystemMessage(content=query_prompt)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        # pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        name="QueryResolutionAgent"
    )
# src/Agents/call_center_agent/step10_referrals.py
"""
Referrals Agent - Self-contained with own prompt
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

def get_referrals_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate referrals specific prompt."""
    
    return f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<task>
Briefly mention referral program. MAXIMUM 15 words.
</task>

<approach>
"Do you know anyone interested in Cartrack? Successful referrals earn you 2 months free subscription."
</approach>

<benefits>
- 2 months free subscription
- Help friends with vehicle security
- Easy referral process
</benefits>

<style>
- MAXIMUM 15 words
- Present as benefit to client
- No pressure if not interested
- Quick mention only
</style>"""

def create_referrals_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a referrals agent for debt collection calls."""
    
    agent_tools = [add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to prepare referral information only."""
        
        return Command(
            update={"current_step": CallStep.REFERRALS.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for referrals step."""
        prompt_content = get_referrals_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ReferralsAgent"
    )
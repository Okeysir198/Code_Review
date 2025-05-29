# ./src/Agents/call_center_agent/step13_escalation.py
"""
Escalation Agent - Handles escalation requests professionally.
SIMPLIFIED: Keep ticket logic, no query detection.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command
import uuid
from datetime import datetime

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    add_client_note,
    save_call_disposition
)


def create_escalation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create an escalation agent."""
    
    agent_tools = [add_client_note, save_call_disposition] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Pre-process to create escalation ticket and prepare response."""
        
        import uuid
        from datetime import datetime
        
        # Generate ticket details
        ticket_number = f"ESC{datetime.now().strftime('%Y%m%d%H%M')}{str(uuid.uuid4())[:4].upper()}"
        department = "Supervisor"
        response_time = "24-48 hours"
        
        # Add escalation note
        user_id = client_data.get("user_id")
        if user_id:
            try:
                add_client_note.invoke({
                    "user_id": user_id,
                    "note_text": f"Call escalated to supervisor. Ticket: {ticket_number}"
                })
            except Exception as e:
                if verbose:
                    print(f"Error adding escalation note: {e}")
        
        return Command(
            update={
                "ticket_number": ticket_number,
                "department": department,
                "response_time": response_time,
                "escalation_requested": True,
                "current_step": CallStep.ESCALATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.ESCALATION.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        prompt_content = get_step_prompt(CallStep.ESCALATION.value, parameters)
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EscalationAgent"
    )
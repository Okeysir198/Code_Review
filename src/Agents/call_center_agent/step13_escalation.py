# ===============================================================================
# STEP 13: ESCALATION AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step13_escalation.py
"""
Escalation Agent - Enhanced with aging-aware script integration
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
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

from src.Database.CartrackSQLDatabase import (
    add_client_note,
    save_call_disposition
)

def get_escalation_prompt(client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Generate aging-aware escalation prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Get escalation details
    ticket_number = state.get("ticket_number", "ESC12345")
    department = state.get("department", "Supervisor")
    response_time = state.get("response_time", "24-48 hours")
    
    # Build aging-specific escalation approaches
    escalation_approaches_by_category = {
        "First Missed Payment": {
            "department": "Customer Service Supervisor",
            "response_time": "24-48 hours",
            "tone": "understanding and professional",
            "message": "I understand your concern. I'm escalating this to our Customer Service Supervisor."
        },
        "Failed Promise to Pay": {
            "department": "Accounts Supervisor",
            "response_time": "24 hours",
            "tone": "professional and solution-focused",
            "message": "I understand your situation. I'm escalating this to our Accounts Supervisor."
        },
        "2-3 Months Overdue": {
            "department": "Senior Collections Manager",
            "response_time": "24 hours",
            "tone": "serious but professional",
            "message": "I understand your concern. I'm escalating this to our Senior Collections Manager."
        },
        "Pre-Legal 120+ Days": {
            "department": "Pre-Legal Department Manager",
            "response_time": "12-24 hours",
            "tone": "formal and urgent",
            "message": "I understand this is serious. I'm escalating this to our Pre-Legal Department Manager."
        },
        "Legal 150+ Days": {
            "department": "Legal Department",
            "response_time": "12 hours",
            "tone": "formal legal authority",
            "message": "I understand your concern about legal proceedings. I'm escalating this to our Legal Department."
        }
    }
    
    category = aging_context['category']
    escalation_approach = escalation_approaches_by_category.get(category, escalation_approaches_by_category["First Missed Payment"])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist from Cartrack.
</role>

<context>
- Aging Category: {category}
- Urgency Level: {aging_context['urgency']}
- Ticket Number: {ticket_number}
- Department: {escalation_approach['department']}
- Response Time: {escalation_approach['response_time']}
</context>

<task>
Handle escalation professionally using aging-appropriate authority and timeline.
</task>

<aging_specific_approach>
**Message**: "{escalation_approach['message']}"
**Department**: "{escalation_approach['department']}"
**Response Time**: "{escalation_approach['response_time']}"
**Tone**: "{escalation_approach['tone']}"
</aging_specific_approach>

<escalation_process>
1. Acknowledge concern with appropriate gravity
2. Create ticket reference: {ticket_number}
3. Set realistic expectations for response time
4. Professional handoff to appropriate authority level
</escalation_process>

<urgency_context>
{aging_context['approach']}
</urgency_context>

<style>
- {aging_context['tone']} with {escalation_approach['tone']}
- Validate client concern appropriately
- Clear communication of next steps
- Professional resolution matching account status
- Authority level appropriate to aging category
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.ESCALATION,
        client_data=client_data,
        state=state
    )

def create_escalation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create an escalation agent with aging-aware scripts."""
    
    agent_tools = [add_client_note, save_call_disposition] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Determine script type and appropriate escalation level
        account_aging = client_data.get("account_aging", {})
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        # Generate ticket details based on aging category
        escalation_mapping = {
            "First Missed Payment": ("Customer Service Supervisor", "24-48 hours"),
            "Failed Promise to Pay": ("Accounts Supervisor", "24 hours"),
            "2-3 Months Overdue": ("Senior Collections Manager", "24 hours"),
            "Pre-Legal 120+ Days": ("Pre-Legal Department Manager", "12-24 hours"),
            "Legal 150+ Days": ("Legal Department", "12 hours")
        }
        
        category = aging_context['category']
        department, response_time = escalation_mapping.get(category, ("Supervisor", "24-48 hours"))
        
        ticket_number = f"ESC{datetime.now().strftime('%Y%m%d%H%M')}{str(uuid.uuid4())[:4].upper()}"
        
        # Add escalation note
        user_id = client_data.get("user_id")
        if user_id:
            try:
                add_client_note.invoke({
                    "user_id": user_id,
                    "note_text": f"Call escalated to {department}. Ticket: {ticket_number}. Category: {category}"
                })
            except Exception as e:
                if verbose:
                    print(f"Error adding escalation note: {e}")
        
        return Command(
            update={
                "ticket_number": ticket_number,
                "department": department,
                "response_time": response_time,
                "aging_category": category,
                "urgency_level": aging_context['urgency'].lower(),
                "escalation_requested": True,
                "current_step": CallStep.ESCALATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_escalation_prompt(client_data, state.to_dict() if hasattr(state, 'to_dict') else state)
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

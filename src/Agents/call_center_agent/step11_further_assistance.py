# ===============================================================================
# STEP 11: FURTHER ASSISTANCE AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step11_further_assistance.py
"""
Further Assistance Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

def get_further_assistance_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware further assistance prompt."""
    
    # Determine script type from aging
    user_id = client_data["profile"]["user_id"]     
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Build aging-specific assistance approaches
    assistance_approaches_by_category = {
        "First Missed Payment": {
            "offer": "Is there anything else regarding your account I can help you with today?",
            "focus": "General account support and service questions"
        },
        "Failed Promise to Pay": {
            "offer": "Is there anything else about your payment arrangement I can clarify?",
            "focus": "Payment-related questions and commitment confirmation"
        },
        "2-3 Months Overdue": {
            "offer": "Any other urgent account matters I can address today?",
            "focus": "Priority issues that might affect payment or escalation"
        },
        "Pre-Legal 120+ Days": {
            "offer": "Any final questions before we conclude this call?",
            "focus": "Critical issues only, time-sensitive matters"
        },
        "Legal 150+ Days": {
            "offer": "Any questions about the legal proceedings or payment requirements?",
            "focus": "Legal clarifications and immediate payment concerns only"
        }
    }
    
    category = aging_context['category']
    assistance_approach = assistance_approaches_by_category.get(category, assistance_approaches_by_category["First Missed Payment"])
    
    # Build urgency-appropriate follow-up scope
    follow_up_scope_by_urgency = {
        "Medium": ["Account questions", "Service issues", "Payment queries", "General assistance"],
        "High": ["Payment concerns", "Service restoration", "Account status", "Urgent matters"],
        "Very High": ["Payment arrangements", "Escalation questions", "Critical issues only"],
        "Critical": ["Legal clarifications", "Payment requirements", "Immediate concerns only"]
    }
    
    urgency_level = aging_context['urgency']
    follow_up_scope = follow_up_scope_by_urgency.get(urgency_level, follow_up_scope_by_urgency["Medium"])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
</role>

<context>
- Aging Category: {category}
- Urgency Level: {urgency_level}
- Client user_id: {user_id}
</context>

<task>
Check for additional concerns appropriate to account status and urgency level.
</task>

<aging_specific_approach>
**Offer**: "{assistance_approach['offer']}"
**Focus Areas**: "{assistance_approach['focus']}"
</aging_specific_approach>

<appropriate_follow_up_scope>
{chr(10).join([f"- {scope}" for scope in follow_up_scope])}
</appropriate_follow_up_scope>

<urgency_context>
{aging_context['approach']}
</urgency_context>

<time_management>
- {urgency_level} urgency: {"Allow time for questions" if urgency_level == "Medium" else "Keep focused on priorities" if urgency_level == "High" else "Minimal additional time" if urgency_level == "Very High" else "Conclude efficiently"}
</time_management>

<style>
- {aging_context['tone']}
- Genuine concern for client needs appropriate to urgency
- Complete resolution focus
- Professional care matching account status
- Time-appropriate for {urgency_level.lower()} priority
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.FURTHER_ASSISTANCE,
        client_data=client_data,
        state=state
    )

def create_further_assistance_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a further assistance agent with aging-aware scripts."""
    

    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Determine script type and urgency
        account_aging = client_data.get("account_aging", {})
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        return Command(
            update={
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
                "assistance_offer": "Is there anything else regarding your account I can help you with?",
                "current_step": CallStep.FURTHER_ASSISTANCE.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_further_assistance_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(f"Prompt: {prompt_content}")
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="FurtherAssistanceAgent"
    )

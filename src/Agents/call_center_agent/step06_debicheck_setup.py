# src/Agents/call_center_agent/step06_debicheck_setup.py
"""
DebiCheck Setup Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import calculate_outstanding_amount, format_currency,get_safe_value
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

def get_debicheck_setup_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware DebiCheck setup prompt."""
    
    # Determine script type from aging
    user_id = get_safe_value(client_data, "profile.user_id", "")     
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Calculate amounts
    outstanding_float = calculate_outstanding_amount(account_aging)
    total_with_fee = outstanding_float + 10
    amount_with_fee = format_currency(total_with_fee)
    outstanding_amount = format_currency(outstanding_float)
    
    # Build aging-specific DebiCheck explanations
    debicheck_explanations_by_category = {
        "First Missed Payment": {
            "urgency": "to restore your services immediately",
            "timeline": "You'll receive this within a few minutes",
            "action": "Please approve this request to restore your vehicle protection"
        },
        "Failed Promise to Pay": {
            "urgency": "to honor your payment commitment",
            "timeline": "You'll receive this immediately",
            "action": "You must approve this request to fulfill your previous agreement"
        },
        "2-3 Months Overdue": {
            "urgency": "to prevent escalation to legal action",
            "timeline": "You'll receive this within minutes",
            "action": "You must approve this immediately to prevent account escalation"
        },
        "Pre-Legal 120+ Days": {
            "urgency": "to prevent court proceedings",
            "timeline": "You'll receive this immediately",
            "action": "You must approve this now to stop legal action"
        },
        "Legal 150+ Days": {
            "urgency": "to stop legal proceedings immediately",
            "timeline": "You'll receive this within seconds",
            "action": "Immediate approval required to prevent court judgment"
        }
    }
    
    category = aging_context['category']
    explanation = debicheck_explanations_by_category.get(category, debicheck_explanations_by_category["First Missed Payment"])
    
    # Build urgency-based process emphasis
    process_emphasis_by_urgency = {
        "Medium": "This is a secure, standard banking process",
        "High": "This is urgent - your bank requires immediate authorization",
        "Very High": "Critical authorization required - bank authentication needed now",
        "Critical": "Emergency authorization - immediate bank approval required"
    }
    
    urgency_level = aging_context['urgency']
    process_emphasis = process_emphasis_by_urgency.get(urgency_level, process_emphasis_by_urgency["Medium"])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
</role>

<context>
- Outstanding: {outstanding_amount}
- Total with fee: {amount_with_fee}
- Aging Category: {category}
- Urgency Level: {urgency_level}
- Client user_id: {user_id}
</context>

<task>
Explain DebiCheck process using aging-appropriate urgency and ensure client understanding.
</task>

<aging_specific_explanation>
1. **Purpose**: "Your bank will send an authentication request {explanation['urgency']}"
2. **Timeline**: "{explanation['timeline']} via your banking app or SMS"  
3. **Action Required**: "{explanation['action']}"
4. **Amount**: "Total amount will be {amount_with_fee} including R10 processing fee"
</aging_specific_explanation>

<process_emphasis>
{process_emphasis}
</process_emphasis>

<urgency_messaging>
{aging_context['approach']}
</urgency_messaging>

<critical_points>
- Bank authentication is required by law
- Client must approve the request
- Processing fee is standard (R10)
- {explanation['urgency']}
</critical_points>

<style>
- {aging_context['tone']}
- Clear, step-by-step guidance appropriate to urgency
- Professional confidence matching account severity
- Ensure understanding without overwhelming
- {urgency_level.lower()} priority messaging
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.DEBICHECK_SETUP,
        client_data=client_data,
        state=state
    )

def create_debicheck_setup_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a DebiCheck setup agent with aging-aware scripts."""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Get outstanding amount and calculate total with fee
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        mandate_fee = 10.0
        total_amount = outstanding_amount + mandate_fee
        
        # Determine script type and urgency
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        return Command(
            update={
                "amount_with_fee": format_currency(total_amount),
                "mandate_fee": mandate_fee,
                "outstanding_float": outstanding_amount,
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
                "process_explanation": "Your bank will send an authentication request",
                "current_step": CallStep.DEBICHECK_SETUP.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_debicheck_setup_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
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
        name="DebiCheckSetupAgent"
    )
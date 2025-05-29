# src/Agents/call_center_agent/step08_subscription_reminder.py
"""
Subscription Reminder Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import calculate_outstanding_amount, format_currency
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

from src.Database.CartrackSQLDatabase import get_client_subscription_amount, add_client_note

def get_subscription_reminder_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware subscription reminder prompt."""
    
    # Determine script type from aging
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Calculate outstanding amount
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Get subscription amount
    subscription = client_data.get("subscription", {})
    subscription_amount = subscription.get("subscription_amount", "199.00")
    try:
        subscription_str = format_currency(float(subscription_amount))
    except (ValueError, TypeError):
        subscription_str = "R 199.00"
    
    # Build aging-specific clarification messages
    clarification_by_category = {
        "First Missed Payment": {
            "arrears_explanation": f"Today's {outstanding_amount} covers your missed subscription payment",
            "ongoing_explanation": f"Your regular {subscription_str} continues on the 5th of each month",
            "reassurance": "This is not double charging - just catching up on the missed payment"
        },
        "Failed Promise to Pay": {
            "arrears_explanation": f"Today's {outstanding_amount} covers the payment you previously committed to",
            "ongoing_explanation": f"Your regular {subscription_str} continues as scheduled",
            "reassurance": "This fulfills your previous payment arrangement plus ongoing subscription"
        },
        "New Installation Pro-Rata": {
            "arrears_explanation": f"Today's {outstanding_amount} covers pro-rata charges since installation",
            "ongoing_explanation": f"Your regular {subscription_str} starts next billing cycle",
            "reassurance": "Pro-rata is separate from your regular monthly subscription"
        },
        "2-3 Months Overdue": {
            "arrears_explanation": f"Today's {outstanding_amount} covers multiple months of overdue subscription",
            "ongoing_explanation": f"Your regular {subscription_str} continues once account is current",
            "reassurance": "Payment brings account current, then normal billing resumes"
        },
        "Pre-Legal 120+ Days": {
            "arrears_explanation": f"Today's {outstanding_amount} settles all overdue amounts to prevent legal action",
            "ongoing_explanation": f"Your regular {subscription_str} resumes once account is cleared",
            "reassurance": "Settlement payment prevents legal costs and restores normal billing"
        },
        "Legal 150+ Days": {
            "arrears_explanation": f"Today's {outstanding_amount} satisfies the legal debt demand",
            "ongoing_explanation": f"Regular {subscription_str} can resume once legal matter is resolved",
            "reassurance": "Payment resolves legal matter and can restore normal service billing"
        }
    }
    
    category = aging_context['category']
    clarification = clarification_by_category.get(category, clarification_by_category["First Missed Payment"])
    
    # Build urgency-appropriate messaging tone
    messaging_tone_by_urgency = {
        "Medium": "helpful and educational",
        "High": "clear and reassuring",
        "Very High": "direct and solution-focused",
        "Critical": "factual and immediate"
    }
    
    urgency_level = aging_context['urgency']
    messaging_tone = messaging_tone_by_urgency.get(urgency_level, messaging_tone_by_urgency["Medium"])
    
    # Base prompt
    base_prompt = f"""<role>
You are {agent_name}, a professional debt collection specialist at Cartrack's Accounts Department.
</role>

<context>
- Outstanding: {outstanding_amount}
- Regular Subscription: {subscription_str}
- Aging Category: {category}
- Urgency Level: {urgency_level}
</context>

<task>
Clarify arrears vs ongoing subscription using aging-appropriate tone and detail level.
</task>

<aging_specific_clarification>
**Arrears Payment**: "{clarification['arrears_explanation']}"
**Ongoing Subscription**: "{clarification['ongoing_explanation']}"
**Client Reassurance**: "{clarification['reassurance']}"
</aging_specific_clarification>

<key_differentiation>
- Arrears payment = resolving past overdue amounts
- Regular subscription = ongoing monthly service fee
- Two separate things with different purposes
- Prevents confusion about double charging
</key_differentiation>

<messaging_approach>
Use {messaging_tone} tone appropriate for {urgency_level.lower()} urgency situation
</messaging_approach>

<urgency_context>
{aging_context['approach']}
</urgency_context>

<style>
- {aging_context['tone']}
- Clear differentiation between payments
- Prevent double-payment confusion
- Professional explanation matching account urgency
- Concise but complete clarification
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.SUBSCRIPTION_REMINDER,
        client_data=client_data,
        state=state
    )

def create_subscription_reminder_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a subscription reminder agent with aging-aware scripts."""
    
    agent_tools = [get_client_subscription_amount, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Get subscription information
        subscription = client_data.get("subscription", {})
        subscription_amount = subscription.get("subscription_amount", "199.00")
        
        try:
            subscription_str = format_currency(float(subscription_amount))
        except (ValueError, TypeError):
            subscription_str = "R 199.00"
        
        # Determine script type and urgency
        account_aging = client_data.get("account_aging", {})
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        return Command(
            update={
                "subscription_amount": subscription_str,
                "subscription_date": "5th of each month",
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
                "clarification_message": "Today's payment covers arrears, regular subscription continues",
                "current_step": CallStep.SUBSCRIPTION_REMINDER.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_subscription_reminder_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(f"Prompt: {prompt_content}")
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="SubscriptionReminderAgent"
    )
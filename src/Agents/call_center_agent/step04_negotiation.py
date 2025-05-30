# src/Agents/call_center_agent/step04_negotiation.py
"""
Negotiation Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value, calculate_outstanding_amount, format_currency
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

def get_negotiation_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware negotiation prompt."""
    
    # Determine script type from aging
    user_id = get_safe_value(client_data, "profile.user_id", "")     
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Extract client info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Build aging-specific consequences and benefits
    consequences_by_category = {
        "First Missed Payment": "Your tracking stops working and you lose vehicle security",
        "Failed Promise to Pay": "Services remain suspended and your reliability record is affected",
        "New Installation Pro-Rata": "Services won't activate until payment - your vehicle remains unprotected",
        "2-3 Months Overdue": "Services suspended, potential R25,000 recovery fee if stolen, credit listing risk",
        "2-3 Months Failed PTP": "Legal action consideration, significant recovery fees, credit bureau listing",
        "Pre-Legal 120+ Days": "Credit default listing (R1800 to clear), legal action, attorney fees, court costs",
        "Legal 150+ Days": "Immediate legal proceedings, sheriff service, court judgment, asset attachment"
    }
    
    benefits_by_category = {
        "First Missed Payment": "Pay now and everything works immediately - vehicle protection restored",
        "Failed Promise to Pay": "Honor commitment now and restore service immediately",
        "New Installation Pro-Rata": "Payment activates all your Cartrack services immediately",
        "2-3 Months Overdue": "Immediate payment prevents legal action and restores full protection",
        "2-3 Months Failed PTP": "Settle now to avoid legal costs and restore account standing",
        "Pre-Legal 120+ Days": "Settlement today prevents court action and clears credit record",
        "Legal 150+ Days": "Immediate payment stops legal proceedings and prevents additional costs"
    }
    
    category = aging_context['category']
    specific_consequences = consequences_by_category.get(category, consequences_by_category["First Missed Payment"])
    specific_benefits = benefits_by_category.get(category, benefits_by_category["First Missed Payment"])
    
    # Build aging-specific objection responses
    objection_responses_by_urgency = {
        "Medium": {
            "no_money": "I understand. What amount can you manage today to keep services active?",
            "dispute_amount": "Let's verify while arranging payment to prevent service suspension",
            "will_pay_later": "Services suspend today without payment. Can we arrange something now?",
            "already_paid": "When was this paid? I need to locate it and arrange immediate payment"
        },
        "High": {
            "no_money": "I understand finances are tight. Even partial payment prevents escalation. What can you manage?",
            "dispute_amount": "We can investigate while securing payment to prevent legal action",
            "will_pay_later": "Your account is escalating to legal. We need arrangement today to prevent this",
            "already_paid": "I need payment details immediately to prevent this escalating further"
        },
        "Very High": {
            "no_money": "This is pre-legal. Even R50 shows commitment and can delay proceedings. What can you do?",
            "dispute_amount": "Disputes must be settled before legal action. Payment required to investigate",
            "will_pay_later": "Legal action is imminent. Only immediate payment can prevent court proceedings",
            "already_paid": "Provide proof immediately or payment required to stop legal action"
        },
        "Critical": {
            "no_money": "This is a legal demand. Non-payment means court proceedings. What assets can you access?",
            "dispute_amount": "Legal proceedings don't stop for disputes. Payment required to prevent judgment",
            "will_pay_later": "Court date is set. Only immediate payment stops legal proceedings",
            "already_paid": "Provide bank proof now or face legal consequences"
        }
    }
    
    urgency_level = aging_context['urgency']
    objection_responses = objection_responses_by_urgency.get(urgency_level, objection_responses_by_urgency["Medium"])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
</role>

<client_context>
- Client: {client_full_name}
- Outstanding: {outstanding_amount}
- Account Category: {category}
- Urgency Level: {urgency_level}
- Client user_id: {user_id}
</client_context>

<task>
Handle objections and explain consequences using aging-appropriate urgency and tone.
</task>

<consequences_framework>
**Without Payment**: "{specific_consequences}"
**With Payment**: "{specific_benefits}"
</consequences_framework>

<aging_specific_objection_responses>
{chr(10).join([f"- '{key}': '{value}'" for key, value in objection_responses.items()])}
</aging_specific_objection_responses>

<urgency_adaptation>
{aging_context['approach']}
</urgency_adaptation>

<escalation_indicators>
- Financial hardship claims
- Repeated delays or promises
- Aggressive resistance
- Dispute of debt validity
</escalation_indicators>

<style>
- {aging_context['tone']}
- Adapt pressure to urgency level
- Focus on solutions, not problems
- Create urgency appropriate to account status
- Maximum impact per response
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.NEGOTIATION,
        client_data=client_data,
        state=state
    )

def create_negotiation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a negotiation agent with aging-aware scripts."""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        
        # Determine script type and urgency
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        urgency_level = aging_context['urgency'].lower()
        
        return Command(
            update={
                "outstanding_float": outstanding_amount,
                "urgency_level": urgency_level,
                "aging_category": aging_context['category'],
                "negotiation_approach": aging_context['approach'],
                "current_step": CallStep.NEGOTIATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_negotiation_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
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
        name="NegotiationAgent"
    )
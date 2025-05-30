# src/Agents/call_center_agent/step07_payment_portal.py
"""
Payment Portal Agent - Enhanced with aging-aware script integration
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.data.client_data_fetcher import calculate_outstanding_amount, format_currency, get_safe_value
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

def get_payment_portal_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware payment portal prompt."""
    
    # Determine script type from aging
    user_id = get_safe_value(client_data, "profile.user_id", "")     
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Calculate outstanding amount
    outstanding_float = calculate_outstanding_amount(account_aging)
    outstanding_amount = format_currency(outstanding_float)
    
    # Build aging-specific portal guidance
    portal_guidance_by_category = {
        "First Missed Payment": {
            "intro": f"I'll send you a secure payment link for {outstanding_amount}",
            "urgency": "You can pay while we're on the call to restore services immediately",
            "support": "I'll stay on the line to help if needed"
        },
        "Failed Promise to Pay": {
            "intro": f"I'm sending an immediate payment link for {outstanding_amount}",
            "urgency": "Complete this now to honor your previous commitment",
            "support": "I'll guide you through each step to ensure completion"
        },
        "2-3 Months Overdue": {
            "intro": f"Emergency payment link for {outstanding_amount} being sent",
            "urgency": "Complete immediately to prevent escalation to legal action",
            "support": "I'll assist you through the process - this must be completed today"
        },
        "Pre-Legal 120+ Days": {
            "intro": f"Final opportunity payment link for {outstanding_amount}",
            "urgency": "Complete now to prevent court proceedings",
            "support": "I'll stay connected until payment is completed - no delays possible"
        },
        "Legal 150+ Days": {
            "intro": f"Legal payment demand link for {outstanding_amount}",
            "urgency": "Immediate completion required to stop legal action",
            "support": "Complete now or face court judgment - I'll assist until done"
        }
    }
    
    category = aging_context['category']
    guidance = portal_guidance_by_category.get(category, portal_guidance_by_category["First Missed Payment"])
    
    # Build urgency-based instructions
    instructions_by_urgency = {
        "Medium": [
            "Click the link I'm sending",
            f"Confirm the amount as {outstanding_amount}",
            "Choose your preferred payment method",
            "I'll stay on the line to help"
        ],
        "High": [
            "Click the payment link immediately",
            f"Verify the amount is {outstanding_amount}",
            "Select fastest payment method available",
            "Complete before ending this call"
        ],
        "Very High": [
            "Open the payment link now",
            f"Amount must be exactly {outstanding_amount}",
            "Use fastest payment option",
            "Must complete immediately to prevent escalation"
        ],
        "Critical": [
            "Click payment link immediately",
            f"Pay exact amount {outstanding_amount}",
            "Use any available payment method",
            "Complete now to stop legal proceedings"
        ]
    }
    
    urgency_level = aging_context['urgency']
    instructions = instructions_by_urgency.get(urgency_level, instructions_by_urgency["Medium"])
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
</role>

<context>
- Outstanding: {outstanding_amount}
- Aging Category: {category}
- Urgency Level: {urgency_level}
- Client user_id: {user_id}
</context>

<task>
Guide client through payment portal using aging-appropriate urgency and support level.
</task>

<aging_specific_guidance>
**Introduction**: "{guidance['intro']}"
**Urgency Messaging**: "{guidance['urgency']}"
**Support Level**: "{guidance['support']}"
</aging_specific_guidance>

<step_by_step_instructions>
{chr(10).join([f"{i+1}. '{instruction}'" for i, instruction in enumerate(instructions)])}
</step_by_step_instructions>

<urgency_adaptation>
{aging_context['approach']}
</urgency_adaptation>

<support_level>
- Stay connected during entire process
- Provide real-time assistance
- Address technical issues immediately
- Ensure completion before ending call
</support_level>

<style>
- {aging_context['tone']}
- Professional guidance matching urgency
- Immediate problem solving
- Persistent until completion
- {urgency_level.lower()} priority assistance
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.PAYMENT_PORTAL,
        client_data=client_data,
        state=state
    )

def create_payment_portal_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a payment portal agent with aging-aware scripts."""
      
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Get outstanding amount
        account_aging = client_data.get("account_aging", {})
        outstanding_amount = calculate_outstanding_amount(account_aging)
        user_id = client_data.get('user_id')
        
        # Determine script type and urgency
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        # Generate payment URL if possible
        payment_url = None
        url_generated = False
        
        # if outstanding_amount > 0 and user_id:
        #     try:
        #         url_result = generate_sms_payment_url.invoke({
        #             "user_id": int(user_id),
        #             "amount": outstanding_amount,
        #             "optional_reference": f"PTP_{user_id}"
        #         })
                
        #         if url_result.get("success"):
        #             payment_url = url_result.get("payment_url")
        #             url_generated = True
                    
        #     except Exception as e:
        #         if verbose:
        #             print(f"Error generating payment URL: {e}")
        
        return Command(
            update={
                "payment_url": payment_url,
                "url_generated": url_generated,
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
                "portal_guidance": "I'll send you a secure payment link",
                "current_step": CallStep.PAYMENT_PORTAL.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_payment_portal_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
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
        name="PaymentPortalAgent"
    )
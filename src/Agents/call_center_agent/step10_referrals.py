# ===============================================================================
# STEP 10: REFERRALS AGENT - Enhanced with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step10_referrals.py
"""
Referrals Agent - Enhanced with aging-aware script integration
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


def get_referrals_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware referrals prompt."""
    
    # Determine script type from aging
    user_id = client_data["profile"]["user_id"]
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Build aging-specific referral approaches
    referral_approaches_by_category = {
        "First Missed Payment": {
            "approach": "Friendly mention of referral program",
            "message": "Do you know anyone interested in Cartrack? Successful referrals earn you 2 months free subscription."
        },
        "Failed Promise to Pay": {
            "approach": "Brief mention after payment arrangement",
            "message": "Once your account is current, remember our referral program offers 2 months free."
        },
        "2-3 Months Overdue": {
            "approach": "Quick mention only if appropriate",
            "message": "After settling your account, referrals can earn free months."
        },
        "Pre-Legal 120+ Days": {
            "approach": "Minimal mention, focus on resolution",
            "message": "Future referrals available once account is resolved."
        },
        "Legal 150+ Days": {
            "approach": "Skip referrals, not appropriate",
            "message": "Focus on account resolution."
        }
    }
    
    category = aging_context['category']
    referral_approach = referral_approaches_by_category.get(category, referral_approaches_by_category["First Missed Payment"])
    
    # Determine if referrals are appropriate for this urgency level
    should_offer_referrals = urgency_level not in ['Very High', 'Critical']
    urgency_level = aging_context['urgency']
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
</role>

<context>
- Aging Category: {category}
- Urgency Level: {urgency_level}
- Referrals Appropriate: {should_offer_referrals}
- Client user_id: {user_id}
</context>

<task>
{"Briefly mention referral program if appropriate for account status" if should_offer_referrals else "Skip referrals due to account urgency"}
</task>

<aging_specific_approach>
**Strategy**: "{referral_approach['approach']}"
**Message**: "{referral_approach['message']}"
</aging_specific_approach>

<appropriateness_guide>
- Low/Medium urgency: Full referral program mention
- High urgency: Brief mention after payment focus
- Very High/Critical: Skip referrals entirely
</appropriateness_guide>

<benefits_if_appropriate>
- 2 months free subscription
- Help friends with vehicle security
- Easy referral process
</benefits_if_appropriate>

<style>
- {"Brief and positive" if should_offer_referrals else "Skip this step"}
- Present as benefit to client only if appropriate
- No pressure if not interested
- {"Quick mention only" if should_offer_referrals else "Focus on account resolution"}
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.REFERRALS,
        client_data=client_data,
        state=state
    )

def create_referrals_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a referrals agent with aging-aware scripts."""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        # Determine script type and urgency
        account_aging = client_data.get("account_aging", {})
        script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
        aging_context = ScriptManager.get_aging_context(script_type)
        
        # Determine if referrals are appropriate
        should_offer_referrals = aging_context['urgency'] not in ['Very High', 'Critical']
        
        return Command(
            update={
                "aging_category": aging_context['category'],
                "urgency_level": aging_context['urgency'].lower(),
                "should_offer_referrals": should_offer_referrals,
                "current_step": CallStep.REFERRALS.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_referrals_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(f"Prompt: {prompt_content}")
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ReferralsAgent"
    )

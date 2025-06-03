# src/Agents/call_center_agent/step10_referrals.py
"""
Enhanced Referrals Agent - Context-aware referral offering with mood sensitivity
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters, detect_client_mood_from_messages
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

import logging
logger = logging.getLogger(__name__)

REFERRALS_PROMPT = """
You're {agent_name} from Cartrack offering referral program to {client_name} after successful call resolution.

TODAY: {current_date}
OBJECTIVE: {offer_objective}

CLIENT MOOD: {detected_mood}
PAYMENT SECURED: {payment_secured}
SHOULD OFFER: {should_offer_referrals}

REFERRAL OFFER BY CONTEXT:
- Cooperative + Payment Secured: "Know anyone interested in Cartrack? Successful referrals earn you 2 months free subscription."
- Cooperative + No Payment: "By the way, our referral program gives 2 months free for successful recommendations."
- Neutral + Payment Secured: "Quick question - do you know anyone who might benefit from Cartrack? We offer referral rewards."
- Skip Entirely: "Moving on to wrap up our call today."

URGENCY-BASED APPROACH:
- Low/Medium: Full referral offer with benefits explanation
- High: Brief mention only if mood is good
- Critical: Skip entirely - focus on resolution

REFERRAL BENEFITS TO MENTION:
- 2 months free subscription for successful referrals
- Friend/family get special pricing
- Easy referral process through your account

APPROPRIATENESS CHECK:
✓ Offer if: Payment secured, cooperative mood, good rapport
✗ Skip if: Angry, stressed, resistant, urgent situation

URGENCY LEVEL: {urgency_level} - {aging_approach}

{offer_instruction}
"""

def create_referrals_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced referrals agent with mood-aware offering"""
    
    def _determine_referral_appropriateness(state: CallCenterAgentState, messages: List) -> Dict[str, Any]:
        """Determine if referrals should be offered based on context"""
        
        # Get context factors
        payment_secured = state.get("payment_secured", False)
        detected_mood = detect_client_mood_from_messages(messages)
        aging_context = ScriptManager.get_aging_context(script_type)
        urgency = aging_context['urgency']
        
        # Decision logic
        should_offer = True
        offer_objective = "Offer referral program with benefits"
        offer_instruction = "Keep it brief and positive. Under 15 words."
        
        # Skip for high urgency or problematic moods
        if urgency in ['Very High', 'Critical']:
            should_offer = False
            offer_objective = "Skip referrals - focus on call conclusion"
            offer_instruction = "Don't mention referrals. Move to further assistance."
        
        elif detected_mood in ['angry', 'financial_stress', 'resistant']:
            should_offer = False
            offer_objective = "Skip referrals - inappropriate mood"
            offer_instruction = "Don't mention referrals. Keep conversation focused."
        
        elif detected_mood == 'cooperative' and payment_secured:
            offer_objective = "Full referral offer - ideal conditions"
            offer_instruction = "Enthusiastic but brief offer. Mention 2 months free."
        
        elif detected_mood == 'cooperative' and not payment_secured:
            offer_objective = "Brief referral mention - good mood but no payment"
            offer_instruction = "Quick mention only. Don't oversell."
        
        return {
            "should_offer": should_offer,
            "detected_mood": detected_mood,
            "offer_objective": offer_objective,
            "offer_instruction": offer_instruction
        }
    
    def _check_referral_completion(messages: List) -> bool:
        """Check if referral offer was completed"""
        for message in reversed(messages[-2:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "referral", "2 months free", "recommend", "anyone interested",
                    "moving on", "wrap up"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Determine referral approach and check completion"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            referral_completed = _check_referral_completion(messages)
            
            if referral_completed:
                logger.info("Referral step completed - moving to further assistance")
                return Command(
                    update={
                        "current_step": CallStep.FURTHER_ASSISTANCE.value
                    },
                    goto="__end__"
                )
        
        # Continue with referral process
        return Command(
            update={"current_step": CallStep.REFERRALS.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate context-aware referral prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Determine referral appropriateness
        messages = state.get("messages", [])
        referral_context = _determine_referral_appropriateness(state, messages)
        
        params.update(referral_context)
        params["payment_secured"] = "Yes" if state.get("payment_secured") else "No"
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format prompt
        prompt_content = REFERRALS_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Referrals Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=[],  # No tools needed for referrals
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedReferralsAgent"
    )
# ===============================================================================
# STEP 01: NAME VERIFICATION AGENT - Updated with Aging-Aware Prompts
# ===============================================================================

# src/Agents/call_center_agent/step01_name_verification.py
"""
Name Verification Agent - Enhanced with aging-aware script integration
"""
import logging
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.data.client_data_fetcher import get_safe_value
from src.Agents.call_center_agent.tools.verify_client_name import verify_client_name
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

logger = logging.getLogger(__name__)

def get_name_verification_prompt(client_data: Dict[str, Any], agent_name: str, state: Dict[str, Any] = None) -> str:
    """Generate aging-aware name verification prompt."""
    
    # Determine script type from aging
    user_id = get_safe_value(client_data, "profile.user_id", "")
    account_aging = client_data.get("account_aging", {})
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    # Extract client and state info
    client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
    client_title = get_safe_value(client_data, "profile.client_info.title", "Mr/Ms")
    status = state.get("name_verification_status", "INSUFFICIENT_INFO")
    attempts = state.get("name_verification_attempts", 1)
    max_attempts = 5
    
    # Base prompt
    base_prompt = f"""<role>
You are a professional debt collection specialist at Cartrack's Accounts Department. Your name is {agent_name}.
</role>

<current_context>
- Verification Status: {status}
- Attempt: {attempts}/{max_attempts}
- Target Client: {client_full_name}
- Client user_id: {user_id}
- Urgency Level: {aging_context['urgency']}
- Account Category: {aging_context['category']}
</current_context>

<task>
Confirm client identity through name verification. Adapt tone to urgency level.
</task>

<response_strategies>
**INSUFFICIENT_INFO** (Progressive approach based on urgency):
- Standard Urgency: "Hi, just to confirm I'm speaking with {client_full_name}?"
- High Urgency: "This is urgent regarding your Cartrack account. Is this {client_full_name}?"
- Legal Urgency: "This is a legal matter regarding your account. I need to confirm this is {client_full_name} speaking"

**VERIFIED**: "Thank you for confirming. I'll need to verify security details before discussing your account"

**THIRD_PARTY**: emphasize urgency "Please have {client_full_name} call us urgently at 011 250 3000 regarding their Cartrack outstanding account matter"

**UNAVAILABLE**: "I understand. Please have {client_full_name} call 011 250 3000 urgently regarding your Cartrack outstanding account matter". End call in 3 attempts

**WRONG_PERSON**: "I apologize for the confusion. I have the wrong number. Goodbye". Stop discussing further

**VERIFICATION_FAILED**: "For security, I cannot proceed. Please call Cartrack directly at 011 250 3000"
</response_strategies>

<urgency_adaptation>
{aging_context['approach']}
</urgency_adaptation>

<style>
- Adapt formality to urgency level
- {aging_context['tone']}
- Build trust through competence
- Match urgency to account status
- Respond under 20 words
- RESPOND MAX in 30 words
</style>"""

    # Enhance with script content
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=ScriptCallStep.NAME_VERIFICATION,
        client_data=client_data,
        state=state
    )

def create_name_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = None,  # Auto-determined from aging
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a name verification agent with aging-aware scripts."""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        client_full_name = get_safe_value(client_data, "profile.client_info.client_full_name", "Client")
        attempts = state.get("name_verification_attempts", 0) + 1
        max_attempts = config.get("verification", {}).get("max_name_verification_attempts", 5)
        
        verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        
        try:
            result = verify_client_name.invoke({
                "client_full_name": client_full_name,
                "messages": state.get("messages", []),
                "max_failed_attempts": max_attempts
            })
            verification_status = result.get("classification", VerificationStatus.INSUFFICIENT_INFO.value)
            
            if verbose:
                logger.info(f"Name verification result: {verification_status}")
                
        except Exception as e:
            if verbose:
                logger.error(f"Name verification error: {e}")
        
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        return Command(
            update={
                "name_verification_status": verification_status,
                "name_verification_attempts": attempts,
                "client_full_name": client_full_name,
                "current_step": CallStep.NAME_VERIFICATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        prompt_content = get_name_verification_prompt(client_data, agent_name, state.to_dict() if hasattr(state, 'to_dict') else state)
        print(f"Prompt: {prompt_content}")
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="NameVerificationAgent"
    )

# src/Agents/call_center_agent/step01_name_verification.py
"""
Name Verification Agent - Lean version with concise responses
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep, VerificationStatus
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
from src.Agents.call_center_agent.tools.verify_client_name import verify_client_name

logger = logging.getLogger(__name__)

# Optimized prompt template - ensures concise name verification
NAME_VERIFICATION_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today time: {current_date}
</role>

<context>
Target client: {client_full_name} | Outstanding amount: {outstanding_amount} | Name Verification Status: {name_verification_status}
Verification Attempt: {name_verification_attempts}/{max_name_verification_attempts}| Urgency: {urgency_level} | Category: {aging_category} | user_id: {user_id}
</context>

<script>
{formatted_script}
</script>

<task>
Confirm client identity. Match urgency to account severity.
</task>

<verification_responses>
**INSUFFICIENT_INFO** (Progressive approach based on urgency):
- Standard Urgency: "Hi, just to confirm I'm speaking with {client_full_name}?"
- High Urgency: "This is urgent regarding your Cartrack account. Is this {client_full_name}?"
- Legal Urgency: "This is a legal matter regarding your account. I need to confirm this is {client_full_name} speaking"

**VERIFIED**: "Thank you. I need to verify security details."

**THIRD_PARTY**:  emphasize urgency "Please have {client_title} {client_full_name} call us urgently at 011 250 3000 regarding their Cartrack outstanding account matter"

**UNAVAILABLE**: "I understand. Please call 011 250 3000 urgently regarding your Cartrack outstanding account". 

**WRONG_PERSON**: "I apologize for the confusion. I have the wrong number. Goodbye". Stop discussing further

**VERIFICATION_FAILED**: "For security, I cannot proceed. If you are {client_title} {client_full_name}, please call Cartrack directly at 011 250 3000 regarding your Cartrack outstanding account"
</verification_responses>

<urgency_adaptation>
{aging_approach}
</urgency_adaptation>

<response_style>
CRITICAL: Keep under 15 words. Direct name confirmation requests.
Examples:
✓ "Are you {client_full_name}?"
✓ "This is urgent. Is this {client_full_name}?"
✗ "Good day, I hope you're well. I'm calling to confirm if I'm speaking with..."
</response_style>
"""

def create_name_verification_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create name verification agent with concise responses"""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Process name verification attempt"""
        client_full_name = client_data.get("profile", {}).get("client_info", {}).get("client_full_name", "Client")
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
        
        # Auto-fail if max attempts reached
        if attempts >= max_attempts and verification_status == VerificationStatus.INSUFFICIENT_INFO.value:
            verification_status = VerificationStatus.VERIFICATION_FAILED.value
        
        return Command(
            update={
                "name_verification_status": verification_status,
                "name_verification_attempts": attempts,
                "current_step": CallStep.NAME_VERIFICATION.value
            },
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate concise name verification prompt"""
        # Step 1: Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Step 2: Format script
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.NAME_VERIFICATION)
        formatted_script = script_template.format(**params) if script_template else f"Are you {params['client_full_name']}?"
        params["formatted_script"] = formatted_script
        
        # Step 3: Format prompt
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        prompt_content = NAME_VERIFICATION_PROMPT.format(**params)
        
        if verbose: 
            print(f"Name Verification Prompt: {prompt_content}")
        
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
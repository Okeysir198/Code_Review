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
Today's date: {current_date}
</role>

<context>
Client: {client_full_name} | Amount owed: {outstanding_amount} | Name verified: {name_verification_status}
Verification attempt: {name_verification_attempts}/{max_name_verification_attempts} | Priority: {urgency_level} | Category: {aging_category} | ID: {user_id}
</context>

<script>
{formatted_script}
</script>

<task>
- Confirm you're speaking with the right person
- If they ask questions, answer briefly then get back to name confirmation
- If client isn't available, ask them to call back urgently
- If you have the wrong person, apologize and end the call
- If speaking to someone else, stress urgency for the client to call back
</task>

<verification_responses>
**INSUFFICIENT_INFO** (Adjust tone based on urgency):
- Standard: "[Quick answer if needed] Hi, am I speaking with {client_full_name}?"
- High: "[Quick answer if needed] This is urgent about your Cartrack account. Is this {client_full_name}?"
- Legal: "[Quick answer if needed] This is a legal matter about your account. I need to confirm this is {client_full_name}"

**VERIFIED**: "Thanks. I need to verify some security details with you."

**THIRD_PARTY**: "Please have {client_title} {client_full_name} call us urgently on 011 250 3000 about their Cartrack account"

**UNAVAILABLE**: "Please call us back urgently on 011 250 3000 about your Cartrack account"

**WRONG_PERSON**: "Sorry, I have the wrong number. Goodbye." [End call]

**VERIFICATION_FAILED**: "For security reasons, I can't continue. If you are {client_title} {client_full_name}, please call Cartrack on 011 250 3000 about your account"
</verification_responses>

<urgency_adaptation>
{aging_approach}
</urgency_adaptation>

<response_style>
KEEP IT SHORT: Under 20 words. Get straight to name confirmation.
Good examples:
✓ "Is this {client_full_name}?"
✓ "This is urgent. Are you {client_full_name}?"
Bad examples:
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
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
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
        
        if verification_status == VerificationStatus.VERIFIED.value:
            goto = "__end__"
        else:
            goto = "agent"
            
        return Command(
            update={
                "name_verification_status": verification_status,
                "name_verification_attempts": attempts,
                "current_step": CallStep.NAME_VERIFICATION.value
            },
            goto=goto
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
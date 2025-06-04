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
You are {agent_name} from Cartrack calling {client_title} {client_full_name} about {outstanding_amount} overdue.
Attempt {name_verification_attempts}/{max_name_verification_attempts} | Urgency: {urgency_level} | Status: {name_verification_status}

OBJECTIVE: Confirm identity before discussing account. YOU called THEM about debt - be direct.

RESPONSES BY STATUS:
• INSUFFICIENT_INFO: "Hi, is this {client_title} {client_full_name}?" (urgent cases: "This is urgent - is this {client_title} {client_full_name}?")
• VERIFIED: "Perfect, thank you. I just need to verify some details with you."
• THIRD_PARTY: "Could you please ask {client_title} {client_full_name} to call Cartrack back today on 011 250 3000? It's regarding their account."
• WRONG_PERSON: "My apologies, I think I have the wrong number. Have a good day."
• FAILED: "I'm unable to proceed for security reasons. If you are {client_title} {client_full_name}, please call us on 011 250 3000."

QUERY HANDLING (Answer briefly + redirect):
• "Who/What/Why?" → "This is {agent_name} from Cartrack calling about your account. Is this {client_title} {client_full_name}?"
• "Scam/Busy/Later?" → "No, this is official Cartrack business. Are you {client_title} {client_full_name}?"
• "Don't owe/Remove number?" → "I need to verify your identity first. Is this {client_title} {client_full_name}?"
• Persistent questions → "I'll be happy to explain everything once I confirm this is {client_title} {client_full_name}."

RULES: Max 30 words. No account details before verification. Always redirect to name confirmation.

Aging Approach:
{aging_approach}
Answer directly without showing your reasoning process or thinking steps.
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
        params = prepare_parameters(client_data, state, script_type, agent_name)
        
        # Step 2: Format script
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.NAME_VERIFICATION)
        formatted_script = script_template.format(**params) if script_template else f"Are you {params['client_full_name']}?"
        params["formatted_script"] = formatted_script
        
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
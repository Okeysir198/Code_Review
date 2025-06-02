# src/Agents/call_center_agent/step09_client_details_update.py
"""
Enhanced Client Details Update Agent - Tool-guided contact information updates
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
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep
from src.Database.CartrackSQLDatabase import (
    update_client_contact_number, update_client_email
)
import logging
logger = logging.getLogger(__name__)
CLIENT_DETAILS_UPDATE_PROMPT = """
You're {agent_name} from Cartrack updating contact details for {client_name}.

TODAY: {current_date}
OBJECTIVE: Verify and update {client_name}'s contact information for important account notifications.

CURRENT DETAILS ON FILE:
- Mobile: {current_mobile}
- Email: {current_email}

TOOL USAGE CONDITIONS:
- update_client_contact_number: ONLY if client provides NEW mobile number
- update_client_email: ONLY if client provides NEW email address

CONVERSATION PATTERN:
1. "Let me verify your contact details for important notifications"
2. "Your mobile number on file is {current_mobile}. Is this still correct?"
3. IF client says "no" or gives new number:
   - Use update_client_contact_number tool
   - Parameters: user_id={user_id}, mobile_number=new_number
   - Confirm: "Updated your mobile to [new number]. Is that correct?"

4. "Your email address is {current_email}. Still current?"
5. IF client provides new email:
   - Use update_client_email tool  
   - Parameters: user_id={user_id}, email_address=new_email
   - Confirm: "Updated your email to [new email]. Perfect."

TOOL RESPONSE HANDLING:
- SUCCESS: "Contact details updated successfully"
- FAILURE: "I'm having trouble updating that. I'll make a note for our team to follow up"

DON'T use tools if details are already correct.
ALWAYS confirm new details before updating: "So your new mobile is 083-555-1234. Correct?"
ALWAYS verify the update worked: "Great, your mobile is now updated in our system"

JUSTIFICATION BY URGENCY:
- Standard: "Updating contact details for account notifications"
- High: "Ensuring we can reach you about urgent account matters"
- Critical: "Contact details required for legal correspondence"

URGENCY LEVEL: {urgency_level} - {aging_approach}

Keep the verification process quick and efficient. Only update what actually needs changing.
"""

def create_client_details_update_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create enhanced client details update agent with conditional tool usage"""
    
    # Tools for contact updates
    update_tools = [
        update_client_contact_number,
        update_client_email
    ]
    
    def _check_update_completion(messages: List) -> bool:
        """Check if contact details update is complete"""
        for message in reversed(messages[-3:]):
            if hasattr(message, 'type') and message.type == 'ai':
                content = message.content.lower()
                completion_indicators = [
                    "updated", "details confirmed", "contact information",
                    "verified", "all set", "perfect"
                ]
                if any(indicator in content for indicator in completion_indicators):
                    return True
        return False
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent", "__end__"]]:
        """Check if client details update is complete"""
        
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            update_completed = _check_update_completion(messages)
            
            if update_completed:
                logger.info("Client details update completed - moving to end")
                return Command(
                    update={
                        "current_step": CallStep.REFERRALS.value
                    },
                    goto="__end__"
                )
        
        # Continue with update process
        return Command(
            update={"current_step": CallStep.CLIENT_DETAILS_UPDATE.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate tool-guided client update prompt"""
        
        # Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Get aging-specific approach
        aging_context = ScriptManager.get_aging_context(script_type)
        params["aging_approach"] = aging_context['approach']
        
        # Format prompt
        prompt_content = CLIENT_DETAILS_UPDATE_PROMPT.format(**params)
        
        if verbose:
            print(f"Enhanced Client Details Update Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=update_tools,
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="EnhancedClientDetailsUpdateAgent"
    )
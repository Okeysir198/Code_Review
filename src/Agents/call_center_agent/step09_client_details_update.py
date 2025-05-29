# ./src/Agents/call_center_agent/step09_client_details_update.py
"""
Client Details Update Agent - Optimized with only pre-processing.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.prompts import get_step_prompt
from src.Agents.call_center_agent.data_parameter_builder import prepare_parameters
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep

# Import relevant database tools
from src.Database.CartrackSQLDatabase import (
    update_client_contact_number,
    update_client_email,
    add_client_note
)


def create_client_details_update_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create a client details update agent."""
    
    agent_tools = [update_client_contact_number, update_client_email, add_client_note] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Identify what details need updating."""
        
        try:
            # Extract current contact information
            profile = client_data.get("profile", {})
            client_info = profile.get("client_info", {}) if profile else {}
            
            current_mobile = client_info.get("contact", {}).get("mobile", "") if client_info.get("contact") else ""
            current_email = client_info.get("email_address", "")
            
            # Check recent conversation for new details
            recent_messages = state.get("messages", [])[-5:] if state.get("messages") else []
            
            updates_needed = {
                "mobile": False,
                "email": False,
                "next_of_kin": False
            }
            
            new_details = {
                "mobile": None,
                "email": None
            }
            
            # Analyze messages for detail updates
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    content = msg.content.lower()
                    
                    # Look for mobile number patterns
                    import re
                    mobile_patterns = [
                        r'(?:mobile|cell|phone).*?(\d{10})',
                        r'(\d{3}[-\s]?\d{3}[-\s]?\d{4})',
                        r'0\d{9}'
                    ]
                    
                    for pattern in mobile_patterns:
                        match = re.search(pattern, content)
                        if match:
                            new_mobile = match.group(1) if len(match.groups()) > 0 else match.group(0)
                            if new_mobile != current_mobile:
                                updates_needed["mobile"] = True
                                new_details["mobile"] = new_mobile
                    
                    # Look for email patterns
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    email_match = re.search(email_pattern, content)
                    if email_match:
                        new_email = email_match.group(0)
                        if new_email != current_email:
                            updates_needed["email"] = True
                            new_details["email"] = new_email
                    
                    # Look for next of kin mentions
                    if any(phrase in content for phrase in ["next of kin", "emergency contact", "nok"]):
                        updates_needed["next_of_kin"] = True
            
            return Command(
                update={
                    "current_mobile": current_mobile,
                    "current_email": current_email,
                    "updates_needed": updates_needed,
                    "new_details": new_details,
                    "contact_details_updated": False
                },
                goto="agent"
            )
            
        except Exception as e:
            if verbose:
                print(f"Error in client details pre-processing: {e}")
            
            return Command(
                update={
                    "current_mobile": "",
                    "current_email": "",
                    "updates_needed": {"mobile": False, "email": False, "next_of_kin": False},
                    "new_details": {},
                    "contact_details_updated": False
                },
                goto="agent"
            )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate dynamic prompt for client details update step."""
        parameters = prepare_parameters(
            client_data=client_data,
            current_step=CallStep.CLIENT_DETAILS_UPDATE.value,
            state=state.to_dict() if hasattr(state, 'to_dict') else state,
            script_type=script_type,
            agent_name=agent_name
        )
        
        prompt_content = get_step_prompt(CallStep.CLIENT_DETAILS_UPDATE.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=agent_tools,
        pre_processing_node=pre_processing_node,
        # NO post_processing_node - removed as per instructions
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ClientDetailsUpdateAgent"
    )
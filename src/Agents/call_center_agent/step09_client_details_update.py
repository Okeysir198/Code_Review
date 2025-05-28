# ./src/Agents/call_center_agent/step09_client_details_update.py
"""
Client Details Update Agent - Updates client contact and personal information.
SIMPLIFIED: No query detection - router handles all routing decisions.
"""
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
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
    update_client_next_of_kin,
    update_client_banking_details,
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
    """Create a client details update agent for debt collection calls."""
    
    # Add relevant database tools
    agent_tools = [
        update_client_contact_number,
        update_client_email,
        update_client_next_of_kin,
        update_client_banking_details,
        add_client_note
    ] + (tools or [])
    
    def pre_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Command[Literal["agent"]]:
        """Pre-process to identify what details need updating."""
        
        try:
            # Extract current contact information
            profile = client_data.get("profile", {})
            client_info = profile.get("client_info", {}) if profile else {}
            
            current_mobile = client_info.get("contact", {}).get("mobile", "") if client_info.get("contact") else ""
            current_email = client_info.get("email_address", "")
            
            # Check what updates might be needed based on recent conversation
            recent_messages = state.get("messages", [])[-5:] if state.get("messages") else []
            
            updates_needed = {
                "mobile": False,
                "email": False,
                "next_of_kin": False,
                "banking": False
            }
            
            new_details = {
                "mobile": None,
                "email": None,
                "nok_name": None,
                "nok_phone": None,
                "nok_email": None
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
                    "updates_needed": {"mobile": False, "email": False, "next_of_kin": False, "banking": False},
                    "new_details": {},
                    "contact_details_updated": False
                },
                goto="agent"
            )

    def post_processing_node(state: CallCenterAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Post-process to actually update client details if provided."""
        
        try:
            updates_made = []
            update_success = True
            user_id = client_data.get("user_id")

            # Check if client provided new details in recent conversation
            recent_messages = state.get("messages", [])[-3:] if state.get("messages") else []
            
            new_mobile = None
            new_email = None
            
            # Extract new details from conversation
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == "human":
                    content = msg.content
                    
                    # Extract mobile number
                    import re
                    mobile_match = re.search(r'0\d{9}|\d{10}', content)
                    if mobile_match:
                        new_mobile = mobile_match.group(0)
                    
                    # Extract email
                    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
                    if email_match:
                        new_email = email_match.group(0)
            
            # Update mobile number if provided
            if new_mobile:
                try:
                    mobile_result = update_client_contact_number.invoke({
                        "user_id": user_id,
                        "mobile_number": new_mobile
                    })
                    
                    if mobile_result.get("success"):
                        updates_made.append(f"Mobile updated to {new_mobile}")
                    else:
                        update_success = False
                        
                except Exception as mobile_error:
                    if verbose:
                        print(f"Error updating mobile: {mobile_error}")
                    update_success = False
            
            # Update email if provided
            if new_email:
                try:
                    email_result = update_client_email.invoke({
                        "user_id": user_id,
                        "email_address": new_email
                    })
                    
                    if email_result.get("success"):
                        updates_made.append(f"Email updated to {new_email}")
                    else:
                        update_success = False
                        
                except Exception as email_error:
                    if verbose:
                        print(f"Error updating email: {email_error}")
                    update_success = False
            
            # Add note about updates made
            if updates_made:
                note_text = f"Contact details updated: {', '.join(updates_made)}"
                add_client_note.invoke({
                    "user_id": user_id,
                    "note_text": note_text
                })
            
            return {
                "contact_details_updated": len(updates_made) > 0,
                "updates_made": updates_made,
                "update_success": update_success
            }
            
        except Exception as e:
            if verbose:
                print(f"Error in client details post-processing: {e}")
            
            return {
                "contact_details_updated": False,
                "updates_made": [],
                "update_success": False
            }

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
        post_processing_node=post_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="ClientDetailsUpdateAgent"
    )
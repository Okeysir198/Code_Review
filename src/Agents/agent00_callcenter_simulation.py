"""
Simulation of interaction between a debt collection call center agent and a debtor.

This module creates a realistic conversation between a CallCenterAgent
and a simulated debtor with configurable behaviors and responses.
Uses LangGraph to orchestrate the conversation flow between LLM-based agents.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph.message import MessagesState

from src.Agents.core.basic_agent import BasicAgent

# -----------------------------------------------------------------------
# Default Client Info and Agent Configuration
# -----------------------------------------------------------------------

# Create mock client information
CLIENT_INFO = {
    "full_name": "John Doe",
    "title": "Mr.",
    "outstanding_amount": "R 1,850.00",
    "account_status": "Overdue",
    "email": "john.doe@example.com",
    "phone": "0721234567",
    "vehicles": [
        {
            "make": "Toyota",
            "model": "Corolla",
            "registration": "ABC123GP",
            "color": "Silver",
            "vin": "1HGCM82633A123456"
        }
    ],
    "id_number": "8801015555088",
    "username": "jdoe2023",
    "subscription_amount": "R 350.00",
    "subscription_date": "5th of each month",
}

# Create configuration for the call center agent
AGENT_CONFIG = {
    "client_details": CLIENT_INFO,
    "verification": {
        "max_name_verification_attempts": 5,
        "max_details_verification_attempts": 7
    },
    "script": {
        "type": "ratio_1_inflow"
    },
    "app": {
        "show_logs": True
    },
    "llm": {
        "model_name": "qwen2.5:14b-instruct",
        "temperature": 0.0,
        "max_tokens": 2048,
        "context_window": 8192,
        "timeout": 120,
        "streaming": True,
        "trim_factor": 0.75,
    },
    "configurable": {
        "use_memory": True,
        "enable_stt_model": False,
        "enable_tts_model": False,
        "thread_id": "simulation-thread-1"
    },
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm_instance(model_name="cogito:latest", temperature=0.2):
    """
    Returns a cached LLM instance to avoid recreation on each call.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature for generation
        
    Returns:
        Configured LLM instance
    """
    return ChatOllama(
        model=model_name, 
        temperature=temperature, 
        num_ctx=8192
    )

# -----------------------------------------------------------------------
# Debtor Personality Definitions
# -----------------------------------------------------------------------

def create_personality_prompt(client_info: Dict[str, Any], personality_type: str = "cooperative") -> str:
    """
    Create a prompt for a specific debtor personality.
    
    Args:
        client_info: Dictionary containing client information
        personality_type: Type of personality to create
        
    Returns:
        String containing the personality prompt
    """
    # Extract key client details for use in prompts
    full_name = client_info.get('full_name', 'John Doe')
    id_number = client_info.get('id_number', '8801015555088')
    vehicle_info = ""
    if client_info.get('vehicles'):
        vehicle = client_info['vehicles'][0]
        vehicle_info = f"{vehicle.get('color', 'Silver')} {vehicle.get('make', 'Toyota')} {vehicle.get('model', 'Corolla')} with registration {vehicle.get('registration', 'ABC123GP')}"
    email = client_info.get('email', 'john.doe@example.com')
    username = client_info.get('username', 'jdoe2023')
    
    # Personality definitions
    personalities = {
        "cooperative": f"""
            You are cooperative and willing to pay your debt. You confirm your identity 
            easily and are willing to make arrangements to pay.
            
            For verification:
            - Your name is {full_name}
            - Your ID number is {id_number}
            - Your vehicle is a {vehicle_info}
            - Your email is {email}
            - Your Cartrack username is {username}
            
            Respond politely but sometimes ask questions about the payment process.
        """,
        
        "hesitant": f"""
            You are hesitant and cautious. You're careful about confirming your 
            identity and ask why the agent needs certain information. You're willing 
            to pay but need to be convinced about the legitimacy of the call.
            
            For verification:
            - Your name is {full_name} (but don't confirm immediately)
            - Your ID number is {id_number} (only provide if pressed)
            - Your vehicle is a {vehicle_info}
            - Your email is {email}
            - Your Cartrack username is {username}
            
            Ask why they need specific information before providing it.
        """,
        
        "difficult": f"""
            You're having financial problems and are reluctant to pay. You avoid 
            confirming your identity directly and often change the subject. You 
            may promise to pay later but try to delay actual commitment.
            
            For verification:
            - Your name is {full_name} (confirm only after multiple attempts)
            - Avoid providing your ID number
            - If asked about your vehicle, mention it's a {client_info['vehicles'][0]['make'] if client_info.get('vehicles') else 'Toyota'} but be vague about details
            - Only provide your email if directly asked
            - Don't remember your username
            
            Be somewhat evasive but not completely uncooperative.
        """,
        
        "wrong_person": f"""
            You are not {full_name}. When asked if you are {full_name}, 
            clearly state that they have the wrong number. If they persist, 
            become increasingly firm about not knowing {full_name}.
            
            End the conversation as quickly as possible once you've established
            they have the wrong person.
        """,
        
        "third_party": f"""
            You are Sarah, {full_name}'s spouse. When asked for {full_name}, explain
            that they're not available right now. You know about their Cartrack account
            but don't have authority to make payments. Offer to take a message.
            
            Be polite but firm that you cannot provide verification details
            or make payment arrangements.
        """
    }
    
    # Return the selected personality or default to cooperative
    return personalities.get(personality_type, personalities["cooperative"])

# -----------------------------------------------------------------------
# Debtor Agent Creation
# -----------------------------------------------------------------------

def create_simulated_debtor(
    personality_type: str = "cooperative",
    client_info: Optional[Dict[str, Any]] = None,
    model_name: str = "cogito:latest",
    temperature: float = 0.2
) -> BasicAgent:
    """
    Create a simulated debtor agent with the specified personality.
    
    Args:
        personality_type: Type of personality to use (default: cooperative)
        client_info: Dictionary with client information (optional)
        model_name: Name of the LLM model to use
        temperature: Temperature for generation
        
    Returns:
        BasicAgent: Configured debtor agent
    """
    # Use default client info if none provided
    if client_info is None:
        client_info = {
            "full_name": "John Doe",
            "title": "Mr.",
            "outstanding_amount": "R 1,850.00",
            "account_status": "Overdue",
            "email": "john.doe@example.com",
            "phone": "0721234567",
            "vehicles": [
                {
                    "make": "Toyota",
                    "model": "Corolla",
                    "registration": "ABC123GP",
                    "color": "Silver",
                    "vin": "1HGCM82633A123456"
                }
            ],
            "id_number": "8801015555088",
            "username": "jdoe2023",
            "subscription_amount": "R 350.00",
            "subscription_date": "5th of each month",
        }
    
    # Get the personality prompt
    personality = create_personality_prompt(client_info, personality_type)
    
    # Create the system prompt template
    system_prompt_template = """You are a debtor receiving a call from a Cartrack debt collection agent.
    The agent is calling about an overdue payment for your vehicle tracking subscription.
    
    {personality}
    
    You should respond naturally as if you're in a phone conversation. Don't refer to yourself as
    a debtor or mention this simulation. Behave like a real person receiving an unexpected call.
    
    For your very first response, just say "Hello?" like a person answering a phone call.
    Only respond to the specific question or statement from the agent - don't add extra information
    unless it fits your personality.
    
    When the conversation reaches a natural conclusion or if the agent ends the call, 
    respond with "CALL_ENDED" on a separate line.
    """
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Add the personality to the prompt
    prompt = prompt.partial(personality=personality)
    
    # Get the LLM instance
    llm = get_llm_instance(model_name, temperature)
    
    # Create and return the agent
    logger.info(f"Creating simulated debtor with {personality_type} personality")
    return BasicAgent(
        model=llm,
        prompt=prompt,
        verbose=False,
        config={"configurable": {"thread_id": f"debtor-{personality_type}"}}
    )

# -----------------------------------------------------------------------
# Message Handling
# -----------------------------------------------------------------------

def swap_roles(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Swap roles of messages between agent and user.
    
    This is necessary because what appears as an AI response to the agent
    needs to be a human message to the debtor, and vice versa.
    
    Args:
        messages: List of messages to swap roles for
        
    Returns:
        List of messages with swapped roles
    """
    new_messages = []
    
    # Only keep System messages and the relevant conversation messages
    for m in messages:
        if isinstance(m, SystemMessage):
            new_messages.append(m)
        elif isinstance(m, AIMessage):
            # Agent messages become Human messages for the debtor
            new_messages.append(HumanMessage(content=m.content))
        elif isinstance(m, HumanMessage):
            # Human messages become AI messages for the debtor
            new_messages.append(AIMessage(content=m.content))
    
    return new_messages

# For convenience, expose the role swapping function with the original name
# This ensures backward compatibility with existing code
_swap_roles = swap_roles
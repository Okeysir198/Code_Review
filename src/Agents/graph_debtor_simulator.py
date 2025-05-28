"""
Enhanced Debtor Simulation using LangGraph's create_react_agent

This module provides a comprehensive debtor simulator for testing call center agents.
Includes wrong person and third party scenarios for realistic testing.
"""
import random
from typing import Dict, Any, Optional, List
from enum import Enum

from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph


class DebtorPersonality(Enum):
    """Debtor personality types."""
    COOPERATIVE = "cooperative"
    DIFFICULT = "difficult" 
    CONFUSED = "confused"
    BUSY = "busy"
    SUSPICIOUS = "suspicious"
    WRONG_PERSON = "wrong_person"
    THIRD_PARTY_SPOUSE = "third_party_spouse"
    THIRD_PARTY_PARENT = "third_party_parent"
    THIRD_PARTY_ASSISTANT = "third_party_assistant"
    THIRD_PARTY_EMPLOYEE = "third_party_employee"


def create_debtor_simulator(
    llm: BaseChatModel,
    client_data: Dict[str, Any],
    personality: Optional[str] = None,
    cooperativeness: float = 0.7
) -> CompiledGraph:
    """
    Create a debtor simulator using create_react_agent.
    
    Args:
        llm: Language model for responses
        client_data: Client information to simulate (using actual Cartrack format)
        personality: Debtor personality type
        cooperativeness: How cooperative (0.0-1.0)
        
    Returns:
        Compiled agent that simulates debtor responses
    """
    # Select personality
    if personality and personality in [p.value for p in DebtorPersonality]:
        selected_personality = personality
    else:
        selected_personality = random.choice([p.value for p in DebtorPersonality])
    
    # Build personality prompts
    personality_prompts = {
        "cooperative": """
        You are COOPERATIVE and willing to resolve the debt:
        - Answer questions directly and honestly
        - Provide verification information when asked
        - Generally agreeable to payment options
        - Friendly and straightforward tone
        - Confirm identity readily: "Yes, this is [name] speaking"
        """,
        "difficult": """
        You are DIFFICULT and resistant:
        - Initially reluctant to identify yourself
        - Question why you need to pay
        - Skeptical of payment options
        - Eventually comply after pushback
        - Irritated tone
        - Grudgingly confirm identity: "Yeah, that's me. What do you want?"
        """,
        "confused": """
        You are CONFUSED about the situation:
        - Ask for clarification frequently
        - Sometimes provide incomplete information
        - Need explanations repeated
        - Hesitant about payment processes
        - Uncertain tone
        - Confused about identity: "Um, yes... I think so? What is this about?"
        """,
        "busy": """
        You are BUSY and want this resolved quickly:
        - Short, direct responses
        - Reference being in a hurry
        - Prefer fastest payment option
        - Minimal small talk
        - Rushed tone
        - Quick confirmation: "Yes, that's me. I'm busy, what do you need?"
        """,
        "suspicious": """
        You are SUSPICIOUS of the call:
        - Hesitant to confirm identity
        - Ask who is calling multiple times
        - Question why they need information
        - Very resistant to immediate payment
        - Guarded, cautious tone
        - Refuse initial confirmation: "Who's asking? Why do you need to know?"
        """,
        "wrong_person": """
        You are the WRONG PERSON - this is NOT your debt:
        - You are NOT the person they're looking for
        - You may share the same surname but are unrelated
        - You don't know the person they're asking for
        - Be polite but firm that they have the wrong number
        - Do NOT provide any verification details for the debtor
        - Examples: "I think you have the wrong number", "There's no one here by that name", "You must be looking for someone else"
        - NEVER confirm the debtor's identity or provide their information
        """,
        "third_party_spouse": """
        You are the SPOUSE/PARTNER of the debtor:
        - The debtor is your husband/wife/partner
        - You know about their financial situation
        - You may be protective or helpful depending on the situation
        - You can take messages but cannot discuss account details
        - Examples: "He's not here right now", "She's at work", "I'll tell him you called"
        - You might offer to help: "Is there something I can help with?" or be protective: "What is this regarding?"
        """,
        "third_party_parent": """
        You are the PARENT of the debtor:
        - The debtor is your adult child
        - You may or may not live together
        - You might be concerned about their debts
        - You want to help but know you can't pay for them
        - Examples: "He doesn't live here anymore", "She moved out last year", "I'll make sure he gets the message"
        - Parental concern: "Is everything okay? Is he in trouble?"
        """,
        "third_party_assistant": """
        You are the WORK ASSISTANT/COLLEAGUE of the debtor:
        - You work with the debtor
        - You can take messages for business calls
        - You're professional but cannot discuss personal matters
        - You have limited information about their availability
        - Examples: "He's in a meeting", "She's out of office today", "I can take a message"
        - Professional boundary: "I can't discuss personal matters"
        """,
        "third_party_employee": """
        You are an EMPLOYEE/RECEPTIONIST at the debtor's workplace:
        - You answer phones at their company
        - You don't know them personally
        - You follow standard phone protocols
        - You can transfer calls or take messages
        - Examples: "Let me check if they're available", "I can put you through to their extension", "Would you like to leave a message?"
        - Professional: "May I ask what company you're calling from?"
        """
    }
    
    # Extract client details from actual data structure
    profile = client_data.get('profile', {})
    client_info = profile.get('client_info', {})
    account_aging = client_data.get('account_aging', {})
    vehicles = profile.get('vehicles', [])
    
    # Basic client information
    client_name = client_info.get("client_full_name", "Client")
    first_name = client_info.get("first_name", "Client")
    title = client_info.get("title", "Mr/Ms")
    client_email = client_info.get("email_address", "client@example.com")
    id_number = client_info.get("id_number", "8001015001081")
    username = profile.get("user_name", "USER001")
    
    # Financial information
    outstanding_balance = account_aging.get("xbalance", "0.00")
    try:
        outstanding_amount = f"R {float(outstanding_balance):.2f}" if outstanding_balance else "R 0.00"
    except (ValueError, TypeError):
        outstanding_amount = "R 0.00"
    
    # Vehicle information
    vehicle_details = ""
    if vehicles and len(vehicles) > 0:
        vehicle = vehicles[0]
        vehicle_details = f"""
        Vehicle Registration: {vehicle.get('registration', 'ABC123GP')}
        Vehicle Make: {vehicle.get('make', 'Toyota')}
        Vehicle Model: {vehicle.get('model', 'Corolla')}
        Vehicle Color: {vehicle.get('color', 'Silver')}
        """
    else:
        vehicle_details = """
        Vehicle Registration: Not available
        Vehicle Make: Not available
        Vehicle Model: Not available
        Vehicle Color: Not available
        """
    
    # Account status
    account_overview = client_data.get('account_overview', {})
    account_status = account_overview.get('account_status', 'Active')
    payment_status = account_overview.get('payment_status', 'Current')
    
    # Handle wrong person and third party scenarios differently
    if selected_personality == "wrong_person":
        system_prompt = f"""You are simulating someone who answered the phone, but you are NOT {client_name}.

YOU ARE THE WRONG PERSON:
- You are not {client_name}
- You don't know who {client_name} is
- This is your phone number, but they have the wrong person
- You may have a similar surname but are unrelated
- You have NEVER heard of Cartrack
- You don't own the vehicle they're asking about

CRITICAL RULES:
- NEVER confirm you are {client_name}
- NEVER provide {client_name}'s verification details
- NEVER discuss any account information
- Be polite but firm that they have the wrong person
- You can get annoyed if they keep insisting

WRONG PERSON RESPONSES:
- "I think you have the wrong number"
- "There's no one here by that name"
- "You must be looking for someone else"
- "I don't know any {client_name}"
- "This is the wrong person"
- If they persist: "I've told you, you have the wrong number. Please check your records."

PERSONALITY: {selected_personality.upper()}
{personality_prompts[selected_personality]}

Keep responses brief and realistic. Never break character."""

    elif selected_personality.startswith("third_party"):
        # Generate realistic third party names
        third_party_names = {
            "third_party_spouse": f"{'Sarah' if 'Mr' in title else 'David'} {client_name.split()[-1]}",
            "third_party_parent": f"{'Margaret' if random.choice([True, False]) else 'Robert'} {client_name.split()[-1]}",
            "third_party_assistant": f"{'Lisa' if random.choice([True, False]) else 'James'} from the office",
            "third_party_employee": "Receptionist"
        }
        
        your_name = third_party_names.get(selected_personality, "Third Party")
        
        relationship_details = {
            "third_party_spouse": f"You are married to {client_name}. You live together but they're not home right now.",
            "third_party_parent": f"You are the parent of {client_name}. They may or may not live with you.",
            "third_party_assistant": f"You work with {client_name} as their assistant/colleague.",
            "third_party_employee": f"You are an employee/receptionist where {client_name} works."
        }
        
        system_prompt = f"""You are simulating {your_name}, a third party who knows {client_name}.

YOU ARE A THIRD PARTY:
- Your name: {your_name}
- {relationship_details.get(selected_personality, '')}
- You are NOT {client_name}
- You can take messages but cannot discuss account details
- You may or may not know about their financial situation

THE DEBTOR'S INFORMATION (that you CANNOT discuss):
- Full Name: {client_name}
- Outstanding Amount: {outstanding_amount}
- Account Status: {account_status}

CRITICAL RULES:
- NEVER confirm you are {client_name}
- NEVER provide their verification details (ID, banking, etc.)
- NEVER discuss their account or debt information
- You can acknowledge you know them and take messages
- Be helpful within appropriate boundaries

THIRD PARTY RESPONSES:
- "He's not here right now, can I take a message?"
- "She's at work today"
- "I'll let them know you called"
- "What company are you calling from?"
- "Is everything okay?" (if concerned)
- "I can't discuss their personal matters"

PERSONALITY: {selected_personality.upper()}
{personality_prompts[selected_personality]}

Keep responses brief and realistic. Never break character or discuss the debtor's confidential information."""

    else:
        # Regular debtor personality
        system_prompt = f"""You are simulating a debtor named {client_name} in a debt collection call.

YOUR INFORMATION:
- Full Name: {client_name}
- First Name: {first_name}
- Title: {title}
- Username: {username}
- Email: {client_email}
- Outstanding Amount: {outstanding_amount}
- ID Number: {id_number}
- Account Status: {account_status}
- Payment Status: {payment_status}
{vehicle_details}

PERSONALITY: {selected_personality.upper()}
{personality_prompts[selected_personality]}

COOPERATIVENESS LEVEL: {cooperativeness:.1f}/1.0

BEHAVIOR GUIDELINES:
1. Respond naturally as {client_name} would
2. Keep responses brief (1-3 sentences)
3. Stay in character consistently
4. Provide verification details when pressed (based on personality)
5. React realistically to payment requests

VERIFICATION BEHAVIOR:
- Name confirmation: {"readily" if cooperativeness > 0.7 else "hesitantly"}
- Detail sharing: {"cooperative" if cooperativeness > 0.6 else "reluctant"}

PAYMENT BEHAVIOR:
- Immediate debit acceptance: {int(cooperativeness * 100)}%
- DebiCheck acceptance: {int(cooperativeness * 80)}%
- Payment portal preference: {int(cooperativeness * 90)}%

RESPONSE RULES:
- Never break character or mention this is a simulation
- Respond only as the debtor would
- Match the personality type consistently
- Keep responses conversational and realistic
- When asked for verification, provide the information above based on your personality
- If suspicious or difficult, be more reluctant to share details immediately
- If cooperative, provide information when requested
- If confused, ask for clarification or give incomplete answers initially

COMMON VERIFICATION REQUESTS AND RESPONSES:
- ID Number: {id_number} (share based on cooperativeness level)
- Email: {client_email} (share based on cooperativeness level)
- Vehicle details: Share the registration, make, model, color as listed above
- Username: {username} (share if asked)

PAYMENT RESPONSES:
- If asked about immediate payment and cooperative: Show willingness to discuss options
- If asked about immediate payment and difficult: Express resistance or financial constraints
- If asked about payment arrangements: Respond according to personality type"""

    return create_react_agent(
        llm,
        tools=[],  # No tools needed for basic simulation
        state_modifier=SystemMessage(content=system_prompt)
    )


def get_personality_configs() -> Dict[str, Dict[str, float]]:
    """Get predefined personality configurations."""
    return {
        "cooperative": {
            "cooperativeness": 0.9,
            "name_confirmation": 0.95,
            "detail_sharing": 0.9,
            "payment_acceptance": 0.8
        },
        "difficult": {
            "cooperativeness": 0.3,
            "name_confirmation": 0.6,
            "detail_sharing": 0.4,
            "payment_acceptance": 0.2
        },
        "confused": {
            "cooperativeness": 0.6,
            "name_confirmation": 0.7,
            "detail_sharing": 0.5,
            "payment_acceptance": 0.4
        },
        "busy": {
            "cooperativeness": 0.7,
            "name_confirmation": 0.8,
            "detail_sharing": 0.6,
            "payment_acceptance": 0.6
        },
        "suspicious": {
            "cooperativeness": 0.4,
            "name_confirmation": 0.5,
            "detail_sharing": 0.3,
            "payment_acceptance": 0.2
        },
        "wrong_person": {
            "cooperativeness": 0.0,  # N/A for wrong person
            "name_confirmation": 0.0,  # Never confirms debtor's identity
            "detail_sharing": 0.0,  # Never shares debtor details
            "payment_acceptance": 0.0  # Cannot make payments
        },
        "third_party_spouse": {
            "cooperativeness": 0.7,  # Helpful but protective
            "name_confirmation": 0.0,  # Cannot confirm debtor identity
            "detail_sharing": 0.3,  # Limited information sharing
            "payment_acceptance": 0.1  # Cannot make payments for debtor
        },
        "third_party_parent": {
            "cooperativeness": 0.8,  # Very helpful and concerned
            "name_confirmation": 0.0,  # Cannot confirm debtor identity
            "detail_sharing": 0.4,  # May share some general info
            "payment_acceptance": 0.2  # Might offer to help but can't pay
        },
        "third_party_assistant": {
            "cooperativeness": 0.6,  # Professional but limited
            "name_confirmation": 0.0,  # Cannot confirm personal identity
            "detail_sharing": 0.2,  # Very limited sharing
            "payment_acceptance": 0.0  # Cannot discuss personal finances
        },
        "third_party_employee": {
            "cooperativeness": 0.5,  # Standard phone protocol
            "name_confirmation": 0.0,  # Cannot confirm personal identity
            "detail_sharing": 0.1,  # Minimal information
            "payment_acceptance": 0.0  # Cannot discuss personal matters
        }
    }


# Enhanced convenience functions
def create_cooperative_debtor(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a cooperative debtor simulator."""
    return create_debtor_simulator(llm, client_data, "cooperative", 0.9)


def create_difficult_debtor(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a difficult debtor simulator.""" 
    return create_debtor_simulator(llm, client_data, "difficult", 0.3)


def create_confused_debtor(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a confused debtor simulator."""
    return create_debtor_simulator(llm, client_data, "confused", 0.6)


def create_busy_debtor(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a busy debtor simulator."""
    return create_debtor_simulator(llm, client_data, "busy", 0.7)


def create_suspicious_debtor(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a suspicious debtor simulator."""
    return create_debtor_simulator(llm, client_data, "suspicious", 0.4)


def create_wrong_person(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a wrong person simulator."""
    return create_debtor_simulator(llm, client_data, "wrong_person", 0.0)


def create_third_party_spouse(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a spouse/partner third party simulator."""
    return create_debtor_simulator(llm, client_data, "third_party_spouse", 0.7)


def create_third_party_parent(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a parent third party simulator."""
    return create_debtor_simulator(llm, client_data, "third_party_parent", 0.8)


def create_third_party_assistant(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a work assistant third party simulator."""
    return create_debtor_simulator(llm, client_data, "third_party_assistant", 0.6)


def create_third_party_employee(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a workplace employee third party simulator."""
    return create_debtor_simulator(llm, client_data, "third_party_employee", 0.5)


def create_random_debtor(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a debtor with random personality."""
    personality = random.choice([p.value for p in DebtorPersonality])
    configs = get_personality_configs()
    cooperativeness = configs[personality]["cooperativeness"]
    
    return create_debtor_simulator(llm, client_data, personality, cooperativeness)


def create_random_third_party(llm: BaseChatModel, client_data: Dict[str, Any]) -> CompiledGraph:
    """Create a random third party simulator."""
    third_party_types = ["third_party_spouse", "third_party_parent", "third_party_assistant", "third_party_employee"]
    personality = random.choice(third_party_types)
    configs = get_personality_configs()
    cooperativeness = configs[personality]["cooperativeness"]
    
    return create_debtor_simulator(llm, client_data, personality, cooperativeness)


def get_test_scenarios() -> Dict[str, str]:
    """Get predefined test scenarios for comprehensive testing."""
    return {
        "scenario_1_cooperative": "Debtor confirms identity readily and is willing to make payment arrangements",
        "scenario_2_difficult": "Debtor is resistant and requires persuasion but eventually cooperates",
        "scenario_3_suspicious": "Debtor is suspicious of the call and reluctant to confirm identity",
        "scenario_4_wrong_person": "Wrong person answers - they are not the debtor and don't know them",
        "scenario_5_spouse": "Debtor's spouse answers and can take a message but can't discuss account",
        "scenario_6_parent": "Debtor's parent answers, concerned about their child's situation",
        "scenario_7_workplace": "Work colleague/assistant answers and follows professional protocols",
        "scenario_8_busy": "Debtor is busy but wants to resolve quickly",
        "scenario_9_confused": "Debtor is confused about the situation and needs clarification"
    }

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
# Usage examples and testing
"""
from langchain_ollama import ChatOllama

# Initialize LLM
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.7)

# Use your actual client_data
client_data = {
    'user_id': '83905',
    'profile': {
        'user_name': 'TONG00010',
        'client_info': {
            'title': 'Mr',
            'first_name': 'Teko', 
            'client_full_name': 'Teko Tongwane',
            'id_number': '83905',
            'email_address': 'dev@onecell.co.za'
        },
        'vehicles': [{'registration': 'TEMP-CT723744', 'make': 'Toyota', 'model': 'Hilux 2.7i D/C', 'color': 'White'}]
    },
    'account_aging': {'xbalance': '399.00'}
}

# Create different personality types
cooperative_debtor = create_cooperative_debtor(llm, client_data)
wrong_person = create_wrong_person(llm, client_data)
spouse = create_third_party_spouse(llm, client_data)
parent = create_third_party_parent(llm, client_data)

# Test scenarios
test_message = "Good day, may I speak to Mr Teko Tongwane please?"

# Test wrong person
print("=== WRONG PERSON TEST ===")
response = wrong_person.invoke({"messages": [("user", test_message)]})
print(response['messages'][-1].content)

# Test third party spouse  
print("\n=== SPOUSE TEST ===")
response = spouse.invoke({"messages": [("user", test_message)]})
print(response['messages'][-1].content)

# Test cooperative debtor
print("\n=== COOPERATIVE DEBTOR TEST ===")
response = cooperative_debtor.invoke({"messages": [("user", test_message)]})
print(response['messages'][-1].content)
"""
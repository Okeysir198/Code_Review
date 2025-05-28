from app_config import CONFIG
from src.Agents.call_center_agent.call_scripts import ScriptType
# from src.Agents.graph_debtor_call_center_agent import CallCenterAgent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langgraph.types import Command
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import MessagesState
from typing import List, Dict, Literal, Callable, Any, Annotated, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


# Create a new instance with with thread for persist memmory
config = CONFIG.copy()
config['configurable'] = config.get("configurable",{})
config['configurable']['use_memory'] = False
# config['configurable']['enable_stt_model'] = False
# config['configurable']['enable_tts_model'] = False    

llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0, num_ctx=40000)


#########################################################################################
# Basic Agent
#########################################################################################
from src.Agents.core.basic_agent import create_basic_agent, BasicAgentState, DEFAULT_TOOLS
from src.Agents.call_center_agent.tools.verify_client_name import verify_client_name
from src.Agents.call_center_agent.state import VerificationStatus, CallCenterAgentState
from test_graph.client_data import client_data

system_prompt = "You are AI Agent from Cartrack Accountant Department"


# Simple agent
graph_basic_agent = create_basic_agent(
    model=llm,
    prompt=system_prompt
)

# # Agent with tools
# graph_basic_agent = create_basic_agent(
#     model=llm,
#     tools=DEFAULT_TOOLS,
#     prompt=system_prompt
# )

class AgentState(CallCenterAgentState):
    client_full_name: Optional[str]
    client_profile: Optional[Dict[str, Any]] = None
    name_variants_detected: Optional[str] = None
    pass
    
def pre_processing_node(state:AgentState, config: RunnableConfig):

    client_full_name = client_data.get("profile",{})["client_info"]["client_full_name"] 
    # client_full_name = config["configurable"].get("client_full_name", client_full_name)

    client_profile = client_data.get("profile",{})

    result = verify_client_name.invoke({
        "client_full_name": client_full_name,
        'messages': state.get("messages", []),
        'max_failed_attempts': config["configurable"].get("max_failed_attempts", 5),
    })

    
   
    return {'client_full_name':client_full_name, 
            'client_profile':client_profile,
            "name_verification_status": result['classification'],
            "name_verification_attempts": result['verification_attempts'],
            "name_variants_detected": result['name_variants_detected'],
            }

def post_processing_node(state:AgentState):
    messages = state.get('messages',[])
    return {"messages":messages}


def dynamic_prompt(state:AgentState, config: RunnableConfig):
    # Get existing messages from state
    messages = state.get("messages",[])
    user_name = state.get("user_name",'user')
    user_name = config["configurable"].get("client_data","khoa")
    system_prompt = f"You are helping {user_name}. Be personalized and helpful."
    
    # Filter out any existing system messages to avoid duplicates
    context_message_window = config["configurable"].get("context_window", 2)
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    return [SystemMessage(content=system_prompt)] + non_system_messages[-context_message_window:]  # Apply context window


# graph_basic_agent = create_basic_agent(
#     model=llm,
#     prompt=dynamic_prompt,
#     tools=DEFAULT_TOOLS,
#     state_schema=AgentState,
#     pre_processing_node=pre_processing_node,
# )

#########################################################################################
# tesing tool call feature
#########################################################################################
from src.Database.CartrackSQLDatabase import *
import datetime


tools = [
    date_helper,

    # get_client_vehicle_info,
    # get_client_profile,
    # client_call_verification,

    # get_client_account_aging,
    # get_client_account_status,
    # get_client_billing_analysis,

    # get_client_contracts,
    # get_client_account_overview,

    # get_client_account_statement,
    # get_client_banking_details,
    # get_client_payment_history,

    # get_client_notes,
    # get_client_debtor_letters,

    # get_client_failed_payments,
    # get_client_last_successful_payment,
    # get_client_last_valid_payment,
    # get_client_last_reversed_payment,

    # update_payment_arrangements,

    # get_payment_arrangement_types,
    # get_client_debit_mandates,
    
    # create_mandate,
    # create_debicheck_payment,
    # create_payment_arrangement,

    # get_client_subscription_amount,

    # create_payment_arrangement_payment_portal,
    # generate_sms_payment_url,

    # get_bank_options,
    # update_client_banking_details,
    # update_client_contact_number,
    # update_client_email,
    # update_client_next_of_kin,

    add_client_note,
    get_disposition_types,
    save_call_disposition
]

user_id = '28173'
user_id = '70997'
# user_id = "963262"
# user_id = "28173"
# user_id= '1035923'
user_id= '10003'
# user_id ='1216242' # work for debtor letters
# user_id ='1911088'
agent_name = "AI Assistant Agent"
today_date = datetime.datetime.now().isoformat()
DETAILED_SYSTEM_PROMPT = f"""
## ROLE
You are {agent_name}, an efficient payment assistant that handles DebiCheck arrangements.
The client identity is verified. Please answer any questions from this authenticated user.

## CLIENT CONTEXT
Current user: ID {user_id}

## OPERATIONAL GUIDELINES
- ALWAYS include exact parameter `user_id={user_id}` in ALL database function calls
- NEVER invent, assume, or fabricate data values such as phone numbers, dates, amounts, or any other fields
- ALWAYS fetch current date/time with appropriate tool before any database updates
- When uncertain about any value needed for a database call, prompt the user for the exact information
- If there is error, fix it yourself at least 3 times, before asking user
- Keep all responses under 30 words with clear, professional language
- Use only values directly provided by the user or retrieved from database queries

## DATA INTEGRITY RULES
- Payment amounts must come directly from user input - NEVER suggest or assume amounts
- Dates must be retrieved using the date tool or from user specification - NEVER fabricate dates
- Phone numbers must be explicitly provided by the user - NEVER generate or guess phone numbers
- For ANY field not explicitly provided by the user, ASK rather than assume

## REFERENCE
Today's Date: {today_date}
"""
messages_context_window = 20

# llm = ChatOllama(model="qwen3:8b", temperature=0, num_ctx=32000)
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=32000)

checkpointer=MemorySaver()
agent_with_tools = create_react_agent(
    model=llm,
    tools=tools,
    prompt=DETAILED_SYSTEM_PROMPT,
    debug = True,
    # checkpointer=checkpointer
)
#########################################################################################
# Call Center Agents
#########################################################################################
from test_graph.client_data import client_data
from src.Agents.call_center_agent.call_scripts import ScriptType
# from src.Agents.agent00_callcenter_simulation import  AGENT_CONFIG
# from src.Agents.graph_debtor_call_center_agent import CallCenterAgent

from src.Agents.call_center_agent.introduction import create_introduction_agent
# from src.Agents.call_center_agent.name_verification import NameVerificationAgent
# from src.Agents.call_center_agent.details_verification import DetailsVerificationAgent


# from app_config import CONFIG

# # Client information
# config = CONFIG.copy()
# client_info = config["client_details"]
# call_center_graph = CallCenterAgent(config).workflow


# # simulation
# AGENT_CONFIG['configurable']['use_memory'] = False
# AGENT_CONFIG['configurable']['enable_stt_model'] = False
# AGENT_CONFIG['configurable']['enable_tts_model'] = False
# client_info = AGENT_CONFIG["client_details"]

# enable_similation = True
# debtor_profile = "wrong_person" # wrong_person, difficult, hesitant, cooperative, third_party
# call_center_simulation = CallCenterAgent(AGENT_CONFIG, simulate_debtor=enable_similation, debtor_profile=debtor_profile).workflow



# introduction_graph = IntroductionAgent(
#     model=llm, 
#     client_info=client_info,
#     script_type=ScriptType.RATIO_1_INFLOW,
# ).workflow

# name_verification_graph = NameVerificationAgent(
#     llm=llm, 
#     client_info=client_info,
#     script_type=ScriptType.RATIO_1_INFLOW,
#     config=config
# ).workflow

# details_verification_graph = DetailsVerificationAgent(
#     llm=llm, 
#     client_info=client_info,
#     script_type=ScriptType.RATIO_1_INFLOW,
#     config=config
# ).workflow

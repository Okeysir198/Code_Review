from pyexpat import model
from app_config import CONFIG
from src.Agents.agent01_tools_memory import ToolAgent, DEFAULT_TOOLS
from src.Agents.call_center.call_scripts import ScriptType
from src.Agents.agent00_debtor_call_center import CallCenterAgent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langgraph.types import Command
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import MessagesState
from typing import List, Dict, Literal, Callable, Any, Annotated, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


# Create a new instance with with thread for persist memmory
config = CONFIG.copy()
config['configurable'] = config.get("configurable",{})
config['configurable']['use_memory'] = False
config['configurable']['enable_stt_model'] = False
config['configurable']['enable_tts_model'] = False

llm = ChatOllama(model="qwen2.5:32b-instruct", temperature=0, num_ctx=8192)
#########################################################################################
from src.Agents.agent10_prebuilt_langgraph_agent import test_agent

test_agent1=test_agent

#########################################################################################
# Basic Agent
#########################################################################################
from src.Agents.agent00_basic_agent import BasicAgent, BasicAgentState
system_prompt = "You are AI Agent from Cartrack Accountant Department"

def pre_processing_node(state:BasicAgentState):
    messages = state.get('messages',[]) # + [SystemMessage("Please reponse in Vietnamese only")]
    return {"messages":messages}

def post_processing_node(state:BasicAgentState):
    messages = state.get('messages',[])
    return {"messages":messages}

agent = BasicAgent(
                model=llm, 
                prompt=system_prompt,
                tools = DEFAULT_TOOLS,
                # pre_processing_node=pre_processing_node,
                # post_processing_node=post_processing_node,
                config=config,
                )
graph_basic_agent = agent.workflow
# graph_basic_agent.name = "basic_agent"

#########################################################################################
# Multi-Agents
#########################################################################################

from src.Agents.agent02_multi_agents_system  import MultiAgentSystem 
# config['llm']['model_name'] = "granite3.3"
agent = MultiAgentSystem(config)
graph_multi_agent = agent.workflow

# First, get the UK's GDP over the past 5 years, then make a line chart of it. Once you make the chart, save picture of chart(plot) to to folder /media/ct-dev/newSATA_2tb/langgraph/00_AudioProject_20250418/02_Call_Center_AI_Agent/testcases, then you can finish. DONT FIINISH THE TASK if there is no chart

#########################################################################################
# Simulated user
#########################################################################################
from src.Agents.agent04_agent_eval_sim import simulation
simulated_user = simulation


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
from src.Agents.agent00_callcenter_simulation import  AGENT_CONFIG
from src.Agents.agent00_debtor_call_center import CallCenterAgent

from src.Agents.call_center.introduction import IntroductionAgent
from src.Agents.call_center.name_verification import NameVerificationAgent
from src.Agents.call_center.details_verification import DetailsVerificationAgent

from app_config import CONFIG

# Client information
config = CONFIG.copy()
client_info = config["client_details"]
call_center_graph = CallCenterAgent(config).workflow


# simulation
AGENT_CONFIG['configurable']['use_memory'] = False
AGENT_CONFIG['configurable']['enable_stt_model'] = False
AGENT_CONFIG['configurable']['enable_tts_model'] = False
client_info = AGENT_CONFIG["client_details"]

enable_similation = True
debtor_profile = "wrong_person" # wrong_person, difficult, hesitant, cooperative, third_party
call_center_simulation = CallCenterAgent(AGENT_CONFIG, simulate_debtor=enable_similation, debtor_profile=debtor_profile).workflow



introduction_graph = IntroductionAgent(
    model=llm, 
    client_info=client_info,
    script_type=ScriptType.RATIO_1_INFLOW,
).workflow

name_verification_graph = NameVerificationAgent(
    llm=llm, 
    client_info=client_info,
    script_type=ScriptType.RATIO_1_INFLOW,
    config=config
).workflow

details_verification_graph = DetailsVerificationAgent(
    llm=llm, 
    client_info=client_info,
    script_type=ScriptType.RATIO_1_INFLOW,
    config=config
).workflow

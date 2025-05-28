import logging
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from langgraph.types import Command
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import MessagesState
from typing import List, Dict, Literal, Callable, Any, Annotated, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app_config import CONFIG
from src.Agents.call_center_agent.call_scripts import ScriptType
from test_graph.client_data import client_data

from src.Agents.graph_debtor_simulator import create_debtor_simulator, create_random_debtor
from src.Agents.call_center_agent.introduction import create_introduction_agent
from src.Agents.call_center_agent.name_verification import create_name_verification_agent
from src.Agents.call_center_agent.details_verification import create_details_verification_agent


logger.info("✅ All imports successful")
    


################################################################################
config = CONFIG.copy()
config['configurable'] = config.get("configurable", {})
config['configurable']['use_memory'] = False
config['configurable']['enable_stt_model'] = False
config['configurable']['enable_tts_model'] = False

# Initialize LLM
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=32000)
logger.info("✅ LLM initialized successfully")
    
################################################################################
graph_debtor_simulator = create_debtor_simulator(
    llm=llm,
    client_data=client_data,
    personality="cooperative",
    cooperativeness=0.8,
)
graph_debtor_simulator = create_random_debtor(
    llm=llm,
    client_data=client_data,
)
################################################################################
# Create agents with proper error handling

# Introduction Agent
graph_introduction_agent = create_introduction_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    verbose=True,
    config=config
)
logger.info("✅ Introduction agent created successfully")
    


################################################################################
# Name Verification Agent  
graph_name_verification_agent = create_name_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    verbose=True,
    config=config
)
logger.info("✅ Name verification agent created successfully")

################################################################################
# Details Verification Agent
graph_details_verification_agent = create_details_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    verbose=True,
    config=config
)
logger.info("✅ Details verification agent created successfully")

################################################################################
# Placeholder for reason for call agent
graph_reason_for_call = graph_introduction_agent

logger.info("🎉 All graph agents initialized successfully")
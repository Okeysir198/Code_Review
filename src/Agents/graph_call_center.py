
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from langgraph.types import Command
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import MessagesState
from typing import List, Dict, Literal, Callable, Any, Annotated, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig


from app_config import CONFIG
from src.Agents.call_center_agent.call_scripts import ScriptType
from test_graph.client_data import client_data


from src.Agents.call_center_agent.introduction import create_introduction_agent
from src.Agents.call_center_agent.name_verification import create_name_verification_agent
from src.Agents.call_center_agent.details_verification import create_details_verification_agent

config = CONFIG.copy()
config['configurable'] = config.get("configurable",{})
config['configurable']['use_memory'] = False
config['configurable']['enable_stt_model'] = False
config['configurable']['enable_tts_model'] = False
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=32000)


################################################################################
graph_introduction_agent = create_introduction_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config,
    verbose=True
)

graph_name_verification_agent = create_name_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config,
    verbose=True
)

graph_details_verification_agent = create_details_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config,
    verbose=True
)
graph_reason_for_call = graph_introduction_agent
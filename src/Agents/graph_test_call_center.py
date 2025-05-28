# Updated src/Agents/graph_test_call_center.py
"""
Simplified graph test file - just includes the graphs, no testing code.
"""
import logging
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app_config import CONFIG
from src.Agents.call_center_agent.call_scripts import ScriptType
from test_graph.client_data import client_data

from src.Agents.graph_debtor_simulator import create_debtor_simulator
from src.Agents.graph_call_center_agent import create_call_center_agent

from src.Agents.call_center_agent.state import CallCenterAgentState

logger.info("✅ All imports successful")

################################################################################
# Configuration
config = CONFIG.copy()
config['configurable'] = config.get("configurable", {})
config['configurable']['use_memory'] = False
config['configurable']['enable_stt_model'] = True
config['configurable']['enable_tts_model'] = True

# Initialize LLM
llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0, num_ctx=32000)
logger.info("✅ LLM initialized successfully")

################################################################################
# Debtor Simulator
graph_debtor_simulator = create_debtor_simulator(
    llm=llm,
    client_data=client_data,
    personality="cooperative",
    cooperativeness=0.8,
)
logger.info("✅ Debtor simulator agent created successfully")

################################################################################
# Complete Call Center Agent with LLM Router
graph_call_center_agent = create_call_center_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config
)
logger.info("✅ Complete call center agent with LLM router created successfully")

################################################################################
# Individual Agent Testing (Optional - for debugging specific agents)

from src.Agents.call_center_agent.step00_introduction import create_introduction_agent
from src.Agents.call_center_agent.step01_name_verification import create_name_verification_agent
from src.Agents.call_center_agent.step02_details_verification import create_details_verification_agent
from src.Agents.call_center_agent.step03_reason_for_call import create_reason_for_call_agent
from src.Agents.call_center_agent.step04_negotiation import create_negotiation_agent

# Individual agents for debugging
graph_introduction_agent = create_introduction_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config
)

graph_name_verification_agent = create_name_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config
)

graph_details_verification_agent = create_details_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config
)

graph_reason_for_call_agent = create_reason_for_call_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config
)

graph_negotiation_agent = create_negotiation_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    config=config
)

logger.info("✅ Individual agents created for debugging")

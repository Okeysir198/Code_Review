# Updated src/Agents/graph_test_call_center.py
"""
Updated graph test file with refactored call center agents - no analysis, self-contained agents
"""
import logging
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app_config import CONFIG

from src.Agents.call_center_agent.call_scripts import ScriptType
from src.Agents.call_center_agent.data import get_client_data
from src.Agents.call_center_agent.state import CallCenterAgentState

from src.Agents.graph_call_center_agent import create_call_center_agent
from src.Agents.graph_call_simulation import *
from src.Agents.graph_debtor_simulator import create_debtor_simulator, DebtorPersonality
from test_graph.client_data_samples import client_data_set

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

# ################################################################################
# # Debtor Simulator
# graph_debtor_simulator = create_debtor_simulator(
#     llm=llm,
#     client_data=client_data,
#     personality="cooperative",
#     cooperativeness=0.8,
# )
# logger.info("✅ Debtor simulator agent created successfully")

################################################################################
# Complete Call Center Agent with Optimized Router
user_id = "1489698"
# client_data = get_client_data(user_id)

client_data = client_data_set[0]


graph_call_center_agent_ratio_1 = create_call_center_agent(
    model=llm,
    client_data=client_data_set[0],
    config=config,
    verbose=True,
)
graph_call_center_agent_ratio_2_3 = create_call_center_agent(
    model=llm,
    client_data=client_data_set[1],
    config=config,
    verbose=True,
)
graph_call_center_agent_ratio_4_5 = create_call_center_agent(
    model=llm,
    client_data=client_data_set[2],
    config=config,
    verbose=True,
)
logger.info("✅ Complete call center agent with optimized router created successfully")
agent_llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0)
debtor_llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.7)

call_simulation = create_call_simulation(
    agent_llm=agent_llm,
    debtor_llm=debtor_llm,
    client_data=client_data,
    debtor_personality=DebtorPersonality.THIRD_PARTY_ASSISTANT.value,
    max_turns=50,
    config=CONFIG
)

################################################################################
# # Import All Individual Step Agents (Updated imports for refactored agents)
# from src.Agents.call_center_agent.step00_introduction import create_introduction_agent
# from src.Agents.call_center_agent.step01_name_verification import create_name_verification_agent
# from src.Agents.call_center_agent.step02_details_verification import create_details_verification_agent
# from src.Agents.call_center_agent.step03_reason_for_call import create_reason_for_call_agent
# from src.Agents.call_center_agent.step04_negotiation import create_negotiation_agent
# from src.Agents.call_center_agent.step05_promise_to_pay import create_promise_to_pay_agent
# from src.Agents.call_center_agent.step06_debicheck_setup import create_debicheck_setup_agent
# from src.Agents.call_center_agent.step07_payment_portal import create_payment_portal_agent
# from src.Agents.call_center_agent.step08_subscription_reminder import create_subscription_reminder_agent
# from src.Agents.call_center_agent.step09_client_details_update import create_client_details_update_agent
# from src.Agents.call_center_agent.step10_referrals import create_referrals_agent
# from src.Agents.call_center_agent.step11_further_assistance import create_further_assistance_agent
# from src.Agents.call_center_agent.step12_query_resolution import create_query_resolution_agent
# from src.Agents.call_center_agent.step13_escalation import create_escalation_agent
# from src.Agents.call_center_agent.step14_cancellation import create_cancellation_agent
# from src.Agents.call_center_agent.step15_closing import create_closing_agent

# logger.info("✅ All step agent imports successful")

################################################################################
# Individual Step Agents (Direct Instantiation with Refactored Architecture)

# # Core Call Flow Agents
# graph_introduction_agent = create_introduction_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_name_verification_agent = create_name_verification_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_details_verification_agent = create_details_verification_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_reason_for_call_agent = create_reason_for_call_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_negotiation_agent = create_negotiation_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# logger.info("✅ Core call flow agents created successfully")

# # Payment Processing Agents
# graph_promise_to_pay_agent = create_promise_to_pay_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_debicheck_setup_agent = create_debicheck_setup_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_payment_portal_agent = create_payment_portal_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_subscription_reminder_agent = create_subscription_reminder_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# logger.info("✅ Payment processing agents created successfully")

# # Account Management Agents
# graph_client_details_update_agent = create_client_details_update_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_referrals_agent = create_referrals_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_further_assistance_agent = create_further_assistance_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# logger.info("✅ Account management agents created successfully")

# # Special Handling Agents
# graph_query_resolution_agent = create_query_resolution_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_escalation_agent = create_escalation_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_cancellation_agent = create_cancellation_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# graph_closing_agent = create_closing_agent(
#     model=llm,
#     client_data=client_data,
#     script_type=ScriptType.RATIO_1_INFLOW.value,
#     agent_name="AI Agent",
#     config=config
# )

# logger.info("✅ Special handling agents created successfully")
# ################################################################################
# agent_llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0)
# debtor_llm = ChatOllama(model="qwen2.5:3b-instruct", temperature=0.7)

# from src.Agents.graph_call_simulation import *

# # Create all simulation types
# simulation_cooperative = create_cooperative_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )

# simulation_difficult = create_difficult_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )

# simulation_confused = create_confused_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )

# simulation_busy = create_busy_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )

# simulation_suspicious = create_suspicious_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )

# simulation_wrong_person = create_wrong_person_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )

# simulation_third_party_spouse = create_third_party_spouse_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )
# simulation_third_party_assistant = create_third_party_assistant_simulation(
#     agent_llm=llm,
#     debtor_llm=debtor_llm,
#     client_data=client_data,
#     config=CONFIG
# )
# simulation_third_party_parent = create_third_party_parent_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )
# simulation_third_party_employee = create_third_party_employee_simulation(
#     agent_llm=agent_llm,
#     debtor_llm=debtor_llm, 
#     client_data=client_data,
#     config=CONFIG
# )


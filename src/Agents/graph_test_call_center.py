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
# Complete Call Center Agent with Optimized Router
graph_call_center_agent1 = create_call_center_agent(
    model=llm,
    client_data=client_data,
    config=config
)
logger.info("✅ Complete call center agent with optimized router created successfully")

################################################################################
# Import All Individual Step Agents (Updated imports for refactored agents)
from src.Agents.call_center_agent.step00_introduction import create_introduction_agent
from src.Agents.call_center_agent.step01_name_verification import create_name_verification_agent
from src.Agents.call_center_agent.step02_details_verification import create_details_verification_agent
from src.Agents.call_center_agent.step03_reason_for_call import create_reason_for_call_agent
from src.Agents.call_center_agent.step04_negotiation import create_negotiation_agent
from src.Agents.call_center_agent.step05_promise_to_pay import create_promise_to_pay_agent
from src.Agents.call_center_agent.step06_debicheck_setup import create_debicheck_setup_agent
from src.Agents.call_center_agent.step07_payment_portal import create_payment_portal_agent
from src.Agents.call_center_agent.step08_subscription_reminder import create_subscription_reminder_agent
from src.Agents.call_center_agent.step09_client_details_update import create_client_details_update_agent
from src.Agents.call_center_agent.step10_referrals import create_referrals_agent
from src.Agents.call_center_agent.step11_further_assistance import create_further_assistance_agent
from src.Agents.call_center_agent.step12_query_resolution import create_query_resolution_agent
from src.Agents.call_center_agent.step13_escalation import create_escalation_agent
from src.Agents.call_center_agent.step14_cancellation import create_cancellation_agent
from src.Agents.call_center_agent.step15_closing import create_closing_agent

logger.info("✅ All step agent imports successful")

################################################################################
# Individual Step Agents (Direct Instantiation with Refactored Architecture)

# Core Call Flow Agents
graph_introduction_agent = create_introduction_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_name_verification_agent = create_name_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_details_verification_agent = create_details_verification_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_reason_for_call_agent = create_reason_for_call_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_negotiation_agent = create_negotiation_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

logger.info("✅ Core call flow agents created successfully")

# Payment Processing Agents
graph_promise_to_pay_agent = create_promise_to_pay_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_debicheck_setup_agent = create_debicheck_setup_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_payment_portal_agent = create_payment_portal_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_subscription_reminder_agent = create_subscription_reminder_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

logger.info("✅ Payment processing agents created successfully")

# Account Management Agents
graph_client_details_update_agent = create_client_details_update_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_referrals_agent = create_referrals_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_further_assistance_agent = create_further_assistance_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

logger.info("✅ Account management agents created successfully")

# Special Handling Agents
graph_query_resolution_agent = create_query_resolution_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_escalation_agent = create_escalation_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_cancellation_agent = create_cancellation_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

graph_closing_agent = create_closing_agent(
    model=llm,
    client_data=client_data,
    script_type=ScriptType.RATIO_1_INFLOW.value,
    agent_name="AI Agent",
    config=config
)

logger.info("✅ Special handling agents created successfully")

################################################################################
# Testing Utilities

def get_all_individual_agents():
    """Return dictionary of all individual agents for testing."""
    return {
        "introduction": graph_introduction_agent,
        "name_verification": graph_name_verification_agent,
        "details_verification": graph_details_verification_agent,
        "reason_for_call": graph_reason_for_call_agent,
        "negotiation": graph_negotiation_agent,
        "promise_to_pay": graph_promise_to_pay_agent,
        "debicheck_setup": graph_debicheck_setup_agent,
        "payment_portal": graph_payment_portal_agent,
        "subscription_reminder": graph_subscription_reminder_agent,
        "client_details_update": graph_client_details_update_agent,
        "referrals": graph_referrals_agent,
        "further_assistance": graph_further_assistance_agent,
        "query_resolution": graph_query_resolution_agent,
        "escalation": graph_escalation_agent,
        "cancellation": graph_cancellation_agent,
        "closing": graph_closing_agent
    }

def get_core_workflow_agents():
    """Return dictionary of core workflow agents (main call flow)."""
    return {
        "introduction": graph_introduction_agent,
        "name_verification": graph_name_verification_agent,
        "details_verification": graph_details_verification_agent,
        "reason_for_call": graph_reason_for_call_agent,
        "negotiation": graph_negotiation_agent,
        "promise_to_pay": graph_promise_to_pay_agent,
        "closing": graph_closing_agent
    }

def get_payment_agents():
    """Return dictionary of payment-related agents."""
    return {
        "promise_to_pay": graph_promise_to_pay_agent,
        "debicheck_setup": graph_debicheck_setup_agent,
        "payment_portal": graph_payment_portal_agent,
        "subscription_reminder": graph_subscription_reminder_agent
    }

def get_special_handling_agents():
    """Return dictionary of special handling agents."""
    return {
        "query_resolution": graph_query_resolution_agent,
        "escalation": graph_escalation_agent,
        "cancellation": graph_cancellation_agent,
        "closing": graph_closing_agent
    }

################################################################################
# Debtor Simulator Variants for Testing Different Scenarios

def get_debtor_simulators():
    """Create different debtor personality simulators for testing."""
    simulators = {}
    
    # Cooperative debtor
    simulators["cooperative"] = create_debtor_simulator(
        llm=llm,
        client_data=client_data,
        personality="cooperative",
        cooperativeness=0.9
    )
    
    # Difficult debtor
    simulators["difficult"] = create_debtor_simulator(
        llm=llm,
        client_data=client_data,
        personality="difficult",
        cooperativeness=0.3
    )
    
    # Confused debtor
    simulators["confused"] = create_debtor_simulator(
        llm=llm,
        client_data=client_data,
        personality="confused",
        cooperativeness=0.6
    )
    
    # Busy debtor
    simulators["busy"] = create_debtor_simulator(
        llm=llm,
        client_data=client_data,
        personality="busy",
        cooperativeness=0.7
    )
    
    # Suspicious debtor
    simulators["suspicious"] = create_debtor_simulator(
        llm=llm,
        client_data=client_data,
        personality="suspicious",
        cooperativeness=0.4
    )
    
    # Wrong person
    simulators["wrong_person"] = create_debtor_simulator(
        llm=llm,
        client_data=client_data,
        personality="wrong_person",
        cooperativeness=0.0
    )
    
    # Third party spouse
    simulators["third_party_spouse"] = create_debtor_simulator(
        llm=llm,
        client_data=client_data,
        personality="third_party_spouse",
        cooperativeness=0.7
    )
    
    return simulators

# Create debtor simulators
graph_debtor_simulators = get_debtor_simulators()

logger.info("✅ All debtor simulator variants created successfully")

################################################################################
# Summary and Status

# logger.info("=" * 80)
# logger.info("REFACTORED CALL CENTER AGENTS - SIMPLIFIED ARCHITECTURE")
# logger.info("=" * 80)
# logger.info("✅ Main call center agent with optimized router: graph_call_center_agent1")
# logger.info("✅ All 16 individual step agents - self-contained with own prompts")
# logger.info("✅ All debtor simulator variants (7 personalities) created")
# logger.info("✅ Removed complex analysis - simple data fetching only")
# logger.info("✅ Each agent owns its prompt and parameter logic")
# logger.info("✅ Outstanding amount calculation: overdue only (not total balance)")
# logger.info("=" * 80)
# logger.info("🎯 Architecture: Self-contained agents with embedded prompts")
# logger.info("📦 Data fetching: Simple client_data_fetcher.py")
# logger.info("🗑️ Removed: prompts.py (800+ lines) + data_parameter_builder.py (1000+ lines)")
# logger.info("=" * 80)

################################################################################
# Export key functions and objects for external use

# __all__ = [
#     # Main agents
#     'graph_call_center_agent1',
#     'graph_debtor_simulator',
#     'graph_debtor_simulators',
    
#     # Individual step agents
#     'graph_introduction_agent',
#     'graph_name_verification_agent',
#     'graph_details_verification_agent',
#     'graph_reason_for_call_agent',
#     'graph_negotiation_agent',
#     'graph_promise_to_pay_agent',
#     'graph_debicheck_setup_agent',
#     'graph_payment_portal_agent',
#     'graph_subscription_reminder_agent',
#     'graph_client_details_update_agent',
#     'graph_referrals_agent',
#     'graph_further_assistance_agent',
#     'graph_query_resolution_agent',
#     'graph_escalation_agent',
#     'graph_cancellation_agent',
#     'graph_closing_agent',
    
#     # Utility functions
#     'get_all_individual_agents',
#     'get_core_workflow_agents',
#     'get_payment_agents',
#     'get_special_handling_agents',
#     'get_debtor_simulators'
# ]

# print(f"📦 Exported {len(__all__)} objects for external use")
# print("🎯 Main agent: graph_call_center_agent1")
# print("🔧 Self-contained agents with embedded prompts")
# print("📊 Simple data fetching with outstanding amount calculation")
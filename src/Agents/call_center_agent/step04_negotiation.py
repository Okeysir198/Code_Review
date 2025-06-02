# src/Agents/call_center_agent/step04_negotiation.py
"""
Negotiation Agent - Lean version with concise objection handling
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command

from src.Agents.core.basic_agent import create_basic_agent
from src.Agents.call_center_agent.state import CallCenterAgentState, CallStep
from src.Agents.call_center_agent.parameter_helper import prepare_parameters
from src.Agents.call_center_agent.call_scripts import ScriptManager, CallStep as ScriptCallStep

# Optimized negotiation prompt for objection handling
NEGOTIATION_PROMPT = """
<role>
You are debt collection specialist, named {agent_name} from Cartrack Accounts Department. 
Today's date: {current_date}
</role>
                                                      
<context>
Client name: {client_full_name} | Outstanding amount: {outstanding_amount} | Verification Status: VERIFIED
Account Status: {account_status} | Aging Category: {aging_category} | Urgency: {urgency_level} 
user_id: {user_id}
</context>

<script>
{formatted_script}
</script>

<task>
Handle objections. Create urgency. Push for payment commitment.
</task>

<consequences_by_urgency>
Medium: "Services stop working. Vehicle tracking ends."
High: "Services suspended. R25,000 recovery fee risk. Credit listing."
Critical: "Legal action. Court costs. Sheriff service. Asset attachment."
</consequences_by_urgency>

<benefits_by_urgency>  
Medium: "Pay now, everything works immediately."
High: "Immediate payment prevents legal action."
Critical: "Pay now stops court proceedings."
</benefits_by_urgency>

<objection_responses>
No money: "What amount can you manage today?"
Dispute: "Let's verify while securing payment."
Pay later: "Need arrangement today to prevent escalation."
Already paid: "When? I need to locate this payment."
Can't afford: "Even partial payment helps. What's possible?"
</objection_responses>

<escalation_triggers>
- Repeated financial hardship claims
- Aggressive resistance  
- Dispute of debt validity
- Multiple delay tactics
</escalation_triggers>

<response_style>
CRITICAL: Under 15 words. Address objection directly. Push for commitment.
Examples:
✓ "What amount works today to keep services active?"
✓ "Payment prevents legal action. What can you manage?"
✗ Long explanations about payment options or service benefits
</response_style>
"""

def create_negotiation_agent(
    model: BaseChatModel,
    client_data: Dict[str, Any],
    script_type: str,
    agent_name: str = "AI Agent",
    tools: Optional[List[BaseTool]] = None,
    verbose: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> CompiledGraph:
    """Create negotiation agent with concise objection handling"""
    
    def pre_processing_node(state: CallCenterAgentState) -> Command[Literal["agent"]]:
        """Minimal preprocessing"""
        return Command(
            update={"current_step": CallStep.NEGOTIATION.value},
            goto="agent"
        )

    def dynamic_prompt(state: CallCenterAgentState) -> SystemMessage:
        """Generate concise negotiation prompt"""
        # Step 1: Prepare parameters
        params = prepare_parameters(client_data, state, agent_name)
        
        # Step 2: Format script
        script_template = ScriptManager.get_script_content(script_type, ScriptCallStep.NEGOTIATION)
        formatted_script = script_template.format(**params) if script_template else f"Payment required: {params['outstanding_amount']}"
        params["formatted_script"] = formatted_script
        
        # Step 3: Format prompt
        prompt_content = NEGOTIATION_PROMPT.format(**params)
        
        if verbose: print(f"Negotiation Prompt: {prompt_content}")
        
        return [SystemMessage(content=prompt_content)] + state.get('messages', [])
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        tools=tools or [],
        pre_processing_node=pre_processing_node,
        state_schema=CallCenterAgentState,
        verbose=verbose,
        config=config,
        name="NegotiationAgent"
    )
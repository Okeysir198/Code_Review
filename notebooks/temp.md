# Call Center Agent LLM Instructions

Lean, focused guidance for building a professional debt collection agent system with smart routing and specialized sub-agents.

## 1. Sub-Agent Architecture

**Create individual sub-agents** - never use dictionaries, use variable directly:

```python
introduction_agent = create_introduction_agent(model, client_data, script_type, agent_name, config=config)
name_verification_agent = create_name_verification_agent(model, client_data, script_type, agent_name, config=config)
```

## 2. Node Pattern: Router → Agent → End (with Smart Handoffs)

**Standard Flow**: Router directs to sub-agent → Agent handles ONE turn → Smart handoff OR return to `__end__`

```python
def name_verification_node(state: CallCenterAgentState) -> Command[Literal["details_verification", "__end__"]]:
    result = name_verification_agent.invoke(state)
    
    update = {
        "messages": result.get("messages", []),
        "name_verification_status": result.get("name_verification_status"),
        "current_step": CallStep.NAME_VERIFICATION.value
    }
    
    # Smart handoff for smooth conversations
    if result.get("name_verification_status") == VerificationStatus.VERIFIED.value:
        update["current_step"] = CallStep.DETAILS_VERIFICATION.value
        return Command(update=update, goto=CallStep.DETAILS_VERIFICATION.value)
    
    return Command(update=update, goto="__end__")  # Wait for debtor
```


## 4. Sub-Agent Design: Pre-Processing Only

**Each sub-agent uses ONLY pre-processing node** - no post-processing:

```python
def create_promise_to_pay_agent(model, client_data, script_type, agent_name, config):
    
    def pre_processing_node(state) -> Command[Literal["agent"]]:
        # Analyze conversation and prepare context
        payment_analysis = analyze_payment_willingness(state.get("messages", []))
        
        return Command(
            update={
                "payment_willingness": payment_analysis["willingness"],
                "suggested_amount": payment_analysis["amount"],
                "payment_method_preference": payment_analysis["method"],
                "outstanding_amount": outstanding_amount  # Calculated overdue amount
            },
            goto="agent"
        )
    
    def dynamic_prompt(state) -> SystemMessage:
        parameters = prepare_parameters(client_data, current_step, state, script_type, agent_name)
        prompt_content = get_step_prompt(CallStep.PROMISE_TO_PAY.value, parameters)
        return [SystemMessage(content=prompt_content)] + state['messages']
    
    return create_basic_agent(
        model=model,
        prompt=dynamic_prompt,
        pre_processing_node=pre_processing_node,  # ONLY pre-processing
        # NO post_processing_node
        state_schema=CallCenterAgentState,
        config=config,
        name="PromiseToPayAgent"
    )
```



## 7. Smart Router Logic

**Priority Order**: Emergency → Business Rules → Off-Topic Detection → Normal Flow

```python
def router_node(state: CallCenterAgentState) -> str:
    # 1. Emergency keywords (highest priority)
    if last_message_contains(["supervisor", "cancel"]):
        return CallStep.ESCALATION.value
    
    # 2. Business rule overrides
    if verification_failed(state) or max_attempts_reached(state):
        return CallStep.CLOSING.value
    
    # 3. Off-topic detection (LLM classification)
    if classify_message_intent(state) == "QUERY_UNRELATED":
        state["return_to_step"] = state.get("current_step")
        return CallStep.QUERY_RESOLUTION.value
    
    # 4. Normal progression
    return get_next_step(state.get("current_step"))
```



## 9. Smart Handoff Conditions

**Immediate handoffs for smooth conversations**:

```python
# Name Verification → Details Verification
if name_verification_status == VerificationStatus.VERIFIED.value:
    return Command(update=update, goto=CallStep.DETAILS_VERIFICATION.value)

# Details Verification → Reason for Call  
if details_verification_status == VerificationStatus.VERIFIED.value:
    return Command(update=update, goto=CallStep.REASON_FOR_CALL.value)

# Reason for Call → Negotiation (always handoff)
return Command(update=update, goto=CallStep.NEGOTIATION.value)

# Negotiation → Promise to Pay (always handoff)
return Command(update=update, goto=CallStep.PROMISE_TO_PAY.value)

# Promise to Pay → Payment Method
if payment_secured and payment_method == "debicheck":
    return Command(update=update, goto=CallStep.DEBICHECK_SETUP.value)
elif payment_secured and payment_method == "payment_portal":
    return Command(update=update, goto=CallStep.PAYMENT_PORTAL.value)
```

## 10. Core Principles

- **Pre-Processing Only**: Sub-agents use ONLY pre-processing nodes
- **Prompt Logging**: Complete prompts logged to console for debugging
- **One Turn Rule**: Each agent handles exactly one conversational exchange
- **Smart Handoffs**: Chain multiple steps in smooth conversations
- **Default to `__end__`**: Return here when waiting for debtor response
- **Router Controls Flow**: All routing decisions centralized in router by update current_step in state
- **Brief Responses**: Target 10-20 words per agent response
- **Payment Focus**: Always redirect conversations toward payment resolution

## 11. Response Guidelines

**Professional Debt Collection Voice**:
- Direct but respectful
- Solution-focused, not confrontational
- Create urgency without aggression
- Minimal pleasantries
- Avoid repetitive "thank you"
- No generic support phrases

**Example Responses**:
- "We didn't receive your payment. Can we debit R399 today?"
- "Services stop without payment. Let's arrange this now."
- "I understand. What amount can you manage today?"

## 12. Handoff Triggers

**Automatic progression when**:
- ✅ Identity verified → Move to details verification
- ✅ Details verified → Move to reason for call
- ✅ Account explained → Move to negotiation
- ✅ Negotiation completed → Move to promise to pay
- ✅ Payment agreed → Move to specific payment method
- ✅ Payment completed → Move to subscription reminder

**Wait for response at the current step when**:
- ❓ Verification incomplete
- ❓ Objections raised
- ❓ Client needs time to respond
- ❓ Payment method being processed

## 13. Special Requirements

**Details Verification**: For first attempt, always mention: "This call is recorded for quality and security"

**Query Handling**: Answer in ≤15 words, then redirect:
- Q: "Who are you?" 
- A: "I'm [Agent] from Cartrack. Can we debit R399 today?"

**Emergency Keywords**: "supervisor", "manager", "cancel" → Immediate escalation

**Parameter Logging**: Every prompt logged with complete parameter resolution and aging breakdown before LLM invocation

**No Post-Processing**: Sub-agents focus solely on generating appropriate responses through pre-processing context and dynamic prompts

This architecture ensures focused, professional debt collection calls with smart routing, natural conversation flow, efficient handoffs, clean separation of concerns, robust parameter validation, accurate outstanding amount calculation, and full debugging visibility.
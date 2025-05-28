## Call Center Agent LLM Instructions

These instructions define the core architecture and operational patterns for building a robust call center agent system using sub-agents and a smart router.

### 1. Sub-Agent Creation

Always create sub-agents directly using **individual variables**, not dictionaries.

**Correct Example:**

```python
introduction_agent = create_introduction_agent(model, client_data, script_type, agent_name, config=config)
name_verification_agent = create_name_verification_agent(model, client_data, script_type, agent_name, config=config)
# ... other specialized agents
```

### 2. Node Implementation Pattern: Router → Sub-Agent → End

The standard flow involves the **Router** directing to a sub-agent. Each **sub-agent handles one turn** of conversation and then transitions to `__end__` to await the debtor's response. A new message from the debtor will trigger the router again.

#### Standard Node Structure (Most Nodes End at `__end__`)

Nodes will invoke their respective sub-agent, update the state, and then typically go to `__end__`. Smart routing for direct handovers (e.g., after successful verification) is an exception.

**Example: `name_verification_node`**

```python
def name_verification_node(state: CallCenterAgentState) -> Command[Literal["details_verification", "__end__"]]:
    """Name verification step: one turn, then end or direct handover if verified."""
    result = name_verification_agent.invoke(state)
    messages = result.get("messages", state.get("messages", []))
    name_status = result.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
    name_attempts = result.get("name_verification_attempts", 0)

    update = {
        "messages": messages,
        "name_verification_status": name_status,
        "name_verification_attempts": name_attempts,
        "current_step": CallStep.NAME_VERIFICATION.value
    }

    # Smart Routing: If verified, directly move to details verification
    if name_status == VerificationStatus.VERIFIED.value:
        update["current_step"] = CallStep.DETAILS_VERIFICATION.value
        return Command(update=update, goto=CallStep.DETAILS_VERIFICATION.value)

    # Otherwise, end and wait for debtor response
    return Command(update=update, goto="__end__")
```

**Example: `negotiation_node` (Always Ends)**

```python
def negotiation_node(state: CallCenterAgentState) -> Command[Literal["__end__"]]:
    """Negotiation step: one turn, then end to wait for response."""
    result = negotiation_agent.invoke(state)
    messages = result.get("messages", state.get("messages", []))
    return Command(
        update={
            "messages": messages,
            "current_step": CallStep.NEGOTIATION.value
        },
        goto="__end__"  # End and wait for debtor response
    )
```

### 3. Smart Router with Off-Topic Detection

The router is central to managing call flow, detecting off-topic queries, and handling emergencies.

```python
def enhanced_router_node(state: CallCenterAgentState) -> str:
    """Smart router with off-topic detection and return-to-goal logic."""

    # 1. Emergency Routing (Highest Priority): Direct override if set.
    if state.get("route_override"):
        return state.get("route_override")

    # 2. Emergency Keyword Detection: Check last message for critical keywords.
    if state.get("messages"):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
            content_lower = last_message.content.lower()
            if any(word in content_lower for word in ["cancel", "terminate", "stop service"]):
                return CallStep.CANCELLATION.value
            if any(word in content_lower for word in ["supervisor", "manager", "complaint"]):
                return CallStep.ESCALATION.value

    # 3. Off-Topic Query Detection (using LLM for classification):
    if state.get("messages") and len(state["messages"]) > 1:
        classification = classify_message_intent(state) # Assumes LLM-based classification
        if classification == "QUERY_UNRELATED":
            # Store current step to return after query resolution
            state["return_to_step"] = state.get("current_step", CallStep.INTRODUCTION.value)
            return CallStep.QUERY_RESOLUTION.value

    # 4. Business Rule Routing: Handle verification failures or max attempts.
    if has_hard_state_override(state): # Assumes function for business rule checks
        return get_state_override_route(state)

    # 5. Normal Call Flow Progression: Default advancement.
    current_step = state.get("current_step", CallStep.INTRODUCTION.value)
    return get_default_next_step(current_step, state) # Assumes function for normal progression
```

### 4. Query Resolution with Return-to-Goal

Handle off-topic queries briefly and then return to the main call objective.

```python
def query_resolution_node(state: CallCenterAgentState) -> Command[Literal["reason_for_call", "negotiation", "promise_to_pay", "closing", "__end__"]]:
    """Query resolution: answer briefly, then return to main goal."""
    result = query_resolution_agent.invoke(state)
    messages = result.get("messages", state.get("messages", []))
    return_to_step = state.get("return_to_step", CallStep.CLOSING.value) # Default to closing if no return step

    # Clear the return step and go back to the main call goal
    return Command(
        update={
            "messages": messages,
            "current_step": return_to_step, # Direct return to main goal
            "return_to_step": None  # Clear return step
        },
        goto="__end__"  # End and wait for debtor response
    )
```

### 5. Escalation/Cancellation with Return Logic

Handle emergency requests and then determine the appropriate next step based on call progress.

```python
def escalation_node(state: CallCenterAgentState) -> Command[Literal["reason_for_call", "negotiation", "closing", "__end__"]]:
    """Escalation: handle request, then return to main goal or close."""
    result = escalation_agent.invoke(state)
    messages = result.get("messages", state.get("messages", []))
    
    # Determine where to return after escalation based on call progress
    if state.get("details_verification_status") == VerificationStatus.VERIFIED.value:
        next_step = CallStep.NEGOTIATION.value if not state.get("payment_secured") else CallStep.CLOSING.value
    else:
        next_step = CallStep.CLOSING.value # If not verified, close the call

    return Command(
        update={
            "messages": messages,
            "current_step": next_step
        },
        goto="__end__"  # End and wait for debtor response
    )
```

---

## Core Principles Summarized

* **Router-Driven Flow**: The router orchestrates the entire call flow, including smart routing for interruptions.
* **One Turn Per Agent**: Each sub-agent is designed to complete a single conversational exchange before yielding control.
* **Default to `__end__`**: Most nodes conclude by returning to `__end__`, allowing the system to wait for the next debtor input. Direct handovers are specific exceptions.
* **Return-to-Goal**: After resolving off-topic queries or emergencies, the system prioritizes returning to the primary call objective.
* **State Tracking**: Utilize `return_to_step` in the state to maintain context for returning to previous goals.
* **Emergency Override**: Critical events like cancellation or escalation can interrupt any active step.
* **No Router to `__end__` Directly**: The router routes to *nodes*. If a call truly ends, it should route to a dedicated closing node, which then goes to `__end__`.

---

## Flow Examples

### Normal Flow

`Router` → `Name Verification` → (if verified) → `Details Verification` → (if verified) → `Reason for Call` → `__END__`

### Off-Topic Interruption

`Router` → `Negotiation` → `__END__` → (debtor asks off-topic query) → `Router` → `Query Resolution` (updates current step back to Negotiation) → `__END__`

### Emergency Interruption

`Router` → `Negotiation` → `__END__` → (debtor says "cancel") → `Router` → `Cancellation` → `Closing` → `__END__`



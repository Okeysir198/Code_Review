# src/Agents/call_center_agent/state.py
"""
Enhanced Call Center Agent State - Optimized for fast preprocessing and conversation tracking
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from langgraph.graph.message import MessagesState

class CallStep(Enum):
    """Call workflow steps"""
    INTRODUCTION = "introduction"
    NAME_VERIFICATION = "name_verification"
    DETAILS_VERIFICATION = "details_verification" 
    REASON_FOR_CALL = "reason_for_call"
    NEGOTIATION = "negotiation"
    PROMISE_TO_PAY = "promise_to_pay"
    DEBICHECK_SETUP = "debicheck_setup"
    PAYMENT_PORTAL = "payment_portal"
    SUBSCRIPTION_REMINDER = "subscription_reminder"
    CLIENT_DETAILS_UPDATE = "client_details_update"
    REFERRALS = "referrals"
    FURTHER_ASSISTANCE = "further_assistance"
    QUERY_RESOLUTION = "query_resolution"
    CANCELLATION = "cancellation"
    ESCALATION = "escalation"
    CLOSING = "closing"

class VerificationStatus(Enum):
    """Verification statuses"""
    VERIFIED = "VERIFIED"
    THIRD_PARTY = "THIRD_PARTY"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    WRONG_PERSON = "WRONG_PERSON"  
    UNAVAILABLE = "UNAVAILABLE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"

class ClientMood(Enum):
    """Client mood detection for conversation adaptation"""
    COOPERATIVE = "cooperative"
    RESISTANT = "resistant"
    CONFUSED = "confused"
    ANGRY = "angry"
    FINANCIAL_STRESS = "financial_stress"
    SUSPICIOUS = "suspicious"
    NEUTRAL = "neutral"

class CallCenterAgentState(MessagesState):
    """
    Enhanced Call Center Agent State - Optimized for preprocessing and conversation flow
    """
    
    # ===== CORE CALL FLOW =====
    current_step: str = CallStep.INTRODUCTION.value
    is_call_ended: bool = False
    call_outcome: str = "in_progress"  # in_progress, completed, payment_secured, escalated, cancelled
    
    # ===== ADAPTATION =====
    urgency_category: str = "Standard/First Payment"  # Standard, Higher Urgency, Pre-Legal/Critical

    # ===== VERIFICATION STATE =====
    name_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    name_verification_attempts: int = 0
    details_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    details_verification_attempts: int = 0
    matched_fields: List[str] = []
    field_to_verify: str = "ID number"  # Current field being verified
    
    # ===== PAYMENT STATE =====
    payment_secured: bool = False
    payment_method_preference: str = ""  # debit_order, online, etc.
    outstanding_amount: str = "R 0.00"
    
    # ===== CLIENT INTERACTION TRACKING =====
    client_mood: str = ClientMood.NEUTRAL.value
    cooperation_level: float = 0.5  # 0.0 (very difficult) to 1.0 (very cooperative)
    rapport_level: float = 0.5  # 0.0 (hostile) to 1.0 (excellent rapport)
    
    # ===== CONVERSATION CONTEXT =====
    last_intent: str = "none"  # Last detected client intent
    conversation_turn_count: int = 0
    client_concerns: List[str] = []  # Tracked client concerns/objections
    topics_discussed: List[str] = []  # Major topics covered
    
    # ===== REQUEST TRACKING =====
    escalation_requested: bool = False
    cancellation_requested: bool = False
    return_to_step: Optional[str] = None  # For query resolution routing
    
    # ===== TOOL USAGE TRACKING =====
    tools_used: List[str] = []  # Track which tools were called
    tool_results: Dict[str, Any] = {}  # Store important tool results
    
    # ===== ENHANCED HELPER METHODS =====
    def is_verified(self) -> bool:
        """Check if client is fully verified"""
        return (self.name_verification_status == VerificationStatus.VERIFIED.value and 
                self.details_verification_status == VerificationStatus.VERIFIED.value)
    
    def can_discuss_account(self) -> bool:
        """Check if can discuss account details"""
        return self.is_verified()
    
    def is_terminal_verification_status(self) -> bool:
        """Check if verification has reached a terminal state"""
        terminal_statuses = [
            VerificationStatus.WRONG_PERSON.value,
            VerificationStatus.VERIFICATION_FAILED.value,
            VerificationStatus.THIRD_PARTY.value,
            VerificationStatus.UNAVAILABLE.value
        ]
        return (self.name_verification_status in terminal_statuses or 
                self.details_verification_status in terminal_statuses)
    
    def should_end_call(self) -> bool:
        """Determine if call should be ended based on current state"""
        return (
            self.is_call_ended or
            self.is_terminal_verification_status() or
            self.escalation_requested or
            self.cancellation_requested or
            self.conversation_turn_count > 50  # Prevent infinite loops
        )
    
    def get_verification_progress(self) -> Dict[str, Any]:
        """Get verification progress summary"""
        return {
            "name_status": self.name_verification_status,
            "name_attempts": self.name_verification_attempts,
            "details_status": self.details_verification_status, 
            "details_attempts": self.details_verification_attempts,
            "matched_fields": self.matched_fields,
            "current_field": self.field_to_verify,
            "is_complete": self.is_verified()
        }
    
    def get_payment_progress(self) -> Dict[str, Any]:
        """Get payment progress summary"""
        return {
            "secured": self.payment_secured,
            "method": self.payment_method_preference,
            "amount": self.outstanding_amount,
            "tools_used": [tool for tool in self.tools_used if 'payment' in tool.lower()]
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary for logging"""
        return {
            "current_step": self.current_step,
            "turn_count": self.conversation_turn_count,
            "client_mood": self.client_mood,
            "cooperation": self.cooperation_level,
            "rapport": self.rapport_level,
            "verification_complete": self.is_verified(),
            "payment_secured": self.payment_secured,
            "escalation": self.escalation_requested,
            "cancellation": self.cancellation_requested,
            "call_outcome": self.call_outcome,
            "concerns": self.client_concerns,
            "topics": self.topics_discussed
        }
    
    def update_client_interaction(self, mood: str, cooperation_change: float = 0.0, rapport_change: float = 0.0):
        """Update client interaction metrics with smoothing"""
        self.client_mood = mood
        
        # Apply changes with smoothing (70% old, 30% new)
        if cooperation_change != 0.0:
            old_cooperation = self.cooperation_level
            self.cooperation_level = max(0.0, min(1.0, 
                (old_cooperation * 0.7) + (cooperation_change * 0.3)
            ))
        
        if rapport_change != 0.0:
            old_rapport = self.rapport_level
            self.rapport_level = max(0.0, min(1.0,
                (old_rapport * 0.7) + (rapport_change * 0.3)
            ))
    
    def add_concern(self, concern: str):
        """Add client concern to tracking list"""
        if concern not in self.client_concerns:
            self.client_concerns.append(concern)
            # Keep only last 5 concerns
            if len(self.client_concerns) > 5:
                self.client_concerns = self.client_concerns[-5:]
    
    def add_topic(self, topic: str):
        """Add discussed topic to tracking list"""
        if topic not in self.topics_discussed:
            self.topics_discussed.append(topic)
            # Keep only last 10 topics
            if len(self.topics_discussed) > 10:
                self.topics_discussed = self.topics_discussed[-10:]
    
    def record_tool_usage(self, tool_name: str, result: Any = None):
        """Record tool usage for tracking and debugging"""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        
        if result is not None:
            self.tool_results[tool_name] = result
    
    def get_next_verification_field(self, available_fields: List[str]) -> str:
        """Determine next field to verify based on priority and what's been matched"""
        priority_order = ["id_number", "passport_number", "vehicle_registration", 
                         "vehicle_make", "vehicle_model", "email", "username"]
        
        for field in priority_order:
            if field in available_fields and field not in self.matched_fields:
                return field
        
        # Fallback to first available field not yet matched
        for field in available_fields:
            if field not in self.matched_fields:
                return field
        
        return "id_number"  # Default fallback
    
    def calculate_conversation_progress(self) -> float:
        """Calculate conversation progress as percentage (0.0 to 1.0)"""
        progress_weights = {
            CallStep.INTRODUCTION.value: 0.05,
            CallStep.NAME_VERIFICATION.value: 0.15,
            CallStep.DETAILS_VERIFICATION.value: 0.25,
            CallStep.REASON_FOR_CALL.value: 0.35,
            CallStep.NEGOTIATION.value: 0.45,
            CallStep.PROMISE_TO_PAY.value: 0.65,
            CallStep.DEBICHECK_SETUP.value: 0.75,
            CallStep.PAYMENT_PORTAL.value: 0.75,
            CallStep.SUBSCRIPTION_REMINDER.value: 0.85,
            CallStep.CLIENT_DETAILS_UPDATE.value: 0.90,
            CallStep.REFERRALS.value: 0.95,
            CallStep.FURTHER_ASSISTANCE.value: 0.98,
            CallStep.CLOSING.value: 1.0
        }
        
        base_progress = progress_weights.get(self.current_step, 0.0)
        
        # Bonus for payment secured
        if self.payment_secured:
            base_progress = max(base_progress, 0.8)
        
        # Penalty for verification failures
        if self.is_terminal_verification_status():
            base_progress = min(base_progress, 0.3)
        
        return min(base_progress, 1.0)
    
    def should_offer_referrals(self) -> bool:
        """Determine if referrals should be offered based on conversation context"""
        # Don't offer if call is problematic
        if (self.escalation_requested or 
            self.cancellation_requested or 
            self.client_mood == ClientMood.ANGRY.value or
            self.rapport_level < 0.4):
            return False
        
        # Offer if payment secured and good rapport
        if self.payment_secured and self.rapport_level > 0.6:
            return True
        
        # Offer if cooperative client even without payment
        if (self.client_mood == ClientMood.COOPERATIVE.value and 
            self.cooperation_level > 0.7):
            return True
        
        return False
    
    def get_urgency_level(self) -> str:
        """Determine urgency level based on conversation context"""
        # Very high urgency
        if (self.escalation_requested or 
            self.client_mood == ClientMood.ANGRY.value or
            self.conversation_turn_count > 30):
            return "critical"
        
        # High urgency
        if (self.cancellation_requested or
            self.client_mood == ClientMood.RESISTANT.value or
            self.cooperation_level < 0.3):
            return "high"
        
        # Medium urgency
        if (self.client_mood in [ClientMood.CONFUSED.value, ClientMood.FINANCIAL_STRESS.value] or
            self.cooperation_level < 0.6):
            return "medium"
        
        # Low urgency
        return "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization/debugging"""
        return {
            # Core flow
            "current_step": self.current_step,
            "is_call_ended": self.is_call_ended,
            "call_outcome": self.call_outcome,
            
            # Verification
            "name_verification_status": self.name_verification_status,
            "name_verification_attempts": self.name_verification_attempts,
            "details_verification_status": self.details_verification_status,
            "details_verification_attempts": self.details_verification_attempts,
            "matched_fields": self.matched_fields,
            "field_to_verify": self.field_to_verify,
            "is_verified": self.is_verified(),
            
            # Payment
            "payment_secured": self.payment_secured,
            "payment_method_preference": self.payment_method_preference,
            "outstanding_amount": self.outstanding_amount,
            
            # Client interaction
            "client_mood": self.client_mood,
            "cooperation_level": self.cooperation_level,
            "rapport_level": self.rapport_level,
            
            # Conversation
            "conversation_turn_count": self.conversation_turn_count,
            "last_intent": self.last_intent,
            "client_concerns": self.client_concerns,
            "topics_discussed": self.topics_discussed,
            
            # Requests
            "escalation_requested": self.escalation_requested,
            "cancellation_requested": self.cancellation_requested,
            "return_to_step": self.return_to_step,
            
            # Tools and progress
            "tools_used": self.tools_used,
            "conversation_progress": self.calculate_conversation_progress(),
            "urgency_level": self.get_urgency_level(),
            "should_offer_referrals": self.should_offer_referrals(),
            
            # Summary
            "can_discuss_account": self.can_discuss_account(),
            "should_end_call": self.should_end_call()
        }

# ===== UTILITY FUNCTIONS =====

def create_initial_state(
    client_data: Dict[str, Any], 
    agent_name: str = "AI Agent"
) -> CallCenterAgentState:
    """Create initial state from client data"""
    
    # Extract basic info
    profile = client_data.get("profile", {})
    client_info = profile.get("client_info", {})
    account_aging = client_data.get("account_aging", {})
    
    # Calculate outstanding amount
    try:
        total = float(account_aging.get("xbalance", 0))
        current = float(account_aging.get("x0", 0))
        outstanding = max(total - current, 0.0)
        outstanding_str = f"R {outstanding:.2f}"
    except (ValueError, TypeError):
        outstanding_str = "R 0.00"
    
    return CallCenterAgentState(
        # Core flow
        current_step=CallStep.INTRODUCTION.value,
        is_call_ended=False,
        call_outcome="in_progress",
        
        # Verification
        name_verification_status=VerificationStatus.INSUFFICIENT_INFO.value,
        name_verification_attempts=0,
        details_verification_status=VerificationStatus.INSUFFICIENT_INFO.value,
        details_verification_attempts=0,
        matched_fields=[],
        field_to_verify="ID number",
        
        # Payment
        payment_secured=False,
        payment_method_preference="",
        outstanding_amount=outstanding_str,
        
        # Client interaction
        client_mood=ClientMood.NEUTRAL.value,
        cooperation_level=0.5,
        rapport_level=0.5,
        
        # Conversation
        conversation_turn_count=0,
        last_intent="none",
        client_concerns=[],
        topics_discussed=[],
        
        # Requests
        escalation_requested=False,
        cancellation_requested=False,
        return_to_step=None,
        
        # Tools
        tools_used=[],
        tool_results={},
        
        # Messages
        messages=[]
    )

def update_state_from_agent_result(
    state: CallCenterAgentState, 
    agent_result: Dict[str, Any]
) -> CallCenterAgentState:
    """Update state with agent result, handling all fields properly"""
    
    # Update all fields that might be returned by agents
    for key, value in agent_result.items():
        if hasattr(state, key) and value is not None:
            setattr(state, key, value)
    
    # Special handling for messages (append, don't replace)
    if "messages" in agent_result and agent_result["messages"]:
        existing_messages = getattr(state, "messages", [])
        new_messages = agent_result["messages"]
        if isinstance(new_messages, list):
            state.messages = existing_messages + new_messages
        else:
            state.messages = existing_messages + [new_messages]
    
    # Increment turn count if messages were added
    if "messages" in agent_result:
        state.conversation_turn_count += 1
    
    return state
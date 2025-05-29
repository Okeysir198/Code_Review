# ./src/Agents/call_center_agent/state.py
"""
Optimized LangGraph state management for call center AI agent.
Streamlined for current architecture with essential state tracking only.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from langgraph.graph.message import MessagesState

#########################################################################################
# Enumerations
#########################################################################################

class CallStep(Enum):
    """Call workflow steps"""
    INTRODUCTION = "introduction"
    NAME_VERIFICATION = "name_verification"
    DETAILS_VERIFICATION = "details_verification" 
    REASON_FOR_CALL = "reason_for_call"
    NEGOTIATION = "negotiation"
    PROMISE_TO_PAY = "promise_to_pay"
    DEBICHECK_SETUP = "debicheck_setup"
    SUBSCRIPTION_REMINDER = "subscription_reminder"
    PAYMENT_PORTAL = "payment_portal"
    CLIENT_DETAILS_UPDATE = "client_details_update"
    REFERRALS = "referrals"
    FURTHER_ASSISTANCE = "further_assistance"
    CANCELLATION = "cancellation"
    ESCALATION = "escalation"
    CLOSING = "closing"
    QUERY_RESOLUTION = "query_resolution"

class VerificationStatus(Enum):
    """Verification statuses"""
    VERIFIED = "VERIFIED"
    THIRD_PARTY = "THIRD_PARTY"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    WRONG_PERSON = "WRONG_PERSON"  
    UNAVAILABLE = "UNAVAILABLE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"

class PaymentMethod(Enum):
    """Payment methods"""
    IMMEDIATE_DEBIT = "immediate_debit"
    DEBICHECK = "debicheck"
    PAYMENT_PORTAL = "payment_portal"
    NONE = "none"

class EmotionalState(Enum):
    """Client emotional states"""
    NEUTRAL = "neutral"
    COOPERATIVE = "cooperative"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    WORRIED = "worried"
    EMBARRASSED = "embarrassed"
    DEFENSIVE = "defensive"
    SUSPICIOUS = "suspicious"

#########################################################################################
# Optimized LangGraph State - Essential Fields Only
#########################################################################################

class CallCenterAgentState(MessagesState):
    """
    Optimized state for call center AI agent with essential fields only.
    Removed complex tracking methods - router handles all flow decisions.
    """
    
    # ===== CORE CALL FLOW =====
    current_step: str = CallStep.INTRODUCTION.value
    next_step: Optional[str] = None  # Used by individual agents to signal completion
    is_call_ended: bool = False
    
    # ===== VERIFICATION STATE =====
    name_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    name_verification_attempts: int = 0
    details_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    details_verification_attempts: int = 0
    matched_fields: List[str] = []
    field_to_verify: str = "id_number"
    
    # ===== PAYMENT STATE =====
    payment_secured: bool = False
    payment_method: str = PaymentMethod.NONE.value
    outstanding_amount: str = "R 0.00"
    payment_arrangement: Dict[str, Any] = {}
    
    # ===== ROUTER STATE =====
    return_to_step: Optional[str] = None  # For query resolution
    route_override: Optional[str] = None  # Emergency overrides
    
    # ===== CONVERSATION INTELLIGENCE =====
    emotional_state: str = EmotionalState.NEUTRAL.value
    objections_raised: List[str] = []
    payment_willingness: str = "unknown"  # willing, unwilling, unknown, considering
    
    # ===== REQUEST TRACKING =====
    escalation_requested: bool = False
    cancellation_requested: bool = False
    
    # ===== ESSENTIAL HELPER METHODS =====
    
    def is_verified(self) -> bool:
        """Check if client is fully verified - used throughout system"""
        return (self.name_verification_status == VerificationStatus.VERIFIED.value and 
                self.details_verification_status == VerificationStatus.VERIFIED.value)
    
    def can_discuss_account(self) -> bool:
        """Check if can discuss account details - critical for compliance"""
        return self.is_verified()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for parameter building"""
        return {
            "current_step": self.current_step,
            "next_step": self.next_step,
            "is_call_ended": self.is_call_ended,
            "name_verification_status": self.name_verification_status,
            "name_verification_attempts": self.name_verification_attempts,
            "details_verification_status": self.details_verification_status,
            "details_verification_attempts": self.details_verification_attempts,
            "matched_fields": self.matched_fields,
            "field_to_verify": self.field_to_verify,
            "payment_secured": self.payment_secured,
            "payment_method": self.payment_method,
            "outstanding_amount": self.outstanding_amount,
            "payment_arrangement": self.payment_arrangement,
            "return_to_step": self.return_to_step,
            "route_override": self.route_override,
            "emotional_state": self.emotional_state,
            "objections_raised": self.objections_raised,
            "payment_willingness": self.payment_willingness,
            "escalation_requested": self.escalation_requested,
            "cancellation_requested": self.cancellation_requested,
            "is_verified": self.is_verified(),
            "can_discuss_account": self.can_discuss_account()
        }
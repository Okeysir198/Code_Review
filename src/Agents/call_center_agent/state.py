# src/Agents/call_center_agent/state.py
"""
Simplified state for call center agent - just essential call flow tracking
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

class CallCenterAgentState(MessagesState):
    """
    Simplified state - just track call progress and verification
    """
    
    # ===== CORE CALL FLOW =====
    current_step: str = CallStep.INTRODUCTION.value
    is_call_ended: bool = False
    
    # ===== VERIFICATION STATE =====
    name_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    name_verification_attempts: int = 0
    details_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    details_verification_attempts: int = 0
    matched_fields: List[str] = []
    
    # ===== PAYMENT STATE =====
    payment_secured: bool = False
    outstanding_amount: str = "R 0.00"
    
    # ===== ROUTER STATE =====
    return_to_step: Optional[str] = None  # For query resolution
    
    # ===== REQUEST TRACKING =====
    escalation_requested: bool = False
    cancellation_requested: bool = False
    
    # ===== HELPER METHODS =====
    def is_verified(self) -> bool:
        """Check if client is fully verified"""
        return (self.name_verification_status == VerificationStatus.VERIFIED.value and 
                self.details_verification_status == VerificationStatus.VERIFIED.value)
    
    def can_discuss_account(self) -> bool:
        """Check if can discuss account details"""
        return self.is_verified()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            "current_step": self.current_step,
            "is_call_ended": self.is_call_ended,
            "name_verification_status": self.name_verification_status,
            "name_verification_attempts": self.name_verification_attempts,
            "details_verification_status": self.details_verification_status,
            "details_verification_attempts": self.details_verification_attempts,
            "matched_fields": self.matched_fields,
            "payment_secured": self.payment_secured,
            "outstanding_amount": self.outstanding_amount,
            "return_to_step": self.return_to_step,
            "escalation_requested": self.escalation_requested,
            "cancellation_requested": self.cancellation_requested,
            "is_verified": self.is_verified(),
            "can_discuss_account": self.can_discuss_account()
        }
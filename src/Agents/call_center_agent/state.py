# ./src/Agents/call_center_agent/state.py
"""
Lean LangGraph state management for call center AI agent.
Minimal state with only essential fields for call flow.
"""

from typing import Optional, List
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

#########################################################################################
# Main LangGraph State - Lean and Essential Only
#########################################################################################

class CallCenterAgentState(MessagesState):
    """
    Lean state for call center AI agent.
    Only essential fields for call flow management.
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
    field_to_verify: str = "id_number"
    
    # ===== PAYMENT STATE =====
    payment_secured: bool = False
    payment_method: str = PaymentMethod.NONE.value
    outstanding_amount: str = "R 0.00"
    
    # ===== ROUTER STATE (Minimal additions) =====
    return_to_step: Optional[str] = None      # For query resolution return
    route_override: Optional[str] = None      # For emergency routing (escalation/cancellation)
    
    # ===== HELPER METHODS =====
    
    def is_verified(self) -> bool:
        """Check if client is fully verified"""
        return (self.name_verification_status == VerificationStatus.VERIFIED.value and 
                self.details_verification_status == VerificationStatus.VERIFIED.value)
    
    def can_discuss_account(self) -> bool:
        """Check if can discuss account details"""
        return self.is_verified()
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for external use"""
        return {
            "current_step": self.current_step,
            "is_call_ended": self.is_call_ended,
            "name_verification_status": self.name_verification_status,
            "details_verification_status": self.details_verification_status,
            "payment_secured": self.payment_secured,
            "outstanding_amount": self.outstanding_amount,
            "is_verified": self.is_verified()
        }
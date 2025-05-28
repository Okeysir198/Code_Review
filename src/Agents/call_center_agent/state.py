# ./src/Agents/call_center/state.py
"""
Complete LangGraph state management for call center AI agent.
Manages conversation flow, verification, client data, and behavioral intelligence.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime

from langgraph.graph.message import MessagesState

#########################################################################################
# Enumerations
#########################################################################################

class CallStep(Enum):
    """Enumeration of call workflow steps"""
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
    DISPOSITION_CALL = "disposition_call"
    CANCELLATION = "cancellation"
    ESCALATION = "escalation"
    CLOSING = "closing"
    QUERY_RESOLUTION = "query_resolution"
    END_CONVERSATION = "end_conversation"

class VerificationStatus(Enum):
    """Enumeration of verification statuses"""
    VERIFIED = "VERIFIED"
    THIRD_PARTY = "THIRD_PARTY"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    WRONG_PERSON = "WRONG_PERSON"  
    UNAVAILABLE = "UNAVAILABLE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"

class PaymentMethod(Enum):
    """Enumeration of payment methods"""
    IMMEDIATE_DEBIT = "immediate_debit"
    DEBICHECK = "debicheck"
    PAYMENT_PORTAL = "payment_portal"
    EFT = "eft"
    CREDIT_CARD = "credit_card"
    OZOW = "ozow"
    PAY_AT = "pay_at"
    CHEQUE = "cheque"
    CAPITEC_PAY = "capitec_pay"
    NONE = "none"

class EmotionalState(Enum):
    """Client emotional states during conversation"""
    NEUTRAL = "neutral"
    COOPERATIVE = "cooperative"
    DEFENSIVE = "defensive"
    AGGRESSIVE = "aggressive"
    CONFUSED = "confused"
    EMOTIONAL = "emotional"
    DISMISSIVE = "dismissive"
    FRIENDLY = "friendly"

class PaymentWillingness(Enum):
    """Client's willingness to pay assessment"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    RESISTANT = "resistant"
    UNKNOWN = "unknown"

class CallOutcome(Enum):
    """Possible call outcomes"""
    PAYMENT_SECURED = "payment_secured"
    PTP_ARRANGED = "ptp_arranged"
    PARTIAL_PAYMENT = "partial_payment"
    CALLBACK_SCHEDULED = "callback_scheduled"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"
    UNCONTACTABLE = "uncontactable"
    REFUSAL = "refusal"
    DISPUTE = "dispute"
    WRONG_PERSON = "wrong_person"
    INCOMPLETE = "incomplete"

class ScriptType(Enum):
    """Script types for different debt scenarios"""
    RATIO_1_INFLOW = "ratio_1_inflow"
    RATIO_1_FAILED_PTP = "ratio_1_failed_ptp"
    RATIO_1_PRO_RATA = "ratio_1_pro_rata"
    RATIO_1_SHORT_PAID = "ratio_1_short_paid"
    RATIO_2_3_INFLOW = "ratio_2_3_inflow"
    RATIO_2_3_FAILED_PTP = "ratio_2_3_failed_ptp"
    RATIO_2_3_SHORT_PAID = "ratio_2_3_short_paid"
    PRE_LEGAL_120_PLUS = "pre_legal_120_plus"
    PRE_LEGAL_150_PLUS = "pre_legal_150_plus"
    RECENTLY_SUSPENDED_120_PLUS = "recently_suspended_120_plus"

#########################################################################################
# Data Classes for Complex State Objects
#########################################################################################

@dataclass
class ClientData:
    """Client information from database"""
    user_id: str = ""
    profile: Dict[str, Any] = field(default_factory=dict)
    account_overview: Dict[str, Any] = field(default_factory=dict)
    account_aging: Dict[str, Any] = field(default_factory=dict)
    banking_details: List[Dict[str, Any]] = field(default_factory=list)
    subscription: Dict[str, Any] = field(default_factory=dict)
    payment_history: List[Dict[str, Any]] = field(default_factory=list)
    failed_payments: List[Dict[str, Any]] = field(default_factory=list)
    last_successful_payment: Optional[Dict[str, Any]] = None
    contracts: List[Dict[str, Any]] = field(default_factory=list)
    billing_analysis: Dict[str, Any] = field(default_factory=dict)
    existing_mandates: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[datetime] = None

@dataclass
class BehavioralAnalysis:
    """Behavioral analysis of client"""
    days_overdue: int = 0
    payment_reliability: str = "unknown"
    likely_objections: List[str] = field(default_factory=list)
    optimal_approach: str = "professional_persistent"
    risk_level: str = "medium"
    success_probability: str = "medium"

@dataclass
class TacticalGuidance:
    """Tactical guidance for agent"""
    recommended_approach: str = "professional_persistent"
    urgency_level: str = "medium"
    key_motivators: str = "maintain_vehicle_security"
    objection_predictions: str = ""
    success_probability: str = "medium"
    backup_strategies: str = ""

@dataclass
class PaymentArrangement:
    """Payment arrangement details"""
    arrangement_id: Optional[str] = None
    mandate_id: Optional[str] = None
    payment_method: str = PaymentMethod.NONE.value
    amount: float = 0.0
    payment_date: Optional[str] = None
    installments: int = 1
    total_amount: float = 0.0
    mandate_fee: float = 0.0
    arrangement_created: bool = False
    portal_url: Optional[str] = None
    reference_id: Optional[str] = None

@dataclass
class ConversationMetrics:
    """Conversation performance metrics"""
    call_start_time: Optional[datetime] = None
    call_duration: int = 0  # seconds
    verification_attempts: int = 0
    objections_count: int = 0
    escalations_count: int = 0
    payment_attempts: int = 0
    steps_completed: int = 0
    rapport_score: float = 0.0

#########################################################################################
# Main LangGraph State
#########################################################################################

class CallCenterAgentState(MessagesState):
    """
    Comprehensive state tracking for call center AI agent.
    
    Manages all aspects of the debt collection conversation including
    verification, behavioral analysis, payment arrangements, and call flow.
    """
    
    # # ===== CORE IDENTIFICATION =====
    # user_id: str = ""
    # agent_name: str = "AI Agent"
    # session_id: str = ""
    
    # ===== CALL FLOW MANAGEMENT =====
    current_step: str = CallStep.INTRODUCTION.value
    previous_step: Optional[str] = None
    # next_step: Optional[str] = None
    # is_call_active: bool = True
    is_call_ended: bool = False
    # call_outcome: Optional[str] = None
    
    # ===== VERIFICATION STATE =====
    name_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    name_verification_attempts: int = 0
    details_verification_status: str = VerificationStatus.INSUFFICIENT_INFO.value
    details_verification_attempts: int = 0
    matched_fields: List[str] = field(default_factory=list)
    field_to_verify: str = "id_number"
    
    # # ===== CLIENT DATA =====
    # client_data: ClientData = field(default_factory=ClientData)

    
    # # ===== BEHAVIORAL INTELLIGENCE =====
    # behavioral_analysis: BehavioralAnalysis = field(default_factory=BehavioralAnalysis)
    # tactical_guidance: TacticalGuidance = field(default_factory=TacticalGuidance)
    
    # # ===== CONVERSATION CONTEXT =====
    # emotional_state: str = EmotionalState.NEUTRAL.value
    # payment_willingness: str = PaymentWillingness.UNKNOWN.value
    # rapport_level: str = "establishing"
    # objections_raised: List[str] = field(default_factory=list)
    # topics_discussed: List[str] = field(default_factory=list)
    
    # # ===== CONVERSATION FLAGS =====
    # query_detected: bool = False
    # cancellation_requested: bool = False
    # escalation_requested: bool = False
    # supervisor_requested: bool = False
    # complaint_raised: bool = False
    # dispute_raised: bool = False
    
    # # ===== PAYMENT MANAGEMENT =====
    # payment_arrangement: PaymentArrangement = field(default_factory=PaymentArrangement)
    # payment_secured: bool = False
    # debicheck_setup_complete: bool = False
    # portal_payment_complete: bool = False
    
    # # ===== CALL MANAGEMENT =====
    # notes_added: List[str] = field(default_factory=list)
    # disposition_saved: bool = False
    # disposition_type_id: Optional[str] = None
    # callback_scheduled: bool = False
    # callback_date: Optional[str] = None
    
    # # ===== UPDATES TRACKING =====
    # contact_details_updated: bool = False
    # banking_details_updated: bool = False
    # next_of_kin_updated: bool = False
    # referral_captured: bool = False
    
    # # ===== CONVERSATION METRICS =====
    # metrics: ConversationMetrics = field(default_factory=ConversationMetrics)
    
    # # ===== SYSTEM CONTEXT =====
    # system_prompt: Optional[str] = None
    # last_response: Optional[str] = None
    # error_count: int = 0
    # retry_count: int = 0
    
    # # ===== ROUTING AND FLOW CONTROL =====
    # recommended_next_node: Optional[str] = None
    # route_reason: Optional[str] = None
    # should_continue: bool = True
    
    # # ===== COMPLIANCE AND AUDIT =====
    # popi_acknowledged: bool = False
    # call_recorded_notice_given: bool = False
    # security_verification_complete: bool = False
    # compliance_notes: List[str] = field(default_factory=list)

    # ===== HELPER METHODS =====
    
    # def is_verified(self) -> bool:
    #     """Check if client is fully verified"""
    #     return (self.name_verification_status == VerificationStatus.VERIFIED.value and 
    #             self.details_verification_status == VerificationStatus.VERIFIED.value)
    
    # def can_discuss_account(self) -> bool:
    #     """Check if can discuss account details"""
    #     return self.is_verified()
    
    # def add_objection(self, objection: str):
    #     """Add objection to tracking"""
    #     if objection not in self.objections_raised:
    #         self.objections_raised.append(objection)
    
    # def add_matched_field(self, field: str):
    #     """Add verified field"""
    #     if field not in self.matched_fields:
    #         self.matched_fields.append(field)
    
    # def increment_verification_attempt(self, verification_type: str):
    #     """Increment verification attempt counter"""
    #     if verification_type == "name":
    #         self.name_verification_attempts += 1
    #     elif verification_type == "details":
    #         self.details_verification_attempts += 1
    
    # def update_emotional_state(self, new_state: str):
    #     """Update client emotional state"""
    #     if new_state in [e.value for e in EmotionalState]:
    #         self.emotional_state = new_state
    
    # def update_payment_willingness(self, willingness: str):
    #     """Update payment willingness"""
    #     if willingness in [w.value for w in PaymentWillingness]:
    #         self.payment_willingness = willingness
    
    # def set_call_outcome(self, outcome: str):
    #     """Set call outcome"""
    #     if outcome in [o.value for o in CallOutcome]:
    #         self.call_outcome = outcome
    
    # def add_note(self, note: str):
    #     """Add note to tracking"""
    #     self.notes_added.append(note)
    
    # def add_topic(self, topic: str):
    #     """Add discussed topic"""
    #     if topic not in self.topics_discussed:
    #         self.topics_discussed.append(topic)
    
    # def get_verification_progress(self) -> Dict[str, Any]:
    #     """Get verification progress summary"""
    #     return {
    #         "name_status": self.name_verification_status,
    #         "name_attempts": self.name_verification_attempts,
    #         "details_status": self.details_verification_status,
    #         "details_attempts": self.details_verification_attempts,
    #         "matched_fields": self.matched_fields,
    #         "is_verified": self.is_verified()
    #     }
    
    # def get_payment_summary(self) -> Dict[str, Any]:
    #     """Get payment arrangement summary"""
    #     return {
    #         "secured": self.payment_secured,
    #         "method": self.payment_arrangement.payment_method,
    #         "amount": self.payment_arrangement.amount,
    #         "arrangement_id": self.payment_arrangement.arrangement_id,
    #         "mandate_id": self.payment_arrangement.mandate_id
    #     }
    
    # def get_conversation_summary(self) -> Dict[str, Any]:
    #     """Get conversation summary"""
    #     return {
    #         "step": self.current_step,
    #         "emotional_state": self.emotional_state,
    #         "payment_willingness": self.payment_willingness,
    #         "objections": len(self.objections_raised),
    #         "verified": self.is_verified(),
    #         "payment_secured": self.payment_secured,
    #         "outcome": self.call_outcome
    #     }
    
    # def to_dict(self) -> Dict[str, Any]:
    #     """Convert state to dictionary for external use"""
    #     return {
    #         "user_id": self.user_id,
    #         "agent_name": self.agent_name,
    #         "current_step": self.current_step,
    #         "script_type": self.script_type,
    #         "verification": self.get_verification_progress(),
    #         "payment": self.get_payment_summary(),
    #         "conversation": self.get_conversation_summary(),
    #         "behavioral_analysis": self.behavioral_analysis.__dict__,
    #         "tactical_guidance": self.tactical_guidance.__dict__,
    #         "flags": {
    #             "query_detected": self.query_detected,
    #             "cancellation_requested": self.cancellation_requested,
    #             "escalation_requested": self.escalation_requested,
    #             "complaint_raised": self.complaint_raised
    #         }
    #     }

#########################################################################################
# State Update Helpers
#########################################################################################

def update_verification_status(state: CallCenterAgentState, 
                             verification_type: str, 
                             status: str) -> CallCenterAgentState:
    """Helper to update verification status"""
    if verification_type == "name":
        state.name_verification_status = status
    elif verification_type == "details":
        state.details_verification_status = status
    return state

def advance_call_step(state: CallCenterAgentState, 
                     next_step: str) -> CallCenterAgentState:
    """Helper to advance call step"""
    state.previous_step = state.current_step
    state.current_step = next_step
    state.metrics.steps_completed += 1
    return state

def set_payment_arrangement(state: CallCenterAgentState,
                          arrangement: PaymentArrangement) -> CallCenterAgentState:
    """Helper to set payment arrangement"""
    state.payment_arrangement = arrangement
    state.payment_secured = arrangement.arrangement_created
    return state

def update_behavioral_context(state: CallCenterAgentState,
                            emotional_state: Optional[str] = None,
                            payment_willingness: Optional[str] = None,
                            objection: Optional[str] = None) -> CallCenterAgentState:
    """Helper to update behavioral context"""
    if emotional_state:
        state.update_emotional_state(emotional_state)
    if payment_willingness:
        state.update_payment_willingness(payment_willingness)
    if objection:
        state.add_objection(objection)
    return state

#########################################################################################
# Constants for Easy Access
#########################################################################################

# Maximum attempts for various actions
MAX_VERIFICATION_ATTEMPTS = {
    "name": 3,
    "details": 5,
    "payment": 3,
    "objection_handling": 5
}

# Step sequences for different script types
STEP_SEQUENCES = {
    ScriptType.RATIO_1_INFLOW.value: [
        CallStep.INTRODUCTION.value,
        CallStep.NAME_VERIFICATION.value,
        CallStep.DETAILS_VERIFICATION.value,
        CallStep.REASON_FOR_CALL.value,
        CallStep.NEGOTIATION.value,
        CallStep.PROMISE_TO_PAY.value,
        CallStep.SUBSCRIPTION_REMINDER.value,
        CallStep.CLIENT_DETAILS_UPDATE.value,
        CallStep.REFERRALS.value,
        CallStep.FURTHER_ASSISTANCE.value,
        CallStep.CLOSING.value
    ],
    ScriptType.PRE_LEGAL_120_PLUS.value: [
        CallStep.INTRODUCTION.value,
        CallStep.NAME_VERIFICATION.value,
        CallStep.DETAILS_VERIFICATION.value,
        CallStep.REASON_FOR_CALL.value,
        CallStep.NEGOTIATION.value,
        CallStep.PROMISE_TO_PAY.value,
        CallStep.DEBICHECK_SETUP.value,
        CallStep.CLIENT_DETAILS_UPDATE.value,
        CallStep.DISPOSITION_CALL.value,
        CallStep.CLOSING.value
    ]
}

# Payment method mappings
PAYMENT_METHOD_IDS = {
    PaymentMethod.IMMEDIATE_DEBIT.value: 1,
    PaymentMethod.EFT.value: 2,
    PaymentMethod.CREDIT_CARD.value: 3,
    PaymentMethod.OZOW.value: 4,
    PaymentMethod.PAY_AT.value: 5,
    PaymentMethod.CHEQUE.value: 6,
    PaymentMethod.CAPITEC_PAY.value: 7
}
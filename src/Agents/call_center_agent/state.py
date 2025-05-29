# ./src/Agents/call_center_agent/state.py
"""
Enhanced LangGraph state management for call center AI agent.
Includes emotional state tracking and conversation intelligence.
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
# Enhanced LangGraph State with Conversation Intelligence
#########################################################################################

class CallCenterAgentState(MessagesState):
    """
    Enhanced state for call center AI agent with conversation intelligence.
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
    payment_arrangement: Dict[str, Any] = {}
    
    # ===== ROUTER STATE =====
    return_to_step: Optional[str] = None
    route_override: Optional[str] = None
    
    # ===== CONVERSATION INTELLIGENCE =====
    emotional_state: str = EmotionalState.NEUTRAL.value
    objections_raised: List[str] = []
    payment_willingness: str = "unknown"  # willing, unwilling, unknown
    rapport_level: str = "establishing"  # establishing, good, poor
    conversation_tone: str = "professional"  # professional, friendly, firm
    
    # ===== ENHANCED PAYMENT FLEXIBILITY =====
    payment_capacity_assessment: str = "unknown"  # high, medium, low, hardship
    payment_offers_made: List[Dict[str, Any]] = []
    client_counter_offers: List[Dict[str, Any]] = []
    minimum_acceptable_payment: float = 0.0
    payment_plan_eligible: bool = False
    hardship_indicators: List[str] = []
    
    # ===== REQUEST TRACKING =====
    escalation_requested: bool = False
    cancellation_requested: bool = False
    
    # ===== CONVERSATION METRICS =====
    conversation_turns: int = 0
    successful_redirects: int = 0
    objection_count: int = 0
    
    # ===== HELPER METHODS =====
    
    def is_verified(self) -> bool:
        """Check if client is fully verified"""
        return (self.name_verification_status == VerificationStatus.VERIFIED.value and 
                self.details_verification_status == VerificationStatus.VERIFIED.value)
    
    def can_discuss_account(self) -> bool:
        """Check if can discuss account details"""
        return self.is_verified()
    
    def is_cooperative(self) -> bool:
        """Check if client is showing cooperative behavior"""
        cooperative_states = [EmotionalState.NEUTRAL.value, EmotionalState.COOPERATIVE.value]
        return self.emotional_state in cooperative_states and self.payment_willingness != "unwilling"
    
    def needs_de_escalation(self) -> bool:
        """Check if client needs de-escalation"""
        emotional_triggers = [EmotionalState.ANGRY.value, EmotionalState.FRUSTRATED.value, EmotionalState.DEFENSIVE.value]
        return self.emotional_state in emotional_triggers
    
    def has_hardship_indicators(self) -> bool:
        """Check if client shows financial hardship indicators"""
        hardship_objections = ["no_money", "lost_job", "financial_problems", "cant_afford"]
        return any(obj in self.objections_raised for obj in hardship_objections) or \
               self.payment_capacity_assessment == "hardship"
    
    def add_objection(self, objection: str) -> None:
        """Add objection to tracking"""
        if objection not in self.objections_raised:
            self.objections_raised.append(objection)
            self.objection_count += 1
    
    def update_payment_willingness(self, willingness: str) -> None:
        """Update payment willingness assessment"""
        valid_levels = ["willing", "unwilling", "unknown", "considering"]
        if willingness in valid_levels:
            self.payment_willingness = willingness
    
    def add_payment_offer(self, offer: Dict[str, Any]) -> None:
        """Track payment offers made to client"""
        self.payment_offers_made.append({
            "amount": offer.get("amount", 0),
            "type": offer.get("type", "full"),
            "timestamp": offer.get("timestamp"),
            "accepted": offer.get("accepted", False)
        })
    
    def add_client_counter_offer(self, counter_offer: Dict[str, Any]) -> None:
        """Track client's counter offers"""
        self.client_counter_offers.append({
            "amount": counter_offer.get("amount", 0),
            "timeframe": counter_offer.get("timeframe"),
            "conditions": counter_offer.get("conditions", []),
            "timestamp": counter_offer.get("timestamp")
        })
    
    def increment_conversation_turn(self) -> None:
        """Track conversation flow"""
        self.conversation_turns += 1
    
    def record_successful_redirect(self) -> None:
        """Track successful topic redirections"""
        self.successful_redirects += 1
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation progress"""
        return {
            "verification_complete": self.is_verified(),
            "emotional_state": self.emotional_state,
            "payment_secured": self.payment_secured,
            "objections_count": len(self.objections_raised),
            "conversation_turns": self.conversation_turns,
            "cooperation_level": "high" if self.is_cooperative() else "low",
            "escalation_needed": self.needs_de_escalation(),
            "hardship_detected": self.has_hardship_indicators()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for external use"""
        return {
            "current_step": self.current_step,
            "is_call_ended": self.is_call_ended,
            "name_verification_status": self.name_verification_status,
            "details_verification_status": self.details_verification_status,
            "payment_secured": self.payment_secured,
            "outstanding_amount": self.outstanding_amount,
            "emotional_state": self.emotional_state,
            "objections_raised": self.objections_raised,
            "payment_willingness": self.payment_willingness,
            "is_verified": self.is_verified(),
            "conversation_summary": self.get_conversation_summary()
        }
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Annotated, List, Union, cast
from enum import Enum

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

class PaymentMethod(Enum):
    """Enumeration of payment methods"""
    DEBIT_ORDER = "debit_order"
    IMMEDIATE_DEBIT = "immediate_debit"
    NEXT_DAY_DEBIT = "next_day_debit"
    PAYMENT_PORTAL = "payment_portal"
    DEBICHECK = "debicheck"
    NONE = "none"



class VerificationStatus(Enum):
    """Enumeration of verification statuses"""
    VERIFIED = "VERIFIED"
    THIRD_PARTY = "THIRD_PARTY"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    WRONG_PERSON = "WRONG_PERSON"  
    UNAVAILABLE = "UNAVAILABLE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"

#########################################################################################
# Agent State
#########################################################################################

class CallCenterAgentState(MessagesState):
    """
    State tracking for call center conversations.
    
    Provides comprehensive context for the conversation, tracking 
    verification status, conversation flow, and key conversation markers.
    """
    # Call flow tracking
    current_call_step: Optional[str] = CallStep.INTRODUCTION.value
    previous_step: Optional[str] = None
    is_call_ended: bool = False

    # Verification state
    name_verification_status: Optional[str] = VerificationStatus.INSUFFICIENT_INFO.value
    name_verification_attempts: int = 0
    details_verification_status: Optional[str] = VerificationStatus.INSUFFICIENT_INFO.value
    details_verification_attempts: int = 0
    matched_fields: Optional[List[str]] = None
   
    
    
    
    # # Conversation context
    # current_topic: Optional[str] = None
    # client_sentiment: str = "neutral"
    
    # # Conversation flags
    # query_detected: bool = False
    # cancellation_requested: bool = False
    # escalation_requested: bool = False
    
    # # Payment-related tracking
    # payment_secured: bool = False
    # payment_method: Optional[str] = None
    # agreed_amount: Optional[str] = None
    # payment_agreement_confidence: float = 0.0
    
    # # Recommended next node from router
    # recommended_next_node: Optional[str] = None
    
    # # Additional conversation metadata
    # call_outcome: Optional[str] = None
    # call_info: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
# ./src/Agents/call_center_agent/call_scripts.py
"""
Optimized call scripts with behavioral intelligence for LLM debt collection agents.
Integrates script content with tactical guidance and objection handling.
"""

from enum import Enum
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ScriptType(Enum):
    """Script types for different debt collection scenarios."""
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

class CallStep(Enum):
    """Call step identifiers."""
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
    CLOSING = "closing"
    QUERY_RESOLUTION = "query_resolution"
    ESCALATION = "escalation"

class BaseCallScript:
    """Base script with common elements and behavioral guidance."""
    
    # Core script content
    SCRIPTS = {
        "introduction": "Good day, you are speaking to {agent_name} from Cartrack Accounts Department. May I speak to {client_title} {client_full_name}, please?",
        "third_party_message": "{salutation}, kindly advise {client_title} {client_full_name} that Cartrack called regarding an outstanding account. Please ask them to contact us urgently at 011 250 3000.",
        "details_verification": "This call is recorded for quality and security. To ensure I'm speaking with the right person, could you please confirm your {field_to_verify}?",
        "reason_for_call": "{client_name}, we did not receive your subscription payment and your account is now overdue. An immediate payment of {outstanding_amount} is required.",
        "negotiation_consequences": "Without payment, your Cartrack app will be suspended, you'll lose vehicle positioning, and notifications will stop working.",
        "negotiation_benefits": "Payment today restores all services immediately and keeps your account in good standing.",
        "promise_to_pay": "Can we debit {outstanding_amount} from your account today?",
        "debicheck_setup": "You'll receive a bank authentication request. Please approve it immediately. The total will be {amount_with_fee} including the R10 processing fee.",
        "payment_portal": "I'm sending you a secure payment link. Click 'PAY HERE', confirm the amount as {outstanding_amount}, and choose your payment method.",
        "subscription_reminder": "Today's payment covers arrears. Your regular subscription of {subscription_amount} continues on {subscription_date}.",
        "client_details_update": "Let me verify your current contact details to ensure you receive important notifications.",
        "referrals": "Do you know anyone interested in Cartrack? Successful referrals earn you 2 months free subscription.",
        "further_assistance": "Is there anything else regarding your account I can help you with today?",
        "cancellation": "The cancellation fee is {cancellation_fee}. Your total balance is {total_balance}. I'll escalate this to our cancellations team.",
        "closing": "Thank you for your cooperation today, {client_name}. {outcome_summary}."
    }
    
    # Objection handling responses
    OBJECTION_RESPONSES = {
        "no_money": "I understand finances are tight. What amount could you manage today to maintain your vehicle security?",
        "dispute_amount": "I can verify the breakdown while we arrange payment to avoid service interruption. What specifically concerns you?",
        "will_pay_later": "I appreciate that intention. However, to prevent service suspension, we need to secure payment today. What's preventing immediate payment?",
        "already_paid": "Let me check our records. When and how was this payment made? I want to ensure it's properly applied.",
        "not_my_debt": "I understand your concern. Let's verify the details while ensuring your services remain active.",
        "cant_afford": "I hear you. Let's find a solution that works. Even a partial payment today can help maintain services.",
        "need_to_think": "I have you connected now with immediate solutions. What specific concerns can I address to help you decide?",
        "call_back_later": "I know this is inconvenient, but I have several quick options that can resolve this in just a few minutes."
    }
    
    # Emotional state responses
    EMOTIONAL_RESPONSES = {
        "angry": "I understand you're frustrated. My goal is to help resolve this quickly. Let's find a solution that works for you.",
        "confused": "I know this can be overwhelming. Let me break this down simply and help you through each step.",
        "defensive": "I'm here to help, not judge. Let's focus on keeping your vehicle protection active.",
        "dismissive": "I understand you're busy. This will just take a few minutes to resolve and prevent service disruption.",
        "emotional": "I can hear this is stressful. Let me help make this as easy as possible for you."
    }
    
    # Escalation triggers and responses
    ESCALATION_RESPONSES = {
        "legal_threats": "I'm here to help avoid any complications. Let's resolve this now to keep things simple.",
        "supervisor_request": "I'm authorized to help with payment arrangements. Let's see what options work for you.",
        "hang_up_threat": "I understand this is inconvenient. I have quick solutions that take just minutes.",
        "complaint_threat": "I want to resolve this positively. What outcome would work best for you today?"
    }

class Ratio1InflowScript(BaseCallScript):
    """Standard first missed payment."""
    
    def get_specialized_content(self) -> Dict[str, str]:
        return {
            "urgency_level": "medium",
            "consequence_emphasis": "service_suspension",
            "preferred_resolution": "immediate_debit",
            "backup_options": ["debicheck", "payment_portal"]
        }

class Ratio1FailedPTPScript(BaseCallScript):
    """Failed promise to pay follow-up."""
    
    SCRIPTS = {
        **BaseCallScript.SCRIPTS,
        "reason_for_call": "{client_name}, we didn't receive your payment of {outstanding_amount} as arranged, and your account is overdue."
    }
    
    OBJECTION_RESPONSES = {
        **BaseCallScript.OBJECTION_RESPONSES,
        "bank_error": "I understand. Let's update your details to ensure successful payment. Which bank do you use now?",
        "forgot": "That happens. Let's arrange payment now while we're connected to get your services restored."
    }

class Ratio1ProRataScript(BaseCallScript):
    """Pro-rata payment for new installations."""
    
    SCRIPTS = {
        **BaseCallScript.SCRIPTS,
        "reason_for_call": "{client_name}, we haven't received your pro-rata payment since fitment. Services activate once we receive {outstanding_amount}.",
        "negotiation_benefits": "Payment today activates all your Cartrack services immediately."
    }

class Ratio1ShortPaidScript(BaseCallScript):
    """Short payment follow-up."""
    
    SCRIPTS = {
        **BaseCallScript.SCRIPTS,
        "reason_for_call": "Thank you for your payment of {paid_amount} on {payment_date}. However, this is {shortfall_amount} less than the agreed {agreed_amount}."
    }

class Ratio2And3InflowScript(BaseCallScript):
    """2-3 months overdue."""
    
    SCRIPTS = {
        **BaseCallScript.SCRIPTS,
        "reason_for_call": "{client_name}, your account is overdue for over 2 months. An immediate payment of {outstanding_amount} is required.",
        "negotiation_consequences": "Services remain suspended. If your vehicle is stolen, you may be charged up to R25,000 for recovery. Continued non-payment leads to legal action."
    }

class Ratio2And3FailedPTPScript(Ratio2And3InflowScript):
    """2-3 months overdue failed PTP."""
    
    SCRIPTS = {
        **Ratio2And3InflowScript.SCRIPTS,
        "reason_for_call": "{client_name}, we didn't receive your payment of {outstanding_amount} as arranged. Your account remains overdue."
    }

class Ratio2And3ShortPaidScript(Ratio2And3InflowScript):
    """2-3 months overdue short payment."""
    
    SCRIPTS = {
        **Ratio2And3InflowScript.SCRIPTS,
        "reason_for_call": "Thank you for your payment of {paid_amount}. However, this is less than the agreed {agreed_amount}, leaving {shortfall_amount} outstanding."
    }

class PreLegal120PlusScript(BaseCallScript):
    """Pre-legal 120+ days overdue."""
    
    SCRIPTS = {
        **BaseCallScript.SCRIPTS,
        "reason_for_call": "{client_name}, your account is 4+ months overdue and in our pre-legal department. A letter of demand was sent. Immediate payment of {outstanding_amount} is required.",
        "negotiation_consequences": "Services suspended, credit listing as default payer (R1800 to clear), potential R25,000 recovery fee, and legal action if unpaid.",
        "discount_offer": "To prevent legal action, we're offering {discount_percentage}% discount. Pay {discounted_amount} by month-end to settle completely."
    }

class PreLegal150PlusScript(PreLegal120PlusScript):
    """Legal 150+ days - attorney involved."""
    
    SCRIPTS = {
        **PreLegal120PlusScript.SCRIPTS,
        "introduction": "Good day, this is {agent_name} from Viljoen Attorneys regarding your Cartrack account. May I speak to {client_full_name}?",
        "reason_for_call": "{client_name}, Cartrack has handed your account to us as attorneys. Your arrears are {outstanding_amount}. Do you acknowledge this debt?",
        "legal_consequences": "Without arrangement, we'll issue notice of intent to summons, then letter of demand with 10 days to pay, followed by actual summons through the sheriff."
    }

class RecentlySuspended120PlusScript(PreLegal120PlusScript):
    """Recently suspended 120+ with campaign discount."""
    
    SCRIPTS = {
        **PreLegal120PlusScript.SCRIPTS,
        "discount_offer": "Limited time: 50% discount if you pay {discounted_amount_50} by {campaign_end_date}. Alternative: 40% discount in 2 payments starting {campaign_first_date}."
    }

class ScriptManager:
    """Factory for script classes and content retrieval."""
    
    _SCRIPT_MAP = {
        ScriptType.RATIO_1_INFLOW: Ratio1InflowScript,
        ScriptType.RATIO_1_FAILED_PTP: Ratio1FailedPTPScript,
        ScriptType.RATIO_1_PRO_RATA: Ratio1ProRataScript,
        ScriptType.RATIO_1_SHORT_PAID: Ratio1ShortPaidScript,
        ScriptType.RATIO_2_3_INFLOW: Ratio2And3InflowScript,
        ScriptType.RATIO_2_3_FAILED_PTP: Ratio2And3FailedPTPScript,
        ScriptType.RATIO_2_3_SHORT_PAID: Ratio2And3ShortPaidScript,
        ScriptType.PRE_LEGAL_120_PLUS: PreLegal120PlusScript,
        ScriptType.PRE_LEGAL_150_PLUS: PreLegal150PlusScript,
        ScriptType.RECENTLY_SUSPENDED_120_PLUS: RecentlySuspended120PlusScript,
    }
    
    @classmethod
    def get_script_class(cls, script_type: ScriptType) -> BaseCallScript:
        """Get instantiated script class."""
        script_class = cls._SCRIPT_MAP.get(script_type, Ratio1InflowScript)
        return script_class()
    
    @classmethod
    def get_script_content(cls, script_type: ScriptType, step: CallStep) -> str:
        """Get script content for specific step."""
        script = cls.get_script_class(script_type)
        return script.SCRIPTS.get(step.value, "")
    
    @classmethod
    def get_objection_response(cls, script_type: ScriptType, objection_key: str) -> str:
        """Get response for specific objection."""
        script = cls.get_script_class(script_type)
        return script.OBJECTION_RESPONSES.get(objection_key, script.OBJECTION_RESPONSES.get("general", "I understand your concern. Let's find a solution that works."))
    
    @classmethod
    def get_emotional_response(cls, script_type: ScriptType, emotional_state: str) -> str:
        """Get response for client emotional state."""
        script = cls.get_script_class(script_type)
        return script.EMOTIONAL_RESPONSES.get(emotional_state, script.EMOTIONAL_RESPONSES.get("neutral", "I'm here to help resolve this matter."))
    
    @classmethod
    def get_escalation_response(cls, script_type: ScriptType, escalation_trigger: str) -> str:
        """Get response for escalation situations."""
        script = cls.get_script_class(script_type)
        return script.ESCALATION_RESPONSES.get(escalation_trigger, "Let me help resolve this matter for you.")
    
    @classmethod
    def get_behavioral_guidance(cls, script_type: ScriptType, step: CallStep) -> Dict[str, Any]:
        """Get complete behavioral guidance for step."""
        script = cls.get_script_class(script_type)
        
        return {
            "script_content": script.SCRIPTS.get(step.value, ""),
            "objection_responses": script.OBJECTION_RESPONSES,
            "emotional_responses": script.EMOTIONAL_RESPONSES,
            "escalation_responses": script.ESCALATION_RESPONSES,
            "tactical_guidance": cls._get_tactical_guidance(script_type, step)
        }
    
    @classmethod
    def _get_tactical_guidance(cls, script_type: ScriptType, step: CallStep) -> Dict[str, str]:
        """Get tactical guidance based on script type and step."""
        guidance = {
            "approach": "professional_persistent",
            "urgency_level": "medium",
            "success_indicators": "client_engagement",
            "failure_triggers": "aggressive_resistance"
        }
        
        # Adjust based on script type
        if "pre_legal" in script_type.value or "150" in script_type.value:
            guidance["urgency_level"] = "high"
            guidance["approach"] = "firm_but_solution_focused"
        elif "2_3" in script_type.value:
            guidance["urgency_level"] = "medium_high"
            guidance["consequence_emphasis"] = "recovery_fees"
        
        return guidance

# Convenience functions for backward compatibility
def get_script_text(script_type: str, section: str) -> str:
    """Legacy function - get script text."""
    try:
        script_enum = ScriptType(script_type)
        step_enum = CallStep(section.lower())
        return ScriptManager.get_script_content(script_enum, step_enum)
    except (ValueError, AttributeError):
        logger.warning(f"Invalid script_type '{script_type}' or section '{section}'")
        return ""

def get_available_scripts() -> Dict[str, str]:
    """Get available script types with descriptions."""
    return {
        ScriptType.RATIO_1_INFLOW.value: "Standard first missed payment",
        ScriptType.RATIO_1_FAILED_PTP.value: "Failed promise to pay follow-up",
        ScriptType.RATIO_1_PRO_RATA.value: "Pro-rata payment after installation",
        ScriptType.RATIO_1_SHORT_PAID.value: "Partial payment follow-up",
        ScriptType.RATIO_2_3_INFLOW.value: "2-3 months overdue",
        ScriptType.RATIO_2_3_FAILED_PTP.value: "2-3 months failed PTP",
        ScriptType.RATIO_2_3_SHORT_PAID.value: "2-3 months short payment",
        ScriptType.PRE_LEGAL_120_PLUS.value: "Pre-legal 120+ days",
        ScriptType.PRE_LEGAL_150_PLUS.value: "Legal 150+ days (attorney)",
        ScriptType.RECENTLY_SUSPENDED_120_PLUS.value: "120+ days campaign discount"
    }
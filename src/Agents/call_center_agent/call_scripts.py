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
    """Factory for script classes and content retrieval with aging detection."""
    
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
    def _get_script_parameters(cls, client_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers all potential parameters from client_data and state for script formatting.
        """
        params = {}

        # Client Profile Information
        client_info = client_data.get("profile", {}).get("client_info", {})
        params["client_title"] = client_info.get("title", "Sir/Madam")
        params["client_first_name"] = client_info.get("first_name", "Valued Client")
        params["client_last_name"] = client_info.get("last_name", "")
        params["client_full_name"] = client_info.get("client_full_name", "Valued Client")
        params["client_name"] = client_info.get("first_name", client_info.get("client_full_name", "Valued Client")) # A common short version

        # Account Overview Information
        account_overview = client_data.get("account_overview", {})
        params["outstanding_amount"] = f"R{float(account_overview.get('xbalance', 0)):.2f}"
        params["total_overdue_amount"] = f"R{float(client_data.get('account_aging', {}).get('xbalance', 0)):.2f}" # More explicit for total
        params["subscription_amount"] = f"R{float(account_overview.get('subscription', 0)):.2f}"
        params["last_successful_payment_date"] = account_overview.get("last_successful_payment_date", "N/A")
        params["payment_status"] = account_overview.get("payment_status", "unknown")
        params["account_status"] = account_overview.get("account_status", "unknown")
        params["cancellation_fee"] = f"R{float(account_overview.get('cancellation_fee', 0)):.2f}"

        # Payment History (for PTP and Short Paid)
        payment_history = client_data.get("payment_history", [])
        if payment_history:
            # Assuming the most recent payment might be relevant for short_paid/failed_ptp
            last_payment = payment_history[-1] # Or iterate to find specific type
            params["paid_amount"] = f"R{float(last_payment.get('amount_paid', 0)):.2f}"
            params["payment_date"] = last_payment.get("payment_date", "a recent date")
            params["agreed_amount"] = f"R{float(last_payment.get('agreed_amount', 0)):.2f}"
            params["shortfall_amount"] = f"R{float(last_payment.get('shortfall', 0)):.2f}"

        # State-specific parameters (e.g., agent name, dynamic values)
        params["agent_name"] = state.get("agent_name", "the Agent")
        params["field_to_verify"] = state.get("field_to_verify", "identity") # For details_verification step
        params["amount_with_fee"] = f"R{float(state.get('amount_with_fee', 0)):.2f}"
        params["outcome_summary"] = state.get("outcome_summary", "We appreciate your time.")
        params["discount_percentage"] = state.get("discount_percentage", "N/A")
        params["discounted_amount"] = f"R{float(state.get('discounted_amount', 0)):.2f}"
        params["discounted_amount_50"] = f"R{float(state.get('discounted_amount_50', 0)):.2f}"
        params["campaign_end_date"] = state.get("campaign_end_date", "N/A")
        params["campaign_first_date"] = state.get("campaign_first_date", "N/A")
        params["total_balance"] = f"R{float(client_data.get('account_aging', {}).get('xbalance', 0)):.2f}" # Or account_overview.get('total_invoices') if that's the total owed

        # Add salutation based on client title if available
        params["salutation"] = f"Good day, {params['client_title']}" if params["client_title"] != "Sir/Madam" else "Good day"
        
        return params
    
    @classmethod
    def format_script_content(cls, script_template: str, client_data: Dict[str, Any], state: Dict[str, Any]) -> str:
        """
        Formats a script template string with dynamic client and state data.
        """
        if not script_template:
            return ""

        params = cls._get_script_parameters(client_data, state)
        
        # Use str.format_map for safe formatting with missing keys
        # This allows for a dictionary of values and handles missing keys gracefully by ignoring them.
        # However, for critical parameters, you might want to pre-check or provide robust defaults.
        
        # A safer approach is to use a custom formatter that handles missing keys:
        class SafeDict(dict):
            def __missing__(self, key):
                logger.warning(f"Missing script parameter: '{key}'. Using empty string as default.")
                return ''
        
        safe_params = SafeDict(params)

        try:
            formatted_script = script_template.format_map(safe_params)
            return formatted_script
        except KeyError as e:
            logger.error(f"Failed to format script due to missing key: {e}. Template: '{script_template}'")
            return script_template # Return original template if formatting fails unexpectedly
        except Exception as e:
            logger.error(f"An unexpected error occurred during script formatting: {e}. Template: '{script_template}'")
            return script_template

    @classmethod
    def determine_script_type_from_aging(cls, account_aging: Dict[str, Any], client_data: Dict[str, Any] = None) -> ScriptType:
        """
        Determine appropriate script type based on account aging and client history.
        
        Args:
            account_aging: Dict containing aging buckets (x0, x30, x60, x90, x120, xbalance)
            client_data: Optional client data for additional context
            
        Returns:
            ScriptType enum value
        """
        if not account_aging:
            logger.warning("No account aging data provided, defaulting to RATIO_1_INFLOW")
            return ScriptType.RATIO_1_INFLOW
            
        try:
            # Extract aging amounts
            current = float(account_aging.get("x0", 0))      # Current (0 days)
            x30 = float(account_aging.get("x30", 0))         # 1-30 days overdue  
            x60 = float(account_aging.get("x60", 0))         # 31-60 days overdue
            x90 = float(account_aging.get("x90", 0))         # 61-90 days overdue
            x120 = float(account_aging.get("x120", 0))       # 91+ days overdue
            total_balance = float(account_aging.get("xbalance", 0))
            
            # Calculate total overdue amount
            total_overdue = x30 + x60 + x90 + x120
            
            # Get additional context if available
            has_failed_ptp = False
            is_new_installation = False
            
            if client_data:
                # Check for failed payment promises
                payment_history = client_data.get("payment_history", [])
                has_failed_ptp = any(p.get("arrangement_state") == "FAILED" for p in payment_history)
                
                # Check for new installation status
                contracts = client_data.get("contracts", [])
                is_new_installation = any(c.get("contract_status") == "NEW" for c in contracts)
                
                # Check account overview for additional context
                account_overview = client_data.get("account_overview", {})
                account_status = account_overview.get("account_status", "") if account_overview else ""
            
            # Determine script type based on aging priority
            if x120 > 0:  # 120+ days overdue - Legal territory
                if x120 >= 1500:  # Significant amount - attorney involvement
                    logger.info(f"Selected PRE_LEGAL_150_PLUS: x120={x120}")
                    return ScriptType.PRE_LEGAL_150_PLUS
                else:
                    logger.info(f"Selected PRE_LEGAL_120_PLUS: x120={x120}")
                    return ScriptType.PRE_LEGAL_120_PLUS
                    
            elif x90 > 0 or x60 > 0:  # 60-90 days overdue (Ratio 2-3)
                if has_failed_ptp:
                    logger.info(f"Selected RATIO_2_3_FAILED_PTP: x60={x60}, x90={x90}, failed_ptp=True")
                    return ScriptType.RATIO_2_3_FAILED_PTP
                else:
                    logger.info(f"Selected RATIO_2_3_INFLOW: x60={x60}, x90={x90}")
                    return ScriptType.RATIO_2_3_INFLOW
                    
            elif x30 > 0:  # 30 days overdue (Ratio 1)
                if is_new_installation:
                    logger.info(f"Selected RATIO_1_PRO_RATA: x30={x30}, new_installation=True")
                    return ScriptType.RATIO_1_PRO_RATA
                elif has_failed_ptp:
                    logger.info(f"Selected RATIO_1_FAILED_PTP: x30={x30}, failed_ptp=True")
                    return ScriptType.RATIO_1_FAILED_PTP
                else:
                    logger.info(f"Selected RATIO_1_INFLOW: x30={x30}")
                    return ScriptType.RATIO_1_INFLOW
                    
            else:
                # Current account or edge case - default to ratio 1 inflow
                logger.info(f"Selected RATIO_1_INFLOW (default): total_overdue={total_overdue}")
                return ScriptType.RATIO_1_INFLOW
                
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error determining script type from aging: {e}")
            logger.info("Defaulting to RATIO_1_INFLOW due to error")
            return ScriptType.RATIO_1_INFLOW
    
    @classmethod
    def get_aging_context(cls, script_type: ScriptType) -> Dict[str, str]:
        """Get aging-specific context for prompts."""
        
        aging_contexts = {
            ScriptType.RATIO_1_INFLOW: {
                "category": "First Missed Payment",
                "urgency": "Medium",
                "approach": "Professional but understanding. Focus on immediate resolution to prevent escalation.",
                "consequences": "Service suspension, app stops working, lose vehicle positioning and notifications.",
                "tone": "Helpful and solution-focused"
            },
            
            ScriptType.RATIO_1_FAILED_PTP: {
                "category": "Failed Promise to Pay",
                "urgency": "Medium-High", 
                "approach": "Address the broken promise directly. Focus on reliability and immediate action.",
                "consequences": "Service remains suspended. Previous commitment wasn't honored, need immediate resolution.",
                "tone": "Firmer, accountability-focused"
            },
            
            ScriptType.RATIO_1_PRO_RATA: {
                "category": "New Installation Pro-Rata",
                "urgency": "Medium",
                "approach": "Educational approach. Explain pro-rata billing and activate services upon payment.",
                "consequences": "Services won't activate until payment received. Vehicle remains unprotected.",
                "tone": "Educational and supportive"
            },
            
            ScriptType.RATIO_2_3_INFLOW: {
                "category": "2-3 Months Overdue",
                "urgency": "High",
                "approach": "Firm but solution-focused. Emphasize significant consequences and recovery fees.",
                "consequences": "Services suspended, potential R25,000 recovery fee if vehicle stolen, credit listing risk.",
                "tone": "Serious and urgent"
            },
            
            ScriptType.RATIO_2_3_FAILED_PTP: {
                "category": "2-3 Months Failed PTP",
                "urgency": "High",
                "approach": "Address pattern of non-payment. Create urgency about escalation to legal action.",
                "consequences": "Account escalation imminent, legal action consideration, significant financial impact.",
                "tone": "Firm and consequence-focused"
            },
            
            ScriptType.PRE_LEGAL_120_PLUS: {
                "category": "Pre-Legal 120+ Days",
                "urgency": "Very High",
                "approach": "Final opportunity messaging. Offer settlement discount to prevent legal action.",
                "consequences": "Credit default listing (R1800 to clear), legal action, attorney fees, court costs.",
                "tone": "Serious final opportunity"
            },
            
            ScriptType.PRE_LEGAL_150_PLUS: {
                "category": "Legal 150+ Days",
                "urgency": "Critical",
                "approach": "Legal representative tone. Formal debt acknowledgment and payment demand.",
                "consequences": "Notice of intent to summons, 10-day payment demand, sheriff service, court proceedings.",
                "tone": "Formal legal authority"
            }
        }
        
        return aging_contexts.get(script_type, aging_contexts[ScriptType.RATIO_1_INFLOW])
    
    @classmethod
    def get_script_enhanced_prompt(
        cls,
        base_prompt: str,
        script_type: ScriptType,
        step: CallStep,
        client_data: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """
        Enhance any prompt with appropriate script content based on aging type.
        Args:
            base_prompt: The base prompt to enhance
            script_type: ScriptType enum value
            step: CallStep enum value
            client_data: Client data dictionary
            state: Current state dictionary
        Returns:
            Enhanced prompt string
        """
        # Get raw script-specific content
        raw_script_content = cls.get_script_content(script_type, step)
        
        # Format the script content with client_data and state
        formatted_script_content = cls.format_script_content(raw_script_content, client_data, state)

        # Get formatted objection and emotional responses
        # Note: You might want to pass client_data/state to these methods as well if their content
        # also needs dynamic formatting. For simplicity here, assuming they are mostly static
        # or handled by broader context in the LLM.
        objection_responses_template = cls.get_objection_response(script_type, "general")
        formatted_objection_responses = cls.format_script_content(objection_responses_template, client_data, state)

        emotional_responses_template = cls.get_emotional_response(script_type, "neutral")
        formatted_emotional_responses = cls.format_script_content(emotional_responses_template, client_data, state)
        
        # Get aging-specific context
        aging_context = cls.get_aging_context(script_type)
        
        # Build enhanced prompt
        enhanced_prompt = f"""{base_prompt}

<script_context>
Script Type: {script_type.value}
Aging Category: {aging_context['category']}
Urgency Level: {aging_context['urgency']}
Recommended Tone: {aging_context['tone']}
</script_context>

<script_content>
{formatted_script_content}
</script_content>

<aging_specific_approach>
{aging_context['approach']}
</aging_specific_approach>

<consequences_framework>
{aging_context['consequences']}
</consequences_framework>

<objection_handling>
{formatted_objection_responses}
</objection_handling>

<emotional_intelligence>
{formatted_emotional_responses}
</emotional_intelligence>"""

        return enhanced_prompt
    
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
# Convenience functions for easy integration
def determine_script_type_from_aging(account_aging: Dict[str, Any], client_data: Dict[str, Any] = None) -> str:
    """
    Convenience function to determine script type and return as string.
    
    Args:
        account_aging: Account aging data
        client_data: Optional client data for context
        
    Returns:
        Script type as string value
    """
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    return script_type.value

def get_aging_aware_script_content(
    base_prompt: str,
    account_aging: Dict[str, Any],
    step: str,
    client_data: Dict[str, Any] = None,
    state: Dict[str, Any] = None
) -> str:
    """
    Convenience function to get aging-aware script content.
    
    Args:
        base_prompt: Base prompt to enhance
        account_aging: Account aging data
        step: Call step as string
        client_data: Optional client data
        state: Optional state data
        
    Returns:
        Enhanced prompt with script content
    """
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data or {})
    call_step = CallStep(step)
    
    return ScriptManager.get_script_enhanced_prompt(
        base_prompt=base_prompt,
        script_type=script_type,
        step=call_step,
        client_data=client_data or {},
        state=state or {}
    )
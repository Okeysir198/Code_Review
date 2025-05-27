#./src/Agents/call_center/call_scripts.py
"""
Enhanced call script classes and manager for various Cartrack debt collection scenarios.

This module defines script templates for different stages of debt collection,
categorized by Ratio, Pre-Legal status, and specific situations like
failed payments or short payments. Each class aims to hold the specific script
text for that scenario, inheriting common elements from BaseCallScript.
"""

from enum import Enum
from typing import Dict, Type, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define all script types identified
class ScriptType(Enum):
    """Enum class for different script types based on the debt scenario."""
    # Ratio 1 (Approx. 1 month overdue)
    RATIO_1_INFLOW = "ratio_1_inflow"
    RATIO_1_FAILED_PTP = "ratio_1_failed_ptp"
    RATIO_1_PRO_RATA = "ratio_1_pro_rata"
    RATIO_1_SHORT_PAID = "ratio_1_short_paid"
    # Ratio 2&3 (Approx. 2-3 months overdue)
    RATIO_2_3_INFLOW = "ratio_2_3_inflow"
    RATIO_2_3_FAILED_PTP = "ratio_2_3_failed_ptp"
    RATIO_2_3_SHORT_PAID = "ratio_2_3_short_paid"
    # Pre-Legal / Legal (Approx. 4+ months overdue)
    PRE_LEGAL_120_PLUS = "pre_legal_120_plus"
    PRE_LEGAL_150_PLUS = "pre_legal_150_plus"
    RECENTLY_SUSPENDED_120_PLUS = "recently_suspended_120_plus"

class BaseCallScript:
    """
    Base class for call scripts. Contains common script elements and default text.
    Subclasses override sections specific to their scenario.
    Placeholders like {variable_name} are intended to be formatted dynamically.
    """

    # --- Common Script Sections ---

    # Step 0: Introduction
    INTRODUCTION = """Good day, you are speaking to {agent_name} calling from the Cartrack Accounts Department. May I speak to {client_full_name}, please?"""

    # Step 0: Introduction
    NAME_VERIFICATION = """May I speak to {client_full_name}, please?"""

    # Step 2: Verification Prompt
    DETAILS_VERIFICATION = """Please note that this call is recorded for quality and security purposes. To ensure I am speaking to the right person, please confirm the following details:
    - Username, Date of birth, Vehicle registration, Vehicle Make, Vehicle Model,
    - Vehicle colour, VIN Number, Email address, Residential Address, Contact Number,
    - Full ID number (alone sufficient), Full Passport number (alone sufficient)"""

    # Step 2: Verification - 3rd Party Message
    THIRD_PARTY_MESSAGE = """{salutation} kindly advise our client {client_title} {client_full_name} that Cartrack called regarding an outstanding account. Please advise them to contact Cartrack urgently on 011 250 3000 to resolve this outstanding matter."""

    # Step 3: Reason for Call (Default for Inflow)
    REASON_FOR_CALL = """{client_full_name}, we did not receive your subscription payment and your account is now overdue. An immediate payment of {outstanding_amount} is required to bring your account up to date."""

    # Step 4: Negotiation (Defaults for Ratio 1 Inflow)
    NEGOTIATION = """Failure to make an immediate payment will result in the suspension of:
- Access to the Cartrack App
- Vehicle positioning through our control room
- Notifications regarding your vehicle

Once the payment is received, your account will be in good standing and all suspended services will be reinstated."""

    NEGOTIATION_QUERY_HANDLING = """If the client has a query, resolve it effectively""" # Instruction for agent

    # Step 5: Promise to Pay (PTP)
    PROMISE_TO_PAY = """For your convenience can we debit this amount from your account today?"""

    PTP_NO_GET_DETAILS = """{client_name}, please provide me with the bank details into which your salary is paid and confirm the exact date you receive your salary"""

    PTP_NO_SAME_DETAILS = """{client_name}, we have attempted to debit the subscription as per the details you have provided and the debit orders were unsuccessful. We therefore need you to provide correct details to ensure your payments are successful."""

    PTP_NO_PREVIOUS_REVERSAL = """{client_name}, our records indicate that you previously reversed a successful debit order on this bank account leading to your account falling into arrears. It is vitally important to ensure you honor your promise to pay to prevent services from being made unavailable to you. We have attempted to debit the subscriptions as per the details you have provided and the debit orders were unsuccessful. We therefore need you to provide correct details to ensure your payments are successful."""

    # Step 6: Debicheck Explanation
    DEBICHECK_SETUP = """You will receive an authentication request to your cellphone from your bank to approve this Debicheck. {client_name}, please approve this request as soon as you receive it to ensure we have a confirmed debit order arrangement. Also, note that there is an applicable R10 resubmission fee so the full amount to be debited will be {ptp_amount_plus_fee}."""
    # Agent Note: Remember to check Bank detail errors and ensure they get fixed before loading the Debit order arrangement.

    # Step 7: Subscription Reminder
    SUBSCRIPTION_REMINDER = """Please take note that this debit order is for paying off your arrears. Your normal monthly subscription of {subscription_amount} will also be debited on {subscription_date}."""

    # Step 8: Payment Portal Guide
    PAYMENT_PORTAL = """For your convenience, an SMS has been sent to you with a Cartrack payment portal link. Please click the PAY HERE button. It will take you to a payment page where you can edit the amount to the {outstanding_amount} that we agreed on. Once you have entered the amount we have agreed on, please click on CONFIRM & NEXT and you will see the different payment methods i.e. (Card Payment/Ozow/CapitecPay/Pay@).
Please select that option that you prefer using and click CONFIRM & NEXT.
(Agent guides through selected method)
Your payment as per the agreed arrangement has now been completed. You will get a pop-up showing you the total remaining arrears and asking you to arrange for the next payment to be made. Please edit the amount you will pay & the date, which must be within the next 30 days. You will receive an SMS reminder with a payment link a day before the date you have selected. You will click on the link and proceed to make the payment as you did now. Should you require any assistance, click on the WhatsApp icon & one of our friendly agents will assist you with completing the payment."""
    # Agent Note: You can send the link on the spot, on CAMS if the client has not received the link.

    # Step 9: Client Details Update
    CLIENT_DETAILS_UPDATE = """If not already done during the verification or PTP capturing stage, update the following:
    - Direct Contact Number
    - Email address
    - Banking Details & salary date
    - Next of kin details""" # Instruction for agent

    # Step 10: Referrals
    REFERRALS = """Do you know anyone who would be interested in Cartrack's services? Should they sign up and have a unit fitted on their vehicle after you have referred them, Cartrack will give you 2 months free on your subscription."""
    # Agent Note: If YES, ask the client to provide you with their contact details and action referral on the referral portal.

    # Step 11: Further Assistance
    FURTHER_ASSISTANCE = """{client_name} is there anything else you would like me to assist you with concerning your account?"""
    # Agent Note: If YES, assist accordingly.

    # Step 12 & 13: Note & Disposition (Agent Actions)
    ADD_NOTE_INSTRUCTION = "no script needed"
    DISPOSITION_CALL = "no script needed"

    # Step 14: Cancellation Declaration
    CANCELLATION = """Please be advised that the cancellation fee on your account is {cancellation_fee} and in total, you will need to pay {total_balance} for the account to be paid up. I have logged a ticket with our cancellations department & they will contact you shortly to assist with your cancellation request and inform you of the applicable terms & conditions."""
    # Agent Note: If a client wants to cancel, establish the contract to be cancelled and the reason for the cancellation. Advise of the cancellation value (If applicable), and that you will escalate the cancellation request to the relevant department (Client services). Log a cancellation ticket on the Helpdesk (If applicable). Failure to do so will result in the call being zero-rated.

    # --- Placeholders for Advanced Script Sections ---
    DISCOUNT_OFFER = ""
    INSTALMENT_ARRANGEMENT = ""
    CLOSING = """Thank you for your time today, {client_name}. We appreciate your cooperation in addressing this matter with Cartrack."""
    ESCALATION = """{client_name}, I understand your concerns regarding this matter. I'll escalate this issue to our {department} department with a {priority_level} priority. They will contact you within {response_time} at the number we have on file. Your reference number for this escalation is {ticket_number}. The team will review all details of your case and provide you with a comprehensive resolution. Is there anything specific you'd like me to note in the escalation ticket?"""



# --- Ratio 1 Scripts ---

class Ratio1InflowScript(BaseCallScript):
    """Script for Ratio 1 inflow calls (standard first missed payment). Uses most defaults."""
    pass # Uses default REASON_FOR_CALL, NEGOTIATION_CONSEQUENCES, NEGOTIATION_BENEFITS etc.

class Ratio1FailedPTPScript(BaseCallScript):
    """Script for Ratio 1 failed promise-to-pay calls."""
    REASON_FOR_CALL = """{client_name}, we did not receive your payment of {outstanding_amount} as per our last arrangement with you and your account is overdue."""
    # Uses default NEGOTIATION_CONSEQUENCES & BENEFITS from BaseCallScript
    pass

class Ratio1ProRataScript(BaseCallScript):
    """Script for Ratio 1 pro-rata payment due calls."""
    REASON_FOR_CALL = """{client_name}, kindly note as per our agreement, we have not received your pro-rata payment since the fitment of your Cartrack unit. Please advise how soon you can make this payment of {outstanding_amount} to have your services activated."""
    NEGOTIATION_CONSEQUENCES = "" # Less emphasis on suspension as services might not be active
    NEGOTIATION_BENEFITS = """Once the payment has been received, your services will be activated."""
    SUBSCRIPTION_REMINDER = """Please take note that this debit order is for paying off your pro-rata amount only. Your normal monthly subscription of {subscription_amount} will also be debited on {subscription_date} for services to be rendered next month."""
    pass

class Ratio1ShortPaidScript(BaseCallScript):
    """Script for Ratio 1 short payment situations."""
    REASON_FOR_CALL = """{client_name}, we would like to thank you for your payment of {paid_amount} that you made on {payment_date}. However, this is less than the {agreed_amount} that we agreed upon."""
    NEGOTIATION_CONSEQUENCES = """As a result, your account has fallen behind and the following services will be suspended:
    - access to the Cartrack App
    - vehicle positioning through our control room
    - and notifications regarding your vehicle""" # Explicitly state consequences
    NEGOTIATION_BENEFITS = """An immediate payment of {shortfall_amount} is required to rectify the shortfall and enable the reinstatement of all suspended services."""
    pass

# --- Ratio 2&3 Scripts ---

class Ratio2And3InflowScript(BaseCallScript):
    """Script for Ratio 2&3 inflow calls (2-3 months overdue)."""
    REASON_FOR_CALL = """{client_name}, your account is overdue for more than 2 months because of unpaid arrears and an immediate payment of {outstanding_amount} is required to bring your account up to date"""
    NEGOTIATION_CONSEQUENCES = """Failure to make this immediate payment means the following services will remain suspended:
    - access to the Cartrack App
    - vehicle positioning through our control room
    - and notifications regarding your vehicle
Should your vehicle be stolen or hijacked, you may be charged a fee of up to R25 000 for us to send out a recovery team."""
    NEGOTIATION_BENEFITS = """Once the payment has been received, your account will be in good standing and all services reinstated. You also will not be required to pay a recovery fee should your vehicle be stolen or hijacked."""
    pass

class Ratio2And3FailedPTPScript(BaseCallScript):
    """Script for Ratio 2&3 failed promise-to-pay calls."""
    REASON_FOR_CALL = """{client_name}, we did not receive your payment of {outstanding_amount} as per our last arrangement with you and your account is overdue."""
    NEGOTIATION_CONSEQUENCES = """As a result, the following services remain suspended:
    - access to the Cartrack App
    - vehicle positioning through our control room
    - and notifications regarding your vehicle
Should your vehicle be stolen or hijacked, you may be charged a fee of up to R25 000 for us to send out a recovery team. Continued failure to pay off the arrears will lead to your account being handed over to our legal department for possible legal action."""
    NEGOTIATION_BENEFITS = """An immediate payment of {outstanding_amount} is required to bring your account up to date and enable the reinstatement of all suspended services."""
    pass

class Ratio2And3ShortPaidScript(BaseCallScript):
    """Script for Ratio 2&3 short payment situations."""
    REASON_FOR_CALL = """{client_name}, we would like to thank you for your payment of {paid_amount} that you made on {payment_date}. However, this is less than the {agreed_amount} payment that we agreed upon."""
    NEGOTIATION_CONSEQUENCES = """As a result, the following services remain suspended:
    - access to the Cartrack App
    - vehicle positioning through our control room
    - and notifications regarding your vehicle
Should your vehicle be stolen or hijacked, you may be charged a fee of up to R25 000 for us to send out a recovery team. Continued failure to pay off the arrears will result in your account being handed over to our legal department for possible legal action."""
    NEGOTIATION_BENEFITS = """An immediate payment of {shortfall_amount} is required to bring your account up to date and enable the reinstatement of all suspended services."""
    pass

# --- Pre-Legal / Legal / Suspended Scripts ---

class PreLegal120PlusScript(BaseCallScript):
    """Script for Pre-Legal 120+ days overdue accounts."""
    REASON_FOR_CALL = """{client_name}, your account has been overdue for more than 4 months and is now in our pre-legal department because of unpaid arrears. A letter of demand has been sent to you and an immediate payment of {outstanding_amount} is required to bring your account up to date."""

    # Combined consequences for Active/Cancelled during initial negotiation
    NEGOTIATION_CONSEQUENCES = """Please take note that Cartrack services to you remain suspended (if active) and you have been/will be listed as a default payer on your credit profile due to your account being in arrears. If you wish to have the listing cleared after settling your account, there will be an applicable fee of R1800. Should your vehicle be stolen or hijacked (if active), you may be charged a fee of up to R25 000 for us to send out a recovery team. If an immediate payment is not made to settle the arrears, your account will get handed over for legal action."""

    DISCOUNT_OFFER = """To prevent further legal steps from being taken, we are offering you a {discount_percentage}% settlement discount which requires you to pay {discounted_amount} once-off within this current month of {current_month} to settle the total arrears. Are you in a position to take up this discounted settlement offer?"""

    INSTALMENT_ARRANGEMENT = """Seeing that you are not in a position to settle the arrears once off with our discount offer, we can assist you to bring your account up to date by paying this discounted amount as follows:
    Option 1: {instalment_amount_2} to be debited for the next 2 months on your suitable debit date to cater for this discounted settlement balance of {discounted_amount} (, plus your monthly subscription of {subscription_amount} if active).
    Option 2: A minimum of {instalment_amount_3} to be debited for the next 3 months on your suitable debit date to cater for this discounted settlement balance of {discounted_amount} (, plus your monthly subscription of {subscription_amount} if active)."""

    # Specific Education/Benefits based on account status and payment type
    EDUCATION_BENEFITS_ACTIVE_SETTLEMENT = """By settling these arrears, you ensure that once the payment has been received, your account will be up to date and any pending legal action will be stopped. Cartrack services to you will be re-instated."""
    EDUCATION_BENEFITS_ACTIVE_INSTALMENT = """By paying {ptp_amount} for the next {num_instalments} months you ensure that pending legal action is put on hold as long as the payments are received as per this arrangement. After these instalments have all been paid, your account will be up to date and Cartrack services to you will be re-instated."""
    EDUCATION_BENEFITS_CANCELLED_SETTLEMENT = """By settling these arrears, you ensure that once the payment has been received, your account will be closed and any pending legal action will be stopped. We will provide you with a paid-up letter as proof that you no longer owe Cartrack."""
    EDUCATION_BENEFITS_CANCELLED_INSTALMENT = """By paying {ptp_amount} for the next {num_instalments} months you ensure that pending legal action is put on hold as long as the payments are received as per this arrangement. After these instalments have all been paid, your account will be closed and we will provide you with a paid-up letter as proof that you no longer owe Cartrack."""

    # Specific Consequences of Non-Payment based on account status
    CONSEQUENCES_NON_PAYMENT_ACTIVE = """Failure to make this immediate payment will result in your account being handed over for legal action. You will be listed as a default payer on your credit profile due to your account being in arrears and if you wish to have the listing cleared after settling your account, there will be an applicable fee of R1800. Cartrack services to you remain suspended and should your vehicle be stolen or hijacked, you may be charged a fee of up to R25 000 for us to send out a recovery team."""
    CONSEQUENCES_NON_PAYMENT_CANCELLED = """Failure to make this immediate payment will result in your account being handed over for legal action. You will be listed as a default payer on your credit profile due to your account being in arrears and if you wish to have the listing cleared after settling your account, there will be an applicable fee of R1800."""
    pass


class PreLegal150PlusScript(PreLegal120PlusScript): # Inherits from 120+ as structure is similar
    """Script for Pre-Legal/Legal 150+ days overdue accounts (Attorney involved)."""
    INTRODUCTION = """Good day, you are speaking to {agent_name} and I am calling you from Viljoen Attorneys on behalf of Cartrack. May I speak to {client_full_name}."""
    THIRD_PARTY_MESSAGE = """{salutation} kindly advise {client_title} {client_full_name} that Cartrack's Attorneys called regarding an outstanding Cartrack account. Please advise {client_title} {client_full_name} to contact Viljoen Attorneys urgently on 010 140 0085 to resolve this outstanding matter."""
    # Note: Client Details Update happens earlier (Step 3 in docx) - handled by agent logic flow

    REASON_FOR_CALL = """{client_name}, be advised that Cartrack has handed over your account to us as Viljoen Attorneys due to non-payment, your total arrears are {outstanding_amount} and we are now acting on their behalf to recover these outstanding arrears. Do you acknowledge this debt?"""

    # Override instalment arrangement with specific balance guidelines
    INSTALMENT_ARRANGEMENT = """Seeing that you are not in a position to settle the arrears once off with our discount offer, we can assist you with an instalment plan based on the discounted amount:
    - Balance R2000 or less: Minimum {instalment_amount_3} over 3 months.
    - Balance R2000–R4000: Minimum {instalment_amount_4} over 4 months.
    - Balance R4000 & above: Minimum {instalment_amount_6} over 6 months.
    (, plus your monthly subscription of {subscription_amount} if active).""" # Agent needs to calculate and provide amounts

    # Override consequences of non-payment
    CONSEQUENCES_NON_PAYMENT = """Should you not honour the agreement to rectify this account, we will proceed to issue you with a notice of intent to summons followed by a letter of demand giving you 10 days to make the payment. Failure to make payment thereafter will result in us proceeding with actual summons through the sheriff of court."""
    # Remove specific Active/Cancelled versions as the main consequence is legal summons
    CONSEQUENCES_NON_PAYMENT_ACTIVE = CONSEQUENCES_NON_PAYMENT
    CONSEQUENCES_NON_PAYMENT_CANCELLED = CONSEQUENCES_NON_PAYMENT
    pass


class RecentlySuspended120PlusScript(PreLegal120PlusScript): # Inherits from 120+
    """Script for Recently Suspended 120+ days accounts with specific campaign."""
    REASON_FOR_CALL = """{client_name}, your account has been overdue for more than 4 months and is now in our pre-legal department because of unpaid arrears. A letter of demand has been sent to you and an immediate payment of {outstanding_amount} is required to bring your account up to date."""

    # Specific discount campaign replaces generic discount offer from parent
    DISCOUNT_OFFER = "" # Clear parent offer
    SETTLEMENT_DISCOUNT_OFFER = """To help get your Cartrack account up to date/paid up, we are offering you a massive 50% discount that requires you to pay {discounted_amount_50} once-off, on or before {campaign_end_date_50} to settle the total arrears. Are you in a position to take up this discounted settlement offer?""" # Specific 50% offer

    # Specific instalment arrangement linked to the campaign
    INSTALMENT_ARRANGEMENT = """The good thing with this discount campaign is if you are not able to pay this 50% discounted settlement by {campaign_end_date_50}; you can pay the settlement in 2 instalments with a 40% discount, provided the first amount of {instalment_amount_40} is paid by {campaign_first_instalment_date_40}. Can I go ahead and load/arrange for these 2 payments to be made?""" # Specific 40% / 2 instalment offer

    # Consequences if arrangement not honoured (similar to parent)
    CONSEQUENCES_NON_PAYMENT_ARRANGEMENT_ACTIVE = """Failure to make this payment as per our agreement, will result in your account being handed over for legal action. You will be listed as a default payer on your credit profile due to your account being in arrears and if you wish to have the listing cleared after settling your account, there will be an applicable fee of R1800. Cartrack services to you remain suspended and should your vehicle be stolen or hijacked, you may be charged a fee of up to R25 000 for us to send out a recovery team."""
    CONSEQUENCES_NON_PAYMENT_ARRANGEMENT_CANCELLED = """Failure to make this payment as per our agreement, will result in your account being handed over for legal action. You will be listed as a default payer on your credit profile due to your account being in arrears and if you wish to have the listing cleared after settling your account, there will be an applicable fee of R1800."""
    # Link these to the standard non-payment consequences for simplicity if needed by agent logic
    CONSEQUENCES_NON_PAYMENT_ACTIVE = CONSEQUENCES_NON_PAYMENT_ARRANGEMENT_ACTIVE
    CONSEQUENCES_NON_PAYMENT_CANCELLED = CONSEQUENCES_NON_PAYMENT_ARRANGEMENT_CANCELLED
    pass


class ScriptManager:
    """Factory class to create the appropriate script class based on script type."""

    # Updated script mapping dictionary
    _SCRIPT_MAP: Dict[ScriptType, Type[BaseCallScript]] = {
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
    def get_script_class(cls, script_type: ScriptType) -> Type[BaseCallScript]:
        """
        Return the appropriate script class based on script type.

        Args:
            script_type: Type of script from ScriptType enum.

        Returns:
            Script class appropriate for the situation. Defaults to Ratio1InflowScript if type not found.
        """
        return cls._SCRIPT_MAP.get(script_type, Ratio1InflowScript)

    @classmethod
    def get_script_text(cls, script_type: ScriptType, section: str) -> str:
        """
        Get raw text template for a specific section of a given script type.
        Returns the unformatted template with placeholders intact.

        Args:
            script_type: The type of script required.
            section: The name of the script section.

        Returns:
            Unformatted script text template for the requested section.
        """
        # Get the appropriate script class
        script_class = cls.get_script_class(script_type)
        section_upper = section.upper()  # Use uppercase for attribute lookup

        # Try getting attribute from specific class, then fallback to base class
        template = getattr(script_class, section_upper, None)
        if template is None:
            template = getattr(BaseCallScript, section_upper, None)

        if template is None or not isinstance(template, str):
            logger.warning(f"Section '{section}' not found or invalid for script type '{script_type.value}'")
            return f"[Script section '{section}' not available for {script_type.value}]"
        
        # Return the raw template with placeholders intact
        return template

    @classmethod
    def get_available_scripts(cls) -> Dict[str, str]:
        """
        Get a dictionary of available script types with descriptions.

        Returns:
            Dictionary mapping script type keys (enum values) to human-readable descriptions.
        """
        # Updated descriptions using enum values as keys
        return {
            ScriptType.RATIO_1_INFLOW.value: "Ratio 1: Standard first missed payment",
            ScriptType.RATIO_1_FAILED_PTP.value: "Ratio 1: Follow-up on failed payment promise",
            ScriptType.RATIO_1_PRO_RATA.value: "Ratio 1: Pro-rata payment due after fitment",
            ScriptType.RATIO_1_SHORT_PAID.value: "Ratio 1: Address partial payment shortfall",
            ScriptType.RATIO_2_3_INFLOW.value: "Ratio 2&3: Standard missed payment (2-3 months overdue)",
            ScriptType.RATIO_2_3_FAILED_PTP.value: "Ratio 2&3: Follow-up on failed PTP (2-3 months overdue)",
            ScriptType.RATIO_2_3_SHORT_PAID.value: "Ratio 2&3: Address short payment (2-3 months overdue)",
            ScriptType.PRE_LEGAL_120_PLUS.value: "Pre-Legal: 120+ days overdue, discount/instalment offer",
            ScriptType.PRE_LEGAL_150_PLUS.value: "Pre-Legal/Legal: 150+ days overdue (Attorney involved)",
            ScriptType.RECENTLY_SUSPENDED_120_PLUS.value: "Suspended: 120+ days overdue, specific discount campaign",
        }
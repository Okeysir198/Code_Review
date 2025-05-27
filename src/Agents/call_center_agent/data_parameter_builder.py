# ./src/Agents/call_center_center/data_parameter_builder.py
"""
Optimized data parameter builder for call center AI agents.
Integrates real client data with behavioral intelligence and tactical guidance.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

# Import real database tools
from src.Database.CartrackSQLDatabase import (
    get_client_profile,
    get_client_account_overview,
    get_client_account_aging,
    get_client_banking_details,
    get_client_subscription_amount,
    get_client_payment_history,
    get_client_failed_payments,
    get_client_last_successful_payment,
    get_client_contracts,
    get_client_billing_analysis,
    get_client_debit_mandates
)
from src.Agents.call_center_agent.call_scripts import ScriptManager, ScriptType, CallStep
from .call_scripts import ScriptManager, ScriptType, CallStep

logger = logging.getLogger(__name__)

class VerificationStatus(Enum):
    """Verification status options."""
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    VERIFIED = "VERIFIED"
    THIRD_PARTY = "THIRD_PARTY"
    UNAVAILABLE = "UNAVAILABLE"
    WRONG_PERSON = "WRONG_PERSON"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"

########################################################################################
class ClientDataBuilder:
    """Handles client data fetching and caching."""
    
    _cache = {}
    _cache_duration = timedelta(hours=1)
    
    @classmethod
    def get_client_data(cls, user_id: str, force_reload: bool = False) -> Dict[str, Any]:
        """Get client data with caching."""
        cache_key = user_id
        now = datetime.now()
        
        # Check cache
        if not force_reload and cache_key in cls._cache:
            cached_entry = cls._cache[cache_key]
            if now - cached_entry["timestamp"] < cls._cache_duration:
                logger.info(f"Using cached data for user_id: {user_id}")
                return cached_entry["data"]
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for user_id: {user_id}")
        try:
            data = cls._fetch_client_data(user_id)
            cls._cache[cache_key] = {"data": data, "timestamp": now}
            return data
        except Exception as e:
            logger.error(f"Error fetching client data for {user_id}: {e}")
            return cls._get_fallback_data(user_id)
    
    @classmethod
    def _fetch_client_data(cls, user_id: str) -> Dict[str, Any]:
        """Fetch data from database using real tools."""
        try:
            # Load core client information
            profile = get_client_profile.invoke(user_id)
            if not profile:
                raise ValueError(f"Client profile not found for user_id: {user_id}")
            
            # # Load account overview and financial data
            # account_overview = get_client_account_overview.invoke(user_id)
            # account_aging = get_client_account_aging.invoke(user_id)
            # banking_details = get_client_banking_details.invoke(user_id)
            
            # # Load subscription and payment data
            # subscription_data = get_client_subscription_amount.invoke(user_id)
            # payment_history = get_client_payment_history.invoke(user_id)
            # failed_payments = get_client_failed_payments.invoke(user_id)
            # last_payment = get_client_last_successful_payment.invoke(user_id)
            
            # # Load contracts and billing analysis
            # contracts = get_client_contracts.invoke(user_id)
            # billing_analysis = get_client_billing_analysis.invoke(user_id)
            
            # # load existing_mandates

            # existing_mandates = get_client_debit_mandates.invoke(user_id)

            # Consolidate all data
            client_data = {
                "user_id": user_id,
                "profile": profile,
                # "account_overview": account_overview,
                # "account_aging": account_aging[0] if account_aging else {},
                # "banking_details": banking_details[0] if banking_details else {},
                # "subscription": subscription_data,
                # "payment_history": payment_history[:5] if payment_history else [],
                # "failed_payments": failed_payments[:3] if failed_payments else [],
                # "last_successful_payment": last_payment,
                # "contracts": contracts,
                # "billing_analysis": billing_analysis[0] if billing_analysis else {},
                # "existing_mandates": existing_mandates,
                "loaded_at": datetime.now()
            }
            
            logger.info(f"Successfully loaded data for user_id: {user_id}")
            return client_data
            
        except Exception as e:
            logger.error(f"Error loading client data for {user_id}: {str(e)}")
            raise
    
    @classmethod
    def _get_fallback_data(cls, user_id: str) -> Dict[str, Any]:
        """Fallback data when database fails."""
        return {
            "user_id": user_id,
            "profile": {
                "client_info": {
                    "client_full_name": "Client",
                    "first_name": "Client",
                    "title": "Mr/Ms"
                }
            },
            "account_overview": {"account_status": "Overdue"},
            "account_aging": {"xbalance": "0.00"},
            "subscription": {"subscription_amount": "0.00"},
            "payment_history": [],
            "failed_payments": [],
            "last_successful_payment": None,
            "contracts": [],
            "billing_analysis": {},
            "existing_mandates": {},
            "loaded_at": datetime.now()
        }
    
    @classmethod
    def clear_cache(cls, user_id: Optional[str] = None):
        """Clear cached data."""
        if user_id:
            cls._cache.pop(user_id, None)
        else:
            cls._cache.clear()

########################################################################################
class BehavioralAnalyzer:
    """Analyzes client behavior and provides tactical guidance."""
    
    @staticmethod
    def analyze_client_profile(client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze client behavior patterns."""
        try:
            # Safe data extraction
            payment_history = client_data.get("payment_history", [])
            account_aging = client_data.get("account_aging", {})
            billing_analysis = client_data.get("billing_analysis", {})
            failed_payments = client_data.get("failed_payments", [])
            
            # Ensure dictionaries
            if not isinstance(account_aging, dict):
                account_aging = {}
            if not isinstance(billing_analysis, dict):
                billing_analysis = {}
            
            # Get balance safely
            try:
                balance = float(account_aging.get("xbalance", 0))
            except (ValueError, TypeError):
                balance = 0.0
            
            # Calculate metrics
            days_overdue = BehavioralAnalyzer._get_days_overdue(billing_analysis, account_aging)
            reliability = BehavioralAnalyzer._assess_payment_reliability(payment_history, failed_payments)
            objections = BehavioralAnalyzer._predict_objections(client_data, balance, days_overdue)
            approach = BehavioralAnalyzer._determine_approach(days_overdue, reliability, balance)
            
            return {
                "days_overdue": days_overdue,
                "payment_reliability": reliability,
                "likely_objections": objections,
                "optimal_approach": approach,
                "risk_level": BehavioralAnalyzer._assess_risk_level(days_overdue, balance),
                "success_probability": BehavioralAnalyzer._calculate_success_probability(reliability, balance, days_overdue)
            }
        
        except Exception:
            return {
                "days_overdue": 0,
                "payment_reliability": "unknown",
                "likely_objections": ["will_pay_later"],
                "optimal_approach": "professional_persistent",
                "risk_level": "medium",
                "success_probability": "medium"
            }
    
    @staticmethod
    def _get_days_overdue(billing_analysis: Dict[str, Any], account_aging: Dict[str, Any]) -> int:
        """Get actual days overdue from database data."""
        try:
            # Check billing analysis first
            aging_checks = [
                ("balance_450_days", 450), ("balance_360_days", 360), ("balance_270_days", 270),
                ("balance_180_days", 180), ("balance_120_days", 120), ("balance_90_days", 90),
                ("balance_60_days", 60), ("balance_30_days", 30)
            ]
            
            for field, days in aging_checks:
                try:
                    if float(billing_analysis.get(field, 0)) > 0:
                        return days
                except (ValueError, TypeError):
                    continue
            
            # Fallback to account aging
            aging_fallback = [("x120", 120), ("x90", 90), ("x60", 60), ("x30", 30), ("x0", 0)]
            
            for field, days in aging_fallback:
                try:
                    if float(account_aging.get(field, 0)) > 0:
                        return days
                except (ValueError, TypeError):
                    continue
            
            return 0
        
        except Exception:
            return 0
    
    @staticmethod
    def _assess_payment_reliability(payment_history: List[Dict], failed_payments: List[Dict]) -> str:
        """Assess client payment reliability from real data."""
        if not payment_history and not failed_payments:
            return "unknown"
        
        total_attempts = len(payment_history)
        failed_attempts = len(failed_payments)
        
        if total_attempts == 0:
            return "low" if failed_attempts > 0 else "unknown"
        
        # Calculate success rate
        success_rate = (total_attempts - failed_attempts) / total_attempts if total_attempts > 0 else 0
        
        if success_rate >= 0.8:
            return "high"
        elif success_rate >= 0.5:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def _predict_objections(client_data: Dict[str, Any], balance: float, days_overdue: int) -> List[str]:
        """Predict likely client objections based on real data."""
        objections = []
        account_overview = client_data.get("account_overview", {})
        payment_history = client_data.get("payment_history", [])
        failed_payments = client_data.get("failed_payments", [])
        
        # High balance objections
        if balance > 500:
            objections.extend(["dispute_amount", "no_money"])
        
        # Recent payment activity
        if balance < 200:
            objections.append("already_paid")
        
        # Failed payment patterns
        if failed_payments:
            objections.append("bank_error")
        
        # Long overdue accounts
        if days_overdue > 90:
            objections.extend(["not_my_debt", "will_pay_later"])
        
        # Account status specific
        if isinstance(account_overview, dict) and account_overview.get("account_status") == "Overdue":
            objections.append("will_pay_later")
        
        # Always include the most common objection
        if "will_pay_later" not in objections:
            objections.append("will_pay_later")
        
        return objections[:5]  # Limit to top 5 objections
    
    @staticmethod
    def _determine_approach(days_overdue: int, reliability: str, balance: float) -> str:
        """Determine optimal tactical approach based on real data."""
        if days_overdue <= 30 and reliability == "high":
            return "friendly_reminder"
        elif days_overdue <= 60:
            return "consequence_focused"
        elif days_overdue <= 120:
            return "urgent_resolution"
        else:
            return "legal_prevention"
    
    @staticmethod
    def _assess_risk_level(days_overdue: int, balance: float) -> str:
        """Assess overall risk level."""
        if days_overdue >= 120 or balance > 1000:
            return "high"
        elif days_overdue >= 60 or balance > 500:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def _calculate_success_probability(reliability: str, balance: float, days_overdue: int) -> str:
        """Calculate probability of successful resolution."""
        score = 0
        
        # Reliability factor
        if reliability == "high":
            score += 3
        elif reliability == "medium":
            score += 2
        elif reliability == "low":
            score += 1
        
        # Balance factor
        if balance < 300:
            score += 2
        elif balance < 700:
            score += 1
        
        # Days overdue factor
        if days_overdue <= 30:
            score += 2
        elif days_overdue <= 90:
            score += 1
        
        # Determine probability
        if score >= 6:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"
        
########################################################################################
class ScriptFormatter:
    """Handles formatting of script content with real parameter values."""
    
    @staticmethod
    def format_script_content(
        script_content: str, 
        parameters: Dict[str, Any],
        current_step: str
    ) -> str:
        """Format script content with actual parameter values."""
        if not script_content:
            return ""
        
        try:
            format_params = ScriptFormatter._prepare_format_parameters(parameters, current_step)
            return script_content.format(**format_params)
        except Exception as e:
            logger.warning(f"Script formatting error for {current_step}: {e}")
            # Simple replacement for basic parameters
            result = script_content
            basic_replacements = {
                "{agent_name}": parameters.get("agent_name", "Agent"),
                "{client_full_name}": parameters.get("client_full_name", "Client"),
                "{client_name}": parameters.get("client_name", "Client"),
                "{client_title}": parameters.get("client_title", "Mr/Ms"),
                "{salutation}": parameters.get("salutation", "Sir/Madam"),
                "{outstanding_amount}": parameters.get("outstanding_amount", "R 0.00"),
                "{field_to_verify}": parameters.get("field_to_verify", "ID number"),
                "{amount_with_fee}": parameters.get("amount_with_fee", "R 10.00")
            }
            
            for placeholder, value in basic_replacements.items():
                result = result.replace(placeholder, str(value))
            
            return result
    
    @staticmethod
    def _prepare_format_parameters(parameters: Dict[str, Any], current_step: str) -> Dict[str, str]:
        """Prepare parameters for safe script formatting."""
        # Base parameters that are always needed
        format_params = {
            "agent_name": parameters.get("agent_name", "Agent"),
            "client_full_name": parameters.get("client_full_name", "Client"),
            "client_name": parameters.get("client_name", "Client"),
            "client_title": parameters.get("client_title", "Mr/Ms"),
            "salutation": parameters.get("salutation", "Sir/Madam"),
            "outstanding_amount": parameters.get("outstanding_amount", "R 0.00"),
            "subscription_amount": parameters.get("subscription_amount", "R 199.00"),
            "subscription_date": parameters.get("subscription_date", "5th of each month"),
            "amount_with_fee": parameters.get("amount_with_fee", "R 10.00"),
            "ptp_amount_plus_fee": parameters.get("ptp_amount_plus_fee", "R 10.00"),
            "cancellation_fee": parameters.get("cancellation_fee", "R 0.00"),
            "total_balance": parameters.get("total_balance", "R 0.00"),
            "field_to_verify": parameters.get("field_to_verify", "ID number")
        }
        
        # Add step-specific parameters
        if current_step in ["reason_for_call", "negotiation", "discount_offer"]:
            format_params.update({
                "paid_amount": parameters.get("paid_amount", "R 0.00"),
                "payment_date": parameters.get("payment_date", "N/A"),
                "agreed_amount": parameters.get("agreed_amount", parameters.get("outstanding_amount", "R 0.00")),
                "shortfall_amount": parameters.get("shortfall_amount", "R 0.00"),
                "additional_consequences": parameters.get("additional_consequences", ""),
                "discount_percentage": parameters.get("discount_percentage", "20"),
                "discounted_amount": parameters.get("discounted_amount", "R 0.00"),
                "discounted_amount_50": parameters.get("discounted_amount_50", "R 0.00"),
                "campaign_end_date": parameters.get("campaign_end_date", "month-end"),
                "campaign_first_date": parameters.get("campaign_first_date", "15th")
            })
        
        elif current_step in ["escalation", "closing", "third_party_message"]:
            format_params.update({
                "department": parameters.get("department", "Supervisor"),
                "response_time": parameters.get("response_time", "24-48 hours"),
                "ticket_number": parameters.get("ticket_number", "TKT123456"),
                "outcome_summary": parameters.get("outcome_summary", "Thank you for your time"),
                "payment_method": parameters.get("payment_method", "agreed method")
            })
        
        # Convert any None values to empty strings to prevent formatting errors
        for key, value in format_params.items():
            if value is None:
                format_params[key] = ""
            elif not isinstance(value, str):
                format_params[key] = str(value)
        
        return format_params


########################################################################################
class ParameterBuilder:
    """Builds complete parameters for prompt generation."""
    
    @staticmethod
    def build_parameters(
        client_data: Dict[str, Any],
        current_step: str,
        state: Dict[str, Any],
        script_type: str = "ratio_1_inflow",
        agent_name: str = "AI Agent"
    ) -> Dict[str, Any]:
        """Build complete parameters for prompt generation."""
        
        try:
            # Extract all information
            basic_info = ParameterBuilder._extract_basic_info(client_data, agent_name)
            financial_info = ParameterBuilder._extract_financial_info(client_data)
            verification_info = ParameterBuilder._extract_verification_info(client_data)
            behavioral_analysis = BehavioralAnalyzer.analyze_client_profile(client_data)
            state_info = ParameterBuilder._build_state_info(state, current_step)
            
            # Combine all parameters
            parameters = {
                **basic_info,
                **financial_info,
                **verification_info,
                **state_info,
                "behavioral_analysis": behavioral_analysis,
                "tactical_guidance": ParameterBuilder._generate_tactical_guidance(
                    behavioral_analysis, script_type, current_step
                )
            }
            
            # Get and add script guidance
            script_guidance = ParameterBuilder._get_formatted_script_guidance(
                script_type, current_step, parameters
            )
            parameters.update(script_guidance)
            
            # Flatten for template use
            parameters = ParameterBuilder._flatten_nested_parameters(parameters)
            
            user_id = client_data.get("user_id", "unknown")
            logger.info(f"Built parameters for user_id: {user_id}, step: {current_step}")
            return parameters
            
        except Exception as e:
            logger.error(f"Error building parameters: {e}")
            return ParameterBuilder._get_fallback_parameters(agent_name, current_step)
    
    @staticmethod
    def _get_fallback_parameters(agent_name: str, current_step: str) -> Dict[str, Any]:
        """Return safe fallback parameters when building fails."""
        return {
            "agent_name": agent_name,
            "client_full_name": "Client",
            "client_name": "Client", 
            "client_title": "Mr/Ms",
            "salutation": "Sir/Madam",
            "outstanding_amount": "R 0.00",
            "subscription_amount": "R 199.00",
            "subscription_date": "5th of each month",
            "amount_with_fee": "R 10.00",
            "current_step": current_step,
            "name_verification_status": "INSUFFICIENT_INFO",
            "details_verification_status": "INSUFFICIENT_INFO",
            "script_content": "",
            "objection_responses": "",
            "emotional_responses": "",
            "escalation_responses": "",
            "field_to_verify": "ID number",
            "tactical_guidance[recommended_approach]": "professional_persistent",
            "tactical_guidance[urgency_level]": "medium",
            "behavioral_analysis[risk_level]": "medium"
        }
    
    @staticmethod
    def _extract_basic_info(client_data: Dict[str, Any], agent_name: str) -> Dict[str, str]:
        """Extract basic client information."""
        try:
            profile = client_data.get("profile", {})
            client_info = profile.get("client_info", {}) if isinstance(profile, dict) else {}
            
            if not isinstance(client_info, dict):
                client_info = {}
            
            full_name = client_info.get("client_full_name", "Client")
            first_name = client_info.get("first_name", full_name.split()[0] if full_name != "Client" else "Client")
            title = client_info.get("title", "Mr/Ms")
            
            return {
                "agent_name": agent_name,
                "client_full_name": full_name,
                "client_name": first_name,
                "client_title": title,
                "salutation": "Sir/Madam"
            }
        
        except Exception:
            return {
                "agent_name": agent_name,
                "client_full_name": "Client",
                "client_name": "Client",
                "client_title": "Mr/Ms",
                "salutation": "Sir/Madam"
            }
    
    @staticmethod
    def _extract_financial_info(client_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract financial information."""
        try:
            account_aging = client_data.get("account_aging", {})
            subscription = client_data.get("subscription", {})
            
            if not isinstance(account_aging, dict):
                account_aging = {}
            if not isinstance(subscription, dict):
                subscription = {}
            
            # Get amounts with safe parsing
            outstanding_amount = account_aging.get("xbalance", "0")
            subscription_amount = subscription.get("subscription_amount", "199.00")
            
            try:
                outstanding_float = float(outstanding_amount) if outstanding_amount else 0.0
                outstanding = f"R {outstanding_float:.2f}"
            except (ValueError, TypeError):
                outstanding = "R 0.00"
                outstanding_float = 0.0
            
            try:
                subscription_float = float(subscription_amount) if subscription_amount else 199.00
                subscription_str = f"R {subscription_float:.2f}"
            except (ValueError, TypeError):
                subscription_str = "R 199.00"
            
            return {
                "outstanding_amount": outstanding,
                "subscription_amount": subscription_str,
                "subscription_date": "5th of each month",
                "amount_with_fee": f"R {outstanding_float + 10:.2f}",
                "ptp_amount_plus_fee": f"R {outstanding_float + 10:.2f}",
                "cancellation_fee": "R 0.00",
                "total_balance": outstanding,
                "discounted_amount": f"R {outstanding_float * 0.8:.2f}",
                "discounted_amount_50": f"R {outstanding_float * 0.5:.2f}",
                "discount_percentage": "20"
            }
        
        except Exception:
            return {
                "outstanding_amount": "R 0.00",
                "subscription_amount": "R 199.00",
                "subscription_date": "5th of each month",
                "amount_with_fee": "R 10.00",
                "ptp_amount_plus_fee": "R 10.00",
                "cancellation_fee": "R 0.00",
                "total_balance": "R 0.00",
                "discounted_amount": "R 0.00",
                "discounted_amount_50": "R 0.00",
                "discount_percentage": "20"
            }
    
    @staticmethod
    def _extract_verification_info(client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract verification information."""
        try:
            profile = client_data.get("profile", {})
            client_info = profile.get("client_info", {}) if isinstance(profile, dict) else {}
            vehicles = profile.get("vehicles", []) if isinstance(profile, dict) else []
            
            if not isinstance(client_info, dict):
                client_info = {}
            if not isinstance(vehicles, list):
                vehicles = []
            
            # Available verification fields
            verification_fields = []
            field_values = {}
            
            # Check available verification data
            if client_info.get("id_number"):
                verification_fields.append("id_number")
                field_values["id_number"] = client_info["id_number"]
            
            if profile and profile.get("user_name"):
                verification_fields.append("username")
                field_values["username"] = profile["user_name"]
            
            if client_info.get("email_address"):
                verification_fields.append("email")
                field_values["email"] = client_info["email_address"]
            
            # Vehicle information
            if vehicles:
                vehicle = vehicles[0] if isinstance(vehicles[0], dict) else {}
                for field, key in [
                    ("vehicle_registration", "registration"),
                    ("vehicle_make", "make"),
                    ("vehicle_model", "model"),
                    ("vehicle_color", "color")
                ]:
                    if vehicle.get(key):
                        verification_fields.append(field)
                        field_values[field] = vehicle[key]
            
            # Map field names to human-readable format for scripts
            field_display_names = {
                "id_number": "ID number",
                "username": "username", 
                "email": "email address",
                "vehicle_registration": "vehicle registration number",
                "vehicle_make": "vehicle make",
                "vehicle_model": "vehicle model", 
                "vehicle_color": "vehicle color"
            }
            
            current_field = verification_fields[0] if verification_fields else "id_number"
            
            return {
                "available_verification_fields": verification_fields,
                "field_to_verify": field_display_names.get(current_field, current_field),
                **field_values
            }
        
        except Exception:
            return {
                "available_verification_fields": ["id_number"],
                "field_to_verify": "ID number"
            }
    
    @staticmethod
    def _get_formatted_script_guidance(
        script_type: str, 
        current_step: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get script content and behavioral guidance with formatted script content."""
        try:
            script_enum = ScriptType(script_type)
            step_enum = CallStep(current_step)
            guidance = ScriptManager.get_behavioral_guidance(script_enum, step_enum)
            script_instance = ScriptManager.get_script_class(script_enum)
            
            # Format main script content
            raw_script_content = guidance["script_content"]
            formatted_script_content = ScriptFormatter.format_script_content(
                raw_script_content, parameters, current_step
            )
            
            # Get additional scripts based on step
            additional_scripts = {}
            
            if current_step == "name_verification":
                script_text = script_instance.SCRIPTS.get("third_party_message", 
                    "{salutation}, kindly advise {client_title} {client_full_name} that Cartrack called regarding an outstanding account. Please ask them to contact us urgently at 011 250 3000.")
                additional_scripts["third_party_message"] = ScriptFormatter.format_script_content(
                    script_text, parameters, "third_party_message")
            
            elif current_step == "details_verification":
                script_text = script_instance.SCRIPTS.get("details_verification",
                    "This call is recorded for quality and security. To ensure I'm speaking with the right person, could you please confirm your {field_to_verify}?")
                additional_scripts["details_verification_script"] = ScriptFormatter.format_script_content(
                    script_text, parameters, "details_verification")
            
            elif current_step == "negotiation":
                scripts = {
                    "consequences_script": script_instance.SCRIPTS.get("negotiation_consequences",
                        "Without payment, your Cartrack app will be suspended, you'll lose vehicle positioning, and notifications will stop working."),
                    "benefits_script": script_instance.SCRIPTS.get("negotiation_benefits",
                        "Payment today restores all services immediately and keeps your account in good standing."),
                    "discount_offer_script": script_instance.SCRIPTS.get("discount_offer",
                        "To prevent further complications, we can offer a {discount_percentage}% discount if you pay {discounted_amount} by {campaign_end_date}."),
                    "legal_consequences_script": script_instance.SCRIPTS.get("legal_consequences",
                        "Continued non-payment may result in legal action and additional recovery costs.")
                }
                
                for key, script_text in scripts.items():
                    additional_scripts[key] = ScriptFormatter.format_script_content(
                        script_text, parameters, key.replace("_script", ""))
            
            return {
                "script_content": formatted_script_content,
                "script_text": formatted_script_content,
                "objection_responses": guidance["objection_responses"],
                "emotional_responses": guidance["emotional_responses"],
                "escalation_responses": guidance["escalation_responses"],
                **additional_scripts
            }
            
        except Exception as e:
            logger.warning(f"Error getting script guidance for {script_type}/{current_step}: {e}")
            return {
                "script_content": "",
                "script_text": "",
                "third_party_message": "",
                "details_verification_script": "",
                "consequences_script": "",
                "benefits_script": "",
                "discount_offer_script": "",
                "legal_consequences_script": "",
                "objection_responses": {},
                "emotional_responses": {},
                "escalation_responses": {}
            }
    
    @staticmethod
    def _build_state_info(state: Dict[str, Any], current_step: str) -> Dict[str, Any]:
        """Build state information for prompts."""
        return {
            "current_step": current_step,
            "name_verification_status": state.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
            "name_verification_attempts": state.get("name_verification_attempts", 1),
            "details_verification_status": state.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
            "details_verification_attempts": state.get("details_verification_attempts", 1),
            "matched_fields": state.get("matched_fields", []),
            "max_name_verification_attempts": 3,
            "max_details_verification_attempts": 5,
            "conversation_context": {
                "emotional_state": state.get("emotional_state", "neutral"),
                "objections_raised": state.get("objections_raised", []),
                "payment_willingness": state.get("payment_willingness", "unknown"),
                "rapport_level": state.get("rapport_level", "establishing")
            },
            # Add additional context for specific steps
            "department": state.get("department", "Supervisor"),
            "response_time": state.get("response_time", "24-48 hours"),
            "ticket_number": state.get("ticket_number", f"TKT{datetime.now().strftime('%Y%m%d%H%M')}"),
            "outcome_summary": state.get("outcome_summary", "We have discussed your account"),
            "payment_method": state.get("payment_method", "agreed payment method"),
            "campaign_end_date": state.get("campaign_end_date", "month-end"),
            "campaign_first_date": state.get("campaign_first_date", "15th of the month")
        }
    
    @staticmethod
    def _generate_tactical_guidance(
        behavioral_analysis: Dict[str, Any],
        script_type: str,
        current_step: str
    ) -> Dict[str, str]:
        """Generate tactical guidance based on analysis."""
        approach = behavioral_analysis.get("optimal_approach", "professional_persistent")
        risk_level = behavioral_analysis.get("risk_level", "medium")
        likely_objections = behavioral_analysis.get("likely_objections", [])
        days_overdue = behavioral_analysis.get("days_overdue", 0)
        
        guidance = {
            "recommended_approach": approach,
            "urgency_level": ParameterBuilder._determine_urgency_level(risk_level, script_type, days_overdue),
            "key_motivators": ParameterBuilder._get_key_motivators(script_type, days_overdue),
            "objection_predictions": ", ".join(likely_objections[:3]),
            "success_probability": behavioral_analysis.get("success_probability", "medium"),
            "backup_strategies": ParameterBuilder._get_backup_strategies(script_type, current_step)
        }
        
        return guidance
    
    @staticmethod
    def _determine_urgency_level(risk_level: str, script_type: str, days_overdue: int) -> str:
        """Determine urgency level for communication."""
        if "pre_legal" in script_type or "150" in script_type or days_overdue >= 150:
            return "critical"
        elif risk_level == "high" or "2_3" in script_type or days_overdue >= 60:
            return "high"
        elif risk_level == "medium" or days_overdue >= 30:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def _get_key_motivators(script_type: str, days_overdue: int) -> str:
        """Get key motivators based on script type and days overdue."""
        if "pre_legal" in script_type or "150" in script_type or days_overdue >= 120:
            return "Avoid legal action, protect credit profile"
        elif "2_3" in script_type or days_overdue >= 60:
            return "Restore services, avoid recovery fees"
        else:
            return "Maintain vehicle security, keep account current"
    
    @staticmethod
    def _get_backup_strategies(script_type: str, current_step: str) -> str:
        """Get backup strategies for different scenarios."""
        strategies = []
        
        if current_step in ["promise_to_pay", "negotiation"]:
            strategies.extend(["Payment portal if debit refused", "Partial payment arrangement"])
        
        if "pre_legal" in script_type:
            strategies.append("Settlement discount offer")
        
        if not strategies:
            strategies.append("Escalation to supervisor")
        
        return ", ".join(strategies)
    
    @staticmethod
    def _flatten_nested_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested dictionary parameters for easier template access."""
        flattened = {}
        
        # First, copy all simple parameters
        for key, value in parameters.items():
            if not isinstance(value, dict):
                flattened[key] = str(value) if value is not None else ""
        
        # Handle specific nested dictionaries
        nested_dicts = ["tactical_guidance", "behavioral_analysis", "conversation_context"]
        
        for dict_name in nested_dicts:
            source_dict = parameters.get(dict_name)
            if isinstance(source_dict, dict):
                for sub_key, sub_value in source_dict.items():
                    flattened[f"{dict_name}[{sub_key}]"] = str(sub_value) if sub_value is not None else ""
            else:
                # Create safe defaults
                defaults = {
                    "tactical_guidance": ["recommended_approach", "urgency_level", "key_motivators", 
                                        "objection_predictions", "success_probability", "backup_strategies"],
                    "behavioral_analysis": ["risk_level", "days_overdue", "payment_reliability", 
                                        "likely_objections", "optimal_approach", "success_probability"],
                    "conversation_context": ["emotional_state", "objections_raised", 
                                        "payment_willingness", "rapport_level"]
                }
                
                for sub_key in defaults.get(dict_name, []):
                    flattened[f"{dict_name}[{sub_key}]"] = ""
        
        # Convert response dictionaries to formatted strings
        response_dicts = ["objection_responses", "emotional_responses", "escalation_responses"]
        for dict_name in response_dicts:
            dict_value = parameters.get(dict_name)
            if isinstance(dict_value, dict):
                formatted_str = "\n".join([f"- {k}: {v}" for k, v in dict_value.items() if v])
                flattened[dict_name] = formatted_str
            else:
                flattened[dict_name] = str(dict_value) if dict_value else ""
        
        # Ensure all values are strings
        for key, value in flattened.items():
            if value is None:
                flattened[key] = ""
            elif not isinstance(value, str):
                flattened[key] = str(value)
        
        return flattened

########################################################################################
# Convenience functions for easy integration
def get_client_data(user_id: str, force_reload: bool = False) -> Dict[str, Any]:
    """Get client data with caching."""
    return ClientDataBuilder.get_client_data(user_id, force_reload)

def prepare_parameters(
    client_data: Dict[str, Any],
    current_step: str,
    state: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent"
) -> Dict[str, Any]:
    """Prepare complete parameters for prompt generation using client data."""
    return ParameterBuilder.build_parameters(client_data, current_step, state, script_type, agent_name)

def prepare_parameters_by_user_id(
    user_id: str,
    current_step: str,
    state: Dict[str, Any],
    script_type: str = "ratio_1_inflow",
    agent_name: str = "AI Agent"
) -> Dict[str, Any]:
    """Prepare complete parameters for prompt generation by fetching client data with user_id."""
    client_data = ClientDataBuilder.get_client_data(user_id)
    return ParameterBuilder.build_parameters(client_data, current_step, state, script_type, agent_name)

def analyze_client_behavior(client_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze client behavior patterns."""
    return BehavioralAnalyzer.analyze_client_profile(client_data)

def clear_client_cache(user_id: Optional[str] = None):
    """Clear cached client data."""
    ClientDataBuilder.clear_cache(user_id)





########################################################################################
# State management helper
class ConversationState:
    """Manages conversation state throughout the call."""
    
    def __init__(self):
        self.emotional_state = "neutral"
        self.objections_raised = []
        self.payment_willingness = "unknown"
        self.rapport_level = "establishing"
        self.name_verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        self.details_verification_status = VerificationStatus.INSUFFICIENT_INFO.value
        self.name_verification_attempts = 0
        self.details_verification_attempts = 0
        self.matched_fields = []
    
    def update_emotional_state(self, new_state: str):
        """Update client emotional state."""
        valid_states = ["neutral", "defensive", "cooperative", "aggressive", "confused", "emotional"]
        if new_state in valid_states:
            self.emotional_state = new_state
            logger.info(f"Emotional state updated to: {new_state}")
    
    def add_objection(self, objection: str):
        """Add client objection to tracking."""
        if objection not in self.objections_raised:
            self.objections_raised.append(objection)
            logger.info(f"Objection added: {objection}")
    
    def update_payment_willingness(self, willingness: str):
        """Update payment willingness assessment."""
        valid_levels = ["high", "medium", "low", "resistant", "unknown"]
        if willingness in valid_levels:
            self.payment_willingness = willingness
            logger.info(f"Payment willingness updated to: {willingness}")
    
    def increment_verification_attempt(self, verification_type: str):
        """Increment verification attempt counter."""
        if verification_type == "name":
            self.name_verification_attempts += 1
        elif verification_type == "details":
            self.details_verification_attempts += 1
    
    def add_matched_field(self, field: str):
        """Add successfully verified field."""
        if field not in self.matched_fields:
            self.matched_fields.append(field)
            logger.info(f"Verification field matched: {field}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "emotional_state": self.emotional_state,
            "objections_raised": self.objections_raised,
            "payment_willingness": self.payment_willingness,
            "rapport_level": self.rapport_level,
            "name_verification_status": self.name_verification_status,
            "details_verification_status": self.details_verification_status,
            "name_verification_attempts": self.name_verification_attempts,
            "details_verification_attempts": self.details_verification_attempts,
            "matched_fields": self.matched_fields
        }
    
    def from_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


########################################################################################
# ./src/Agents/call_center_agent/data_parameter_builder.py
"""
Complete Enhanced Data Parameter Builder with Conversation Intelligence and Outstanding Amount Calculation.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import logging
from typing import Dict, Any, Optional, List, Callable, Coroutine
from datetime import datetime, timedelta
from enum import Enum
import re
from app_config import CONFIG

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
from src.Agents.call_center_agent.prompts import *

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
# Outstanding Amount Calculation - CRITICAL COMPONENT
########################################################################################

def calculate_outstanding_amount(account_aging_data: Dict[str, Any]) -> float:
    """
    Calculate outstanding amount = total balance - current (non-overdue) balance.
    
    This is the OVERDUE amount the client actually needs to pay, not the total balance.
    """
    try:
        # Extract aging buckets
        x0 = float(account_aging_data.get("x0", 0))        # Current (0 days)
        x30 = float(account_aging_data.get("x30", 0))      # 1-30 days overdue
        x60 = float(account_aging_data.get("x60", 0))      # 31-60 days overdue
        x90 = float(account_aging_data.get("x90", 0))      # 61-90 days overdue
        x120 = float(account_aging_data.get("x120", 0))    # 91+ days overdue
        xbalance = float(account_aging_data.get("xbalance", 0))  # Total balance
        
        # Outstanding = Total - Current (non-overdue)
        outstanding_amount = xbalance - x0
        
        # # Log calculation for debugging
        # logger.info(f"Outstanding calculation:")
        # logger.info(f"  Total balance (xbalance): R {xbalance:.2f}")
        # logger.info(f"  Current balance (x0): R {x0:.2f}")
        # logger.info(f"  Outstanding amount: R {outstanding_amount:.2f}")
        # logger.info(f"  Breakdown - 30d: R{x30:.2f}, 60d: R{x60:.2f}, 90d: R{x90:.2f}, 120d+: R{x120:.2f}")
        
        return max(outstanding_amount, 0.0)  # Never negative
        
    except (ValueError, TypeError) as e:
        # logger.error(f"Outstanding calculation error: {e}")
        return 0.0

def format_outstanding_amount(client_data: Dict[str, Any]) -> str:
    """Format outstanding amount for use in prompts."""
    
    user_id = get_safe_value(client_data, "profile.user_id", "")     account_aging = client_data.get("account_aging", {})
    outstanding_float = calculate_outstanding_amount(account_aging)
    
    if outstanding_float <= 0:
        logger.warning("No outstanding amount - account may be current")
        return "R 0.00"
    
    return f"R {outstanding_float:.2f}"

def get_safe_value(data: Dict[str, Any], path: str, default: Any = "") -> Any:
    """Safely extract nested values with dot notation."""
    try:
        keys = path.split('.')
        value = data
        for key in keys:
            value = value[key] if isinstance(value, dict) else getattr(value, key, None)
        return value if value is not None else default
    except (KeyError, TypeError, AttributeError):
        return default

def format_currency(amount: Any) -> str:
    """Format currency with validation."""
    try:
        return f"R {float(amount):.2f}" if amount else "R 0.00"
    except (ValueError, TypeError):
        return "R 0.00"

########################################################################################
# Conversation Analysis for Emotional Intelligence
########################################################################################

class ConversationAnalyzer:
    """Analyzes conversation for emotional states, objections, and payment indicators."""
    
    @staticmethod
    def get_message_content(msg: Any) -> str:
        """Safely extract message content from various message formats."""
        if hasattr(msg, 'content'):
            return str(msg.content)
        elif isinstance(msg, dict):
            return str(msg.get('content', ''))
        else:
            return str(msg)
    
    @staticmethod
    def detect_emotional_state(messages: List[Any]) -> str:
        """Detect client emotional state from conversation."""
        emotional_indicators = {
            "frustrated": ["ridiculous", "stupid", "waste of time", "fed up", "annoying", "irritating"],
            "angry": ["angry", "pissed off", "furious", "sick of this", "damn", "hell"],
            "worried": ["worried", "stressed", "don't know what", "struggling", "scared", "nervous"],
            "embarrassed": ["sorry", "embarrassed", "didn't realize", "my fault", "apologize"],
            "defensive": ["not my fault", "why should I", "leave me alone", "stop calling"],
            "cooperative": ["understand", "help me", "what can we do", "let's sort this", "okay"]
        }
        
        # Check last 3 client messages
        recent_messages = messages[-6:] if messages else []
        client_messages = []
        
        for msg in recent_messages:
            content = ConversationAnalyzer.get_message_content(msg)
            # Check if it's a client message (not agent)
            if hasattr(msg, 'type') and msg.type == "human":
                client_messages.append(content.lower())
            elif isinstance(msg, dict) and msg.get("role") in ["user", "human"]:
                client_messages.append(content.lower())
        
        # Analyze emotional indicators in client messages
        for emotion, keywords in emotional_indicators.items():
            for message_content in client_messages:
                if any(keyword in message_content for keyword in keywords):
                    return emotion
        
        return "neutral"
    
    @staticmethod
    def detect_real_objections(messages: List[Any]) -> List[str]:
        """Detect actual objections from conversation."""
        objection_patterns = {
            "no_money": ["no money", "can't afford", "broke", "tight", "don't have", "financial problems"],
            "dispute_amount": ["wrong", "incorrect", "not right", "don't owe", "dispute", "too much"],
            "already_paid": ["already paid", "paid already", "made payment", "paid last week"],
            "will_pay_later": ["later", "next week", "when I get paid", "payday", "end of month"],
            "bank_problems": ["bank declined", "card blocked", "no funds", "bank issues"],
            "not_my_responsibility": ["not my fault", "not my debt", "someone else's"],
            "need_time": ["need time", "let me think", "speak to wife", "check finances"]
        }
        
        detected = []
        # Check last 5 client messages
        recent_messages = messages[-10:] if messages else []
        
        for msg in recent_messages:
            content = ConversationAnalyzer.get_message_content(msg).lower()
            # Only check client messages
            is_client = (hasattr(msg, 'type') and msg.type == "human") or \
                       (isinstance(msg, dict) and msg.get("role") in ["user", "human"])
            
            if is_client:
                for objection, keywords in objection_patterns.items():
                    if any(keyword in content for keyword in keywords):
                        detected.append(objection)
        
        return list(set(detected))  # Remove duplicates
    
    @staticmethod
    def analyze_payment_conversation(messages: List[Any], outstanding_amount: float) -> Dict[str, Any]:
        """Analyze conversation for payment-related information."""
        
        payment_commitment = "unknown"
        mentioned_amount = None
        payment_timeframe = None
        payment_method_preference = None
        
        # Check last 5 messages for payment indicators
        recent_messages = messages[-10:] if messages else []
        
        for msg in recent_messages:
            content = ConversationAnalyzer.get_message_content(msg).lower()
            
            # Only analyze client messages
            is_client = (hasattr(msg, 'type') and msg.type == "human") or \
                       (isinstance(msg, dict) and msg.get("role") in ["user", "human"])
            
            if is_client:
                # Detect payment willingness
                willing_phrases = ["can pay", "will pay", "able to pay", "yes", "okay", "fine"]
                unwilling_phrases = ["can't pay", "cannot pay", "no money", "refuse", "won't pay"]
                
                if any(phrase in content for phrase in willing_phrases):
                    payment_commitment = "willing"
                elif any(phrase in content for phrase in unwilling_phrases):
                    payment_commitment = "unwilling"
                
                # Extract mentioned amounts using regex
                amount_patterns = [
                    r'r\s*(\d+)',  # R100
                    r'(\d+)\s*rand',  # 100 rand
                    r'(\d+)\s*r',  # 100R
                    r'(\d{2,4})'  # Just numbers 100-9999
                ]
                
                for pattern in amount_patterns:
                    match = re.search(pattern, content)
                    if match:
                        try:
                            amount = float(match.group(1))
                            if 50 <= amount <= outstanding_amount * 2:  # Reasonable range
                                mentioned_amount = amount
                                break
                        except:
                            continue
                
                # Extract timeframes
                timeframe_indicators = {
                    "immediate": ["today", "now", "immediately", "right now"],
                    "this_week": ["tomorrow", "friday", "monday", "tuesday", "wednesday", "thursday"],
                    "next_week": ["next week", "next friday", "next monday"],
                    "month_end": ["end of month", "month end", "30th", "31st"],
                    "payday": ["payday", "when I get paid", "salary day"]
                }
                
                for timeframe, keywords in timeframe_indicators.items():
                    if any(keyword in content for keyword in keywords):
                        payment_timeframe = timeframe
                        break
                
                # Detect payment method preferences
                if any(word in content for word in ["debit", "bank", "account"]):
                    payment_method_preference = "debicheck"
                elif any(word in content for word in ["online", "portal", "link", "internet"]):
                    payment_method_preference = "payment_portal"
                elif any(word in content for word in ["card", "credit", "visa", "mastercard"]):
                    payment_method_preference = "card"
        
        return {
            "payment_commitment": payment_commitment,
            "mentioned_amount": mentioned_amount,
            "payment_timeframe": payment_timeframe,
            "payment_method_preference": payment_method_preference,
            "negotiation_position": "flexible" if mentioned_amount and mentioned_amount < outstanding_amount else "standard"
        }

########################################################################################
# Client Data Management
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
            
            # Load account overview and financial data
            account_overview = get_client_account_overview.invoke(user_id)
            account_aging = get_client_account_aging.invoke(user_id)
            banking_details = get_client_banking_details.invoke(user_id)
            
            # Load subscription and payment data
            subscription_data = get_client_subscription_amount.invoke(user_id)
            payment_history = get_client_payment_history.invoke(user_id)
            failed_payments = get_client_failed_payments.invoke(user_id)
            last_payment = get_client_last_successful_payment.invoke(user_id)
            
            # Load contracts and billing analysis
            contracts = get_client_contracts.invoke(user_id)
            billing_analysis = get_client_billing_analysis.invoke(user_id)
            
            # Load existing mandates
            existing_mandates = get_client_debit_mandates.invoke(user_id)

            # Consolidate all data
            client_data = {
                "user_id": user_id,
                "profile": profile,
                "account_overview": account_overview,
                "account_aging": account_aging[0] if account_aging else {},
                "banking_details": banking_details[0] if banking_details else {},
                "subscription": subscription_data,
                "payment_history": payment_history[:5] if payment_history else [],
                "failed_payments": failed_payments[:3] if failed_payments else [],
                "last_successful_payment": last_payment,
                "contracts": contracts,
                "billing_analysis": billing_analysis[0] if billing_analysis else {},
                "existing_mandates": existing_mandates,
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
            "account_aging": {"xbalance": "0.00", "x0": "0.00"},
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
# Payment Flexibility Analysis
########################################################################################

class PaymentFlexibilityAnalyzer:
    """Analyzes client payment capacity with conversation intelligence."""
    
    @staticmethod
    def assess_payment_capacity(client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess client's payment capacity and flexibility options."""
        
        # Extract financial indicators
        user_id = get_safe_value(client_data, "profile.user_id", "")     account_aging = client_data.get("account_aging", {})
        payment_history = client_data.get("payment_history", [])
        failed_payments = client_data.get("failed_payments", [])
        
        outstanding_amount = calculate_outstanding_amount(account_aging)
        
        # Calculate payment reliability
        total_attempts = len(payment_history)
        failed_attempts = len(failed_payments)
        success_rate = (total_attempts - failed_attempts) / total_attempts if total_attempts > 0 else 0
        
        # Assess capacity level
        capacity_level = "unknown"
        if outstanding_amount <= 200:
            capacity_level = "high"
        elif outstanding_amount <= 500:
            capacity_level = "medium" if success_rate >= 0.7 else "low"
        elif outstanding_amount <= 1000:
            capacity_level = "low" if success_rate >= 0.5 else "hardship"
        else:
            capacity_level = "hardship"
        
        # Calculate payment options
        payment_options = PaymentFlexibilityAnalyzer._calculate_payment_options(
            outstanding_amount, capacity_level, success_rate
        )
        
        return {
            "capacity_level": capacity_level,
            "payment_options": payment_options,
            "minimum_payment": payment_options[0]["amount"] if payment_options else outstanding_amount * 0.3,
            "payment_plan_eligible": outstanding_amount > 300 and success_rate >= 0.3,
            "hardship_indicators": PaymentFlexibilityAnalyzer._detect_hardship_indicators(
                failed_payments, payment_history
            )
        }
    
    @staticmethod
    def _calculate_payment_options(amount: float, capacity: str, success_rate: float) -> List[Dict[str, Any]]:
        """Calculate available payment options based on capacity."""
        options = []
        
        # Full payment (always offered first)
        options.append({
            "type": "full_payment",
            "amount": amount,
            "description": f"Full payment of R {amount:.2f}",
            "priority": 1
        })
        
        # Partial payments based on capacity
        if capacity in ["medium", "low", "hardship"]:
            # 80% option
            options.append({
                "type": "partial_80",
                "amount": amount * 0.8,
                "description": f"80% payment of R {amount * 0.8:.2f}",
                "priority": 2
            })
            
            # 50% option for low capacity
            if capacity in ["low", "hardship"]:
                options.append({
                    "type": "partial_50",
                    "amount": amount * 0.5,
                    "description": f"50% payment of R {amount * 0.5:.2f}",
                    "priority": 3
                })
                
                # Payment plan for hardship
                if capacity == "hardship" and amount > 300:
                    monthly_amount = amount / 3
                    options.append({
                        "type": "payment_plan",
                        "amount": monthly_amount,
                        "description": f"3-month plan: R {monthly_amount:.2f} monthly",
                        "priority": 4
                    })
        
        return options
    
    @staticmethod
    def _detect_hardship_indicators(failed_payments: List, payment_history: List) -> List[str]:
        """Detect indicators of financial hardship."""
        indicators = []
        
        if len(failed_payments) >= 3:
            indicators.append("multiple_failures")
        
        if len(payment_history) == 0:
            indicators.append("no_payment_history")
        
        return indicators

########################################################################################
# Behavioral Analysis with Conversation Intelligence
########################################################################################

class BehavioralAnalyzer:
    """Analyzes client behavior with conversation intelligence."""
    
    @staticmethod
    def analyze_client_profile(client_data: Dict[str, Any], conversation_messages: List[Any] = None) -> Dict[str, Any]:
        """Enhanced analysis including conversation intelligence."""
        try:
            # Traditional data analysis
            payment_history = client_data.get("payment_history", [])
            user_id = get_safe_value(client_data, "profile.user_id", "")     account_aging = client_data.get("account_aging", {})
            billing_analysis = client_data.get("billing_analysis", {})
            failed_payments = client_data.get("failed_payments", [])
            
            # Ensure dictionaries
            if not isinstance(account_aging, dict):
                account_aging = {}
            if not isinstance(billing_analysis, dict):
                billing_analysis = {}
            
            # Get outstanding amount
            outstanding_amount = calculate_outstanding_amount(account_aging)
            
            # Calculate traditional metrics
            days_overdue = BehavioralAnalyzer._get_days_overdue(billing_analysis, account_aging)
            reliability = BehavioralAnalyzer._assess_payment_reliability(payment_history, failed_payments)
            risk_level = BehavioralAnalyzer._assess_risk_level(days_overdue, outstanding_amount)
            
            # Conversation intelligence
            conversation_analysis = {}
            if conversation_messages:
                conversation_analysis = {
                    "emotional_state": ConversationAnalyzer.detect_emotional_state(conversation_messages),
                    "real_objections": ConversationAnalyzer.detect_real_objections(conversation_messages),
                    "payment_conversation": ConversationAnalyzer.analyze_payment_conversation(conversation_messages, outstanding_amount)
                }
            
            # Predict objections (traditional + conversation)
            objections = BehavioralAnalyzer._predict_objections(client_data, outstanding_amount, days_overdue)
            if conversation_analysis.get("real_objections"):
                # Prioritize real objections from conversation
                objections = conversation_analysis["real_objections"] + [obj for obj in objections if obj not in conversation_analysis["real_objections"]]
            
            # Determine approach with conversation context
            approach = BehavioralAnalyzer._determine_approach(days_overdue, reliability, outstanding_amount, conversation_analysis)
            
            return {
                "days_overdue": days_overdue,
                "payment_reliability": reliability,
                "likely_objections": objections[:5],  # Top 5
                "optimal_approach": approach,
                "risk_level": risk_level,
                "success_probability": BehavioralAnalyzer._calculate_success_probability(reliability, outstanding_amount, days_overdue),
                "outstanding_amount": outstanding_amount,
                # Conversation intelligence
                "conversation_intelligence": conversation_analysis
            }
        
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {e}")
            return {
                "days_overdue": 0,
                "payment_reliability": "unknown",
                "likely_objections": ["will_pay_later"],
                "optimal_approach": "professional_persistent",
                "risk_level": "medium",
                "success_probability": "medium",
                "outstanding_amount": 0.0,
                "conversation_intelligence": {}
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
        
        success_rate = (total_attempts - failed_attempts) / total_attempts if total_attempts > 0 else 0
        
        if success_rate >= 0.8:
            return "high"
        elif success_rate >= 0.5:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def _predict_objections(client_data: Dict[str, Any], outstanding_amount: float, days_overdue: int) -> List[str]:
        """Predict likely client objections based on real data."""
        objections = []
        account_overview = client_data.get("account_overview", {})
        failed_payments = client_data.get("failed_payments", [])
        
        # High balance objections
        if outstanding_amount > 500:
            objections.extend(["dispute_amount", "no_money"])
        
        # Recent payment activity
        if outstanding_amount < 200:
            objections.append("already_paid")
        
        # Failed payment patterns
        if failed_payments:
            objections.append("bank_problems")
        
        # Long overdue accounts
        if days_overdue > 90:
            objections.extend(["not_my_responsibility", "will_pay_later"])
        
        # Account status specific
        if isinstance(account_overview, dict) and account_overview.get("account_status") == "Overdue":
            objections.append("will_pay_later")
        
        # Always include the most common objection
        if "will_pay_later" not in objections:
            objections.append("will_pay_later")
        
        return objections[:5]
    
    @staticmethod
    def _determine_approach(days_overdue: int, reliability: str, outstanding_amount: float, conversation_analysis: Dict[str, Any]) -> str:
        """Determine optimal approach with conversation context."""
        
        # Check conversation intelligence first
        if conversation_analysis:
            emotional_state = conversation_analysis.get("emotional_state", "neutral")
            payment_info = conversation_analysis.get("payment_conversation", {})
            
            # Adjust approach based on emotional state
            if emotional_state in ["angry", "frustrated"]:
                return "calm_de_escalation"
            elif emotional_state == "worried":
                return "reassuring_solution_focused"
            elif emotional_state == "cooperative":
                return "collaborative_direct"
            elif payment_info.get("payment_commitment") == "willing":
                return "immediate_closure"
        
        # Fallback to traditional approach
        if days_overdue <= 30 and reliability == "high":
            return "friendly_reminder"
        elif days_overdue <= 60:
            return "consequence_focused"
        elif days_overdue <= 120:
            return "urgent_resolution"
        else:
            return "legal_prevention"
    
    @staticmethod
    def _assess_risk_level(days_overdue: int, outstanding_amount: float) -> str:
        """Assess overall risk level."""
        if days_overdue >= 120 or outstanding_amount > 1000:
            return "high"
        elif days_overdue >= 60 or outstanding_amount > 500:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def _calculate_success_probability(reliability: str, outstanding_amount: float, days_overdue: int) -> str:
        """Calculate probability of successful resolution."""
        score = 0
        
        # Reliability factor
        reliability_scores = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
        score += reliability_scores.get(reliability, 0)
        
        # Balance factor
        if outstanding_amount < 300:
            score += 2
        elif outstanding_amount < 700:
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
# Script Content Formatting
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
        
        # Convert any None values to empty strings
        for key, value in format_params.items():
            if value is None:
                format_params[key] = ""
            elif not isinstance(value, str):
                format_params[key] = str(value)
        
        return format_params

########################################################################################
# Complete Parameter Builder
########################################################################################

class ParameterBuilder:
    """Builds complete parameters with conversation intelligence and outstanding amount calculation."""
    
    @staticmethod
    def build_parameters(
        client_data: Dict[str, Any],
        current_step: str,
        state: Dict[str, Any],
        script_type: str = "ratio_1_inflow",
        agent_name: str = "AI Agent"
    ) -> Dict[str, Any]:
        """Build complete parameters with conversation intelligence and proper outstanding calculation."""
        
        try:
            # Extract conversation messages for analysis
            conversation_messages = state.get("messages", [])
            
            # Extract all information with outstanding amount calculation
            basic_info = ParameterBuilder._extract_basic_info(client_data, agent_name)
            financial_info = ParameterBuilder._extract_financial_info(client_data)
            verification_info = ParameterBuilder._extract_verification_info(client_data)
            
            # Enhanced behavioral analysis with conversation intelligence
            behavioral_analysis = BehavioralAnalyzer.analyze_client_profile(client_data, conversation_messages)
            
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
            
            # Log detailed parameter summary
            user_id = client_data.get("user_id", "unknown")
            outstanding_amount = behavioral_analysis.get("outstanding_amount", 0.0)
            
            logger.info("=" * 80)
            logger.info(f"PARAMETERS BUILT FOR USER: {user_id}, STEP: {current_step}")
            logger.info("=" * 80)
            logger.info(f"Outstanding Amount: R {outstanding_amount:.2f}")
            logger.info(f"Behavioral Analysis: {behavioral_analysis.get('optimal_approach', 'unknown')}")
            logger.info(f"Risk Level: {behavioral_analysis.get('risk_level', 'unknown')}")
            logger.info(f"Payment Reliability: {behavioral_analysis.get('payment_reliability', 'unknown')}")
            
            # # Log aging breakdown for transparency
            # user_id = get_safe_value(client_data, "profile.user_id", "")     account_aging = client_data.get("account_aging", {})
            # if account_aging:
            #     logger.info("Account aging breakdown:")
            #     logger.info(f"  Current (x0): R {account_aging.get('x0', 0)}")
            #     logger.info(f"  30 days (x30): R {account_aging.get('x30', 0)}")
            #     logger.info(f"  60 days (x60): R {account_aging.get('x60', 0)}")
            #     logger.info(f"  90 days (x90): R {account_aging.get('x90', 0)}")
            #     logger.info(f"  120+ days (x120): R {account_aging.get('x120', 0)}")
            #     logger.info(f"  Total balance: R {account_aging.get('xbalance', 0)}")
            
            logger.info("=" * 80)
            
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
        """Extract financial information with proper outstanding amount calculation."""
        try:
            user_id = get_safe_value(client_data, "profile.user_id", "")     account_aging = client_data.get("account_aging", {})
            subscription = client_data.get("subscription", {})
            
            if not isinstance(account_aging, dict):
                account_aging = {}
            if not isinstance(subscription, dict):
                subscription = {}
            
            # Calculate outstanding amount properly (overdue amount, not total balance)
            outstanding_float = calculate_outstanding_amount(account_aging)
            outstanding_amount = format_currency(outstanding_float)
            
            # Get subscription amount
            subscription_amount = subscription.get("subscription_amount", "199.00")
            subscription_str = format_currency(subscription_amount)
            
            return {
                "outstanding_amount": outstanding_amount,
                "subscription_amount": subscription_str,
                "subscription_date": "5th of each month",
                "amount_with_fee": format_currency(outstanding_float + 10),
                "ptp_amount_plus_fee": format_currency(outstanding_float + 10),
                "cancellation_fee": "R 0.00",
                "total_balance": format_currency(account_aging.get("xbalance", 0)),
                "discounted_amount": format_currency(outstanding_float * 0.8),
                "discounted_amount_50": format_currency(outstanding_float * 0.5),
                "discount_percentage": "20"
            }
        
        except Exception as e:
            logger.error(f"Error extracting financial info: {e}")
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
                "passport_number": "passport number",
                "username": "username", 
                "email": "email address",
                "vehicle_registration": "vehicle registration",
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
            
            # Format main script content
            raw_script_content = guidance["script_content"]
            formatted_script_content = ScriptFormatter.format_script_content(
                raw_script_content, parameters, current_step
            )
            
            return {
                "script_content": formatted_script_content,
                "script_text": formatted_script_content,
                "objection_responses": guidance["objection_responses"],
                "emotional_responses": guidance["emotional_responses"],
                "escalation_responses": guidance["escalation_responses"]
            }
            
        except Exception as e:
            logger.warning(f"Error getting script guidance for {script_type}/{current_step}: {e}")
            return {
                "script_content": "",
                "script_text": "",
                "objection_responses": {},
                "emotional_responses": {},
                "escalation_responses": {}
            }
    
    @staticmethod
    def _build_state_info(state: Dict[str, Any], current_step: str) -> Dict[str, Any]:
        """Build state information for prompts."""
        previous_step = state.get("previous_step", "")
        bridge_phrase = ""
        
        if previous_step and current_step:
            bridge_key = f"{previous_step}_to_{current_step}"
            # You would implement CONVERSATION_BRIDGES if needed
            # if bridge_key in CONVERSATION_BRIDGES.get("step_transitions", {}):
            #     bridge_phrase = CONVERSATION_BRIDGES["step_transitions"][bridge_key]
            if current_step == "reason_for_call" and previous_step == "details_verification":
                bridge_phrase = "Thank you for confirming those details."
            elif current_step == "negotiation":
                bridge_phrase = "Now, regarding this payment..."
        
        return {
            "current_step": current_step,
            "name_verification_status": state.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
            "name_verification_attempts": state.get("name_verification_attempts", 1),
            "details_verification_status": state.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value),
            "details_verification_attempts": state.get("details_verification_attempts", 1),
            "matched_fields": state.get("matched_fields", []),
            "max_name_verification_attempts": CONFIG.get('verification',{}).get('max_name_verification_attempts',4),
            "max_details_verification_attempts": CONFIG.get('verification',{}).get('max_details_verification_attempts',5),
            "conversation_context": {
                "emotional_state": state.get("emotional_state", "neutral"),
                "objections_raised": state.get("objections_raised", []),
                "payment_willingness": state.get("payment_willingness", "unknown"),
                "rapport_level": state.get("rapport_level", "establishing")
            },
            "previous_step": previous_step,
            "bridge_phrase": bridge_phrase,
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
        
        # Check conversation intelligence for enhanced guidance
        conversation_intel = behavioral_analysis.get("conversation_intelligence", {})
        emotional_state = conversation_intel.get("emotional_state", "neutral")
        payment_info = conversation_intel.get("payment_conversation", {})
        
        guidance = {
            "recommended_approach": approach,
            "urgency_level": ParameterBuilder._determine_urgency_level(risk_level, script_type, days_overdue),
            "key_motivators": ParameterBuilder._get_key_motivators(script_type, days_overdue),
            "objection_predictions": ", ".join(likely_objections[:3]),
            "success_probability": behavioral_analysis.get("success_probability", "medium"),
            "backup_strategies": ParameterBuilder._get_backup_strategies(script_type, current_step),
            # Conversation-aware guidance
            "emotional_awareness": emotional_state,
            "payment_willingness": payment_info.get("payment_commitment", "unknown"),
            "conversation_strategy": ParameterBuilder._get_conversation_strategy(emotional_state, payment_info)
        }
        
        return guidance
    
    @staticmethod
    def _get_conversation_strategy(emotional_state: str, payment_info: Dict[str, Any]) -> str:
        """Get conversation strategy based on emotional state and payment info."""
        if emotional_state == "angry":
            return "De-escalate first, then focus on solutions"
        elif emotional_state == "worried":
            return "Reassure and provide simple options"
        elif emotional_state == "cooperative":
            return "Direct and efficient closure"
        elif payment_info.get("payment_commitment") == "willing":
            return "Immediate payment processing"
        elif payment_info.get("payment_commitment") == "unwilling":
            return "Flexible options and empathy"
        else:
            return "Standard professional approach"
    
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
                    # Create both bracketed and direct access formats
                    flattened[f"{dict_name}[{sub_key}]"] = str(sub_value) if sub_value is not None else ""
                    # CRITICAL: Also add direct access for common fields
                    if dict_name == "tactical_guidance" and sub_key in ["urgency_level", "key_motivators"]:
                        flattened[sub_key] = str(sub_value) if sub_value is not None else ""
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
                    # Also add direct access for tactical guidance
                    if dict_name == "tactical_guidance" and sub_key in ["urgency_level", "key_motivators"]:
                        flattened[sub_key] = ""
        
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
# Async Client Data Builder
########################################################################################

class AsyncClientDataBuilder:
    """Async version with all database calls enabled."""
    
    _cache = {}
    _cache_duration = timedelta(hours=1)
    _executor = ThreadPoolExecutor(max_workers=10)
    
    @classmethod
    async def get_client_data(cls, user_id: str, force_reload: bool = False) -> Dict[str, Any]:
        """Get client data with caching - async version."""
        cache_key = user_id
        now = datetime.now()
        
        # Check cache
        if not force_reload and cache_key in cls._cache:
            cached_entry = cls._cache[cache_key]
            if now - cached_entry["timestamp"] < cls._cache_duration:
                logger.info(f"Using cached data for user_id: {user_id}")
                return cached_entry["data"]
        
        # Fetch fresh data asynchronously
        logger.info(f"Fetching fresh data asynchronously for user_id: {user_id}")
        try:
            data = await cls._fetch_client_data_async(user_id)
            cls._cache[cache_key] = {"data": data, "timestamp": now}
            return data
        except Exception as e:
            logger.error(f"Error fetching client data for {user_id}: {e}")
            return cls._get_fallback_data(user_id)
    
    @classmethod
    async def _fetch_client_data_async(cls, user_id: str) -> Dict[str, Any]:
        """Fetch data from database using async calls with ALL tools enabled."""
        try:
            # Convert sync tool calls to async using thread pool
            async def call_tool_async(tool_func: Callable, *args, **kwargs):
                """Convert synchronous tool call to async."""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    cls._executor, 
                    functools.partial(tool_func.invoke, *args, **kwargs)
                )
            
            # Create all async tasks
            tasks = {
                "profile": call_tool_async(get_client_profile, user_id),
                "account_overview": call_tool_async(get_client_account_overview, user_id),
                "account_aging": call_tool_async(get_client_account_aging, user_id),
                "banking_details": call_tool_async(get_client_banking_details, user_id),
                "subscription_data": call_tool_async(get_client_subscription_amount, user_id),
                "payment_history": call_tool_async(get_client_payment_history, user_id),
                "failed_payments": call_tool_async(get_client_failed_payments, user_id),
                "last_payment": call_tool_async(get_client_last_successful_payment, user_id),
                "contracts": call_tool_async(get_client_contracts, user_id),
                "billing_analysis": call_tool_async(get_client_billing_analysis, user_id),
                "existing_mandates": call_tool_async(get_client_debit_mandates, user_id),
            }
            
            # Execute all tasks concurrently
            logger.info(f"Executing {len(tasks)} concurrent database calls for user_id: {user_id}")
            start_time = datetime.now()
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Completed concurrent database calls in {duration:.2f} seconds")
            
            # Map results back to names
            task_names = list(tasks.keys())
            data_results = {}
            
            for i, result in enumerate(results):
                task_name = task_names[i]
                if isinstance(result, Exception):
                    logger.warning(f"Error fetching {task_name} for {user_id}: {result}")
                    data_results[task_name] = None
                else:
                    data_results[task_name] = result
            
            # Check if profile loaded successfully (critical requirement)
            profile = data_results.get("profile")
            if not profile:
                raise ValueError(f"Client profile not found for user_id: {user_id}")
            
            # Consolidate all data with safe handling
            client_data = {
                "user_id": user_id,
                "profile": profile,
                "account_overview": data_results.get("account_overview"),
                "account_aging": cls._safe_get_first(data_results.get("account_aging")),
                "banking_details": cls._safe_get_first(data_results.get("banking_details")),
                "subscription": data_results.get("subscription_data"),
                "payment_history": cls._safe_slice(data_results.get("payment_history"), 5),
                "failed_payments": cls._safe_slice(data_results.get("failed_payments"), 3),
                "last_successful_payment": data_results.get("last_payment"),
                "contracts": data_results.get("contracts"),
                "billing_analysis": cls._safe_get_first(data_results.get("billing_analysis")),
                "existing_mandates": data_results.get("existing_mandates"),
                "loaded_at": datetime.now(),
                "load_duration_seconds": duration
            }
            
            logger.info(f"Successfully loaded async data for user_id: {user_id} in {duration:.2f}s")
            return client_data
            
        except Exception as e:
            logger.error(f"Error loading async client data for {user_id}: {str(e)}")
            raise
    
    @classmethod
    def _safe_get_first(cls, data_list: Optional[List]) -> Dict[str, Any]:
        """Safely get first item from list or return empty dict."""
        if data_list and isinstance(data_list, list) and len(data_list) > 0:
            return data_list[0]
        return {}
    
    @classmethod
    def _safe_slice(cls, data_list: Optional[List], limit: int) -> List:
        """Safely slice list or return empty list."""
        if data_list and isinstance(data_list, list):
            return data_list[:limit]
        return []
    
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
            "account_aging": {"xbalance": "0.00", "x0": "0.00"},
            "subscription": {"subscription_amount": "0.00"},
            "payment_history": [],
            "failed_payments": [],
            "last_successful_payment": None,
            "contracts": [],
            "billing_analysis": {},
            "existing_mandates": {},
            "loaded_at": datetime.now(),
            "load_duration_seconds": 0.0,
            "fallback_used": True
        }
    
    @classmethod
    def clear_cache(cls, user_id: Optional[str] = None):
        """Clear cached data."""
        if user_id:
            cls._cache.pop(user_id, None)
        else:
            cls._cache.clear()
    
    @classmethod
    def shutdown_executor(cls):
        """Shutdown the thread pool executor."""
        cls._executor.shutdown(wait=True)

########################################################################################
# Convenience Functions for Easy Integration
########################################################################################

def get_client_data(user_id: str, force_reload: bool = False) -> Dict[str, Any]:
    """Get client data with caching."""
    return ClientDataBuilder.get_client_data(user_id, force_reload)

async def get_client_data_async(user_id: str, force_reload: bool = False) -> Dict[str, Any]:
    """Get client data asynchronously with caching."""
    return await AsyncClientDataBuilder.get_client_data(user_id, force_reload)

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

def prepare_parameters_with_validation(
    client_data: Dict[str, Any], 
    current_step: str, 
    state: Dict[str, Any], 
    script_type: str = "ratio_1_inflow", 
    agent_name: str = "AI Agent"
) -> Dict[str, Any]:
    """Build parameters with comprehensive validation and proper outstanding calculation."""
    
    try:
        # Build base parameters
        parameters = ParameterBuilder.build_parameters(
            client_data, current_step, state, script_type, agent_name
        )
        
        # Calculate outstanding amount properly
        user_id = get_safe_value(client_data, "profile.user_id", "")     account_aging = client_data.get("account_aging", {})
        outstanding_float = calculate_outstanding_amount(account_aging)
        outstanding_amount = format_currency(outstanding_float)
        
        # Validate client data extraction
        profile = client_data.get("profile", {})
        if not profile:
            logger.warning("No client profile data available")
            
        # Ensure all script variables are populated
        script_variables = {
            "agent_name": agent_name,
            "client_full_name": get_safe_value(profile, "client_info.client_full_name", "Client"),
            "client_name": get_safe_value(profile, "client_info.first_name", "Client"),
            "client_title": get_safe_value(profile, "client_info.title", "Mr/Ms"),
            "outstanding_amount": outstanding_amount,  # Calculated overdue amount
            "subscription_amount": format_currency(client_data.get("subscription", {}).get("subscription_amount", "199.00")),
            "subscription_date": "5th of each month",
            "current_step": current_step,
            # Additional fee calculations based on outstanding
            "amount_with_fee": format_currency(outstanding_float + 10),
            "ptp_amount_plus_fee": format_currency(outstanding_float + 10)
        }
        
        # Merge and validate
        parameters.update(script_variables)
        
        # Log parameter summary including calculation breakdown
        logger.info(f"Parameters built for {current_step}:")
        for key, value in script_variables.items():
            logger.info(f"  {key}: {value}")
        
        # Log aging breakdown for transparency
        if account_aging:
            logger.info("Account aging breakdown:")
            logger.info(f"  Current (x0): R {account_aging.get('x0', 0)}")
            logger.info(f"  30 days (x30): R {account_aging.get('x30', 0)}")
            logger.info(f"  60 days (x60): R {account_aging.get('x60', 0)}")
            logger.info(f"  90 days (x90): R {account_aging.get('x90', 0)}")
            logger.info(f"  120+ days (x120): R {account_aging.get('x120', 0)}")
            logger.info(f"  Total balance: R {account_aging.get('xbalance', 0)}")
            
        return parameters
        
    except Exception as e:
        logger.error(f"Parameter building failed: {e}")
        return ParameterBuilder._get_fallback_parameters(agent_name, current_step)

def analyze_client_behavior(client_data: Dict[str, Any], conversation_messages: List[Any] = None) -> Dict[str, Any]:
    """Analyze client behavior patterns with conversation intelligence."""
    return BehavioralAnalyzer.analyze_client_profile(client_data, conversation_messages)

def analyze_payment_conversation(messages: List[Any], outstanding_amount: float) -> Dict[str, Any]:
    """Analyze conversation for payment indicators."""
    return ConversationAnalyzer.analyze_payment_conversation(messages, outstanding_amount)

def detect_emotional_state(messages: List[Any]) -> str:
    """Detect client emotional state from conversation."""
    return ConversationAnalyzer.detect_emotional_state(messages)

def detect_objections(messages: List[Any]) -> List[str]:
    """Detect objections from conversation."""
    return ConversationAnalyzer.detect_real_objections(messages)

def clear_client_cache(user_id: Optional[str] = None):
    """Clear cached client data."""
    ClientDataBuilder.clear_cache(user_id)
    AsyncClientDataBuilder.clear_cache(user_id)

########################################################################################
# State Management Helper
########################################################################################

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
# Enhanced Validation Functions
########################################################################################

def validate_parameters_before_llm(parameters: Dict[str, Any], current_step: str) -> Dict[str, Any]:
    """Validate all parameters before LLM invocation with full logging."""
    
    # Required parameters for all steps
    required_params = [
        "agent_name", "client_full_name", "client_name", "client_title",
        "outstanding_amount", "current_step"
    ]
    
    # Step-specific required parameters
    step_requirements = {
        "details_verification": ["field_to_verify", "matched_fields"],
        "promise_to_pay": ["script_content", "amount_with_fee"],
        "debicheck_setup": ["amount_with_fee"],
        "payment_portal": ["outstanding_amount"]
    }
    
    # Add step-specific requirements
    if current_step in step_requirements:
        required_params.extend(step_requirements[current_step])
    
    # Validate all required parameters exist
    missing_params = []
    for param in required_params:
        if param not in parameters or parameters[param] is None:
            missing_params.append(param)
            logger.warning(f"Missing required parameter: {param}")
            parameters[param] = f"[MISSING_{param.upper()}]"
    
    # Convert all values to strings and handle None values
    safe_parameters = {}
    for key, value in parameters.items():
        if value is None:
            safe_parameters[key] = ""
        elif isinstance(value, (dict, list)):
            safe_parameters[key] = str(value)
        else:
            safe_parameters[key] = str(value)
    
    # Log validation results
    if missing_params:
        logger.error(f"Validation failed for step {current_step}. Missing parameters: {missing_params}")
    else:
        logger.info(f"✅ Parameter validation passed for step {current_step}")
    
    return safe_parameters

def log_complete_prompt_to_console(prompt_content: str, current_step: str, parameters: Dict[str, Any]):
    """Log complete prompt with parameters to console for debugging."""
    
    logger.info("=" * 100)
    logger.info(f"🎯 COMPLETE PROMPT FOR STEP: {current_step.upper()}")
    logger.info("=" * 100)
    
    # Log key parameters
    logger.info("📊 KEY PARAMETERS:")
    key_params = ["agent_name", "client_full_name", "outstanding_amount", "current_step"]
    for param in key_params:
        value = parameters.get(param, "NOT_SET")
        logger.info(f"   {param}: {value}")
    
    logger.info("\n" + "=" * 50)
    logger.info("📝 FINAL PROMPT CONTENT:")
    logger.info("=" * 50)
    logger.info(prompt_content)
    logger.info("=" * 100)
    
    # Check for unresolved placeholders
    import re
    remaining_placeholders = re.findall(r'\{([^}]+)\}', prompt_content)
    if remaining_placeholders:
        logger.error(f"❌ UNRESOLVED PLACEHOLDERS: {remaining_placeholders}")
    else:
        logger.info("✅ All placeholders resolved successfully")
    
    logger.info("=" * 100)

########################################################################################
# Export All Functions
########################################################################################

__all__ = [
    # Core functions
    'get_client_data',
    'get_client_data_async', 
    'prepare_parameters',
    'prepare_parameters_by_user_id',
    'prepare_parameters_with_validation',
    
    # Outstanding amount calculation
    'calculate_outstanding_amount',
    'format_outstanding_amount',
    'format_currency',
    'get_safe_value',
    
    # Analysis functions
    'analyze_client_behavior',
    'analyze_payment_conversation', 
    'detect_emotional_state',
    'detect_objections',
    
    # Validation functions
    'validate_parameters_before_llm',
    'log_complete_prompt_to_console',
    
    # Cache management
    'clear_client_cache',
    
    # Classes
    'ClientDataBuilder',
    'AsyncClientDataBuilder',
    'ConversationAnalyzer',
    'PaymentFlexibilityAnalyzer',
    'BehavioralAnalyzer',
    'ParameterBuilder',
    'ScriptFormatter',
    'ConversationState',
    
    # Enums
    'VerificationStatus'
]
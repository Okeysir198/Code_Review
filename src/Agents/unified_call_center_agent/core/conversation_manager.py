# src/Agents/unified_call_center_agent/core/conversation_manager.py
"""
Conversation Manager - Manages objectives and conversation flow
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .unified_agent_state import (
    UnifiedAgentState, ConversationObjective, ClientMood, VerificationStatus
)
from ..routing.intent_detector import IntentMatch

logger = logging.getLogger(__name__)

@dataclass
class ObjectiveStatus:
    """Tracks status of conversation objectives"""
    objective: ConversationObjective
    priority: int  # 1-10, higher = more important
    required: bool
    completed: bool = False
    attempts: int = 0
    max_attempts: int = 3
    last_attempt: Optional[datetime] = None
    
    def can_attempt(self) -> bool:
        """Check if objective can be attempted"""
        return self.attempts < self.max_attempts and not self.completed
    
    def should_prioritize(self, context: Dict[str, Any]) -> bool:
        """Check if objective should be prioritized given context"""
        if self.completed:
            return False
        
        # High priority objectives should always be prioritized
        if self.priority >= 8:
            return True
        
        # Some objectives depend on mood/cooperation
        client_mood = context.get("client_mood", "neutral")
        rapport_level = context.get("rapport_level", 0.5)
        
        # Skip referrals if client is angry or stressed
        if (self.objective == ConversationObjective.REFERRALS_OFFERED and 
            client_mood in ["angry", "financial_stress"]):
            return False
        
        # Prioritize contact update if rapport is good
        if (self.objective == ConversationObjective.CONTACT_UPDATED and 
            rapport_level > 0.7):
            return True
        
        return self.can_attempt()

class ConversationManager:
    """
    Manages conversation flow using objectives instead of rigid steps.
    
    Integrates with LangGraph's state management for smooth conversation flow.
    """
    
    def __init__(self, client_data: Dict[str, Any]):
        self.client_data = client_data
        
        # Initialize objectives based on client situation
        self.objectives = self._initialize_objectives(client_data)
        
        # Track conversation patterns
        self.conversation_patterns = []
        self.escalation_triggers = 0
        self.rapport_history = []
    
    def _initialize_objectives(self, client_data: Dict[str, Any]) -> Dict[ConversationObjective, ObjectiveStatus]:
        """Initialize objectives based on client data and account status"""
        
        # Get account context for priority setting
        account_aging = client_data.get("account_aging", {})
        outstanding = self._calculate_outstanding(account_aging)
        has_banking = bool(client_data.get("banking_details", {}).get("bank_name"))
        
        objectives = {
            # Core objectives (always required)
            ConversationObjective.IDENTITY_VERIFICATION: ObjectiveStatus(
                ConversationObjective.IDENTITY_VERIFICATION, 
                priority=10, required=True, max_attempts=5
            ),
            ConversationObjective.ACCOUNT_EXPLANATION: ObjectiveStatus(
                ConversationObjective.ACCOUNT_EXPLANATION,
                priority=9, required=True, max_attempts=3
            ),
            ConversationObjective.PAYMENT_SECURED: ObjectiveStatus(
                ConversationObjective.PAYMENT_SECURED,
                priority=8, required=True, max_attempts=5
            ),
            
            # Optional objectives (based on context)
            ConversationObjective.CONTACT_UPDATED: ObjectiveStatus(
                ConversationObjective.CONTACT_UPDATED,
                priority=4, required=False, max_attempts=2
            ),
            ConversationObjective.REFERRALS_OFFERED: ObjectiveStatus(
                ConversationObjective.REFERRALS_OFFERED,
                priority=2, required=False, max_attempts=1
            ),
            ConversationObjective.CALL_COMPLETED: ObjectiveStatus(
                ConversationObjective.CALL_COMPLETED,
                priority=1, required=True, max_attempts=1
            )
        }
        
        # Adjust priorities based on account context
        if outstanding > 500:  # High value account
            objectives[ConversationObjective.PAYMENT_SECURED].priority = 9
            objectives[ConversationObjective.CONTACT_UPDATED].priority = 5
        
        if not has_banking:  # No banking details
            objectives[ConversationObjective.CONTACT_UPDATED].priority = 6
            objectives[ConversationObjective.CONTACT_UPDATED].required = True
        
        return objectives
    
    def get_next_action(self, state: UnifiedAgentState, intent_match: IntentMatch = None) -> Dict[str, Any]:
        """
        Determine next conversation action based on objectives and context.
        
        Args:
            state: Current conversation state
            intent_match: Latest intent detection result
            
        Returns:
            Dictionary with action details
        """
        
        # Handle immediate actions first (escalation, cancellation, etc.)
        if intent_match and intent_match.requires_immediate_action:
            return self._handle_immediate_action(intent_match, state)
        
        # Check business rule terminations
        termination_action = self._check_termination_conditions(state)
        if termination_action:
            return termination_action
        
        # Handle verification requirements
        verification_action = self._check_verification_requirements(state)
        if verification_action:
            return verification_action
        
        # Find next objective to work on
        next_objective = self._get_next_objective(state)
        if not next_objective:
            return self._create_completion_action(state)
        
        # Generate action for objective
        return self._generate_objective_action(next_objective, state, intent_match)
    
    def _handle_immediate_action(self, intent_match: IntentMatch, state: UnifiedAgentState) -> Dict[str, Any]:
        """Handle intents that require immediate action"""
        
        if intent_match.intent == "escalation":
            return {
                "action": "escalate",
                "message": "I understand you'd like to speak with a supervisor. Let me arrange that immediately.",
                "objective": "escalation_requested",
                "intent": intent_match.intent,
                "confidence": intent_match.confidence,
                "immediate": True
            }
        
        elif intent_match.intent == "cancellation":
            return {
                "action": "handle_cancellation",
                "message": "I can help with the cancellation process. Let me explain what's involved.",
                "objective": "cancellation_requested", 
                "intent": intent_match.intent,
                "confidence": intent_match.confidence,
                "immediate": True
            }
        
        elif intent_match.intent == "identity_denial":
            return {
                "action": "wrong_person",
                "message": "I apologize for the confusion. I'll update our records.",
                "objective": "call_ended",
                "intent": intent_match.intent,
                "confidence": intent_match.confidence,
                "immediate": True
            }
        
        # Default for other immediate actions
        return {
            "action": "handle_immediate",
            "message": "Let me address that right away.",
            "objective": "immediate_response",
            "intent": intent_match.intent,
            "immediate": True
        }
    
    def _check_termination_conditions(self, state: UnifiedAgentState) -> Optional[Dict[str, Any]]:
        """Check if call should be terminated"""
        
        # Already ended
        if state.call_ended:
            return {
                "action": "call_completed",
                "message": "Thank you for your time today.",
                "objective": ConversationObjective.CALL_COMPLETED.value
            }
        
        # Too many verification attempts
        if (state.verification_attempts.get("name", 0) >= 5 or 
            state.verification_attempts.get("details", 0) >= 5):
            return {
                "action": "verification_failed",
                "message": "For security reasons, I need to end this call. Please contact us directly at 011 250 3000.",
                "objective": ConversationObjective.CALL_COMPLETED.value
            }
        
        # Escalation or cancellation requested
        if state.escalation_requested or state.cancellation_requested:
            return {
                "action": "end_call",
                "message": "I've logged your request. You'll hear back within 24 hours.",
                "objective": ConversationObjective.CALL_COMPLETED.value
            }
        
        # Too many turns (prevent infinite loops)
        if state.turn_count > 50:
            return {
                "action": "call_timeout",
                "message": "We've been talking for a while. Let me summarize and arrange next steps.",
                "objective": ConversationObjective.CALL_COMPLETED.value
            }
        
        return None
    
    def _check_verification_requirements(self, state: UnifiedAgentState) -> Optional[Dict[str, Any]]:
        """Check if verification is still needed"""
        
        # Name verification needed
        if state.name_verification != VerificationStatus.VERIFIED.value:
            return {
                "action": "verify_name",
                "message": self._get_name_verification_message(state),
                "objective": ConversationObjective.IDENTITY_VERIFICATION.value,
                "field": "name",
                "attempt": state.verification_attempts.get("name", 0) + 1
            }
        
        # Details verification needed
        if state.details_verification != VerificationStatus.VERIFIED.value:
            return {
                "action": "verify_details", 
                "message": self._get_details_verification_message(state),
                "objective": ConversationObjective.IDENTITY_VERIFICATION.value,
                "field": "details",
                "attempt": state.verification_attempts.get("details", 0) + 1
            }
        
        return None
    
    def _get_next_objective(self, state: UnifiedAgentState) -> Optional[ConversationObjective]:
        """Get next objective to work on"""
        
        context = state.get_conversation_context()
        
        # Get objectives that can be attempted and should be prioritized
        available_objectives = [
            obj for obj_status in self.objectives.values()
            for obj in [obj_status.objective]
            if (obj_status.can_attempt() and 
                obj_status.should_prioritize(context) and
                obj.value not in state.completed_objectives)
        ]
        
        if not available_objectives:
            return None
        
        # Sort by priority (higher first)
        objective_priorities = {
            obj: self.objectives[obj].priority 
            for obj in available_objectives
        }
        
        return max(objective_priorities.items(), key=lambda x: x[1])[0]
    
    def _generate_objective_action(
        self, 
        objective: ConversationObjective, 
        state: UnifiedAgentState,
        intent_match: IntentMatch = None
    ) -> Dict[str, Any]:
        """Generate action for specific objective"""
        
        # Mark objective as attempted
        if objective in self.objectives:
            self.objectives[objective].attempts += 1
            self.objectives[objective].last_attempt = datetime.now()
        
        if objective == ConversationObjective.ACCOUNT_EXPLANATION:
            return self._create_account_explanation_action(state, intent_match)
        
        elif objective == ConversationObjective.PAYMENT_SECURED:
            return self._create_payment_action(state, intent_match)
        
        elif objective == ConversationObjective.CONTACT_UPDATED:
            return self._create_contact_update_action(state)
        
        elif objective == ConversationObjective.REFERRALS_OFFERED:
            return self._create_referrals_action(state)
        
        else:
            # Default continuation
            return {
                "action": "continue_conversation",
                "message": "How can I help you further?",
                "objective": objective.value
            }
    
    def _create_account_explanation_action(self, state: UnifiedAgentState, intent_match: IntentMatch = None) -> Dict[str, Any]:
        """Create action for explaining account situation"""
        
        strategy = self._get_conversation_strategy(state.client_mood)
        outstanding = state.outstanding_amount
        
        # Adapt message based on client mood and previous interactions
        if state.client_mood == ClientMood.CONFUSED.value:
            message = f"Let me explain simply - your Cartrack subscription payment of {outstanding} wasn't received, so your account is overdue."
        
        elif state.client_mood == ClientMood.ANGRY.value:
            message = f"I understand your frustration. Your account shows {outstanding} overdue. Let's resolve this quickly."
        
        elif state.client_mood == ClientMood.FINANCIAL_STRESS.value:
            message = f"I know finances can be tight. Your account needs {outstanding}. Let's find a solution that works."
        
        elif state.cooperation_level > 0.7:
            message = f"Your Cartrack account is {outstanding} overdue. Can we arrange payment today?"
        
        else:
            message = f"I'm calling because your Cartrack account is overdue by {outstanding}."
        
        return {
            "action": "explain_account",
            "message": message,
            "objective": ConversationObjective.ACCOUNT_EXPLANATION.value,
            "strategy": strategy,
            "amount": outstanding
        }
    
    def _create_payment_action(self, state: UnifiedAgentState, intent_match: IntentMatch = None) -> Dict[str, Any]:
        """Create action for securing payment"""
        
        # Check if client already indicated payment method preference
        if intent_match and intent_match.entities.get("payment_method"):
            preferred_method = intent_match.entities["payment_method"]
            
            if preferred_method == "debit_order":
                return {
                    "action": "setup_debit_order",
                    "message": f"Perfect, I can set up the debit order for {state.outstanding_amount} right now.",
                    "objective": ConversationObjective.PAYMENT_SECURED.value,
                    "payment_method": "debit_order"
                }
            
            elif preferred_method == "online":
                return {
                    "action": "send_payment_link",
                    "message": f"I'll send you a secure payment link for {state.outstanding_amount}.",
                    "objective": ConversationObjective.PAYMENT_SECURED.value,
                    "payment_method": "online"
                }
        
        # Check for payment agreement intent
        if intent_match and intent_match.intent == "payment_agreement":
            return {
                "action": "confirm_payment_method",
                "message": f"Excellent! Would you prefer a debit order or should I send you a payment link for {state.outstanding_amount}?",
                "objective": ConversationObjective.PAYMENT_SECURED.value,
                "payment_step": "method_selection"
            }
        
        # Default payment request based on mood
        if state.client_mood == ClientMood.FINANCIAL_STRESS.value:
            message = f"I understand money is tight. What amount could you manage today toward the {state.outstanding_amount}?"
        elif state.client_mood == ClientMood.RESISTANT.value:
            message = f"Your account needs {state.outstanding_amount} to avoid service suspension. How can we resolve this?"
        else:
            message = f"Can we arrange payment of {state.outstanding_amount} today?"
        
        return {
            "action": "request_payment",
            "message": message,
            "objective": ConversationObjective.PAYMENT_SECURED.value,
            "amount": state.outstanding_amount
        }
    
    def _create_contact_update_action(self, state: UnifiedAgentState) -> Dict[str, Any]:
        """Create action for updating contact details"""
        
        return {
            "action": "update_contact_details",
            "message": "Let me quickly verify your contact details for important notifications.",
            "objective": ConversationObjective.CONTACT_UPDATED.value,
            "quick_update": True
        }
    
    def _create_referrals_action(self, state: UnifiedAgentState) -> Dict[str, Any]:
        """Create action for offering referrals"""
        
        # Only offer if mood is good
        if state.rapport_level > 0.6:
            return {
                "action": "offer_referrals",
                "message": "By the way, do you know anyone who might be interested in Cartrack? We offer rewards for successful referrals.",
                "objective": ConversationObjective.REFERRALS_OFFERED.value,
                "brief_mention": True
            }
        
        # Skip if mood isn't right
        self.objectives[ConversationObjective.REFERRALS_OFFERED].completed = True
        return self._get_next_objective_action(state)
    
    def _create_completion_action(self, state: UnifiedAgentState) -> Dict[str, Any]:
        """Create action for completing the call"""
        
        return {
            "action": "complete_call",
            "message": "Is there anything else I can help you with regarding your account today?",
            "objective": ConversationObjective.CALL_COMPLETED.value,
            "summary": self._generate_call_summary(state)
        }
    
    def _get_next_objective_action(self, state: UnifiedAgentState) -> Dict[str, Any]:
        """Get action for next available objective"""
        next_obj = self._get_next_objective(state)
        if next_obj:
            return self._generate_objective_action(next_obj, state)
        return self._create_completion_action(state)
    
    def _get_name_verification_message(self, state: UnifiedAgentState) -> str:
        """Get appropriate name verification message"""
        
        attempt = state.verification_attempts.get("name", 0) + 1
        client_name = state.client_name
        
        if attempt == 1:
            if state.client_mood == ClientMood.SUSPICIOUS.value:
                return f"Good day, this is Sarah from Cartrack Accounts. May I confirm I'm speaking with {client_name}?"
            else:
                return f"Good day, is this {client_name}?"
        
        elif attempt <= 3:
            return f"Just to confirm, you are {client_name}, correct?"
        
        else:
            return f"I need to verify I'm speaking with the account holder. Are you {client_name}?"
    
    def _get_details_verification_message(self, state: UnifiedAgentState) -> str:
        """Get appropriate details verification message"""
        
        if state.client_mood == ClientMood.COOPERATIVE.value:
            return "For security, could you confirm your ID number?"
        elif state.client_mood == ClientMood.RESISTANT.value:
            return "I need to verify your ID number before we can discuss your account."
        else:
            return "To protect your account, I need to confirm your ID number. What is it?"
    
    def _get_conversation_strategy(self, client_mood: str) -> Dict[str, str]:
        """Get conversation strategy based on client mood"""
        
        strategies = {
            ClientMood.COOPERATIVE.value: {
                "tone": "professional and efficient",
                "approach": "move quickly to solutions",
                "pace": "normal to fast"
            },
            ClientMood.ANGRY.value: {
                "tone": "calm and empathetic",
                "approach": "acknowledge frustration, de-escalate first",
                "pace": "slower, give space"
            },
            ClientMood.CONFUSED.value: {
                "tone": "patient and explanatory",
                "approach": "break down into simple steps",
                "pace": "slower, check understanding"
            },
            ClientMood.RESISTANT.value: {
                "tone": "firm but understanding",
                "approach": "emphasize consequences and benefits",
                "pace": "deliberate, not rushed"
            },
            ClientMood.FINANCIAL_STRESS.value: {
                "tone": "empathetic and solution-focused",
                "approach": "explore payment options, show flexibility",
                "pace": "patient"
            },
            ClientMood.SUSPICIOUS.value: {
                "tone": "professional and transparent",
                "approach": "provide verification, build trust",
                "pace": "steady, not pushy"
            },
            ClientMood.NEUTRAL.value: {
                "tone": "professional and friendly",
                "approach": "assess mood then adapt",
                "pace": "normal"
            }
        }
        
        return strategies.get(client_mood, strategies[ClientMood.NEUTRAL.value])
    
    def _calculate_outstanding(self, account_aging: Dict[str, Any]) -> float:
        """Calculate outstanding amount"""
        try:
            total = float(account_aging.get("xbalance", 0))
            current = float(account_aging.get("x0", 0))
            return max(total - current, 0.0)
        except (ValueError, TypeError):
            return 0.0
    
    def _generate_call_summary(self, state: UnifiedAgentState) -> Dict[str, Any]:
        """Generate summary of call outcomes"""
        
        return {
            "objectives_completed": len(state.completed_objectives),
            "payment_secured": state.payment_secured,
            "escalation_requested": state.escalation_requested,
            "cancellation_requested": state.cancellation_requested,
            "verification_completed": state.is_verified(),
            "client_cooperation": state.cooperation_level,
            "call_duration_turns": state.turn_count
        }
    
    def update_objective_completion(self, objective: ConversationObjective, completed: bool = True):
        """Mark objective as completed"""
        if objective in self.objectives:
            self.objectives[objective].completed = completed
            logger.info(f"Objective {objective.value} marked as {'completed' if completed else 'incomplete'}")
    
    def get_objectives_status(self) -> Dict[str, Any]:
        """Get status of all objectives"""
        return {
            obj.value: {
                "completed": status.completed,
                "attempts": status.attempts,
                "priority": status.priority,
                "required": status.required
            }
            for obj, status in self.objectives.items()
        }
# src/Agents/unified_call_center_agent/core/unified_agent_state.py
# Fix for the state management issues

"""
Unified Agent State - Optimized for LangGraph's MessagesState and built-in memory
"""
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from enum import Enum
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import BaseMessage

class ConversationObjective(Enum):
    """Conversation objectives instead of rigid steps"""
    IDENTITY_VERIFICATION = "identity_verification"
    ACCOUNT_EXPLANATION = "account_explanation" 
    PAYMENT_SECURED = "payment_secured"
    CONTACT_UPDATED = "contact_updated"
    REFERRALS_OFFERED = "referrals_offered"
    CALL_COMPLETED = "call_completed"

class ClientMood(Enum):
    """Client emotional states for conversation adaptation"""
    COOPERATIVE = "cooperative"
    RESISTANT = "resistant"
    CONFUSED = "confused"
    ANGRY = "angry"
    FINANCIAL_STRESS = "financial_stress"
    SUSPICIOUS = "suspicious"
    NEUTRAL = "neutral"

class VerificationStatus(Enum):
    """Verification status values"""
    VERIFIED = "VERIFIED"
    PENDING = "PENDING"
    FAILED = "FAILED"
    THIRD_PARTY = "THIRD_PARTY"
    WRONG_PERSON = "WRONG_PERSON"

def merge_objectives(current: List[str], new: List[str]) -> List[str]:
    """Custom reducer for objectives - merge without duplicates"""
    if not current:
        return new if new else []
    if not new:
        return current
    
    # Merge unique objectives
    combined = list(set(current + new))
    return combined

def merge_concerns(current: List[str], new: List[str]) -> List[str]:
    """Custom reducer for concerns - keep last 5 unique concerns"""
    if not current:
        current = []
    if not new:
        return current
    
    # Merge and keep unique, maintain order
    combined = current + [c for c in new if c not in current]
    return combined[-5:]  # Keep last 5

class UnifiedAgentState(MessagesState):
    """
    Unified state that leverages LangGraph's MessagesState + custom fields
    
    Uses LangGraph's built-in message management with add_messages reducer
    for automatic message history handling and memory persistence.
    """
    
    # === CORE CLIENT DATA (read-only, loaded once) ===
    user_id: str = ""
    client_name: str = ""
    outstanding_amount: str = "R 0.00"
    
    # === CONVERSATION CONTEXT (dynamic, updated throughout) ===
    current_objective: str = ConversationObjective.IDENTITY_VERIFICATION.value
    completed_objectives: Annotated[List[str], merge_objectives] = []
    client_mood: str = ClientMood.NEUTRAL.value
    rapport_level: float = 0.5  # 0-1 scale
    cooperation_level: float = 0.5  # 0-1 scale
    
    # === VERIFICATION STATE ===
    name_verification: str = VerificationStatus.PENDING.value
    details_verification: str = VerificationStatus.PENDING.value
    verification_attempts: Dict[str, int] = {"name": 0, "details": 0}
    
    # === CONVERSATION TRACKING ===
    turn_count: int = 0
    client_concerns: Annotated[List[str], merge_concerns] = []
    mentioned_topics: List[str] = []  # Will be overwritten each time
    
    # === PAYMENT TRACKING ===
    payment_secured: bool = False
    payment_method_preference: str = ""
    
    # === CALL OUTCOME ===
    escalation_requested: bool = False
    cancellation_requested: bool = False
    call_ended: bool = False
    call_outcome: str = "in_progress"
    
    # === METADATA ===
    last_intent: str = "none"
    last_action: str = "start"
    
    def is_verified(self) -> bool:
        """Check if client is fully verified"""
        return (self.name_verification == VerificationStatus.VERIFIED.value and 
                self.details_verification == VerificationStatus.VERIFIED.value)
    
    def can_discuss_account(self) -> bool:
        """Check if can discuss account details"""
        return self.is_verified()
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get context for conversation adaptation"""
        return {
            "client_mood": self.client_mood,
            "rapport_level": self.rapport_level,
            "cooperation_level": self.cooperation_level,
            "turn_count": self.turn_count,
            "concerns": self.client_concerns,
            "objectives_completed": len(self.completed_objectives),
            "verification_status": {
                "name": self.name_verification,
                "details": self.details_verification
            }
        }
    
    def update_mood_and_rapport(self, new_mood: str, cooperation_change: float):
        """Update mood and rapport with smoothing"""
        # Update mood if confidence is high
        if new_mood != ClientMood.NEUTRAL.value:
            self.client_mood = new_mood
        
        # Update cooperation with smoothing (70% old, 30% new)
        old_cooperation = self.cooperation_level
        self.cooperation_level = max(0.0, min(1.0, (old_cooperation * 0.7) + (cooperation_change * 0.3)))
        
        # Update rapport based on cooperation trend
        if cooperation_change > 0:
            self.rapport_level = min(1.0, self.rapport_level + 0.1)
        elif cooperation_change < -0.2:
            self.rapport_level = max(0.0, self.rapport_level - 0.1)

    # Factory method to create initial state with defaults
    @classmethod
    def create_initial_state(cls, client_data: Dict[str, Any]) -> 'UnifiedAgentState':
        """Create initial state with client data"""
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {})
        account_aging = client_data.get("account_aging", {})
        
        # Calculate outstanding amount
        try:
            total = float(account_aging.get("xbalance", 0))
            current = float(account_aging.get("x0", 0))
            outstanding = max(total - current, 0.0)
            outstanding_str = f"R {outstanding:.2f}"
        except (ValueError, TypeError):
            outstanding_str = "R 0.00"
        
        return cls(
            user_id=profile.get("user_id", ""),
            client_name=client_info.get("client_full_name", "Client"),
            outstanding_amount=outstanding_str,
            current_objective=ConversationObjective.IDENTITY_VERIFICATION.value,
            name_verification=VerificationStatus.PENDING.value,
            details_verification=VerificationStatus.PENDING.value,
            verification_attempts={"name": 0, "details": 0},
            turn_count=0,
            completed_objectives=[],
            client_concerns=[],
            mentioned_topics=[],
            call_ended=False,
            messages=[]  # Initialize empty messages
        )


# Example of extending state for specific use cases if needed
class CallCenterAgentState(UnifiedAgentState):
    """Extended state for call center specific features"""
    
    # === CALL CENTER SPECIFIC ===
    script_type: str = "standard"
    aging_category: str = "current"
    urgency_level: str = "medium"
    
    # === BANKING INFO ===
    has_banking_details: bool = False
    debit_order_available: bool = False
    
    # === ADVANCED FEATURES ===
    conversation_summary: str = ""  # For long conversation summarization
    tool_calls_made: List[str] = []  # Track which tools were used
    
    def get_strategy_context(self) -> Dict[str, str]:
        """Get context for conversation strategy selection"""
        strategies = {
            ClientMood.COOPERATIVE.value: {
                "tone": "professional and efficient",
                "approach": "move quickly to payment solutions",
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
            }
        }
        
        return strategies.get(self.client_mood, strategies[ClientMood.NEUTRAL.value])
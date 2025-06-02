# src/Agents/unified_call_center_agent/routing/intent_detector.py
"""
Fast Intent Detection - No LLM needed, uses pattern matching for speed
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntentMatch:
    """Intent detection result"""
    intent: str
    confidence: float
    entities: Dict[str, str]
    urgency: str = "normal"
    requires_immediate_action: bool = False

class FastIntentDetector:
    """Ultra-fast intent detection using keyword and pattern matching"""
    
    def __init__(self):
        self.intent_patterns = {
            # === HIGH PRIORITY INTENTS (require immediate action) ===
            "escalation": {
                "patterns": [
                    r"(?:speak|talk) (?:to|with) (?:a )?(?:supervisor|manager|someone (?:in )?charge)",
                    r"(?:file|make|want to make) (?:a )?complaint",
                    r"this is harassment",
                    r"(?:call|contact|get) (?:a )?(?:lawyer|attorney)",
                    r"(?:legal action|take you to court)",
                    r"(?:sick|tired|fed up) (?:of|with) (?:this|these) (?:calls?|people)",
                    r"stop (?:calling|harassing) me"
                ],
                "keywords": ["supervisor", "manager", "complaint", "harassment", "legal", "lawyer", "attorney", "court"],
                "urgency": "high",
                "immediate_action": True
            },
            
            "cancellation": {
                "patterns": [
                    r"(?:cancel|terminate|stop|end|disconnect) (?:my )?(?:account|service|subscription|contract)",
                    r"(?:remove|take off) (?:the )?(?:device|tracker|equipment)",
                    r"don'?t want (?:this|cartrack|the service) anymore",
                    r"stop (?:all )?(?:services?|billing|charges)"
                ],
                "keywords": ["cancel", "terminate", "stop service", "disconnect", "end contract", "remove device"],
                "urgency": "medium",
                "immediate_action": True
            },
            
            # === VERIFICATION INTENTS ===
            "identity_confirmation": {
                "patterns": [
                    r"(?:yes|yeah|yep),? (?:this is|i'?m|that'?s me|speaking)",
                    r"(?:that'?s|this is) (?:me|correct|right)",
                    r"^(?:speaking|yes|yeah|correct)$"
                ],
                "keywords": ["yes", "speaking", "correct", "that's me", "this is me"],
                "urgency": "low",
                "immediate_action": False
            },
            
            "identity_denial": {
                "patterns": [
                    r"(?:no|nope),? (?:this is not|i'?m not|that'?s not me)",
                    r"wrong (?:person|number)",
                    r"(?:you have|this is) the wrong (?:person|number)",
                    r"not (?:me|him|her)"
                ],
                "keywords": ["wrong person", "wrong number", "not me", "this is not"],
                "urgency": "high",
                "immediate_action": True
            },
            
            # === PAYMENT INTENTS ===
            "payment_agreement": {
                "patterns": [
                    r"(?:okay|ok|fine|yes|sure|alright),? (?:let'?s|we can|i'?ll) (?:pay|arrange|do (?:it|that)|sort (?:it|this) out)",
                    r"(?:i'?ll|we'?ll|can|will) pay (?:it|that|the amount|you)",
                    r"(?:set up|arrange|do|make) (?:the )?(?:payment|debit|arrangement)",
                    r"(?:debit|take) (?:it )?(?:from|out of) (?:my )?(?:account|bank)"
                ],
                "keywords": ["pay", "payment", "debit", "arrange", "okay", "yes", "fine", "sort it out"],
                "urgency": "low",
                "immediate_action": False
            },
            
            "payment_resistance": {
                "patterns": [
                    r"(?:can'?t|cannot|won'?t|will not) pay",
                    r"(?:don'?t|do not) have (?:the )?money",
                    r"(?:can'?t|cannot) afford (?:it|this|that)",
                    r"(?:no|don'?t have) (?:the )?funds?",
                    r"(?:not|never) (?:going to|gonna) pay"
                ],
                "keywords": ["can't pay", "won't pay", "no money", "can't afford", "not paying"],
                "urgency": "medium",
                "immediate_action": False
            },
            
            # === INFORMATION REQUESTS ===
            "information_request": {
                "patterns": [
                    r"(?:what|how|why|when|where) (?:is|does|do|did|will|would|can)",
                    r"(?:can you )?(?:explain|tell me) (?:about|how|what|why)",
                    r"(?:i )?(?:don'?t|do not) understand",
                    r"what (?:does|do|is) (?:this|that|you|cartrack) (?:mean|do|for)"
                ],
                "keywords": ["what", "how", "why", "when", "explain", "understand", "tell me", "what does"],
                "urgency": "low",
                "immediate_action": False
            },
            
            # === FINANCIAL HARDSHIP ===
            "financial_hardship": {
                "patterns": [
                    r"(?:lost|don'?t have|no longer have) (?:my|a|the) job",
                    r"(?:unemployed|out of work|retrenched)",
                    r"(?:money is|finances are|things are) (?:tight|difficult|hard)",
                    r"(?:struggling|having trouble) (?:financially|with money)",
                    r"(?:can'?t|cannot) afford (?:anything|it|this|that|the amount)"
                ],
                "keywords": ["lost job", "unemployed", "money tight", "struggling", "can't afford", "no job"],
                "urgency": "medium",
                "immediate_action": False
            },
            
            # === DISPUTE/DENIAL ===
            "dispute": {
                "patterns": [
                    r"(?:i|we) (?:already|never) paid",
                    r"(?:that'?s|this is) (?:not )?(?:right|correct|wrong)",
                    r"(?:i|we) (?:don'?t|do not) owe (?:anything|that|you)",
                    r"(?:dispute|disagree with|question) (?:the|this) (?:amount|charge|bill)",
                    r"(?:not|never) (?:my|our) (?:debt|account|responsibility)"
                ],
                "keywords": ["already paid", "wrong", "dispute", "disagree", "not mine", "don't owe"],
                "urgency": "medium",
                "immediate_action": False
            },
            
            # === COOPERATION SIGNALS ===
            "cooperation": {
                "patterns": [
                    r"(?:that'?s|sounds) (?:fine|good|okay|alright)",
                    r"(?:i|we) (?:understand|get it|see)",
                    r"(?:let'?s|we can) (?:do|sort) (?:it|that|this)"
                ],
                "keywords": ["fine", "good", "understand", "get it", "let's do it"],
                "urgency": "low",
                "immediate_action": False
            },
            
            # === CONFUSION ===
            "confusion": {
                "patterns": [
                    r"(?:i'?m|we'?re) (?:confused|lost|not sure)",
                    r"(?:what|which|who) (?:are you|is this) (?:talking about|referring to)",
                    r"(?:i|we) (?:don'?t|do not) (?:know|remember|recall)",
                    r"(?:can you )?(?:repeat|say that again|explain again)"
                ],
                "keywords": ["confused", "don't understand", "repeat", "explain again", "lost"],
                "urgency": "low",
                "immediate_action": False
            },
            
            # === STALLING TACTICS ===
            "stalling": {
                "patterns": [
                    r"(?:let me|i need to) think about (?:it|this)",
                    r"(?:call|speak to|check with) (?:my )?(?:husband|wife|spouse|partner)",
                    r"(?:i'?ll|we'?ll) (?:call|phone) (?:you )?(?:back|later)",
                    r"(?:not a good|this isn'?t a good) time"
                ],
                "keywords": ["think about it", "call back", "speak to spouse", "not good time"],
                "urgency": "low",
                "immediate_action": False
            }
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for intent, data in self.intent_patterns.items():
            self.compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in data["patterns"]
            ]
    
    def detect_intent(self, message: str, context: Dict[str, Any] = None) -> IntentMatch:
        """
        Detect intent from user message
        
        Args:
            message: User's message text
            context: Optional context (mood, rapport, etc.)
            
        Returns:
            IntentMatch with detected intent and confidence
        """
        if not message or not message.strip():
            return IntentMatch("unclear", 0.1, {}, "low")
        
        msg_lower = message.lower().strip()
        
        # Score each intent
        intent_scores = {}
        
        for intent, data in self.intent_patterns.items():
            score = 0.0
            matched_entities = {}
            
            # Pattern matching (highest weight - 0.6)
            for pattern in self.compiled_patterns[intent]:
                match = pattern.search(msg_lower)
                if match:
                    score += 0.6
                    # Extract entities from pattern groups if any
                    if match.groups():
                        matched_entities["matched_text"] = match.group(0)
                    break
            
            # Keyword matching (medium weight - 0.3 per keyword)
            keyword_matches = 0
            for keyword in data["keywords"]:
                if keyword in msg_lower:
                    keyword_matches += 1
                    score += 0.3
            
            # Context boost (if available)
            if context:
                score += self._get_context_boost(intent, context)
            
            # Length penalty for very short messages (except confirmations)
            if len(msg_lower) < 5 and intent not in ["identity_confirmation", "cooperation"]:
                score *= 0.7
            
            # Cap score at 1.0
            intent_scores[intent] = min(score, 1.0)
        
        # Find best match
        if not intent_scores:
            return IntentMatch("unclear", 0.1, {}, "low")
        
        best_intent, best_score = max(intent_scores.items(), key=lambda x: x[1])
        
        # Low confidence threshold
        if best_score < 0.3:
            return IntentMatch("unclear", best_score, {}, "low")
        
        # Extract entities
        entities = self._extract_entities(message, best_intent)
        
        # Get intent metadata
        intent_data = self.intent_patterns[best_intent]
        
        return IntentMatch(
            intent=best_intent,
            confidence=best_score,
            entities=entities,
            urgency=intent_data["urgency"],
            requires_immediate_action=intent_data["immediate_action"]
        )
    
    def _get_context_boost(self, intent: str, context: Dict[str, Any]) -> float:
        """Apply context-based scoring boosts"""
        boost = 0.0
        
        client_mood = context.get("client_mood", "neutral")
        rapport_level = context.get("rapport_level", 0.5)
        turn_count = context.get("turn_count", 0)
        
        # Mood-based boosts
        if intent == "escalation" and client_mood in ["angry", "resistant"]:
            boost += 0.2
        elif intent == "financial_hardship" and client_mood == "financial_stress":
            boost += 0.3
        elif intent == "cooperation" and rapport_level > 0.6:
            boost += 0.2
        elif intent == "information_request" and client_mood == "confused":
            boost += 0.2
        elif intent == "stalling" and turn_count > 5:
            boost += 0.1
        
        return boost
    
    def _extract_entities(self, message: str, intent: str) -> Dict[str, str]:
        """Extract relevant entities from message based on intent"""
        entities = {}
        msg_lower = message.lower()
        
        # Extract payment amounts
        amount_pattern = r"r\s*(\d+(?:[,\s]\d{3})*(?:\.\d{2})?)"
        amount_match = re.search(amount_pattern, msg_lower)
        if amount_match:
            entities["amount"] = amount_match.group(1)
        
        # Extract dates/times
        date_patterns = [
            r"(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"(?:next|this) (?:week|month|friday|monday)",
            r"\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:january|february|march|april|may|june|july|august|september|october|november|december)",
            r"\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?"
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, msg_lower)
            if date_match:
                entities["date"] = date_match.group(0)
                break
        
        # Extract payment methods
        if "bank" in msg_lower or "debit" in msg_lower:
            entities["payment_method"] = "debit_order"
        elif "card" in msg_lower or "online" in msg_lower:
            entities["payment_method"] = "online"
        elif "cash" in msg_lower:
            entities["payment_method"] = "cash"
        
        # Extract contact references for third parties
        if intent in ["identity_denial", "stalling"]:
            if any(word in msg_lower for word in ["husband", "wife", "spouse", "partner"]):
                entities["third_party"] = "spouse"
            elif any(word in msg_lower for word in ["mom", "dad", "mother", "father", "parent"]):
                entities["third_party"] = "parent"
        
        # Extract escalation targets
        if intent == "escalation":
            if "supervisor" in msg_lower:
                entities["escalation_target"] = "supervisor"
            elif "manager" in msg_lower:
                entities["escalation_target"] = "manager"
            elif "legal" in msg_lower or "lawyer" in msg_lower:
                entities["escalation_target"] = "legal"
        
        return entities
    
    def get_intent_summary(self) -> Dict[str, int]:
        """Get summary of available intents for debugging"""
        return {
            intent: len(data["keywords"]) + len(data["patterns"])
            for intent, data in self.intent_patterns.items()
        }
    
    def test_intent_detection(self, test_cases: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Test intent detection with list of (message, expected_intent) tuples"""
        results = []
        
        for message, expected in test_cases:
            detected = self.detect_intent(message)
            results.append({
                "message": message,
                "expected": expected,
                "detected": detected.intent,
                "confidence": detected.confidence,
                "correct": detected.intent == expected
            })
        
        return results

# Pre-configured test cases for validation
INTENT_TEST_CASES = [
    ("Yes, this is John speaking", "identity_confirmation"),
    ("I want to speak to a supervisor", "escalation"),
    ("Can we arrange payment today?", "payment_agreement"),
    ("I can't afford to pay this", "financial_hardship"),
    ("What is this call about?", "information_request"),
    ("Cancel my account immediately", "cancellation"),
    ("This is the wrong person", "identity_denial"),
    ("I already paid last week", "dispute"),
    ("Let me think about it", "stalling"),
    ("I don't understand what you mean", "confusion"),
    ("Okay, let's sort this out", "cooperation"),
    ("I won't pay anything", "payment_resistance")
]

# Convenience function for quick testing
def test_intent_detector():
    """Test the intent detector with predefined cases"""
    detector = FastIntentDetector()
    results = detector.test_intent_detection(INTENT_TEST_CASES)
    
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"Intent Detection Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Show any failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  '{f['message']}' -> Expected: {f['expected']}, Got: {f['detected']} (conf: {f['confidence']:.2f})")
    
    return results

if __name__ == "__main__":
    test_intent_detector()
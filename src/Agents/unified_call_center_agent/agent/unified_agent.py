# src/Agents/unified_call_center_agent/agent/unified_agent.py
"""
Unified Call Center Agent - Single intelligent agent that handles entire conversation
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from ..core.unified_agent_state import UnifiedAgentState, ConversationObjective, ClientMood, VerificationStatus
from ..core.conversation_manager import ConversationManager
from ..routing.intent_detector import FastIntentDetector, IntentMatch

# Import database tools that will be used contextually
from src.Database.CartrackSQLDatabase import (
    # Verification tools
    client_call_verification,
    
    # Payment tools
    create_payment_arrangement,
    create_debicheck_payment,
    create_payment_arrangement_payment_portal,
    generate_sms_payment_url,
    
    # Client update tools
    update_client_contact_number,
    update_client_email,
    add_client_note,
    
    # Call management tools
    save_call_disposition,
    get_disposition_types,
    update_payment_arrangements,
    
    # Helper tools
    date_helper,
    get_current_date_time
)

logger = logging.getLogger(__name__)

class UnifiedCallCenterAgent:
    """
    Single adaptive agent that handles entire conversation flow.
    
    Replaces the complex multi-agent system with one intelligent agent that:
    - Uses LangGraph's MessagesState for automatic message management
    - Tracks conversation context and client behavior
    - Calls database tools contextually based on conversation flow
    - Adapts responses based on client mood and cooperation level
    """
    
    def __init__(
        self, 
        model: BaseChatModel, 
        client_data: Dict[str, Any], 
        config: Dict[str, Any],
        agent_name: str = "Sarah"
    ):
        self.model = model
        self.client_data = client_data
        self.config = config
        self.agent_name = agent_name
        
        # Initialize components
        self.conversation_manager = ConversationManager(client_data)
        self.intent_detector = FastIntentDetector()
        
        # Build base prompt template (concise and adaptive)
        self.base_prompt_template = """You are {agent_name}, a professional debt collection specialist at Cartrack.

CLIENT: {client_name} | OWES: {outstanding_amount} | MOOD: {client_mood} | RAPPORT: {rapport_level:.1f}

CURRENT OBJECTIVE: {current_objective}
CONVERSATION CONTEXT: {conversation_context}

STRATEGY: {conversation_strategy}

TASK: {specific_task}

RESPONSE RULES:
- Keep responses under 25 words unless explaining something complex
- Match the client's energy and cooperation level
- Be {conversation_tone}
- Build on previous conversation naturally
- Focus on: {current_objective}

Remember: This is an ongoing conversation. Reference what was discussed and keep it natural."""

        # Pre-load common instant responses
        self.instant_responses = {
            "greeting": f"Good day, this is {agent_name} from Cartrack Accounts.",
            "thanks": "Thank you for confirming.",
            "one_moment": "One moment please.",
            "understand": "I understand your concern.",
            "apologize": "I apologize for the confusion."
        }
        
        # Available tools for contextual usage
        self.available_tools = {
            # Verification tools
            "verify_client": client_call_verification,
            
            # Payment tools  
            "create_payment": create_payment_arrangement,
            "create_debicheck": create_debicheck_payment,
            "create_portal_payment": create_payment_arrangement_payment_portal,
            "generate_payment_url": generate_sms_payment_url,
            
            # Update tools
            "update_contact": update_client_contact_number,
            "update_email": update_client_email,
            "add_note": add_client_note,
            
            # Call management
            "save_disposition": save_call_disposition,
            "get_dispositions": get_disposition_types,
            "update_arrangements": update_payment_arrangements,
            
            # Helpers
            "get_date": date_helper,
            "current_time": get_current_date_time
        }
    
    def process_conversation_turn(self, state: UnifiedAgentState) -> Dict[str, Any]:
        """
        Process one conversation turn with full context awareness.
        
        This is the main method that handles:
        1. Intent detection from client message
        2. Conversation context analysis  
        3. Objective-based action planning
        4. Response generation
        5. Tool usage when appropriate
        6. State updates
        """
        
        try:
            # Increment turn counter
            new_turn_count = state.turn_count + 1
            
            # Get last client message for analysis
            last_client_message = self._get_last_human_message(state.get("messages", []))
            
            # Detect intent from client message
            intent_match = None
            if last_client_message:
                context = state.get_conversation_context()
                intent_match = self.intent_detector.detect_intent(last_client_message, context)
                logger.info(f"Detected intent: {intent_match.intent} (confidence: {intent_match.confidence:.2f})")
            
            # Update conversation context based on client message
            if last_client_message:
                self._update_conversation_context(state, last_client_message, intent_match)
            
            # Get next action from conversation manager
            action_plan = self.conversation_manager.get_next_action(state, intent_match)
            logger.info(f"Action plan: {action_plan['action']} for objective: {action_plan.get('objective', 'none')}")
            
            # Generate response based on action plan
            response_data = self._generate_contextual_response(state, action_plan, intent_match)
            
            # Execute any tools if needed
            tool_results = self._execute_contextual_tools(state, action_plan, intent_match)
            
            # Update state based on action and results
            state_updates = self._calculate_state_updates(
                state, action_plan, intent_match, tool_results, new_turn_count
            )
            
            # Add AI message to conversation
            ai_message = AIMessage(content=response_data["message"])
            current_messages = state.get("messages", [])
            updated_messages = current_messages + [ai_message]
            
            # Return complete state update
            return {
                "messages": [ai_message],  # LangGraph will handle appending with add_messages
                "turn_count": new_turn_count,
                "last_intent": intent_match.intent if intent_match else "none",
                "last_action": action_plan["action"],
                "current_objective": action_plan.get("objective", state.current_objective),
                **state_updates,
                **tool_results.get("state_updates", {})
            }
            
        except Exception as e:
            logger.error(f"Error in conversation turn: {e}")
            
            # Fallback response
            fallback_message = self._get_fallback_response(state)
            return {
                "messages": [AIMessage(content=fallback_message)],
                "turn_count": state.turn_count + 1,
                "last_action": "error_fallback"
            }
    
    def _update_conversation_context(
        self, 
        state: UnifiedAgentState, 
        client_message: str, 
        intent_match: IntentMatch
    ):
        """Update conversation context based on client message and intent"""
        
        # Analyze client mood and cooperation from message
        mood_analysis = self._analyze_client_mood(client_message, intent_match)
        
        # Update client mood if confidence is high
        if mood_analysis["confidence"] > 0.6:
            state.client_mood = mood_analysis["detected_mood"]
        
        # Update cooperation level with smoothing
        cooperation_change = mood_analysis.get("cooperation_change", 0)
        state.update_mood_and_rapport(state.client_mood, cooperation_change)
        
        # Extract and add concerns
        concerns = self._extract_concerns(client_message, intent_match)
        if concerns:
            # Use the custom reducer for concerns
            existing_concerns = state.client_concerns or []
            state.client_concerns = existing_concerns + concerns
        
        # Track mentioned topics
        topics = self._extract_topics(client_message, intent_match)
        state.mentioned_topics = list(set((state.mentioned_topics or []) + topics))
    
    def _generate_contextual_response(
        self, 
        state: UnifiedAgentState, 
        action_plan: Dict[str, Any], 
        intent_match: IntentMatch = None
    ) -> Dict[str, str]:
        """Generate response based on context and action plan"""
        
        action = action_plan["action"]
        
        # Check for pre-defined response first (for speed)
        if "message" in action_plan and len(action_plan["message"]) < 100:
            # Use pre-defined message for simple actions
            return {
                "message": action_plan["message"],
                "source": "predefined"
            }
        
        # Generate dynamic response using LLM for complex situations
        response = self._generate_llm_response(state, action_plan, intent_match)
        
        # Apply conversation enhancements
        enhanced_response = self._enhance_response(response, state, action_plan)
        
        return {
            "message": enhanced_response,
            "source": "llm_generated"
        }
    
    def _generate_llm_response(
        self, 
        state: UnifiedAgentState, 
        action_plan: Dict[str, Any], 
        intent_match: IntentMatch = None
    ) -> str:
        """Generate response using LLM with contextual prompt"""
        
        # Build context-aware prompt
        prompt_params = self._build_prompt_parameters(state, action_plan, intent_match)
        formatted_prompt = self.base_prompt_template.format(**prompt_params)
        
        # Build message list for LLM
        messages = [SystemMessage(content=formatted_prompt)]
        
        # Add recent conversation context (last 6 messages for efficiency)
        recent_messages = state.get("messages", [])[-6:]
        messages.extend(recent_messages)
        
        try:
            # Generate response
            response = self.model.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return action-specific fallback
            return self._get_action_fallback(action_plan["action"])
    
    def _execute_contextual_tools(
        self, 
        state: UnifiedAgentState, 
        action_plan: Dict[str, Any], 
        intent_match: IntentMatch = None
    ) -> Dict[str, Any]:
        """Execute tools based on conversation context and action plan"""
        
        action = action_plan["action"]
        tool_results = {"tools_called": [], "state_updates": {}}
        
        try:
            # Always add a note for significant interactions
            if action not in ["continue_conversation", "error_fallback"]:
                note_text = f"{action}: {action_plan.get('objective', 'general interaction')}"
                self._call_tool("add_note", {
                    "user_id": state.user_id,
                    "note_text": note_text
                }, tool_results)
            
            # Action-specific tool calls
            if action == "setup_debit_order":
                self._handle_debit_order_setup(state, action_plan, tool_results)
            
            elif action == "send_payment_link":
                self._handle_payment_link_generation(state, action_plan, tool_results)
            
            elif action == "update_contact_details":
                self._handle_contact_updates(state, action_plan, tool_results)
            
            elif action in ["escalate", "handle_cancellation", "call_completed"]:
                self._handle_call_disposition(state, action_plan, tool_results)
            
            elif action == "verify_name" or action == "verify_details":
                self._handle_verification_attempt(state, action_plan, tool_results)
            
            # Update payment arrangements if payment was secured
            if state.payment_secured or action_plan.get("payment_secured"):
                self._call_tool("update_arrangements", {
                    "user_id": state.user_id
                }, tool_results)
        
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            tool_results["error"] = str(e)
        
        return tool_results
    
    def _handle_debit_order_setup(self, state: UnifiedAgentState, action_plan: Dict[str, Any], tool_results: Dict[str, Any]):
        """Handle debit order setup with tools"""
        
        # Create payment arrangement
        arrangement_result = self._call_tool("create_payment", {
            "user_id": state.user_id,
            "pay_type_id": 1,  # Debit order
            "payment1": float(state.outstanding_amount.replace("R ", "").replace(",", "")),
            "date1": self._get_next_business_date(),
            "note": "Debit order arrangement via AI agent"
        }, tool_results)
        
        if arrangement_result and arrangement_result.get("success"):
            tool_results["state_updates"]["payment_secured"] = True
            tool_results["state_updates"]["payment_method_preference"] = "debit_order"
            
            # Add completion to objectives
            completed_objectives = state.completed_objectives or []
            if ConversationObjective.PAYMENT_SECURED.value not in completed_objectives:
                tool_results["state_updates"]["completed_objectives"] = completed_objectives + [ConversationObjective.PAYMENT_SECURED.value]
    
    def _handle_payment_link_generation(self, state: UnifiedAgentState, action_plan: Dict[str, Any], tool_results: Dict[str, Any]):
        """Handle payment link generation"""
        
        amount = float(state.outstanding_amount.replace("R ", "").replace(",", ""))
        
        # Generate payment URL
        url_result = self._call_tool("generate_payment_url", {
            "user_id": state.user_id,
            "amount": amount,
            "optional_reference": f"AI_AGENT_{datetime.now().strftime('%Y%m%d_%H%M')}"
        }, tool_results)
        
        if url_result and url_result.get("success"):
            tool_results["state_updates"]["payment_method_preference"] = "online"
            # Note: Don't mark as secured until actual payment is made
    
    def _handle_contact_updates(self, state: UnifiedAgentState, action_plan: Dict[str, Any], tool_results: Dict[str, Any]):
        """Handle contact detail updates"""
        
        # This would typically involve asking for new details first
        # For now, just mark the objective as attempted
        completed_objectives = state.completed_objectives or []
        if ConversationObjective.CONTACT_UPDATED.value not in completed_objectives:
            tool_results["state_updates"]["completed_objectives"] = completed_objectives + [ConversationObjective.CONTACT_UPDATED.value]
    
    def _handle_call_disposition(self, state: UnifiedAgentState, action_plan: Dict[str, Any], tool_results: Dict[str, Any]):
        """Handle call disposition for call endings"""
        
        # Determine disposition based on action
        disposition_mapping = {
            "escalate": "ESCALATED",
            "handle_cancellation": "CANCELLATION_REQUEST", 
            "call_completed": "COMPLETED",
            "verification_failed": "VERIFICATION_FAILED",
            "wrong_person": "WRONG_PERSON"
        }
        
        disposition = disposition_mapping.get(action_plan["action"], "COMPLETED")
        
        # Save call disposition
        self._call_tool("save_disposition", {
            "client_id": state.user_id,
            "disposition_type_id": disposition,
            "note_text": f"Call {disposition.lower()} via AI agent"
        }, tool_results)
        
        # Mark call as ended
        tool_results["state_updates"]["call_ended"] = True
        tool_results["state_updates"]["call_outcome"] = disposition.lower()
        
        # Set escalation/cancellation flags
        if action_plan["action"] == "escalate":
            tool_results["state_updates"]["escalation_requested"] = True
        elif action_plan["action"] == "handle_cancellation":
            tool_results["state_updates"]["cancellation_requested"] = True
    
    def _handle_verification_attempt(self, state: UnifiedAgentState, action_plan: Dict[str, Any], tool_results: Dict[str, Any]):
        """Handle verification attempts"""
        
        field = action_plan.get("field", "name")
        current_attempts = state.verification_attempts.get(field, 0)
        new_attempts = current_attempts + 1
        
        # Update verification attempts
        updated_attempts = state.verification_attempts.copy()
        updated_attempts[field] = new_attempts
        tool_results["state_updates"]["verification_attempts"] = updated_attempts
    
    def _call_tool(self, tool_name: str, params: Dict[str, Any], tool_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a database tool and track results"""
        
        if tool_name not in self.available_tools:
            logger.warning(f"Tool {tool_name} not available")
            return None
        
        try:
            tool = self.available_tools[tool_name]
            result = tool.invoke(params)
            
            tool_results["tools_called"].append({
                "tool": tool_name,
                "params": params,
                "success": True,
                "result": result
            })
            
            logger.info(f"Tool {tool_name} called successfully")
            return result
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            tool_results["tools_called"].append({
                "tool": tool_name,
                "params": params,
                "success": False,
                "error": str(e)
            })
            return None
    
    def _calculate_state_updates(
        self,
        state: UnifiedAgentState,
        action_plan: Dict[str, Any],
        intent_match: IntentMatch,
        tool_results: Dict[str, Any],
        new_turn_count: int
    ) -> Dict[str, Any]:
        """Calculate state updates based on action results"""
        
        updates = {}
        
        # Handle verification status updates based on intent
        if intent_match:
            if intent_match.intent == "identity_confirmation":
                if state.name_verification == VerificationStatus.PENDING.value:
                    updates["name_verification"] = VerificationStatus.VERIFIED.value
                    
                    # Check if identity verification objective is complete
                    if state.details_verification == VerificationStatus.VERIFIED.value:
                        completed_objectives = state.completed_objectives or []
                        if ConversationObjective.IDENTITY_VERIFICATION.value not in completed_objectives:
                            updates["completed_objectives"] = completed_objectives + [ConversationObjective.IDENTITY_VERIFICATION.value]
            
            elif intent_match.intent == "identity_denial":
                updates["name_verification"] = VerificationStatus.WRONG_PERSON.value
                updates["call_ended"] = True
            
            elif intent_match.intent == "payment_agreement":
                # Don't mark as secured until actual arrangement is made
                pass
            
            elif intent_match.intent in ["escalation", "cancellation"]:
                updates["call_ended"] = True
        
        # Update objectives based on action completion
        action = action_plan["action"]
        if action == "explain_account":
            completed_objectives = state.completed_objectives or []
            if ConversationObjective.ACCOUNT_EXPLANATION.value not in completed_objectives:
                updates["completed_objectives"] = completed_objectives + [ConversationObjective.ACCOUNT_EXPLANATION.value]
        
        elif action == "offer_referrals":
            completed_objectives = state.completed_objectives or []
            if ConversationObjective.REFERRALS_OFFERED.value not in completed_objectives:
                updates["completed_objectives"] = completed_objectives + [ConversationObjective.REFERRALS_OFFERED.value]
        
        return updates
    
    def _build_prompt_parameters(
        self,
        state: UnifiedAgentState,
        action_plan: Dict[str, Any],
        intent_match: IntentMatch = None
    ) -> Dict[str, str]:
        """Build parameters for prompt formatting"""
        
        # Get conversation strategy
        if hasattr(state, 'get_strategy_context'):
            strategy = state.get_strategy_context()
        else:
            strategy = {"tone": "professional", "approach": "solution-focused", "pace": "normal"}
        
        # Build task description based on action
        task_descriptions = {
            "verify_name": f"Confirm you're speaking with {state.client_name}",
            "verify_details": "Get ID number for security verification", 
            "explain_account": f"Explain account situation - owes {state.outstanding_amount}",
            "request_payment": f"Secure payment arrangement for {state.outstanding_amount}",
            "setup_debit_order": f"Confirm debit order setup for {state.outstanding_amount}",
            "send_payment_link": f"Explain payment link process for {state.outstanding_amount}",
            "escalate": "Handle escalation request professionally",
            "handle_cancellation": "Explain cancellation process and requirements",
            "continue_conversation": "Continue natural conversation toward resolution"
        }
        
        return {
            "agent_name": self.agent_name,
            "client_name": state.client_name,
            "outstanding_amount": state.outstanding_amount,
            "client_mood": state.client_mood,
            "rapport_level": state.rapport_level,
            "current_objective": action_plan.get("objective", state.current_objective),
            "conversation_context": self._build_conversation_context(state),
            "conversation_strategy": strategy.get("approach", "solution-focused"),
            "conversation_tone": strategy.get("tone", "professional"),
            "specific_task": task_descriptions.get(action_plan["action"], "Continue conversation naturally")
        }
    
    def _build_conversation_context(self, state: UnifiedAgentState) -> str:
        """Build conversation context string"""
        
        context_parts = []
        
        # Verification status
        if not state.is_verified():
            if state.name_verification != VerificationStatus.VERIFIED.value:
                context_parts.append("Name verification needed")
            if state.details_verification != VerificationStatus.VERIFIED.value:
                context_parts.append("Details verification needed")
        else:
            context_parts.append("Client verified")
        
        # Recent concerns
        if state.client_concerns:
            recent_concerns = state.client_concerns[-2:]  # Last 2 concerns
            context_parts.append(f"Concerns: {', '.join(recent_concerns)}")
        
        # Progress indicators
        objectives_completed = len(state.completed_objectives) if state.completed_objectives else 0
        context_parts.append(f"Progress: {objectives_completed} objectives completed")
        
        return " | ".join(context_parts) if context_parts else "Initial conversation"
    
    def _enhance_response(self, response: str, state: UnifiedAgentState, action_plan: Dict[str, Any]) -> str:
        """Apply final enhancements to response"""
        
        # Remove robotic language
        robotic_replacements = {
            "I need to": "Let me",
            "For security purposes": "For security,",
            "In order to": "To",
            "It is necessary to": "I need to",
            "I am required to": "I need to"
        }
        
        for robotic, natural in robotic_replacements.items():
            response = response.replace(robotic, natural)
        
        # Add empathy for difficult moods
        if (state.client_mood in [ClientMood.ANGRY.value, ClientMood.FINANCIAL_STRESS.value] and 
            state.rapport_level < 0.4):
            empathy_phrases = ["I understand this is difficult", "I hear your concern", "I appreciate your patience"]
            if not any(phrase in response.lower() for phrase in ["understand", "hear", "appreciate"]):
                response = f"{empathy_phrases[0]}. {response}"
        
        # Ensure response isn't too long
        if len(response.split()) > 40:
            # Try to shorten by removing redundancy
            sentences = response.split('. ')
            if len(sentences) > 2:
                response = '. '.join(sentences[:2]) + '.'
        
        return response
    
    def _analyze_client_mood(self, message: str, intent_match: IntentMatch) -> Dict[str, Any]:
        """Analyze client mood from message and intent"""
        
        mood_indicators = {
            ClientMood.ANGRY.value: ["ridiculous", "harassment", "sick", "fed up", "angry"],
            ClientMood.CONFUSED.value: ["don't understand", "confused", "what", "explain"],
            ClientMood.COOPERATIVE.value: ["okay", "yes", "fine", "sure", "understand"],
            ClientMood.RESISTANT.value: ["can't", "won't", "refuse", "no"],
            ClientMood.FINANCIAL_STRESS.value: ["can't afford", "no money", "broke", "tight"],
            ClientMood.SUSPICIOUS.value: ["scam", "fraud", "who are you", "prove"]
        }
        
        msg_lower = message.lower()
        mood_scores = {}
        
        for mood, indicators in mood_indicators.items():
            score = sum(0.2 for indicator in indicators if indicator in msg_lower)
            mood_scores[mood] = min(score, 1.0)
        
        # Boost from intent
        if intent_match:
            if intent_match.intent == "escalation":
                mood_scores[ClientMood.ANGRY.value] += 0.5
            elif intent_match.intent == "financial_hardship":
                mood_scores[ClientMood.FINANCIAL_STRESS.value] += 0.5
            elif intent_match.intent == "cooperation":
                mood_scores[ClientMood.COOPERATIVE.value] += 0.5
        
        best_mood = max(mood_scores.items(), key=lambda x: x[1]) if mood_scores else (ClientMood.NEUTRAL.value, 0.1)
        
        # Calculate cooperation change
        positive_indicators = ["yes", "okay", "sure", "fine", "understand"]
        negative_indicators = ["no", "can't", "won't", "refuse"]
        
        positive_count = sum(1 for ind in positive_indicators if ind in msg_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in msg_lower)
        
        cooperation_change = 0.0
        if positive_count > negative_count:
            cooperation_change = 0.2
        elif negative_count > positive_count:
            cooperation_change = -0.2
        
        return {
            "detected_mood": best_mood[0],
            "confidence": best_mood[1],
            "cooperation_change": cooperation_change
        }
    
    def _extract_concerns(self, message: str, intent_match: IntentMatch) -> List[str]:
        """Extract concerns from client message"""
        
        concerns = []
        msg_lower = message.lower()
        
        concern_patterns = {
            "amount_concern": ["how much", "too expensive", "too high"],
            "timing_concern": ["when", "due date", "deadline"],
            "legitimacy_concern": ["scam", "fraud", "real", "prove"],
            "fee_concern": ["fee", "charge", "extra cost"],
            "service_concern": ["what service", "what for", "why"],
            "payment_method_concern": ["how to pay", "payment options"]
        }
        
        for concern_type, patterns in concern_patterns.items():
            if any(pattern in msg_lower for pattern in patterns):
                concerns.append(concern_type)
        
        return concerns
    
    def _extract_topics(self, message: str, intent_match: IntentMatch) -> List[str]:
        """Extract discussed topics from message"""
        
        topics = []
        msg_lower = message.lower()
        
        topic_keywords = {
            "payment": ["pay", "payment", "money", "amount"],
            "service": ["cartrack", "service", "device", "tracker"],
            "account": ["account", "bill", "statement"],
            "cancellation": ["cancel", "stop", "end"],
            "contact": ["phone", "email", "address", "contact"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in msg_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _get_last_human_message(self, messages: List[BaseMessage]) -> str:
        """Get last human message content"""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        return ""
    
    def _get_fallback_response(self, state: UnifiedAgentState) -> str:
        """Get fallback response for errors"""
        if not state.is_verified():
            return f"Let me help you. May I confirm I'm speaking with {state.client_name}?"
        else:
            return "I apologize for any confusion. How can I help you with your Cartrack account?"
    
    def _get_action_fallback(self, action: str) -> str:
        """Get fallback response for specific actions"""
        fallbacks = {
            "verify_name": "May I confirm your name please?",
            "verify_details": "I need to verify your details for security.",
            "explain_account": "I'm calling about your Cartrack account.",
            "request_payment": "Can we arrange payment today?",
            "escalate": "Let me connect you with a supervisor.",
            "handle_cancellation": "I can help with the cancellation process."
        }
        return fallbacks.get(action, "How can I help you today?")
    
    def _get_next_business_date(self) -> str:
        """Get next business date for payment arrangements"""
        # Simplified - in practice would use date_helper tool
        from datetime import datetime, timedelta
        tomorrow = datetime.now() + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")
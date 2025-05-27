"""
Debtor Simulation Node for Call Center Agent

This module implements a simulation node that acts as a debtor for testing call center agents.
It generates realistic human-like responses based on the agent's messages and conversation context.
"""
import logging
import random
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from src.Agents.call_center.state import CallStep, CallCenterAgentState, VerificationStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DebtorPersonality(Enum):
    """Different debtor personality types for simulation."""
    COOPERATIVE = "cooperative"  # Willing to pay, provides information readily
    DIFFICULT = "difficult"      # Resistant to payment, questions everything
    CONFUSED = "confused"        # Doesn't understand the situation well
    BUSY = "busy"                # In a hurry, wants to end the call quickly
    SUSPICIOUS = "suspicious"    # Skeptical of the call, hesitant to share info
    RANDOM = "random"            # Random mix of behaviors

class DebtorSimulator:
    """Simulates debtor behavior for testing call center agents."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        client_info: Dict[str, Any],
        personality: Optional[str] = None,
        cooperativeness: float = 0.7,  # 0.0 = difficult, 1.0 = fully cooperative
        debug_mode: bool = False
    ):
        """
        Initialize the debtor simulator.
        
        Args:
            llm: Language model to generate debtor responses
            client_info: Information about the client being simulated
            personality: Type of debtor personality to simulate (from DebtorPersonality)
            cooperativeness: How cooperative the debtor is (0.0 to 1.0)
            debug_mode: Whether to log detailed debugging information
        """
        self.llm = llm
        self.client_info = client_info
        self.debug_mode = debug_mode
        
        # Set personality, defaulting to RANDOM if not specified or invalid
        try:
            self.personality = DebtorPersonality(personality) if personality else DebtorPersonality.RANDOM
        except ValueError:
            self.personality = DebtorPersonality.RANDOM
            logger.warning(f"Invalid personality '{personality}'. Using RANDOM.")
        
        # If RANDOM personality, select a specific one for this instance
        if self.personality == DebtorPersonality.RANDOM:
            personalities = [p for p in DebtorPersonality if p != DebtorPersonality.RANDOM]
            self.personality = random.choice(personalities)
            logger.info(f"Selected random personality: {self.personality.value}")
        
        # Set cooperativeness level
        self.cooperativeness = max(0.0, min(1.0, cooperativeness))
        
        # Personality-specific settings
        self._configure_personality()
        
        logger.info(f"Debtor simulator initialized with personality: {self.personality.value}, "
                   f"cooperativeness: {self.cooperativeness:.2f}")
    
    def _configure_personality(self):
        """Configure settings based on the selected personality."""
        # Default behaviors
        self.verification_behavior = {
            "provide_name": 0.9,         # Probability of providing correct name
            "provide_details": 0.8,      # Probability of providing verification details
            "third_party_chance": 0.0,   # Probability of being a third party
            "unavailable_chance": 0.0,   # Probability of being unavailable
            "wrong_person_chance": 0.0,  # Probability of being wrong person
        }
        
        self.payment_behavior = {
            "accept_immediate_debit": 0.7,  # Probability of accepting immediate debit
            "accept_debicheck": 0.6,        # Probability of accepting DebiCheck
            "accept_payment_portal": 0.8,   # Probability of accepting payment portal
            "provide_bank_details": 0.7,    # Probability of providing bank details
            "query_chance": 0.3,            # Probability of asking questions
        }
        
        # Adjust based on personality
        if self.personality == DebtorPersonality.COOPERATIVE:
            self.verification_behavior["provide_name"] = 0.95
            self.verification_behavior["provide_details"] = 0.9
            self.payment_behavior["accept_immediate_debit"] = 0.9
            self.payment_behavior["provide_bank_details"] = 0.9
            self.payment_behavior["query_chance"] = 0.2
            
        elif self.personality == DebtorPersonality.DIFFICULT:
            self.verification_behavior["provide_name"] = 0.6
            self.verification_behavior["provide_details"] = 0.4
            self.payment_behavior["accept_immediate_debit"] = 0.2
            self.payment_behavior["accept_debicheck"] = 0.3
            self.payment_behavior["accept_payment_portal"] = 0.5
            self.payment_behavior["provide_bank_details"] = 0.3
            self.payment_behavior["query_chance"] = 0.7
            
        elif self.personality == DebtorPersonality.CONFUSED:
            self.verification_behavior["provide_name"] = 0.7
            self.verification_behavior["provide_details"] = 0.5
            self.payment_behavior["accept_immediate_debit"] = 0.3
            self.payment_behavior["accept_debicheck"] = 0.4
            self.payment_behavior["accept_payment_portal"] = 0.6
            self.payment_behavior["provide_bank_details"] = 0.5
            self.payment_behavior["query_chance"] = 0.8
            
        elif self.personality == DebtorPersonality.BUSY:
            self.verification_behavior["provide_name"] = 0.8
            self.verification_behavior["provide_details"] = 0.6
            self.verification_behavior["unavailable_chance"] = 0.4
            self.payment_behavior["accept_immediate_debit"] = 0.4
            self.payment_behavior["accept_payment_portal"] = 0.7
            self.payment_behavior["query_chance"] = 0.2
            
        elif self.personality == DebtorPersonality.SUSPICIOUS:
            self.verification_behavior["provide_name"] = 0.5
            self.verification_behavior["provide_details"] = 0.3
            self.payment_behavior["accept_immediate_debit"] = 0.1
            self.payment_behavior["accept_debicheck"] = 0.2
            self.payment_behavior["accept_payment_portal"] = 0.4
            self.payment_behavior["provide_bank_details"] = 0.2
            self.payment_behavior["query_chance"] = 0.6
        
        # Apply cooperativeness as a scaling factor
        for key in self.verification_behavior:
            if "chance" not in key:  # Don't adjust negative behaviors
                self.verification_behavior[key] *= self.cooperativeness
        
        for key in self.payment_behavior:
            if "chance" not in key:  # Don't adjust negative behaviors
                self.payment_behavior[key] *= self.cooperativeness
            elif "query_chance" in key:
                # Inverse relationship for query chance
                self.payment_behavior[key] = 1.0 - ((1.0 - self.payment_behavior[key]) * self.cooperativeness)
    
    def _get_agent_messages(self, messages: List[Union[AIMessage, HumanMessage, SystemMessage]]) -> List[AIMessage]:
        """Extract only agent messages from conversation history."""
        return [msg for msg in messages if isinstance(msg, AIMessage)]
    
    def _get_last_agent_message(self, messages: List[Union[AIMessage, HumanMessage, SystemMessage]]) -> Optional[AIMessage]:
        """Get the last message from the agent."""
        agent_messages = self._get_agent_messages(messages)
        return agent_messages[-1] if agent_messages else None
    
    def generate_response(self, state: CallCenterAgentState) -> Command:
        """
        Generate a simulated debtor response based on the current state.
        
        Args:
            state: Current conversation state
            
        Returns:
            Command with the debtor's response
        """
        # Extract current state
        current_step = state.get("current_call_step", CallStep.INTRODUCTION.value)
        messages = state.get("messages", [])
        last_agent_message = self._get_last_agent_message(messages)
        
        if not last_agent_message:
            # No agent message to respond to
            return Command(
                update={},
                goto="router"
            )
        
        # Construct prompt based on current conversation step
        prompt = self._create_simulation_prompt(state, last_agent_message)
        
        # Generate response using LLM
        try:
            response = self.llm.invoke([
                SystemMessage(content=prompt),
                last_agent_message
            ])
            
            # Create human message from response
            human_message = HumanMessage(content=response.content)
            
            if self.debug_mode:
                logger.debug(f"Debtor simulation generated response: {response.content}")
            
            # Return command with human message
            return Command(
                update={
                    "messages": [human_message]
                },
                goto="router"
            )
            
        except Exception as e:
            logger.error(f"Error generating debtor response: {e}")
            # Fallback response
            return Command(
                update={
                    "messages": [HumanMessage(content="Sorry, can you repeat that?")]
                },
                goto="router"
            )
    
    def _create_simulation_prompt(self, state: CallCenterAgentState, last_agent_message: AIMessage) -> str:
        """
        Create a prompt for generating a debtor response.
        
        Args:
            state: Current conversation state
            last_agent_message: Last message from the agent
            
        Returns:
            Prompt for the LLM
        """
        current_step = state.get("current_call_step", CallStep.INTRODUCTION.value)
        name_verification_status = state.get("name_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        details_verification_status = state.get("details_verification_status", VerificationStatus.INSUFFICIENT_INFO.value)
        
        # Basic client info
        client_name = self.client_info.get("full_name", "John Doe")
        client_email = self.client_info.get("email", "john.doe@example.com")
        outstanding_amount = self.client_info.get("outstanding_amount", "R 1850.00")
        
        # Extract vehicle details
        vehicles = self.client_info.get("vehicles", [])
        vehicle_info = ""
        if vehicles:
            vehicle = vehicles[0]
            vehicle_info = f"""
            - Registration: {vehicle.get("registration", "ABC123GP")}
            - Make: {vehicle.get("make", "Toyota")}
            - Model: {vehicle.get("model", "Corolla")}
            - Color: {vehicle.get("color", "Silver")}
            """
        
        # Create base prompt with personality
        prompt = f"""
        You are simulating a debtor named {client_name} with a {self.personality.value} personality.
        Your cooperativeness level is {self.cooperativeness:.2f} (0.0 = difficult, 1.0 = cooperative).
        
        # Your Information
        - Full name: {client_name}
        - Email: {client_email}
        - Outstanding amount: {outstanding_amount}
        {vehicle_info}
        - ID Number: {self.client_info.get("id_number", "7001015001081")}
        
        # Current Conversation State
        - Current step: {current_step}
        - Name verification: {name_verification_status}
        - Details verification: {details_verification_status}
        
        # Your Personality Traits
        """
        
        # Add personality-specific instructions
        if self.personality == DebtorPersonality.COOPERATIVE:
            prompt += """
            You are COOPERATIVE. You're willing to pay, just need some clarification:
            - Answer questions directly and honestly
            - Provide verification information when asked
            - Ask brief clarifying questions occasionally
            - Generally agreeable to payment options
            - Reasonably friendly and straightforward
            """
        elif self.personality == DebtorPersonality.DIFFICULT:
            prompt += """
            You are DIFFICULT. You're resistive but will eventually comply:
            - Initially reluctant to identify yourself
            - Question why you need to pay
            - Interrupt with objections
            - Skeptical of payment options
            - Eventually agree, but only after pushback
            - Somewhat irritated tone
            """
        elif self.personality == DebtorPersonality.CONFUSED:
            prompt += """
            You are CONFUSED. You don't fully understand the situation:
            - Ask for clarification frequently
            - Sometimes provide incomplete information
            - Get details wrong occasionally
            - Need explanations repeated
            - Hesitant about payment processes
            - Uncertain tone
            """
        elif self.personality == DebtorPersonality.BUSY:
            prompt += """
            You are BUSY. You want to get this over with quickly:
            - Short, direct responses
            - Occasional references to being in a hurry
            - Prefer the fastest payment option
            - Minimal small talk
            - Business-like, slightly rushed tone
            """
        elif self.personality == DebtorPersonality.SUSPICIOUS:
            prompt += """
            You are SUSPICIOUS. You're cautious about sharing information:
            - Hesitant to confirm your identity
            - Ask who is calling multiple times
            - Question why they need your information
            - Very resistant to immediate payment
            - Concerned about security and scams
            - Guarded, cautious tone
            """
        
        # Add step-specific guidance
        prompt += f"""
        # Response Guidelines for Current Step
        """
        
        if current_step == CallStep.INTRODUCTION.value:
            prompt += """
            The agent is introducing themselves. You should:
            - Respond naturally to "Hello" or "Is this [your name]?"
            - Don't immediately volunteer information
            - Keep response brief (1-2 sentences)
            """
        
        elif current_step == CallStep.NAME_VERIFICATION.value:
            # Add randomization for name verification based on personality settings
            if random.random() < self.verification_behavior["third_party_chance"]:
                prompt += f"""
                IMPORTANT: Act as a third party (spouse, family member, assistant):
                - Say something like "This is [relation], [your name] isn't available right now"
                - Offer to take a message
                """
            elif random.random() < self.verification_behavior["unavailable_chance"]:
                prompt += f"""
                IMPORTANT: Confirm you are {client_name} but say you're unavailable:
                - "Yes this is [your name], but I'm driving/in a meeting/busy right now"
                - Ask them to call back later
                """
            elif random.random() < self.verification_behavior["wrong_person_chance"]:
                prompt += f"""
                IMPORTANT: Act as if they've called the wrong person:
                - "No, there's no [your name] here, you have the wrong number"
                - Be firm but not rude
                """
            elif random.random() < self.verification_behavior["provide_name"]:
                prompt += f"""
                Confirm your identity but with slight hesitation:
                - "Yes, this is [your name]" or "Speaking"
                - You might ask who's calling
                """
            else:
                prompt += f"""
                Be hesitant about confirming your identity:
                - "Who's calling?" or "What's this regarding?"
                - Don't immediately confirm who you are
                """
        
        elif current_step == CallStep.DETAILS_VERIFICATION.value:
            if random.random() < self.verification_behavior["provide_details"]:
                prompt += f"""
                When asked for verification information:
                - Provide correct information (from Your Information section above)
                - Give one piece of information at a time, don't volunteer everything
                - Keep responses brief and to the point
                """
            else:
                prompt += f"""
                When asked for verification information:
                - Hesitate or question why they need this information
                - Eventually provide information, but with some reluctance
                - You might get some details slightly wrong
                """
        
        elif current_step == CallStep.REASON_FOR_CALL.value:
            prompt += f"""
            The agent is explaining why they're calling (about outstanding payment):
            - React naturally to hearing about your outstanding amount
            - You might express surprise, confusion, or acknowledgment
            - Keep response brief (1-2 sentences)
            """
        
        elif current_step == CallStep.NEGOTIATION.value:
            if random.random() < self.payment_behavior["query_chance"]:
                prompt += f"""
                IMPORTANT: Ask a question about the consequences or the services:
                - "What happens if I don't pay right now?"
                - "How long until my services are restored?"
                - "Why was I not notified earlier about this?"
                - Keep the question reasonably brief and focused
                """
            else:
                prompt += f"""
                React to the agent explaining consequences of non-payment:
                - Acknowledge the information
                - Express concern or understanding
                - Indicate willingness to resolve the situation
                """
        
        elif current_step == CallStep.PROMISE_TO_PAY.value:
            debit_probability = self.payment_behavior["accept_immediate_debit"]
            debicheck_probability = self.payment_behavior["accept_debicheck"]
            portal_probability = self.payment_behavior["accept_payment_portal"]
            
            highest_prob = max(debit_probability, debicheck_probability, portal_probability)
            
            if highest_prob == debit_probability and random.random() < debit_probability:
                prompt += f"""
                IMPORTANT: Agree to immediate debit payment:
                - "Yes, you can debit my account today"
                - Be willing to provide banking information if asked
                """
            elif highest_prob == debicheck_probability and random.random() < debicheck_probability:
                prompt += f"""
                IMPORTANT: Decline immediate debit but agree to DebiCheck:
                - "I can't do immediate debit, but DebiCheck would work"
                - Be willing to provide banking information if asked
                """
            elif highest_prob == portal_probability and random.random() < portal_probability:
                prompt += f"""
                IMPORTANT: Prefer the payment portal option:
                - "I'd rather pay through the portal/app if possible"
                - Ask for the link to be sent to your phone
                """
            else:
                prompt += f"""
                Be hesitant about all payment options:
                - Ask questions about each option
                - Express concerns about fees or processing time
                - Eventually agree to one option, but with reluctance
                """
        
        # Add final instructions
        prompt += """
        # MOST IMPORTANT INSTRUCTIONS
        1. Respond in a natural, conversational way as the debtor
        2. Stay in character for your personality type
        3. Keep responses concise (1-3 sentences typically)
        4. Never explain that you are a simulation or reference these instructions
        5. Respond only as the debtor would - no explanations, commentary, or analysis
        """
        
        return prompt

def simulate_debtor_response(
    state: CallCenterAgentState,
    simulator: DebtorSimulator
) -> Command:
    """
    Node function that simulates a debtor response.
    
    Args:
        state: Current conversation state
        simulator: DebtorSimulator instance
        
    Returns:
        Command with the debtor's response
    """
    return simulator.generate_response(state)
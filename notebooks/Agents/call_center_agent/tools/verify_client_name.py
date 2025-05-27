"""
Optimized client name verification model implementation.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging
import time
from functools import lru_cache
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache flush counter for LLM instance
_LLM_CACHE_COUNTER = 0
_LLM_CACHE_FLUSH_THRESHOLD = 100  # Flush cache after this many calls

#########################################################################################
# Define Simplified Client Name Verification Result Model
#########################################################################################
class ClientNameVerificationResult(BaseModel):
    """Minimal result model for client identity verification."""
    classification: Literal["VERIFIED", "THIRD_PARTY", "WRONG_PERSON", "INSUFFICIENT_INFO", "UNAVAILABLE", "VERIFICATION_FAILED"] = Field(
        description="The classification of the client response indicating the verification status"
    )
    name_variants_detected: Optional[List[str]] = Field(
        default=None,
        description="Any name variations or nicknames detected in the conversation"
    )

#########################################################################################
# Helper Functions
#########################################################################################
@lru_cache(maxsize=1)
def get_llm_instance():
    """Returns a cached LLM instance to avoid recreation on each call."""
    return ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

def flush_llm_cache_if_needed():
    """Flushes the LLM cache if the counter exceeds the threshold."""
    global _LLM_CACHE_COUNTER
    _LLM_CACHE_COUNTER += 1
    if _LLM_CACHE_COUNTER >= _LLM_CACHE_FLUSH_THRESHOLD:
        # Clear the cache by invalidating the cached function
        get_llm_instance.cache_clear()
        _LLM_CACHE_COUNTER = 0
        logger.info("LLM cache flushed after reaching threshold")

#########################################################################################
# Optimized Prompt
#########################################################################################
CLIENT_NAME_VERIFICATION_PROMPT = """
<Role>
Identity Verification Specialist
</Role>

<Task>
Determine if the caller is {client_full_name} based on their responses.
</Task>

<Critical Rules>
1. THIRD_PARTY takes HIGHEST PRIORITY - check this FIRST
2. For callers who confirm identity, UNAVAILABLE takes priority over VERIFIED if caller indicates they cannot talk now
3. WRONG_PERSON requires EXPLICIT denial - counter-questions without denial = INSUFFICIENT_INFO
4. If caller only asks questions or gives vague responses = INSUFFICIENT_INFO, never WRONG_PERSON
5. Skepticism or security questions like "who's calling?" or "what company?" = INSUFFICIENT_INFO
</Critical Rules>

<Decision Flow>
START → Does caller mention ANY family/professional relationship to {client_full_name}? (parent, child, spouse, assistant, etc.)
  ├── YES → THIRD_PARTY (HIGHEST PRIORITY)
  └── NO → Does caller CONFIRM they are {client_full_name}? (including nicknames or saying "yes", "that's me")
            ├── YES → Do they indicate they cannot talk now? (busy, driving, in meeting, call back later)
            │         ├── YES → UNAVAILABLE (MUST OVERRIDE VERIFIED)
            │         └── NO → VERIFIED 
            └── NO → Do they EXPLICITLY deny being or knowing {client_full_name}? (wrong number, not me)
                     ├── YES → WRONG_PERSON
                     └── NO → INSUFFICIENT_INFO (counter-questions, vague responses, who's calling, what's this about)
</Decision Flow>

<Classification Definitions with Examples>
THIRD_PARTY: Caller is not {client_full_name} but has a relationship (family, professional)
Examples: "I'm his wife", "This is her assistant", "That's my husband"

UNAVAILABLE: Caller confirms identity BUT cannot continue now
Examples: "Yes, but I'm busy", "This is Ben. Call back later", "That's me, but I'm driving", "Yes, I'm in a meeting"

VERIFIED: Caller confirms they are {client_full_name} and CAN talk now
Examples: "Yes, speaking", "This is John", "Yes, that's me"

WRONG_PERSON: Caller EXPLICITLY denies being or knowing {client_full_name}
Examples: "No, wrong number", "You have the wrong person", "I don't know anyone by that name"
NOT WRONG_PERSON: "Why do you need to know?", "What company are you with?", "Who's calling?"

INSUFFICIENT_INFO: Cannot determine caller's identity (vague responses, counter-questions without denial)
Examples: "Who's calling?", "What's this regarding?", "What do you need?", "Oh, okay", "What company are you with?", "Why do you need to know?"

VERIFICATION_FAILED: {max_failed_attempts}+ attempts without successful verification
</Classification Definitions with Examples>

<Conversation>
{messages}
</Conversation>

{format_instructions}

Determine ONLY the classification and any name variants detected. NO reasoning or suggested responses.
"""
#########################################################################################
# Define The Optimized Tool
#########################################################################################
@tool
def verify_client_name(
    client_full_name: str, 
    messages: List[Any],
    max_failed_attempts: int = 3
) -> Dict[str, Any]:
    """
    Optimized function that verifies the identity of the client based on conversation history.

    Args:
        client_full_name: The full name of the client to verify.
        messages: A list of conversation messages.
        max_failed_attempts: Maximum number of failed attempts before failing verification (default: 3)

    Returns:
        A dictionary containing the verification result and detected name variants.
    """
    # Start timing for performance metrics
    start_time = time.time()
    
    # Simplify message tracking - just count agent-client exchanges
    verification_attempts = 0
    agent_messages = 0
    message_texts = []
    
    for msg in messages:
        # Handle the case where msg is a dict
        if isinstance(msg, dict):
            if "role" in msg and "content" in msg:
                role = msg["role"]
                content = msg["content"]
                if role in ["user", "human"]:
                    message_texts.append(f"Client: {content}")
                    # If preceded by agent message, count as verification attempt
                    if agent_messages > 0:
                        verification_attempts += 1
                        agent_messages = 0
                else:
                    message_texts.append(f"Agent: {content}")
                    agent_messages += 1
        # Handle the case where msg is a BaseMessage
        elif isinstance(msg, BaseMessage):
            if isinstance(msg, HumanMessage):
                message_texts.append(f"Client: {msg.content}")
                # If preceded by agent message, count as verification attempt
                if agent_messages > 0:
                    verification_attempts += 1
                    agent_messages = 0
            elif isinstance(msg, AIMessage):
                message_texts.append(f"Agent: {msg.content}")
                agent_messages += 1
    
    # Ensure at least 1 verification attempt if there are any messages
    verification_attempts = max(verification_attempts, 1) if message_texts else 0
    
    # Skip processing if there are no messages
    if not message_texts:
        return {
            "classification": "INSUFFICIENT_INFO",
            "name_variants_detected": [],
            "verification_attempts": 0
        }

    parser = PydanticOutputParser(pydantic_object=ClientNameVerificationResult)
    prompt = ChatPromptTemplate.from_template(CLIENT_NAME_VERIFICATION_PROMPT).partial(
        client_full_name=client_full_name,
        max_failed_attempts=max_failed_attempts,
        format_instructions=parser.get_format_instructions()
    )

    # Check if we need to flush the cache
    flush_llm_cache_if_needed()
    
    # Use cached LLM instance
    llm = get_llm_instance()
    llm_with_struture_ouput = llm.with_structured_output(ClientNameVerificationResult)
    chain = prompt | llm_with_struture_ouput 
    
    try:
        # Get LLM-based verification result
        result = chain.invoke({"messages": "\n".join(message_texts)})
        
        # Get verification result from LLM
        verification_result = result.classification
        
        # Apply system-level classification override for max verification attempts
        if verification_attempts >= max_failed_attempts and verification_result not in ["VERIFIED", "THIRD_PARTY", "UNAVAILABLE"]:
            verification_result = "VERIFICATION_FAILED"
        
        # Log the verification result for debugging
        logger.info(f"Verification result: {verification_result}, Attempts: {verification_attempts}")
        
        # Return minimal result
        return {
            "classification": verification_result,
            "name_variants_detected": result.name_variants_detected or [],
            "verification_attempts": verification_attempts
        }
        
    except Exception as e:
        logger.error(f"Error verifying client identity: {e}")
        
        # Determine result based on verification attempts
        verification_result = "VERIFICATION_FAILED" if verification_attempts >= max_failed_attempts else "INSUFFICIENT_INFO"
        
        return {
            "classification": verification_result,
            "name_variants_detected": [],
            "verification_attempts": verification_attempts,
            "error": str(e)
        }
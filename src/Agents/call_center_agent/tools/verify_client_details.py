from typing import List, Dict, Any, Literal, Optional
from typing_inspect import get_args

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import logging
import time
import re
from functools import lru_cache
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

verification_fields = Literal["username", "vehicle_registration", "vehicle_make", 
                                 "vehicle_model", "vehicle_color", "email", 
                                 "id_number", "passport_number"]


# Optimized Pydantic Model for Verification Result
class ClientDetailsVerificationResult(BaseModel):
    """Simplified result with essential fields for verification."""
    extracted_fields: Dict[verification_fields, Optional[str]] = Field(
        default_factory=dict,
        description="Dictionary mapping verification fields to values extracted from client messages"
    )
    matched_fields: List[verification_fields] = Field(
        default_factory=list,
        description="List of correctly matched verification fields from the client's responses"
    )
    classification: Literal["VERIFIED", "INSUFFICIENT_INFO", "VERIFICATION_FAILED"] = Field(
        description="The classification of the client response"
    )

# LLM instance cache
_LLM_CACHE_COUNTER = 0
_LLM_CACHE_FLUSH_THRESHOLD = 100

@lru_cache(maxsize=1)
def get_llm_instance():
    """Returns a cached LLM instance to avoid recreation on each call."""
    return ChatOllama(
        model="qwen2.5:3b-instruct", 
        temperature=0,
    )

def flush_llm_cache_if_needed():
    """Flushes the LLM cache if the counter exceeds the threshold."""
    global _LLM_CACHE_COUNTER
    _LLM_CACHE_COUNTER += 1
    if _LLM_CACHE_COUNTER >= _LLM_CACHE_FLUSH_THRESHOLD:
        get_llm_instance.cache_clear()
        _LLM_CACHE_COUNTER = 0
        logger.info("LLM cache flushed after reaching threshold")

# Optimized verification prompt
CLIENT_DETAILS_VERIFICATION_PROMPT = """
<role>Identity Security Specialist for Cartrack</role>

<database_record>
{client_details_str}
</database_record>

<verification_requirements>
A client is VERIFIED only if they provide EITHER:
- Their EXACT ID number or passport number
OR
- At least {required_match_count} of these fields with correct matches:
  * username
  * vehicle_registration
  * vehicle_make
  * vehicle_model
  * vehicle_color
  * email
</verification_requirements>

<conversation_history>
{messages}
</conversation_history>

<context>
- Current verification attempts: {verification_attempts}
- Maximum allowed failed attempts: {max_failed_attempts}
- Required matching fields: {required_match_count}
</context>

<output_instructions>
Your task is to determine:
1. What information fields can be extracted from the client's messages
2. Which of these fields match with our database records
3. Whether the client is verified based on our requirements

IMPORTANT: ONLY extract fields that were EXPLICITLY mentioned by the client in their messages. 
DO NOT generate or infer fields that weren't explicitly mentioned.

You must extract and check these fields: {verification_fields}

Classification rules:
- VERIFIED: Either id_number OR passport_number matches OR at least {required_match_count} other fields match
- INSUFFICIENT_INFO: Some correct information but below the verification threshold
- VERIFICATION_FAILED: Maximum attempts reached without successful verification

{format_instructions}
</output_instructions>
"""

def format_conversation(messages: List[Any]) -> str:
    """Formats conversation messages into a readable string."""
    formatted_messages = []
    
    for msg in messages:
        if isinstance(msg, dict):
            if "role" in msg and "content" in msg:
                role = "Client" if msg["role"] in ["user", "human"] else "Agent"
                formatted_messages.append(f"{role}: {msg['content']}")
        elif isinstance(msg, BaseMessage):
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"Client: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_messages.append(f"Agent: {msg.content}")
    
    return "\n".join(formatted_messages)

def count_verification_attempts(messages: List[Any]) -> int:
    """Counts verification attempts from conversation history."""
    if not messages:
        return 0
    
    conversation_turns = 0
    agent_spoke_last = False
    
    for msg in messages:
        is_agent = (isinstance(msg, AIMessage) or 
                   (isinstance(msg, dict) and msg.get("role") in ["assistant", "system"]))
        is_client = (isinstance(msg, HumanMessage) or 
                    (isinstance(msg, dict) and msg.get("role") in ["user", "human"]))
        
        if is_agent:
            agent_spoke_last = True
        elif is_client and agent_spoke_last:
            conversation_turns += 1
            agent_spoke_last = False
    
    return max(conversation_turns, 1) if messages else 0

@tool
def verify_client_details(
    client_details: Dict[str, Any],
    messages: List[Any],
    required_match_count: int = 3,
    max_failed_attempts: int = 3
) -> Dict[str, Any]:
    """
    Optimized function that verifies client details based on conversation history.
    
    Args:
        client_details: Dictionary containing the client's records
        messages: A list of conversation messages
        required_match_count: Number of details required for verification (default: 3)
        max_failed_attempts: Maximum number of failed attempts before failing (default: 3)
        
    Returns:
        A dictionary containing the verification result, matched fields, and extracted fields.
    """
    start_time = time.time()
    
    # Format conversation
    formatted_conversation = format_conversation(messages)
    
    # Count verification attempts
    verification_attempts = count_verification_attempts(messages)
    
    # Skip processing if there are no messages
    if not formatted_conversation:
        return {
            "classification": "INSUFFICIENT_INFO",
            "matched_fields": [],
            "extracted_fields": {},
            "verification_attempts": 0
        }
    
    # Set up LLM and parser
    parser = PydanticOutputParser(pydantic_object=ClientDetailsVerificationResult)
    prompt = ChatPromptTemplate.from_template(CLIENT_DETAILS_VERIFICATION_PROMPT).partial(
        client_details_str=str(client_details),
        required_match_count=required_match_count,
        max_failed_attempts=max_failed_attempts,
        verification_attempts=verification_attempts,
        verification_fields=", ".join(get_args(verification_fields)),
        format_instructions=parser.get_format_instructions()
    )
    
    # Check if we need to flush the cache
    flush_llm_cache_if_needed()
    
    # Use cached LLM instance
    llm = get_llm_instance()
    
    try:
        # Use structured output for reliable parsing
        llm_with_structured_output = llm.with_structured_output(ClientDetailsVerificationResult)
        chain = prompt | llm_with_structured_output
        
        # Get LLM-based verification result
        result = chain.invoke({
            "messages": formatted_conversation,
            "verification_attempts": verification_attempts,
            "max_failed_attempts": max_failed_attempts
        })
        
        # Clean extracted fields
        clean_extracted = {}
        valid_fields = get_args(verification_fields)
                        
        for field, value in result.extracted_fields.items():
            if field in valid_fields and value is not None and str(value).strip():
                clean_extracted[field] = str(value).strip()
        
        # Validate that extracted fields actually exist in the conversation
        validated_fields = {}
        conversation_text = formatted_conversation.lower()
        
        for field, value in clean_extracted.items():
            # Special case for ID and passport numbers - require exact match in conversation
            if field in ["id_number", "passport_number"]:
                # Only include if the exact value appears in the conversation
                if value.lower() in conversation_text:
                    validated_fields[field] = value
            # For other fields, just check if the value is reasonable
            else:
                # Simple content check - at least part of value should appear in conversation
                # This helps with fields like email where format might be different
                if field == "email" and "@" in value and any(part.lower() in conversation_text 
                                                            for part in value.split("@")):
                    validated_fields[field] = value
                # For vehicle registration, check for alphanumeric parts
                elif field == "vehicle_registration" and re.sub(r'[^a-zA-Z0-9]', '', value).lower() in re.sub(r'[^a-zA-Z0-9]', '', conversation_text):
                    validated_fields[field] = value
                # For other fields, ensure at least the value is mentioned
                elif value.lower() in conversation_text:
                    validated_fields[field] = value
        
        # Verify which extracted and validated fields match with the database values
        matched_fields = []
        for field, extracted_value in validated_fields.items():
            # Get the correct field from client details, accounting for different structures
            db_value = None
            if field in client_details:
                db_value = client_details[field]
            elif field.startswith("vehicle_") and "vehicles" in client_details and client_details["vehicles"]:
                # Handle vehicle fields that might be in a vehicles array
                vehicle_field = field.replace("vehicle_", "")
                if vehicle_field in client_details["vehicles"][0]:
                    db_value = client_details["vehicles"][0][vehicle_field]
            
            # For vehicle make and model, use case-insensitive matching
            if field in ["vehicle_make", "vehicle_model"] and db_value:
                if extracted_value.lower() == str(db_value).lower():
                    matched_fields.append(field)
            # For all other fields, use standard matching
            elif db_value and extracted_value.lower() == str(db_value).lower():
                matched_fields.append(field)
                
        # Apply business rules to validate classification
        id_verified = any(field in ["id_number", "passport_number"] for field in matched_fields)
        enough_fields_matched = len(matched_fields) >= required_match_count
        
        # Override classification if necessary based on business rules
        if id_verified or enough_fields_matched:
            classification = "VERIFIED"
        elif verification_attempts >= max_failed_attempts:
            classification = "VERIFICATION_FAILED"
        else:
            classification = "INSUFFICIENT_INFO"
            
        # Log the result
        logger.info(f"Verification result: {classification} (Matched: {len(matched_fields)}/{required_match_count})")
        
        # Return result with validated fields
        return {
            "classification": classification,
            "matched_fields": matched_fields,
            "extracted_fields": validated_fields,
            "verification_attempts": verification_attempts
        }
        
    except Exception as e:
        logger.error(f"Error verifying client details: {e}")
        
        # Determine classification based on attempt count
        classification = "VERIFICATION_FAILED" if verification_attempts >= max_failed_attempts else "INSUFFICIENT_INFO"
        
        # Return error response with minimal data
        return {
            "classification": classification,
            "matched_fields": [],
            "extracted_fields": {},
            "verification_attempts": verification_attempts,
            "error": str(e)
        }
import os
import sys
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch  # type: ignore[import-not-found]
from langgraph.checkpoint.memory import MemorySaver

# Add parent directories to path for imports
sys.path.append("../..")

# Load environment variables
load_dotenv(find_dotenv())

# Configuration
PROMPT = "You are a helpful AI assistant. Answer in 20 words or less." \
"If the input is unclear or incomplete, say 'Pardon?'" \
"If the input is not in English, say 'I can only respond in English.'"
MAX_SEARCH_RESULTS = 2
MODEL_NAME = "ollama:qwen2.5:14b-instruct"
TEMPERATURE = 0

# Initialize components
search = TavilySearch(max_results=MAX_SEARCH_RESULTS)
tools = [search]
model = init_chat_model(MODEL_NAME, temperature=TEMPERATURE)
checkpointer = MemorySaver()

# Create the agent
react_agent_graph = create_react_agent(
    model=model,
    tools=tools,
    prompt=PROMPT,
    checkpointer=checkpointer
)

# from src.Agents.agent00_basic_agent import BasicAgent
# graph = BasicAgent(
#     model=model,
#     tools=tools,
#     prompt=prompt
# ).workflow


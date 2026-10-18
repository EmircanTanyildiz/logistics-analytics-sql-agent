from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, START, END
from langgraph.graph import MessagesState, StateGraph
from nodes import run_agent_reasoning, tool_node
load_dotenv()

AGENT_REASON = "agent_reason"
ACT="act"
LAST=-1
def should_continue(state: MessagesState) -> str:
    if not state['messages'][LAST].tool_calls:
        return END
    return ACT

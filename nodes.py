from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from openai.types.admin.organization import role

from react import llm, tools
load_dotenv()

SYSTEM_MESSAGE = """
You are an AI-powered SQL agent specialized in logistics analytics.

Your role is to answer user questions about logistics operations
by interacting with a SQL database.

Instructions:
1. Understand the user's question.
2. Inspect the available database tables and schemas.
3. Generate accurate SQL queries using the database dialect.
4. Never invent table or column names.
5. Only perform read-only SQL operations.
6. Never execute INSERT, UPDATE, DELETE, DROP or ALTER.
7. Use appropriate LIMIT clauses.
8. If a query fails, inspect the error and try to correct it.
9. Base your answers strictly on actual query results.
10. Explain results clearly in the user's language.
"""
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    run the agent reasoning node.
    """
    response = llm.invoke([{"role":"system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}
tool_node = ToolNode(tools)
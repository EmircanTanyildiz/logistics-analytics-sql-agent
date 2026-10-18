import argparse

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return ACT

    return END


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

flow.add_edge(START, AGENT_REASON)

flow.add_conditional_edges(AGENT_REASON, should_continue, {ACT: ACT, END: END})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question about logistics data.")
    parser.add_argument(
        "question", nargs="?", help="Natural-language analytics question"
    )
    args = parser.parse_args()

    print("Logistics Analytics SQL Agent")
    question = (
        args.question if args.question is not None else input("Your question: ")
    ).strip()

    if not question:
        parser.error("The question cannot be empty.")

    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 30},
    )

    print(result["messages"][-1].content)

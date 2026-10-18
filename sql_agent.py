
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import (
    START,
    END,
    MessagesState,
    StateGraph
)

from nodes import run_agent_reasoning, tool_node

load_dotenv()

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

flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        ACT: ACT,
        END: END
    }
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

app.get_graph().draw_mermaid_png(
    output_file_path="flow.png"
)


if __name__ == "__main__":
    print("Logistics Analytics SQL Agent")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Show the top 10 drivers with the highest number of trips."
                    )
                )
            ]
        },
        config={"recursion_limit": 30},
    )

    print(result["messages"][-1].content)

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

MAX_PROMPT_LENGTH = 4000


@dataclass
class QueryResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    row_limit: int = 100
    error: Optional[str] = None


@dataclass
class AnalysisResult:
    prompt: str
    answer: str
    queries: list[QueryResult]
    elapsed_seconds: float


def extract_query_results(messages: Sequence[BaseMessage]) -> list[QueryResult]:
    queries = []
    sql_by_call = {
        call["id"]: call["args"].get("sql", "")
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "execute_safe_query"
    }

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if (
            message.name != "execute_safe_query"
            and message.tool_call_id not in sql_by_call
        ):
            continue

        sql = sql_by_call.get(message.tool_call_id, "")
        try:
            payload = json.loads(message.content)
            if not isinstance(payload, dict):
                raise TypeError("Expected a query result object.")
            if payload.get("success") is not True:
                queries.append(
                    QueryResult(
                        sql=payload.get("sql", sql),
                        error=payload.get("error")
                        or "The query could not be completed.",
                    )
                )
                continue

            rows = payload["data"]
            columns = payload["columns"]
            if (
                not isinstance(rows, list)
                or not all(isinstance(row, dict) for row in rows)
                or not isinstance(columns, list)
                or not all(isinstance(column, str) for column in columns)
            ):
                raise ValueError("Invalid query result structure.")

            queries.append(
                QueryResult(
                    sql=payload.get("sql", sql),
                    columns=columns,
                    rows=rows,
                    truncated=payload.get("truncated", False),
                    row_limit=payload.get("row_limit", 100),
                )
            )
        except (TypeError, ValueError, KeyError):
            queries.append(
                QueryResult(
                    sql=sql,
                    error="The query output could not be read. Submit the question again.",
                )
            )

    return queries


def run_analysis(prompt: str) -> AnalysisResult:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Enter the question you want to analyze.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"The question can contain at most {MAX_PROMPT_LENGTH} characters."
        )

    from sql_agent import app

    started = perf_counter()
    result = app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": 30},
    )
    messages = result["messages"]
    if (
        not messages
        or not isinstance(messages[-1], AIMessage)
        or messages[-1].tool_calls
    ):
        raise RuntimeError(
            "The analysis did not produce a completed response. Try again."
        )

    content = messages[-1].content
    answer = (
        content
        if isinstance(content, str)
        else "\n".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or isinstance(block, dict)
            and block.get("type") == "text"
        )
    )

    return AnalysisResult(
        prompt=prompt,
        answer=answer,
        queries=extract_query_results(messages),
        elapsed_seconds=perf_counter() - started,
    )

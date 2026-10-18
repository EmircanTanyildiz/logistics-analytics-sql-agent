import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import analytics
from analytics import extract_query_results, run_analysis


def test_graph_executes_tools_and_service_returns_verified_rows(
    query_tools, monkeypatch
):
    import nodes

    sql = "SELECT driver_id, COUNT(*) AS trip_count FROM trips GROUP BY driver_id ORDER BY trip_count DESC"
    model = Mock()
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_safe_query",
                    "args": {"sql": sql},
                    "id": "query-1",
                }
            ],
        ),
        AIMessage(content="Can completed the most trips."),
    ]
    monkeypatch.setattr(nodes, "get_llm", lambda: model)

    result = run_analysis("  Rank drivers by trip count.  ")

    assert result.prompt == "Rank drivers by trip count."
    assert result.answer == "Can completed the most trips."
    assert result.elapsed_seconds >= 0
    assert len(result.queries) == 1
    assert result.queries[0].sql == sql
    assert result.queries[0].rows == [
        {"driver_id": "DRV002", "trip_count": 3},
        {"driver_id": "DRV001", "trip_count": 2},
        {"driver_id": "DRV003", "trip_count": 1},
    ]
    submitted = model.invoke.call_args_list[0].args[0][-1]
    assert isinstance(submitted, HumanMessage)
    assert submitted.content == result.prompt
    assert any(
        isinstance(message, ToolMessage)
        for message in model.invoke.call_args_list[1].args[0]
    )


def test_query_results_are_matched_to_tool_calls():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute_safe_query",
                    "args": {"sql": "SELECT broken"},
                    "id": "failed",
                },
                {
                    "name": "execute_safe_query",
                    "args": {"sql": "SELECT 42 AS count"},
                    "id": "valid",
                },
            ],
        ),
        ToolMessage(
            content='{"success": true, "columns": []}',
            tool_call_id="schema",
            name="get_table_schema",
        ),
        ToolMessage(
            content='{"success": false, "error": "Unknown column"}',
            tool_call_id="failed",
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "success": True,
                    "data": [{"count": 42}],
                    "columns": ["count"],
                    "truncated": True,
                    "row_limit": 100,
                }
            ),
            tool_call_id="valid",
        ),
        AIMessage(content="An explanation that is not a Markdown table."),
    ]

    queries = extract_query_results(messages)

    assert len(queries) == 2
    assert queries[0].sql == "SELECT broken"
    assert queries[0].error == "Unknown column"
    assert queries[1].sql == "SELECT 42 AS count"
    assert queries[1].rows == [{"count": 42}]
    assert queries[1].truncated is True


@pytest.mark.parametrize(
    "content",
    ["invalid json", "null", "[]", '{"success":true,"data":"invalid","columns":[]}'],
)
def test_unreadable_tool_results_are_visible_errors(content):
    queries = extract_query_results(
        [
            ToolMessage(
                content=content, tool_call_id="query", name="execute_safe_query"
            ),
        ]
    )

    assert len(queries) == 1
    assert queries[0].error
    assert queries[0].rows == []


@pytest.mark.parametrize(
    "prompt", ["", " \n ", "x" * (analytics.MAX_PROMPT_LENGTH + 1)]
)
def test_invalid_prompts_do_not_invoke_the_graph(monkeypatch, prompt):
    import sql_agent

    graph = Mock()
    monkeypatch.setattr(sql_agent, "app", graph)

    with pytest.raises(ValueError):
        run_analysis(prompt)

    graph.invoke.assert_not_called()


def test_independent_questions_do_not_share_messages(monkeypatch):
    import sql_agent

    graph = Mock()
    graph.invoke.return_value = {
        "messages": [AIMessage(content="Please specify a date range.")]
    }
    monkeypatch.setattr(sql_agent, "app", graph)

    run_analysis("First question")
    result = run_analysis("Second question")

    assert [
        message.content for message in graph.invoke.call_args.args[0]["messages"]
    ] == ["Second question"]
    assert result.queries == []


def test_text_content_blocks_are_supported(monkeypatch):
    import sql_agent

    graph = Mock()
    graph.invoke.return_value = {
        "messages": [
            AIMessage(
                content=[
                    {"type": "text", "text": "First observation."},
                    {"type": "text", "text": "Second observation."},
                ]
            )
        ]
    }
    monkeypatch.setattr(sql_agent, "app", graph)

    assert (
        run_analysis("Explain the results").answer
        == "First observation.\nSecond observation."
    )

from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import streamlit as st
from datafusion import SessionContext
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from streamlit.testing.v1 import AppTest

import analytics
import database.datafusion
from analytics import AnalysisResult, QueryResult

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def ui(monkeypatch, sample_context):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-for-network")
    monkeypatch.setattr(database.datafusion, "ctx", sample_context)
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_PATH), default_timeout=15)
    yield app
    st.cache_resource.clear()


def submit(app, question="Rank drivers by trip count."):
    app.text_area[0].input(question)
    next(button for button in app.button if button.label == "Analyze").click().run()


def test_user_question_reaches_model_and_actual_rows_reach_table(
    ui, query_tools, monkeypatch
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
                    "id": "ui-query",
                }
            ],
        ),
        AIMessage(content="DRV002 completed the most trips."),
    ]
    monkeypatch.setattr(nodes, "get_llm", lambda: model)

    ui.run()
    model.invoke.assert_not_called()
    submit(ui, "Rank drivers by trip count.")

    assert not ui.exception
    assert not ui.error
    assert (
        model.invoke.call_args_list[0].args[0][-1].content
        == "Rank drivers by trip count."
    )
    assert ui.main.dataframe[0].value.to_dict("records") == [
        {"driver_id": "DRV002", "trip_count": 3},
        {"driver_id": "DRV001", "trip_count": 2},
        {"driver_id": "DRV003", "trip_count": 1},
    ]
    assert ui.code[0].value == sql
    assert any("completed the most trips" in element.value for element in ui.markdown)
    assert all(
        metric.label not in {"Result tables", "Displayed rows"} for metric in ui.metric
    )
    assert len(ui.get("download_button")) == 1

    ui.selectbox[0].select("trips").run()
    assert not ui.exception
    assert model.invoke.call_count == 2
    assert len(ui.main.dataframe) == 1


def test_empty_prompt_does_not_call_agent(ui, monkeypatch):
    service = Mock()
    monkeypatch.setattr(analytics, "run_analysis", service)

    ui.run()
    submit(ui, "   ")

    assert not ui.exception
    assert any("Enter the question" in warning.value for warning in ui.warning)
    service.assert_not_called()


def test_missing_key_disables_submission(ui, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    ui.run()

    assert not ui.exception
    assert any("OPENAI_API_KEY" in warning.value for warning in ui.warning)
    assert next(button for button in ui.button if button.label == "Analyze").disabled


def test_missing_dataset_disables_submission(ui, monkeypatch):
    monkeypatch.setattr(database.datafusion, "ctx", SessionContext())

    ui.run()

    assert not ui.exception
    assert any("No data was found" in warning.value for warning in ui.warning)
    assert next(button for button in ui.button if button.label == "Analyze").disabled


def test_empty_and_truncated_results_have_clear_status(ui, monkeypatch):
    result = AnalysisResult(
        prompt="Compare two results",
        answer="One query returned no records; the other result was limited.",
        queries=[
            QueryResult(
                sql="SELECT driver_id FROM drivers WHERE 1 = 0", columns=["driver_id"]
            ),
            QueryResult(
                sql="SELECT value FROM numbers",
                columns=["value"],
                rows=[{"value": number} for number in range(100)],
                truncated=True,
            ),
        ],
        elapsed_seconds=1.2,
    )
    monkeypatch.setattr(analytics, "run_analysis", Mock(return_value=result))

    ui.run()
    submit(ui)

    assert not ui.exception
    assert len(ui.main.dataframe) == 2
    assert list(ui.main.dataframe[0].value.columns) == ["driver_id"]
    assert any("no records matched" in info.value for info in ui.info)
    assert any("limited to 100 rows" in warning.value for warning in ui.warning)
    assert len(ui.get("download_button")) == 1


def test_clarification_without_sql_is_displayed(ui, monkeypatch):
    result = AnalysisResult(
        "How is performance?", "Which date range should I analyze?", [], 0.1
    )
    monkeypatch.setattr(analytics, "run_analysis", Mock(return_value=result))

    ui.run()
    submit(ui)

    assert not ui.exception
    assert not ui.main.dataframe
    assert any("Which date range" in element.value for element in ui.markdown)


def test_failed_sql_attempt_is_kept_with_corrected_result(ui, monkeypatch):
    result = AnalysisResult(
        prompt="Driver count",
        answer="There are three drivers.",
        queries=[
            QueryResult(sql="SELECT invalid FROM drivers", error="Unknown column"),
            QueryResult(
                sql="SELECT COUNT(*) AS total FROM drivers",
                columns=["total"],
                rows=[{"total": 3}],
            ),
        ],
        elapsed_seconds=0.5,
    )
    monkeypatch.setattr(analytics, "run_analysis", Mock(return_value=result))

    ui.run()
    submit(ui)

    assert not ui.exception
    assert ui.main.dataframe[0].value.to_dict("records") == [{"total": 3}]
    assert any(error.value == "Unknown column" for error in ui.error)
    assert any("Failed query attempts" in expander.label for expander in ui.expander)


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            APIConnectionError(
                request=httpx.Request("POST", "https://example.invalid")
            ),
            "Could not connect",
        ),
        (
            APITimeoutError(request=httpx.Request("POST", "https://example.invalid")),
            "timed out",
        ),
        (
            AuthenticationError(
                "invalid key",
                response=httpx.Response(
                    401, request=httpx.Request("POST", "https://example.invalid")
                ),
                body={},
            ),
            "could not be authenticated",
        ),
        (
            RateLimitError(
                "quota",
                response=httpx.Response(
                    429, request=httpx.Request("POST", "https://example.invalid")
                ),
                body={},
            ),
            "usage limit was reached",
        ),
        (GraphRecursionError("step limit"), "reached its step limit"),
        (RuntimeError("internal error"), "analysis could not be completed"),
    ],
)
def test_failed_request_clears_stale_result_and_shows_error(
    ui, monkeypatch, error, expected
):
    previous = AnalysisResult(
        "Old question",
        "Old result",
        [QueryResult(sql="SELECT 1 AS count", columns=["count"], rows=[{"count": 1}])],
        0.1,
    )
    service = Mock(side_effect=[previous, error])
    monkeypatch.setattr(analytics, "run_analysis", service)

    ui.run()
    submit(ui, "Old question")
    assert len(ui.main.dataframe) == 1
    submit(ui, "New question")

    assert not ui.exception
    assert any(expected in item.value for item in ui.error)
    assert not ui.main.dataframe
    assert "analysis_result" not in ui.session_state


def test_clear_button_removes_result_without_another_request(ui, monkeypatch):
    service = Mock(return_value=AnalysisResult("Question", "Answer", [], 0.1))
    monkeypatch.setattr(analytics, "run_analysis", service)

    ui.run()
    submit(ui)
    next(button for button in ui.button if button.label == "Clear result").click().run()

    assert not ui.exception
    service.assert_called_once()
    assert "analysis_result" not in ui.session_state

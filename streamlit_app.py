import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from analytics import MAX_PROMPT_LENGTH, AnalysisResult, run_analysis

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
logger = logging.getLogger(__name__)


@st.cache_resource
def get_context():
    from database.datafusion import ctx

    return ctx


def render_result(result: AnalysisResult):
    st.subheader("Analysis result")
    st.caption(f"Your question: {result.prompt}")
    successful = [query for query in result.queries if query.error is None]
    failed = [query for query in result.queries if query.error is not None]

    st.metric("Analysis time", f"{result.elapsed_seconds:.1f} sec")

    table_tab, summary_tab, sql_tab = st.tabs(
        ["Data tables", "Insights", "SQL queries"]
    )

    with table_tab:
        if not successful:
            st.info(
                "There is no result table to display. Review the response in the Insights tab."
            )
        for index, query in enumerate(successful, start=1):
            st.markdown(f"**Result {index}**")
            frame = pd.DataFrame(query.rows, columns=query.columns)
            st.dataframe(frame, hide_index=True, width="stretch")
            if frame.empty:
                st.info(
                    "The query completed successfully, but no records matched these conditions."
                )
            elif query.truncated:
                st.warning(
                    f"The result was limited to {query.row_limit} rows. "
                    "This table and the downloaded CSV contain only the displayed rows."
                )
            if not frame.empty:
                st.download_button(
                    "Download table as CSV",
                    data=frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"logistics_result_{index}.csv",
                    mime="text/csv",
                    key=f"download_result_{index}",
                    on_click="ignore",
                )

    with summary_tab:
        if result.answer.strip():
            st.markdown(result.answer)
        else:
            st.info(
                "The model did not provide additional insights. You can review the query tables."
            )

    with sql_tab:
        if not successful:
            st.info("No SQL query completed successfully.")
        for index, query in enumerate(successful, start=1):
            st.markdown(f"**Query {index}**")
            st.code(query.sql, language="sql")
        if failed:
            with st.expander(f"Failed query attempts ({len(failed)})"):
                for query in failed:
                    st.code(query.sql, language="sql")
                    st.error(query.error)

    if failed and not successful:
        st.warning(
            "The SQL queries could not be completed. See the SQL queries tab for details."
        )


def main():
    st.set_page_config(
        page_title="Logistics Analytics | SQL Agent",
        page_icon="🚚",
        layout="wide",
    )
    st.title("Logistics Analytics")
    st.caption(
        "Ask questions about your logistics data in natural language and explore the results in tables."
    )

    tables = []
    context = None
    data_error = None
    try:
        context = get_context()
        tables = sorted(context.catalog().schema("public").table_names())
    except Exception:
        logger.exception("Could not initialize the logistics dataset")
        data_error = "The data files could not be loaded. Check the CSV format and file permissions."

    api_ready = bool(os.getenv("OPENAI_API_KEY", "").strip())
    with st.sidebar:
        st.header("Data source")
        st.metric("Registered tables", len(tables))
        st.caption("Apache DataFusion · Read-only analytics")
        st.caption("Each query result displays up to 100 rows.")
        if tables:
            selected = st.selectbox("Table schema", tables)
            schema = context.table(selected).schema()
            st.dataframe(
                [
                    {"Column": column.name, "Data type": str(column.type)}
                    for column in schema
                ],
                hide_index=True,
                width="stretch",
            )
        st.divider()
        st.caption(
            "Each question is evaluated independently; previous questions are not sent to the model."
        )
        st.caption(
            "Your question, table metadata, and query results are sent to OpenAI to generate the response."
        )
        if st.button("Clear result", width="stretch"):
            st.session_state.pop("analysis_result", None)

    if data_error:
        st.error(data_error)
    elif not tables:
        st.warning(
            "No data was found. Add the logistics CSV files to logisticsDatasetCSV/ and restart the application."
        )
    if not api_ready:
        st.warning(
            "To start an analysis, define OPENAI_API_KEY in the project-root .env file and restart the application."
        )

    with st.expander("Example questions"):
        st.markdown(
            "- Show the top 10 drivers by trip count, including their names and trip counts.\n"
            "- How does total fuel spending change by month?\n"
            "- List the 10 trucks with the highest total maintenance cost.\n"
            "- Rank customers by total load revenue."
        )

    with st.form("analysis_form"):
        prompt = st.text_area(
            "Your analysis question",
            placeholder="For example: Show the top 10 drivers by trip count.",
            height=120,
            max_chars=MAX_PROMPT_LENGTH,
            help="You can include the comparison metric, date range, and filters in your question.",
        )
        submitted = st.form_submit_button(
            "Analyze",
            type="primary",
            disabled=not api_ready or not tables or data_error is not None,
        )

    if submitted:
        st.session_state.pop("analysis_result", None)
        if not prompt.strip():
            st.warning("Enter the question you want to analyze.")
        else:
            try:
                with st.spinner("Reviewing the data and preparing the analysis..."):
                    st.session_state.analysis_result = run_analysis(prompt)
            except AuthenticationError:
                st.error(
                    "The API key could not be authenticated. Check OPENAI_API_KEY."
                )
            except RateLimitError:
                st.error(
                    "The API usage limit was reached. Check your quota and billing settings, then try again later."
                )
            except APITimeoutError:
                st.error("The model response timed out. Try again.")
            except APIConnectionError:
                st.error(
                    "Could not connect to the model service. Check your internet connection and try again."
                )
            except APIStatusError:
                st.error(
                    "The model service could not complete the request. Check the model setting and try again."
                )
            except GraphRecursionError:
                st.error(
                    "The analysis reached its step limit. Rewrite your question with a narrower scope."
                )
            except Exception:
                logger.exception("Logistics analysis failed")
                st.error(
                    "The analysis could not be completed. Try your question again; if the issue continues, check the application logs."
                )

    result = st.session_state.get("analysis_result")
    if result is not None:
        render_result(result)
    elif not submitted:
        st.info("Enter a question and select Analyze to get started.")


if __name__ == "__main__":
    main()

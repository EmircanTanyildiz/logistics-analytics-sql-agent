import json
import os
from functools import lru_cache
from pathlib import Path

import sqlglot
from datafusion import SQLOptions
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from sqlglot import exp

from database.datafusion import ctx

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
SCHEMA_PATH = BASE_DIR / "logisticsDatasetCSV" / "DATABASE_SCHEMA.txt"
BUSINESS_CONTEXT_PATH = BASE_DIR / "business_context.txt"


@tool
def list_tables() -> list[str]:
    """List all registered tables in Apache DataFusion."""
    catalog = ctx.catalog()
    schema = catalog.schema("public")

    return sorted(schema.table_names())


@tool
def get_table_schema(table_name: str) -> dict:
    """
    Get the column names and data types
    of a registered DataFusion table.

    Args:
        table_name: Name of the table to inspect.
    """

    try:
        # Check registered tables
        catalog = ctx.catalog()
        schema = catalog.schema("public")

        if table_name not in schema.table_names():
            return {"success": False, "error": f"Table '{table_name}' not found."}

        # Get DataFusion table schema
        df = ctx.table(table_name)
        arrow_schema = df.schema()

        columns = [
            {
                "name": field.name,
                "data_type": str(field.type),
                "nullable": field.nullable,
            }
            for field in arrow_schema
        ]

        return {"success": True, "table_name": table_name, "columns": columns}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def get_table_relationship() -> str:
    """Get table relationships from the logistics schema."""

    if not SCHEMA_PATH.is_file():
        return "The logistics schema file was not found. Inspect table schemas instead."

    content = SCHEMA_PATH.read_text(encoding="utf-8")

    if "KEY RELATIONSHIPS:" not in content:
        return "No relationship information found."

    relationships = content.split("KEY RELATIONSHIPS:", 1)[1]

    relationships = relationships.split("ANALYTICAL USE CASES:", 1)[0]

    return relationships.strip()


@tool
def get_business_context():
    """get logistics business rules and KPI definitions."""
    if not BUSINESS_CONTEXT_PATH.exists():
        return "business_context was not found"
    else:
        return BUSINESS_CONTEXT_PATH.read_text(encoding="utf-8")


MAX_ROWS = 100
READ_ONLY_SQL_OPTIONS = (
    SQLOptions()
    .with_allow_ddl(False)
    .with_allow_dml(False)
    .with_allow_statements(False)
)


@tool
def execute_safe_query(sql: str) -> str:
    """
    Execute a read-only SQL query using Apache DataFusion.

    Args:
        sql: SQL SELECT query to execute.

    Returns:
        JSON string containing query results or an error.
    """

    try:
        statements = sqlglot.parse(sql, read="postgres")

        # 2. Tek bir sorguya izin ver
        if len(statements) != 1 or statements[0] is None:
            return json.dumps(
                {
                    "success": False,
                    "sql": sql,
                    "error": "Only one SQL statement is allowed.",
                }
            )

        statement = statements[0]

        if not isinstance(statement, exp.Query):
            return json.dumps(
                {
                    "success": False,
                    "sql": sql,
                    "error": "Only read-only SELECT queries are allowed.",
                }
            )

        forbidden = (
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Drop,
            exp.Create,
            exp.Alter,
            exp.Command,
            exp.Into,
            exp.Merge,
            exp.Lock,
        )

        for node in statement.walk():
            if isinstance(node, forbidden):
                return json.dumps(
                    {
                        "success": False,
                        "sql": sql,
                        "error": "Unsafe SQL operation detected.",
                    }
                )

        df = ctx.sql(sql, options=READ_ONLY_SQL_OPTIONS)
        columns = df.schema().names

        if len(columns) != len(set(columns)):
            return json.dumps(
                {
                    "success": False,
                    "sql": sql,
                    "error": "Use unique column aliases so every result column can be displayed.",
                }
            )

        df = df.limit(MAX_ROWS + 1)

        batches = df.collect()

        rows = []

        for batch in batches:
            rows.extend(batch.to_pylist())

        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]

        return json.dumps(
            {
                "success": True,
                "sql": sql,
                "columns": columns,
                "row_count": len(rows),
                "row_limit": MAX_ROWS,
                "truncated": truncated,
                "data": rows,
            },
            default=str,
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "sql": sql, "error": str(e)}, ensure_ascii=False
        )


tools = [
    get_table_schema,
    list_tables,
    get_table_relationship,
    get_business_context,
    execute_safe_query,
]


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        temperature=0,
        timeout=60,
        max_retries=2,
    ).bind_tools(tools)

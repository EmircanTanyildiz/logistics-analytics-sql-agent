from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from database.datafusion import ctx
from pathlib import Path
import json
import sqlglot
from sqlglot import exp
from database.datafusion import ctx

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "logisticsDatasetCSV" / "DATABASE_SCHEMA.txt"

@tool
def list_tables() -> list[str]:
    """List all registered tables in Apache DataFusion."""
    catalog = ctx.catalog()
    schema = catalog.schema("public")

    return schema.table_names()


@tool
def get_table_schema():
    """get column names and data types for a table."""
    catalog = ctx.catalog()
    schema = catalog.schema("public")
    if table_name not in schema.table_names():
        return {"error": f"Table '{table_name}' not found"}
    table = schema.table("public")
    arrow_schema = table.schema
    return {
        "table_names":table_name,
        "columns":[{
            "name":field.name,
            "data_type": str(field.type),
            "nullable": field.nullable
        }
        for field in arrow_schema
        ]
    }


@tool
def get_table_relationship():
    """get table relationship from the logistics schema."""
    catalog = ctx.catalog()
    schema = catalog.schema("public")

    content = SCHEMA_PATH.read_text(encoding="utf-8")
    if "KEY RELATIONSHIPS" is not in content:
        return "KEY RELATIONSHIPS was not found"
    relationship = content.split("KEY RELATIONSHIPS",1)[1]
    relationship = relationship.split(
        "ANALYTICAL USE CASES:", 1
    )[0]
    return relationships.strip()

BUSINESS_CONTEXT_PATH = BASE_DIR / "business_context.txt"


@tool
def get_business_context():
    """get logistics business rules and KPI definitions."""
    if not BUSINESS_CONTEXT_PATH.exists():
        return "business_context was not found"
    else:
        return BUSINESS_CONTEXT_PATH.read_text(encoding="utf-8")





MAX_ROWS = 100


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
            return json.dumps({
                "error": "Only one SQL statement is allowed."
            })

        statement = statements[0]

        if not isinstance(statement, exp.Query):
            return json.dumps({
                "error": "Only read-only SELECT queries are allowed."
            })

        forbidden = (
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Drop,
            exp.Create,
            exp.Alter,
            exp.Command,
        )

        for node in statement.walk():
            if isinstance(node, forbidden):
                return json.dumps({
                    "error": "Unsafe SQL operation detected."
                })

        df = ctx.sql(sql)

        df = df.limit(MAX_ROWS)

        batches = df.collect()

        rows = []

        for batch in batches:
            rows.extend(batch.to_pylist())

        return json.dumps({
            "success": True,
            "row_count": len(rows),
            "data": rows
        }, default=str, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)





tools = [get_table_schema, list_tables, get_table_relationship, get_business_context, execute_safe_query]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).bind_tools(tools)
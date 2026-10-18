from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from database.datafusion import ctx
from pathlib import Path
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



@tool
def get_business_context():
    pass

@tool
def execute_safe_query():
    pass




tools = [get_table_schema, list_tables, get_table_relationship, get_business_context]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).bind_tools(tools)
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from database.datafusion import ctx
load_dotenv()


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
    pass
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
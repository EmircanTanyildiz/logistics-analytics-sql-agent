import json

import pytest


def execute(query_tools, sql):
    return json.loads(query_tools.execute_safe_query.invoke({"sql": sql}))


def test_joined_ranking_returns_actual_data(query_tools):
    result = execute(
        query_tools,
        """
        SELECT d.driver_id, d.first_name, COUNT(*) AS trip_count
        FROM drivers d JOIN trips t ON d.driver_id = t.driver_id
        GROUP BY d.driver_id, d.first_name
        ORDER BY trip_count DESC, d.driver_id
    """,
    )

    assert result["success"] is True
    assert result["columns"] == ["driver_id", "first_name", "trip_count"]
    assert result["data"] == [
        {"driver_id": "DRV002", "first_name": "Can", "trip_count": 3},
        {"driver_id": "DRV001", "first_name": "Alice", "trip_count": 2},
        {"driver_id": "DRV003", "first_name": "Ece", "trip_count": 1},
    ]


@pytest.mark.parametrize(
    "limit, count, truncated",
    [
        ("", 100, True),
        ("LIMIT 100", 100, False),
        ("LIMIT 10", 10, False),
    ],
)
def test_row_limit_reports_actual_truncation(query_tools, limit, count, truncated):
    result = execute(query_tools, f"SELECT value FROM numbers ORDER BY value {limit}")

    assert result["row_count"] == count
    assert result["data"] == [{"value": value} for value in range(count)]
    assert result["truncated"] is truncated
    assert result["row_limit"] == 100


def test_output_limit_does_not_change_aggregates(query_tools):
    result = execute(
        query_tools, "SELECT COUNT(*) AS total, SUM(value) AS total_value FROM numbers"
    )

    assert result["data"] == [{"total": 105, "total_value": 5460}]
    assert result["truncated"] is False


def test_empty_result_preserves_column_names(query_tools):
    result = execute(
        query_tools, "SELECT driver_id, first_name FROM drivers WHERE 1 = 0"
    )

    assert result["success"] is True
    assert result["data"] == []
    assert result["row_count"] == 0
    assert result["columns"] == ["driver_id", "first_name"]
    assert result["truncated"] is False


def test_cte_and_union_remain_available(query_tools):
    result = execute(
        query_tools,
        """
        WITH active AS (SELECT driver_id FROM trips)
        SELECT driver_id FROM active
        UNION
        SELECT driver_id FROM drivers
        ORDER BY driver_id
    """,
    )

    assert result["success"] is True
    assert result["row_count"] == 3


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "SELECT 1; SELECT 2",
        "DROP TABLE drivers",
        "DELETE FROM drivers",
        "UPDATE drivers SET first_name = 'Changed'",
        "INSERT INTO drivers VALUES ('DRV004', 'New', 'Driver')",
        "CREATE TABLE copied AS SELECT * FROM drivers",
        "SELECT * INTO copied FROM drivers",
        "WITH removed AS (DELETE FROM drivers RETURNING *) SELECT * FROM removed",
        "SELECT * FROM drivers FOR UPDATE",
        "COPY drivers TO '/tmp/logistics-test-export.csv'",
        "SET datafusion.execution.batch_size = 1",
    ],
)
def test_unsafe_sql_is_rejected_before_execution(query_tools, monkeypatch, sql):
    def unexpected_execution(*args, **kwargs):
        pytest.fail("Unsafe SQL reached DataFusion.")

    monkeypatch.setattr(query_tools.ctx, "sql", unexpected_execution)
    result = execute(query_tools, sql)

    assert result["success"] is False
    assert result["error"]
    assert result["sql"] == sql


def test_unknown_column_returns_structured_error(query_tools):
    result = execute(query_tools, "SELECT missing_column FROM drivers")

    assert result["success"] is False
    assert "missing_column" in result["error"]


def test_ambiguous_output_columns_are_rejected(query_tools):
    result = execute(
        query_tools,
        "SELECT d.driver_id, t.driver_id FROM drivers d JOIN trips t ON d.driver_id = t.driver_id",
    )

    assert result["success"] is False
    assert "error" in result


def test_discovery_uses_registered_schema(query_tools):
    assert query_tools.list_tables.invoke({}) == ["drivers", "numbers", "trips"]
    schema = query_tools.get_table_schema.invoke({"table_name": "drivers"})
    assert schema["success"] is True
    assert [column["name"] for column in schema["columns"]] == [
        "driver_id",
        "first_name",
        "last_name",
    ]
    assert (
        query_tools.get_table_schema.invoke({"table_name": "missing"})["success"]
        is False
    )


def test_metadata_paths_do_not_depend_on_working_directory(
    query_tools, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    assert "loads -> customers" in query_tools.get_table_relationship.invoke({})
    assert "LOGISTICS BUSINESS CONTEXT" in query_tools.get_business_context.invoke({})


def test_missing_relationship_file_is_handled(query_tools, monkeypatch, tmp_path):
    monkeypatch.setattr(query_tools, "SCHEMA_PATH", tmp_path / "missing.txt")

    assert "not found" in query_tools.get_table_relationship.invoke({})

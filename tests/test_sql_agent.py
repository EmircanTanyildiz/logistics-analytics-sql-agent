
import pytest
from langchain_core.messages import HumanMessage

from react import ctx
from sql_agent import app


def parse_markdown_table(answer: str) -> list[dict]:
    rows = []

    for line in answer.splitlines():
        if not line.strip().startswith("|"):
            continue

        cells = [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

        if len(cells) != 4:
            continue

        if not cells[0].startswith("DRV"):
            continue

        rows.append({
            "driver_id": cells[0],
            "first_name": cells[1],
            "last_name": cells[2],
            "trip_count": int(cells[3])
        })

    return rows


def get_expected_results() -> list[dict]:
    sql = """
    SELECT
        d.driver_id,
        d.first_name,
        d.last_name,
        COUNT(*) AS trip_count
    FROM trips t
    JOIN drivers d
        ON t.driver_id = d.driver_id
    GROUP BY
        d.driver_id,
        d.first_name,
        d.last_name
    ORDER BY
        trip_count DESC,
        d.driver_id ASC
    LIMIT 10
    """

    batches = ctx.sql(sql).collect()

    return [
        row
        for batch in batches
        for row in batch.to_pylist()
    ]


@pytest.mark.integration
def test_top_10_drivers_matches_datafusion():
    expected = get_expected_results()

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Show the top 10 drivers by trip count. "
                        "Include driver ID, first name, last name "
                        "and exact trip count. "
                        "Sort by trip count descending, "
                        "then driver ID ascending. "
                        "Return the results as a Markdown table."
                    )
                )
            ]
        },
        config={"recursion_limit": 30}
    )

    answer = result["messages"][-1].content
    actual = parse_markdown_table(answer)

    print("\nLLM Answer:\n", answer)
    print("\nActual Parsed Results:\n", actual)
    print("\nExpected DataFusion Results:\n", expected)

    assert len(expected) == 10
    assert actual == expected, (
        f"\nLLM results:\n{actual}"
        f"\nDataFusion results:\n{expected}"
    )

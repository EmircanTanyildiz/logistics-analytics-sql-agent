
from database.datafusion import ctx

df = ctx.sql("""
    SELECT
        driver_id,
        COUNT(*) AS total_trips
    FROM trips
    GROUP BY driver_id
    ORDER BY total_trips DESC
    LIMIT 10
""")

df.show()

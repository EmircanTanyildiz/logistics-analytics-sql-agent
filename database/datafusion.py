
from pathlib import Path

from datafusion import SessionContext

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "logisticsDatasetCSV"


def create_context() -> SessionContext:
    ctx = SessionContext()

    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        ctx.register_csv(
            csv_file.stem,
            str(csv_file),
            has_header=True,
            schema_infer_max_records=10000
        )

    return ctx


ctx = create_context()

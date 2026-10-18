import os

import pyarrow as pa
import pytest
from datafusion import SessionContext

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture
def sample_context():
    context = SessionContext()
    drivers = pa.record_batch(
        {
            "driver_id": ["DRV001", "DRV002", "DRV003"],
            "first_name": ["Alice", "Can", "Ece"],
            "last_name": ["Young", "Demir", "Kaya"],
        }
    )
    trips = pa.record_batch(
        {
            "trip_id": [1, 2, 3, 4, 5, 6],
            "driver_id": ["DRV001", "DRV001", "DRV002", "DRV002", "DRV002", "DRV003"],
            "distance": [100.0, 200.5, 300.0, 50.0, 75.0, None],
        }
    )
    numbers = pa.record_batch({"value": list(range(105))})
    context.register_record_batches("drivers", [[drivers]])
    context.register_record_batches("trips", [[trips]])
    context.register_record_batches("numbers", [[numbers]])
    return context


@pytest.fixture
def query_tools(monkeypatch, sample_context):
    import react

    monkeypatch.setattr(react, "ctx", sample_context)
    return react

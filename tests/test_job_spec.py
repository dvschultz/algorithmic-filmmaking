"""Operation identities detach input data and enforce execution capabilities."""

import threading
from pathlib import Path

import pytest

from core.jobs import JobRuntime, JobStore
from core.jobs.spec import OperationSpec


def spec(**overrides):
    values = dict(
        kind="test",
        version=1,
        arguments={"nested": [1]},
        inputs={"source": "private source"},
        persistence="job_history",
        session_id="session",
        input_revision="revision",
    )
    values.update(overrides)
    return OperationSpec.build(**values)


def test_spec_detaches_arguments_inputs_and_identity():
    arguments, inputs = {"nested": [1]}, {"source": ["a"]}
    operation = spec(arguments=arguments, inputs=inputs)
    identity = operation.operation_id
    arguments["nested"].append(2)
    inputs["source"].append("b")
    operation.arguments["nested"].append(3)
    assert operation.arguments == {"nested": [1]}
    assert operation.operation_id == identity
    assert OperationSpec.from_json(operation.to_json()) == operation
    assert spec(input_revision="different").operation_id != spec().operation_id


@pytest.mark.parametrize(
    "arguments", [{1: "coerced"}, {"bad": float("nan")}, {"bad": Path("private")}]
)
def test_spec_rejects_ambiguous_or_non_json_arguments(arguments):
    with pytest.raises(ValueError):
        spec(arguments=arguments)


def test_runtime_records_private_metadata_and_refuses_unsupported_cancel(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    runtime = JobRuntime(store)
    entered, release = threading.Event(), threading.Event()
    operation = spec(cancellable=False)

    def run(progress, cancel):
        entered.set()
        assert release.wait(5)
        assert not cancel.is_set()
        return {}

    try:
        task = runtime.submit(
            kind="test", args=operation.arguments, run=run, operation=operation
        )["task_id"]
        assert entered.wait(5)
        assert runtime.cancel(task) is False
        row = store.get(task)
        assert OperationSpec.from_json(row.operation_json) == operation
        projection = row.to_safe_projection()
        assert projection["operation"]["cancellable"] is False
        assert "private source" not in str(projection)
        assert "nested" not in str(projection)
    finally:
        release.set()
        runtime.shutdown()
    assert store.get(task).status == "completed"


def test_runtime_rejects_spec_mismatch_before_insertion():
    runtime = JobRuntime.for_session()
    try:
        with pytest.raises(ValueError, match="persistence"):
            runtime.submit(
                kind="test",
                args=spec().arguments,
                run=lambda p, c: {},
                operation=spec(),
            )
        with pytest.raises(ValueError, match="metadata"):
            runtime.submit(kind="wrong", args={}, run=lambda p, c: {}, operation=spec())
        assert runtime.store.list() == []
        operation = spec(persistence="session_only", cancellable=False)
        with pytest.raises(ValueError, match="cancellation event"):
            runtime.submit(
                kind="test",
                args=operation.arguments,
                run=lambda p, c: {},
                operation=operation,
                cancellation_event=threading.Event(),
            )
        assert runtime.store.list() == []
    finally:
        runtime.close_session()

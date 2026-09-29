import json
import logging

import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.ci.run_file import app
from tests.ci.test.conftest import _SnapshotFileCase
from typer.testing import CliRunner

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


@pytest.mark.parametrize("golden_value", ["actual", "different"])
def test_successful_child_completes_attempt_and_checks_the_golden(
    snapshot_file_case: _SnapshotFileCase, golden_value: str
) -> None:
    """A real successful child completes its attempt even when the snapshot comparison fails."""
    snapshot_file_case.write_golden(value=golden_value)
    before = snapshot_file_case.golden.read_bytes()

    result = CliRunner().invoke(app, ["--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "10"])

    assert result.exit_code == (0 if golden_value == "actual" else -1), result.output
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text()) == {"test": str(snapshot_file_case.test_file), "completed": True}
    assert (attempt.parent / "records/record.json").is_file()
    assert snapshot_file_case.golden.read_bytes() == before


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_failed_or_timed_out_child_keeps_an_incomplete_attempt(
    snapshot_file_case: _SnapshotFileCase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    """Real child failure and timeout retain raw records without claiming successful completion."""
    snapshot_file_case.write_golden(value="actual")
    caplog.set_level(logging.INFO)
    monkeypatch.setenv(
        "FILE_RUN_TEST_EXIT" if failure == "exit" else "FILE_RUN_TEST_SLEEP", "7" if failure == "exit" else "60"
    )

    result = CliRunner().invoke(app, ["--test-file", str(snapshot_file_case.test_file), "--timeout-seconds", "2"])

    assert result.exit_code == -1, result.output
    assert ("returned exit code 7" if failure == "exit" else "after 2 seconds") in caplog.text
    [attempt] = snapshot_file_case.record_root.glob("*/*/attempt.json")
    assert json.loads(attempt.read_text()) == {"test": str(snapshot_file_case.test_file), "completed": False}
    assert (attempt.parent / "records/record.json").is_file()

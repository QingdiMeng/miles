"""Check termination handling and real distributed-checkpoint discovery."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp

from examples.decision_calibration.eval_after_training import checkpoint_complete, ready_checkpoints, retry, run, training_terminated, wait_for_training


class WatcherTests(unittest.TestCase):
    def test_terminal_states(self) -> None:
        for status in ("SUCCEEDED", "FAILED", "STOPPED"):
            self.assertTrue(training_terminated(status, False, 0))
            self.assertFalse(training_terminated(status, True, 0))
        for status in ("PENDING", "RUNNING"):
            self.assertFalse(training_terminated(status, False, 10))
        self.assertFalse(training_terminated(None, False, 2))
        self.assertTrue(training_terminated(None, False, 3))

    def test_checkpoint_discovery_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            for step in (0, 127, 255, 383):
                dcp.save({"model": {"weight": torch.ones(2)}}, checkpoint_id=checkpoints / f"iter_{step:07d}")
            incomplete = checkpoints / "iter_0000383"
            next(incomplete.glob("*.distcp")).unlink()
            self.assertTrue(checkpoint_complete(checkpoints / "iter_0000127"))
            self.assertFalse(checkpoint_complete(incomplete))
            self.assertFalse(checkpoint_complete(checkpoints / "iter_0000511"))
            self.assertEqual([p.name for p in ready_checkpoints(checkpoints, 128)], ["iter_0000127", "iter_0000255"])
            args = SimpleNamespace(result_dir=root / "results", no_wait=True, dry_run=True,
                                   checkpoint_dir=checkpoints, min_updates=128)
            run(args)
            state = json.loads((args.result_dir / "status.json").read_text())
            self.assertEqual(state["training_status"], "DRY_RUN")
            self.assertEqual(len(state["checkpoints"]), 2)
            args.no_wait = False
            args.dry_run = False
            args.include_baseline = True
            args.model = root / "model"
            with patch("examples.decision_calibration.eval_after_training.wait_for_training", return_value="FAILED"), \
                 patch("examples.decision_calibration.eval_after_training.serve_and_evaluate") as baseline, \
                 patch("examples.decision_calibration.eval_after_training.evaluate_checkpoint") as evaluate:
                run(args)
            self.assertEqual(baseline.call_count, 1)
            self.assertEqual([c.args[1].name for c in evaluate.call_args_list], ["iter_0000127", "iter_0000255"])
            self.assertEqual(json.loads((args.result_dir / "status.json").read_text())["completed"], ["baseline", "step128", "step256"])

    def test_actual_wait_loop_on_failure_and_lost_ray(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(result_dir=Path(directory), ray_address="http://localhost:8265", run_id="test", ray_job_id="test", poll_seconds=20)
            with patch("examples.decision_calibration.eval_after_training.driver_alive", return_value=False), \
                 patch("examples.decision_calibration.eval_after_training.time.sleep"), \
                 patch("examples.decision_calibration.eval_after_training.JobSubmissionClient") as client:
                client.return_value.get_job_status.return_value = "FAILED"
                self.assertEqual(wait_for_training(args), "FAILED")
                client.side_effect = ConnectionError("Ray offline")
                self.assertEqual(wait_for_training(args), "DRIVER_ABSENT")

    def test_retry_continues_without_exception_scope_bug(self) -> None:
        with patch("examples.decision_calibration.eval_after_training.time.sleep"):
            error = retry("bad", lambda: (_ for _ in ()).throw(ValueError("probe")))
        self.assertEqual(error, "bad: ValueError: probe")
        self.assertIsNone(retry("good", lambda: None))


if __name__ == "__main__":
    unittest.main()

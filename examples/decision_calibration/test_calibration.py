"""Checks for the dataset, probability protocol and actual policy gradients."""

import csv
import io
import json
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from examples.decision_calibration.loss import action_surrogate, policy_loss
from examples.decision_calibration.prepare import build
from examples.decision_calibration.probabilities import brier, extract_probabilities, score_metrics
from miles.backends.training_utils.data import DataIterator, get_batch


def _archive() -> bytes:
    csv_text = io.StringIO()
    writer = csv.writer(csv_text)
    writer.writerow(["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"])
    writer.writerows([[f"Distinct fixture question {i}", "right", "wrong1", "wrong2", "wrong3"] for i in range(448)])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dataset/gpqa_main.csv", csv_text.getvalue())
    return buffer.getvalue()


class CalibrationTests(unittest.TestCase):
    def test_disjoint_reproducible_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            archive = _archive()
            manifest = build(first, archive, 261001)
            self.assertEqual(manifest, build(second, archive, 261001))
            seen = set()
            for name, count in (("train", 1024), ("validation", 256), ("test", 512)):
                rows = [json.loads(line) for line in (first / f"{name}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), count)
                ids = {row["metadata"]["id"] for row in rows}
                self.assertFalse(seen & ids)
                seen.update(ids)
                for row in rows:
                    self.assertAlmostEqual(sum(row["metadata"]["target"]), 1)
                    self.assertTrue(all(p >= 0 for p in row["metadata"]["target"]))

    def test_protocol_reorders_ids_and_stabilizes_softmax(self) -> None:
        output = {"meta_info": {"output_token_ids_logprobs": [[[-1002, 3], [-1000, 1], [-1001, 2], [-1003, 4]]]}}
        actual = extract_probabilities(output, [1, 2, 3, 4])
        expected = torch.tensor([0.0, -1.0, -2.0, -3.0]).softmax(0).tolist()
        for a, e in zip(actual, expected, strict=True):
            self.assertAlmostEqual(a, e, places=7)

    def test_soft_targets_and_irreducible_brier(self) -> None:
        target = [0.25] * 4
        self.assertEqual(brier(target, target), 0)
        self.assertEqual(score_metrics(target, target)["brier_expected"], 0.75)
        self.assertEqual(brier([1, 0, 0, 0], target), 0.75)

    def test_expected_rl_gradient_equals_brier_gradient(self) -> None:
        for target in ([1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4], [0.25] * 4):
            logits = torch.tensor([0.5, -0.2, 1.2, -1.5], requires_grad=True)
            target_tensor = torch.tensor(target)
            old = logits.detach().softmax(0)
            expected_surrogate = sum(old[action] * action_surrogate(logits, old, target_tensor, action) for action in range(4))
            actual = torch.autograd.grad(expected_surrogate, logits)[0]
            direct = (logits.softmax(0) - target_tensor).square().sum()
            expected = torch.autograd.grad(direct, logits)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_stale_actions_still_give_current_brier_gradient(self) -> None:
        logits = torch.tensor([1.5, -0.8, 0.2, -1.5], requires_grad=True)
        old = torch.tensor([0.1, 0.4, 0.3, 0.2])
        target = torch.tensor([0.1, 0.2, 0.3, 0.4])
        objective = sum(old[a] * action_surrogate(logits, old, target, a) for a in range(4))
        actual = torch.autograd.grad(objective, logits)[0]
        expected = torch.autograd.grad((logits.softmax(0) - target).square().sum(), logits)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "Real get_batch consumer requires CUDA")
    def test_metadata_survives_real_batch_and_loss(self) -> None:
        state = SimpleNamespace(tp=SimpleNamespace(size=1), cp=SimpleNamespace(size=1, rank=0))
        metadata = {"candidate_token_ids": [0, 1, 2, 3], "probabilities": [0.25] * 4, "target": [1, 0, 0, 0], "action": 2, "reward": -0.5}
        data = {"tokens": [torch.tensor([4, 5, 2], device="cuda")], "total_lengths": [3], "response_lengths": [1], "loss_masks": [torch.tensor([1], device="cuda")], "metadata": [metadata]}
        with patch("miles.backends.training_utils.parallel._parallel_state", state):
            iterator = DataIterator(data, micro_batch_size=1)
            batch = get_batch(iterator, ["tokens", "total_lengths", "response_lengths", "loss_masks"], pad_multiplier=1)
            self.assertEqual(batch["metadata"], [metadata])
            logits = torch.zeros(1, 3, 6, device="cuda", requires_grad=True)
            args = Namespace(qkv_format="thd", true_on_policy_mode=False, rollout_temperature=1)
            loss, metrics = policy_loss(args, batch, logits, torch.mean)
            loss.backward()
            self.assertEqual(set(metrics), {"loss", "brier_excess", "decision_entropy", "decision_probability_abs_diff"})
            self.assertGreater(logits.grad[0, 1, 2].item(), 0)
            self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

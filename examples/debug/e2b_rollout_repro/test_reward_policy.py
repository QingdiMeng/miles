import json
import tempfile
import unittest
from pathlib import Path

from reward_policy import apply_reward_policy


class RewardPolicyTest(unittest.TestCase):
    def test_timeout_types_and_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("AgentTimeoutError", "VerifierTimeoutError", "EnvironmentStartTimeoutError", "TimeoutException"):
                for reward in (None, 0.0, 1.0):
                    with self.subTest(kind=kind, reward=reward):
                        verifier = None if reward is None else {"rewards": {"reward": reward}}
                        (Path(directory) / "result.json").write_text(json.dumps({
                            "exception_info": {"exception_type": kind}, "verifier_result": verifier}))
                        original = {"trial_dir": directory, "exit_status": "TimeLimitExceeded", "reward": 0.0}
                        result = apply_reward_policy(original)
                        self.assertEqual(result["pilot_valid_for_training"], kind == "AgentTimeoutError")
                        if kind == "AgentTimeoutError":
                            self.assertEqual(result["reward"], reward if reward is not None else 0.0)
                        self.assertNotIn("pilot_valid_for_training", original)

    def test_truncation_and_unknown_timeout(self) -> None:
        result = apply_reward_policy({"exit_status": "SequenceLengthLimitExceeded", "reward": 1.0})
        self.assertTrue(result["pilot_valid_for_training"])
        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(apply_reward_policy({"exit_status": "TimeLimitExceeded"})["pilot_valid_for_training"])

    def test_context_and_specific_parser_failure(self) -> None:
        parser_trace = 'terminus_xml_plain_parser.py in _find_top_level_tags: tag_content.split()[0]'
        cases = (
            ("ContextLengthExceededError", "", True),
            ("IndexError", parser_trace, True),
            ("IndexError", "another_module.py: values[0]", False),
            ("IndexError", "terminus_xml_plain_parser.py: another_function", False),
            ("TimeoutException", "", False),
            ("InternalServerError", "", False),
        )
        with tempfile.TemporaryDirectory() as directory:
            for kind, trace, accepted in cases:
                with self.subTest(kind=kind, trace=trace):
                    (Path(directory) / "result.json").write_text(json.dumps({
                        "exception_info": {"exception_type": kind, "exception_traceback": trace},
                        "verifier_result": {"rewards": {"reward": 1.0}},
                    }))
                    original = {"trial_dir": directory, "exit_status": "AgentError", "reward": 1.0}
                    result = apply_reward_policy(original)
                    self.assertEqual(result["pilot_valid_for_training"], accepted)
                    if accepted:
                        self.assertEqual(result["reward"], 0.0)
                        self.assertEqual(apply_reward_policy(result), result)
                    self.assertEqual(original["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()

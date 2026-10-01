"""Boundary checks for probability-report scoring and prompt conversion."""

import unittest

from examples.decision_calibration.report_reward import parse_report, report_messages, score_report


class ReportTests(unittest.TestCase):
    def test_scores(self) -> None:
        target = [1.0, 0.0, 0.0, 0.0]
        self.assertEqual(score_report('{"A":1,"B":0,"C":0,"D":0}', target)["reward"], 0)
        self.assertEqual(score_report('{"A":0,"B":1,"C":0,"D":0}', target)["reward"], -2)
        self.assertEqual(score_report('{"A":0.25,"B":0.25,"C":0.25,"D":0.25}', target)["reward"], -0.75)
        self.assertEqual(score_report('{"A":0.1,"B":0.2,"C":0.3,"D":0.4}', [0.1, 0.2, 0.3, 0.4])["reward"], 0)

    def test_invalid_reports(self) -> None:
        for text in ('', '{}', '[]', 'null', '{"A":1,"A":0,"B":0,"C":0,"D":0}', '{"A":true,"B":0,"C":0,"D":0}', '{"A":"1","B":0,"C":0,"D":0}', '{"A":NaN,"B":0,"C":0,"D":0}', '{"A":Infinity,"B":0,"C":0,"D":0}', '{"A":-0.1,"B":1.1,"C":0,"D":0}', '{"A":0.2,"B":0.2,"C":0.2,"D":0.2}', '{"A":1,"B":0,"C":0,"D":0,"E":0}', '```json\n{"A":1,"B":0,"C":0,"D":0}\n```'):
            with self.subTest(text=text):
                self.assertEqual(score_report(text, [1, 0, 0, 0])["reward"], -3)

    def test_order_and_targets(self) -> None:
        self.assertEqual(parse_report('{"D":0.4,"C":0.3,"B":0.2,"A":0.1}'), [0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            score_report('{}', [0, 0, 0, 0])

    def test_prompt_conversion(self) -> None:
        messages = [{"role": "user", "content": "Take a guess.\n\nChoose one option as your guess. Output only its letter (A, B, C, or D)."}]
        result = report_messages(messages)
        self.assertNotIn("Output only its letter", result[0]["content"])
        self.assertIn("Output only JSON", result[0]["content"])
        self.assertIn("Output only its letter", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()

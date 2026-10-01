"""Score generated JSON probability reports with negative Brier loss."""

import json
import math
from collections.abc import Sequence

LETTERS = ("A", "B", "C", "D")
REPORT_INSTRUCTION = 'Report your probability for each option as a JSON object with exactly the keys "A", "B", "C", "D". Use numbers between 0 and 1 that sum to 1. Output only JSON, without explanation or Markdown.'


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_report(text: str) -> list[float]:
    obj = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(obj, dict) or set(obj) != set(LETTERS):
        raise ValueError("Expected exactly A, B, C, D")
    values = [obj[key] for key in LETTERS]
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Probabilities must be finite numbers in [0,1]")
    if not math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one")
    return [float(x) for x in values]


def score_report(text: str, target: Sequence[float]) -> dict:
    if len(target) != 4 or any(not math.isfinite(x) or x < 0 for x in target) or not math.isclose(sum(target), 1):
        raise ValueError("Invalid target distribution")
    try:
        probabilities = parse_report(text)
    except (ValueError, TypeError):
        # Valid four-choice Brier loss is at most 2; malformed reports are worse.
        return {"reward": -3.0, "valid": False, "probabilities": None, "brier": None}
    loss = sum((p - t) ** 2 for p, t in zip(probabilities, target, strict=True))
    return {"reward": -loss, "valid": True, "probabilities": probabilities, "brier": loss}


def report_messages(messages: list[dict]) -> list[dict]:
    result = [dict(m) for m in messages]
    text = result[-1]["content"]
    for suffix in ("Choose one option. Output only its letter (A, B, C, or D).", "Choose one option as your guess. Output only its letter (A, B, C, or D)."):
        text = text.removesuffix(suffix).rstrip()
    result[-1]["content"] = text + "\n\n" + REPORT_INSTRUCTION
    return result

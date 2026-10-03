"""Score generated JSON probability reports with negative Brier loss."""

import json
import math
from collections.abc import Sequence

LETTERS = ("A", "B", "C", "D")
LEGACY_REPORT_INSTRUCTION = 'Your final answer will be assessed by how close your reported probabilities are to the actual ground-truth answer distribution, using negative Brier loss: -sum_i (p_i - t_i)^2. Higher scores are better. For single-correct-answer questions, the target assigns probability 1 to the correct option and 0 to the others. For random-outcome questions, the target is the true outcome distribution. Report your uncertainty honestly: putting probability 1 on an incorrect option is penalized more than spreading probability across plausible options. Report your probability for each option as a JSON object with exactly the keys "A", "B", "C", "D". Use numbers between 0 and 1 that sum to 1. Output only JSON, without explanation or Markdown.'
PREVIOUS_REPORT_INSTRUCTION = LEGACY_REPORT_INSTRUCTION.replace(
    'Report your probability for each option as a JSON object with exactly the keys "A", "B", "C", "D". Use numbers between 0 and 1 that sum to 1.',
    'Report your probabilities as a JSON object using only option keys "A", "B", "C", "D". Omitted options have probability zero. Use finite nonnegative numbers with a positive total; we normalize them to sum to 1 before scoring. Negative values are invalid.',
)
REPORT_INSTRUCTION = PREVIOUS_REPORT_INSTRUCTION.replace(
    ', using negative Brier loss: -sum_i (p_i - t_i)^2', '',
).replace('For single-correct-answer questions,', 'If there is only one correct answer,')


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_report(text: str, option_count: int = 4) -> list[float]:
    """Fill omitted options with zero and normalize nonnegative JSON weights."""
    if not 2 <= option_count <= 10:
        raise ValueError("Expected 2 to 10 options")
    letters = tuple(chr(65 + i) for i in range(option_count))
    obj = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(obj, dict) or not set(obj).issubset(letters):
        raise ValueError(f"Expected only option keys from {letters}")
    values = [obj.get(key, 0) for key in letters]
    if any(type(x) not in (int, float) for x in values):
        raise ValueError("Probabilities must be numeric")
    try:
        values = [float(x) for x in values]
    except OverflowError as exc:
        raise ValueError("Probabilities must be finite numbers") from exc
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Probabilities must be finite nonnegative numbers")
    maximum = max(values)
    if maximum == 0:
        raise ValueError("Probabilities must have a positive total")
    try:
        total = math.fsum(values)
    except OverflowError:
        # Scale before summing when finite weights would overflow their total.
        values = [x / maximum for x in values]
        total = math.fsum(values)
    return [x / total for x in values]


def score_report(text: str, target: Sequence[float]) -> dict:
    if not 2 <= len(target) <= 10 or any(not math.isfinite(x) or x < 0 for x in target) or not math.isclose(sum(target), 1):
        raise ValueError("Invalid target distribution")
    try:
        probabilities = parse_report(text, len(target))
    except (ValueError, TypeError):
        # Valid Brier loss is at most 2 for any option count.
        return {"reward": -3.0, "valid": False, "probabilities": None, "brier": None}
    loss = sum((p - t) ** 2 for p, t in zip(probabilities, target, strict=True))
    return {"reward": -loss, "valid": True, "probabilities": probabilities, "brier": loss}


def report_messages(messages: list[dict], option_count: int = 4) -> list[dict]:
    if not 2 <= option_count <= 10:
        raise ValueError("Expected 2 to 10 options")
    result = [dict(m) for m in messages]
    text = result[-1]["content"]
    for suffix in ("Choose one option. Output only its letter (A, B, C, or D).", "Choose one option as your guess. Output only its letter (A, B, C, or D)."):
        text = text.removesuffix(suffix).rstrip()
    keys = ", ".join(json.dumps(chr(65 + i)) for i in range(option_count))
    for previous in (LEGACY_REPORT_INSTRUCTION, PREVIOUS_REPORT_INSTRUCTION):
        text = text.removesuffix(previous.replace('"A", "B", "C", "D"', keys)).rstrip()
    instruction = REPORT_INSTRUCTION.replace('"A", "B", "C", "D"', keys)
    result[-1]["content"] = text if text.endswith(instruction) else text + "\n\n" + instruction
    return result

"""Decision probability protocol and metrics shared by rollout and evaluation."""

import math
from collections.abc import Sequence

LETTERS = ("A", "B", "C", "D")


def candidate_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(value) != 1 for value in ids) or len({value[0] for value in ids}) != 4:
        raise ValueError("Decision letters must be four distinct single tokens")
    return [value[0] for value in ids]


def decision_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    if not isinstance(messages, list):
        raise ValueError("Supply unrendered messages; omit --apply-chat-template")
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


def request_payload(prompt_ids: list[int], token_ids: list[int]) -> dict:
    # The server's generated token is ignored: sample locally from the exact
    # candidate-normalized distribution to avoid grammar/backend semantics.
    return {
        "input_ids": prompt_ids,
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": -1,
        "token_ids_logprob": token_ids,
    }


def extract_candidate_logprobs(output: dict, token_ids: list[int]) -> list[float]:
    scores = output["meta_info"]["output_token_ids_logprobs"]
    if len(scores) != 1:
        raise ValueError(f"Expected one decision position, received {len(scores)}")
    by_id = {int(item[1]): float(item[0]) for item in scores[0]}
    values = [by_id[token] for token in token_ids]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite candidate log probabilities: {values}")
    return values


def extract_probabilities(output: dict, token_ids: list[int]) -> list[float]:
    values = extract_candidate_logprobs(output, token_ids)
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    return [weight / sum(weights) for weight in weights]


def brier(probabilities: Sequence[float], target: Sequence[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(probabilities, target, strict=True))


def score_metrics(probabilities: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    excess = brier(probabilities, target)
    return {
        "brier_excess": excess,
        "brier_expected": excess + 1 - sum(t * t for t in target),
        "cross_entropy": -sum(t * math.log(max(p, 1e-30)) for p, t in zip(probabilities, target, strict=True)),
        "entropy": -sum(p * math.log(max(p, 1e-30)) for p in probabilities),
        "top_choice_success_probability": target[max(range(len(probabilities)), key=probabilities.__getitem__)],
    }

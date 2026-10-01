"""Sample one candidate action and attach a detached Brier-gradient reward."""

import random

from examples.decision_calibration.probabilities import (
    LETTERS,
    brier,
    candidate_token_ids,
    decision_prompt_ids,
    extract_candidate_logprobs,
    extract_probabilities,
    request_payload,
    score_metrics,
)
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import compute_routing_headers
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample, tokenizer = input.sample, input.state.tokenizer
    tokens = candidate_token_ids(tokenizer)
    prompt = decision_prompt_ids(tokenizer, sample.prompt)
    if len(prompt) + 1 > input.args.rollout_max_context_len:
        raise ValueError("Question exceeds context budget; refusing silent truncation")
    url = f"http://{input.args.sglang_router_ip}:{input.args.sglang_router_port}/generate"
    output = await post(url, request_payload(prompt, tokens), headers=compute_routing_headers(input.args, sample))
    probabilities = extract_probabilities(output, tokens)
    rng = random.Random(f"{input.args.seed}:{sample.index}:{sample.rollout_id}")
    action = rng.choices(range(4), weights=probabilities, k=1)[0]
    target = sample.metadata["target"]
    reward = 2 * (target[action] - probabilities[action])
    sample.tokens = prompt + [tokens[action]]
    sample.response = LETTERS[action]
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.reward = -brier(probabilities, target) if input.evaluation else reward
    # Standard Miles diagnostics compare full-vocabulary log probabilities.
    # The custom objective uses candidate-normalized probabilities separately.
    sample.rollout_log_probs = [extract_candidate_logprobs(output, tokens)[action]]
    sample.train_metadata = {
        "candidate_token_ids": tokens,
        "probabilities": probabilities,
        "target": target,
        "action": action,
        "reward": reward,
    }
    sample.metadata["calibration_metrics"] = score_metrics(probabilities, target)
    sample.update_from_meta_info(input.args, output["meta_info"])
    sample.status = Sample.Status.COMPLETED
    return GenerateFnOutput(samples=sample)


async def reward(args, sample: Sample, **kwargs) -> float:
    if sample.reward is None:
        raise ValueError("Calibration generation did not provide a reward")
    return sample.reward

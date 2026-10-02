"""Generate probability reports for conventional sequence-level GRPO."""

from examples.decision_calibration.probabilities import decision_prompt_ids
from examples.decision_calibration.report_reward import report_messages, score_report
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import compute_routing_headers, update_sample_from_response
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample, args = input.sample, input.args
    prompt = decision_prompt_ids(input.state.tokenizer, report_messages(sample.prompt))
    cap = args.eval_max_response_len if input.evaluation else args.rollout_max_response_len
    if len(prompt) + cap > args.rollout_max_context_len:
        raise ValueError("Question exceeds context budget")
    payload = {"input_ids": prompt, "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": cap}, "return_logprob": True, "logprob_start_len": -1}
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True
    output = await post(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate", payload, headers=compute_routing_headers(args, sample))
    await update_sample_from_response(args, sample, payload, output, update_loss_mask=True)
    if not sample.response_length or len(sample.rollout_log_probs) != sample.response_length:
        raise ValueError("Missing generated token log probabilities")
    metrics = score_report(sample.response, sample.metadata["target"])
    # A token-limited unfinished report receives the format penalty.
    sample.reward = metrics["reward"]
    sample.metadata["report_metrics"] = metrics
    return GenerateFnOutput(samples=sample)


async def reward(args, sample: Sample, **kwargs) -> float:
    if sample.reward is None:
        raise ValueError("Missing probability-report reward")
    return sample.reward

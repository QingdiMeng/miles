"""One-token Qwen3.6-35B-A3B Brier policy-gradient pilot on two 8-GPU nodes.

Requires a converted original checkpoint, prepared decision-calibration data,
and an externally joined Ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1).
Prints the configuration unless --launch is supplied. CLI parsing uses Tap.

Args:
    model_dir: Parent of original HF and *_torch_dist checkpoints.
    data_dir: Directory containing train.jsonl and validation.jsonl.
    output_dir: Writable experiment root with space for full checkpoints.
    run_id: Reproducible run identifier, generated before every new launch.

Example:
    python scripts/run_qwen3_6_decision_calibration.py \
        --model-dir /path/to/models --data-dir /path/to/data \
        --output-dir /path/to/results --run-id YYMMDD-0123abcd
"""

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from tap import Tap

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    num_nodes: int = 2
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets/decision-calibration"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8
    num_rollout: int = 200
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    learning_rate: float = 1e-6
    context_length: int = 65536
    max_tokens_per_gpu: int = 8192
    eval_interval: int = 25
    save_interval: int = 50
    seed: int = 261001
    wandb_project: str | None = None
    dump_details: str | None = None
    launch: bool = False
    node_order: str | None = None
    wandb_team: str | None = None
    probability_report: bool = False
    report_max_response_len: int = 128
    routing_replay: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id


class CLI(Tap):
    run_id: str
    model_dir: str = ScriptArgs.model_dir
    data_dir: str = ScriptArgs.data_dir
    output_dir: str = ScriptArgs.output_dir
    megatron_path: str = ScriptArgs.megatron_path
    num_rollout: int = ScriptArgs.num_rollout
    rollout_batch_size: int = ScriptArgs.rollout_batch_size
    n_samples_per_prompt: int = ScriptArgs.n_samples_per_prompt
    learning_rate: float = ScriptArgs.learning_rate
    context_length: int = ScriptArgs.context_length
    max_tokens_per_gpu: int = ScriptArgs.max_tokens_per_gpu
    eval_interval: int = ScriptArgs.eval_interval
    save_interval: int = ScriptArgs.save_interval
    seed: int = ScriptArgs.seed
    wandb_project: str | None = ScriptArgs.wandb_project
    dump_details: str | None = ScriptArgs.dump_details
    launch: bool = ScriptArgs.launch
    node_order: str | None = ScriptArgs.node_order
    wandb_team: str | None = ScriptArgs.wandb_team
    probability_report: bool = ScriptArgs.probability_report
    report_max_response_len: int = ScriptArgs.report_max_response_len
    routing_replay: bool = ScriptArgs.routing_replay


def _wandb_args(args: ScriptArgs) -> str:
    defaults = shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_id))
    # Authentication stays in the environment/netrc, never printed argv.
    if "--wandb-key" in defaults:
        index = defaults.index("--wandb-key")
        del defaults[index : index + 2]
    if args.wandb_project is not None:
        if "--wandb-project" in defaults:
            defaults[defaults.index("--wandb-project") + 1] = args.wandb_project
        else:
            defaults = ["--use-wandb", "--wandb-project", args.wandb_project, "--wandb-group", args.run_id, "--disable-wandb-random-suffix"]
    if args.wandb_team is not None:
        defaults.extend(["--wandb-team", args.wandb_team])
    return shlex.join(defaults)


def _train_args(args: ScriptArgs) -> str:
    if args.num_nodes != 2 or args.num_gpus_per_node != 8:
        raise ValueError("This pilot recipe requires two nodes with eight GPUs each")
    quote = shlex.quote
    model = Path(args.model_dir) / "Qwen3.6-35B-A3B"
    checkpoint = f"""
        --hf-checkpoint {quote(str(model))} --ref-load {quote(str(model) + "_torch_dist")}
        --save {quote(str(args.run_dir / "checkpoints"))} --save-interval {args.save_interval}
    """
    rollout = f"""
        --prompt-data {quote(str(Path(args.data_dir) / "train.jsonl"))}
        --input-key prompt --metadata-key metadata --rollout-shuffle
        --num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size}
        --n-samples-per-prompt {args.n_samples_per_prompt} --num-steps-per-rollout 1
        --rollout-max-response-len 1 --rollout-max-context-len {args.context_length}
        --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1
        --custom-generate-function-path examples.decision_calibration.rollout.generate
        --custom-rm-path examples.decision_calibration.rollout.reward
    """
    algorithm = """
        --loss-type custom_loss
        --custom-loss-function-path examples.decision_calibration.loss.policy_loss
        --disable-compute-advantages-and-returns --entropy-coef 0 --kl-coef 0
        --mtp-num-layers 0
    """
    optimizer = f"""
        --optimizer adam --lr {args.learning_rate} --lr-decay-style constant
        --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.98 --use-distributed-optimizer
    """
    performance = f"""
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
        --context-parallel-size 1 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
        --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
        --use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu}
        --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8
    """
    sglang = f"""
        --rollout-num-gpus 8 --rollout-num-gpus-per-engine 1 --sglang-tp-size 1
        --sglang-mem-fraction-static 0.75 --sglang-context-length {args.context_length}
        --sglang-max-running-requests 32 --sglang-chunked-prefill-size 4096
        --sglang-router-policy cache_aware
    """
    evaluation = f"""
        --eval-interval {args.eval_interval}
        --eval-prompt-data validation {quote(str(Path(args.data_dir) / "validation.jsonl"))}
        --n-samples-per-eval-prompt 1 --eval-max-response-len 1
    """
    traces = args.dump_details or str(args.run_dir / "traces")
    misc = f"""
        --bf16 --attention-dropout 0 --hidden-dropout 0 --attention-backend flash
        --seq-length {args.context_length}
        --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
        --seed {args.seed} --dump-details {quote(traces)} --use-miles-dashboard
        --observe-training-entropy --use-rollout-entropy --use-prometheus
    """
    if args.probability_report:
        rollout = rollout.replace("--rollout-max-response-len 1", f"--rollout-max-response-len {args.report_max_response_len}")
        rollout = rollout.replace("examples.decision_calibration.rollout.", "examples.decision_calibration.report_rollout.")
        algorithm = "--loss-type policy_loss --advantage-estimator grpo --entropy-coef 0 --kl-coef 0 --mtp-num-layers 0"
        misc += " --custom-rollout-log-function-path examples.decision_calibration.report_logging.log_rollouts --custom-eval-rollout-log-function-path examples.decision_calibration.report_logging.log_evaluation"
        evaluation = evaluation.replace("--eval-max-response-len 1", f"--eval-max-response-len {args.report_max_response_len}")
    if args.routing_replay:
        algorithm += " --use-rollout-routing-replay"
    return " ".join(shlex.join(shlex.split(block)) for block in (checkpoint, rollout, algorithm, optimizer, performance, sglang, evaluation, misc, _wandb_args(args)))


def execute(args: ScriptArgs) -> None:
    backend = args.create_backend()
    extra_env_vars = {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    if args.node_order is not None:
        extra_env_vars["MILES_RAY_NODE_ORDER"] = args.node_order
    backend.execute_train(
        train_script="train_async.py",
        extra_env_vars=extra_env_vars,
        train_args=_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.6-35B-A3B",
        megatron_path=args.megatron_path,
    )


def main() -> None:
    cli = CLI(underscores_to_dashes=True).parse_args()
    args = ScriptArgs(**cli.as_dict())
    if args.launch:
        execute(args)
    else:
        print(_train_args(args))


if __name__ == "__main__":
    main()

"""Log JSON report validity and exact probability-one collapse percentages."""

from collections.abc import Sequence

from examples.decision_calibration.report_reward import score_report
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking


def report_statistics(samples: Sequence) -> dict[str, float]:
    groups = {"all": list(samples)}
    for sample in samples:
        source = sample.metadata["source"]
        groups.setdefault(source, []).append(sample)
    result = {}
    for source, group in groups.items():
        if not group:
            continue
        reports = [score_report(s.response, s.metadata["target"]) for s in group]
        valid = [r for r in reports if r["valid"]]
        collapsed = sum(max(r["probabilities"]) == 1.0 for r in valid)
        prefix = "" if source == "all" else source + "/"
        result[prefix + "report_count"] = len(group)
        result[prefix + "probability_collapse_count"] = collapsed
        result[prefix + "probability_collapse_pct"] = 100 * collapsed / len(group)
        result[prefix + "report_valid_pct"] = 100 * len(valid) / len(group)
        if valid:
            result[prefix + "probability_collapse_valid_pct"] = 100 * collapsed / len(valid)
    return result


def log_rollouts(rollout_id: int, args, samples: Sequence, rollout_extra_metrics: dict | None, rollout_time: float) -> bool:
    metrics = {"rollout/" + k: v for k, v in report_statistics(samples).items()}
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking.log(args, metrics, step_key="rollout/step")
    return False


def log_evaluation(rollout_id: int, args, data: dict, extra_metrics: dict | None) -> bool:
    metrics = {}
    for name, value in data.items():
        samples = value.get("samples")
        if samples is not None:
            metrics.update({f"eval/{name}/" + k: v for k, v in report_statistics(samples).items()})
    metrics["eval/step"] = compute_rollout_step(args, rollout_id)
    tracking.log(args, metrics, step_key="eval/step")
    return False

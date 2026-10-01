"""Record decision probabilities and compare paired held-out Brier scores."""

import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx
import polars as pl
from tap import Tap
from transformers import AutoTokenizer

from examples.decision_calibration.probabilities import (
    candidate_token_ids,
    decision_prompt_ids,
    extract_probabilities,
    request_payload,
    score_metrics,
)


class Args(Tap):
    data: Path
    output: Path
    model: str
    endpoint: str
    concurrency: int = 16
    baseline: Path | None = None
    seed: int = 261001


def _ece(rows: list[dict]) -> float:
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        p, target = row["probabilities"], row["target"]
        action = max(range(len(p)), key=p.__getitem__)
        bins[min(int(p[action] * 10), 9)].append((p[action], target[action]))
    return sum(abs(sum(p - t for p, t in group)) for group in bins.values()) / len(rows)


def _paired_delta(rows: list[dict], baseline: dict[str, dict], seed: int) -> dict:
    deltas = []
    for row in rows:
        before = baseline[row["id"]]
        if row["target"] != before["target"] or row["source"] != before["source"]:
            raise ValueError("Baseline target/source mismatch")
        deltas.append(row["metrics"]["brier_excess"] - before["metrics"]["brier_excess"])
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
    return {"after_minus_before": sum(deltas) / len(deltas), "paired_bootstrap_95pct": [estimates[49], estimates[1949]]}


def summarize(rows: list[dict], baseline: dict[str, dict] | None, seed: int) -> dict:
    if baseline is not None and set(baseline) != {row["id"] for row in rows}:
        raise ValueError("Baseline must contain exactly the same held-out examples")
    groups = {"all": rows, "synthetic": [row for row in rows if row["source"] != "gpqa"]}
    groups.update({source: [row for row in rows if row["source"] == source] for source in ("gpqa", "coin", "dice", "urn")})
    report = {}
    for name, group in groups.items():
        if not group:
            continue
        metrics = {key: sum(row["metrics"][key] for row in group) / len(group) for key in group[0]["metrics"]}
        metrics["top_label_ece_10_bins"] = _ece(group)
        report[name] = {"rows": len(group), **metrics}
        if baseline is not None:
            report[name]["brier_change"] = _paired_delta(group, baseline, seed)
    return report


async def evaluate(args: Args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    candidates = candidate_token_ids(tokenizer)
    rows = pl.read_ndjson(args.data).to_dicts()
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=10800) as client:

        async def score(row: dict) -> dict:
            async with semaphore:
                prompt = decision_prompt_ids(tokenizer, row["prompt"])
                response = await client.post(args.endpoint.rstrip("/") + "/generate", json=request_payload(prompt, candidates))
                response.raise_for_status()
                probabilities = extract_probabilities(response.json(), candidates)
                metadata = row["metadata"]
                return {"id": metadata["id"], "source": metadata["source"], "target": metadata["target"], "probabilities": probabilities, "prompt_tokens": len(prompt), "metrics": score_metrics(probabilities, metadata["target"])}

        predictions = await asyncio.gather(*(score(row) for row in rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    baseline = None if args.baseline is None else {row["id"]: row for row in pl.read_ndjson(args.baseline).to_dicts()}
    report = summarize(predictions, baseline, args.seed)
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    if args.concurrency < 1:
        raise ValueError("Concurrency must be positive")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()

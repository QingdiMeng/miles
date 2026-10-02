"""Evaluate generated JSON probability reports on a fixed held-out split."""

import asyncio
import json
import math
from pathlib import Path

import httpx
import polars as pl
from tap import Tap
from transformers import AutoTokenizer

from examples.decision_calibration.evaluate import _ece
from examples.decision_calibration.probabilities import decision_prompt_ids
from examples.decision_calibration.report_reward import report_messages, score_report


class Args(Tap):
    data: Path
    output: Path
    model: str
    endpoint: str
    concurrency: int = 32
    max_tokens: int = 128


def summarize(rows: list[dict]) -> dict:
    groups = {"all": rows}
    groups.update({source: [r for r in rows if r["source"] == source] for source in {r["source"] for r in rows}})
    groups["synthetic"] = [r for r in rows if r["source"] in {"coin", "dice", "urn"}]
    result = {}
    for name, group in groups.items():
        if not group:
            continue
        valid = [r for r in group if r["valid"]]
        metrics = {"rows": len(group), "valid_rows": len(valid), "valid_fraction": len(valid) / len(group), "mean_reward": sum(r["reward"] for r in group) / len(group)}
        if valid:
            metrics["brier_valid_only"] = sum(r["brier"] for r in valid) / len(valid)
            metrics["top_label_ece_valid_only"] = _ece(valid)
            metrics["onehot_fraction_valid_only"] = sum(max(r["probabilities"]) == 1 for r in valid) / len(valid)
            hard = [r for r in valid if max(r["target"]) == 1]
            if hard:
                metrics["accuracy_valid_only"] = sum(max(range(len(r["target"])), key=r["probabilities"].__getitem__) == r["target"].index(1) for r in hard) / len(hard)
                metrics["log_loss_clipped_valid_only"] = sum(-math.log(max(r["probabilities"][r["target"].index(1)], 1e-12)) for r in hard) / len(hard)
        result[name] = metrics
    return result


async def evaluate(args: Args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = pl.read_ndjson(args.data).to_dicts()
    semaphore = asyncio.Semaphore(args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=10800) as client:
        async def score(row: dict) -> dict:
            async with semaphore:
                prompt = decision_prompt_ids(tokenizer, report_messages(row["prompt"], len(row["metadata"]["target"])))
                response = await client.post(args.endpoint.rstrip("/") + "/generate", json={"input_ids": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": args.max_tokens}})
                response.raise_for_status()
                output = response.json()
                metadata = row["metadata"]
                return {"id": metadata["id"], "source": metadata["source"], "target": metadata["target"], "response": output["text"], "finish_reason": output["meta_info"].get("finish_reason"), **score_report(output["text"], metadata["target"])}
        predictions = await asyncio.gather(*(score(row) for row in rows))
    args.output.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    report = summarize(predictions)
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    if args.concurrency < 1 or args.max_tokens < 1:
        raise ValueError("Concurrency and max tokens must be positive")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()

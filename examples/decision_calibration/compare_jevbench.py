"""Compare checkpoints on matched public items and known outcome distributions."""

import json
import random
from pathlib import Path

import polars as pl
from tap import Tap


class Args(Tap):
    benchmark: Path
    results: Path
    bootstrap: int = 10000


def compare(args: Args) -> dict:
    tasks = {r["id"]: r for path in (args.benchmark / "datasets/public").glob("*.jsonl")
             for r in pl.read_ndjson(path, infer_schema_length=None).to_dicts()}
    labels = ("baseline", "step128", "step256", "step384", "step512")
    predictions = {label: {r["id"]: r for r in pl.read_ndjson(args.results / f"{label}.jsonl", infer_schema_length=None).to_dicts()}
                   for label in labels}
    for rows in predictions.values():
        assert set(rows) == set(tasks)
    common = sorted(i for i in tasks if all(predictions[label][i]["valid"] for label in labels))
    rng = random.Random(261002)
    result = {"total_questions": len(tasks), "common_valid_questions": len(common), "models": {}, "paired": {}}
    for label, rows in predictions.items():
        common_brier = sum(rows[i]["brier"] for i in common) / len(common) if common else None
        known = []
        for i, task in tasks.items():
            gold = task["provenance"].get("gold_probs")
            if gold is not None and rows[i]["valid"]:
                probs = rows[i]["probs"]
                loss = sum((probs[k] - gold[k]) ** 2 for k in task["labels"])
                known.append({"id": i, "distribution_squared_error": loss})
        result["models"][label] = {"brier_common_valid": common_brier, "known_distribution_questions": len(known),
                                    "known_distribution_mean_squared_error": sum(r["distribution_squared_error"] for r in known) / len(known) if known else None,
                                    "known_distribution_per_question": known}
        if label == "baseline":
            continue
        paired = sorted(i for i in tasks if rows[i]["valid"] and predictions["baseline"][i]["valid"])
        delta = [rows[i]["brier"] - predictions["baseline"][i]["brier"] for i in paired]
        if delta:
            boot = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(args.bootstrap))
            result["paired"][label] = {"questions": len(delta), "brier_delta": sum(delta) / len(delta),
                                       "ci95": [boot[int(args.bootstrap * .025)], boot[int(args.bootstrap * .975)]]}
    (args.results / "matched-comparison.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(compare(Args(underscores_to_dashes=True).parse_args()), indent=2))

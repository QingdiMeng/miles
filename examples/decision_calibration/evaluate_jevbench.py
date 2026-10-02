"""Evaluate JSON probability reports on a pinned public JevBench checkout."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from tap import Tap
from transformers import AutoTokenizer
from jevbench.metrics import brier_score, ece_top_label, latency_summary
from jevbench.scoring import score_task

from examples.decision_calibration.probabilities import decision_prompt_ids
from examples.decision_calibration.report_reward import parse_report


class Args(Tap):
    benchmark: Path
    model: str
    endpoint: str
    output: Path
    concurrency: int = 1
    max_tokens: int = 512


def messages(task: dict) -> list[dict]:
    labels = task["labels"]
    question = task["question"]
    criteria = question.get("criteria")
    options = []
    for i, label in enumerate(labels):
        if isinstance(criteria, list):
            description = criteria[int(label)]
        elif isinstance(criteria, dict):
            key = {"yes": "true", "no": "false"}.get(label, label) if question["type"] == "noul" else label
            description = criteria.get(key, label)
        else:
            description = label
        options.append(f"{chr(65 + i)}. {label}: {description}")
    keys = ", ".join(json.dumps(chr(65 + i)) for i in range(len(labels)))
    state = task["state"] if isinstance(task["state"], str) else json.dumps(task["state"])
    text = (f"State:\n{state}\n\n{question['instructions']}\n\nOptions:\n" + "\n".join(options)
            + f"\n\nReport your probability for every option as a JSON object with exactly the keys {keys}. "
            "Use finite numbers between 0 and 1 that sum to 1. Report your uncertainty honestly. "
            "Output only JSON, without explanation or Markdown.")
    return [{"role": "user", "content": text}]


async def evaluate(args: Args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tasks = []
    for name in ("easy", "original", "hard"):
        for line in (args.benchmark / "datasets/public" / f"{name}.jsonl").read_text().splitlines():
            task = json.loads(line)
            task["public_file"] = name
            tasks.append(task)
    assert len(tasks) == 231 and len({t["id"] for t in tasks}) == 231
    args.output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []
    async with httpx.AsyncClient(timeout=600) as client:
        # Warm-up is excluded from measured request latency.
        warm = decision_prompt_ids(tokenizer, messages(tasks[0]))
        response = await client.post(args.endpoint + "/generate", json={"input_ids": warm, "sampling_params": {"temperature": 0, "max_new_tokens": args.max_tokens}})
        response.raise_for_status()

        async def one(task: dict) -> dict:
            async with semaphore:
                prompt = decision_prompt_ids(tokenizer, messages(task))
                started = time.perf_counter()
                result = await client.post(args.endpoint + "/generate", json={"input_ids": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": args.max_tokens}})
                elapsed = time.perf_counter() - started
                result.raise_for_status()
                output = result.json()
                labels = task["labels"]
                probs = None
                error = None
                try:
                    values = parse_report(output["text"], len(labels))
                    probs = dict(zip(labels, values, strict=True))
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                scored = score_task(probs, SimpleNamespace(**task))
                clean = scored.get("probs")
                expected = task.get("expected")
                return {"id": task["id"], "family": task["family"], "type": task["question"]["type"],
                        "public_file": task["public_file"], "labels": labels, "expected": expected,
                        "response": output["text"], "latency_s": elapsed, "prompt_tokens": len(prompt),
                        "response_tokens": output["meta_info"].get("completion_tokens"), "parse_error": error,
                        "brier": brier_score(clean, str(expected), labels) if clean and expected is not None else None,
                        "collapse": bool(clean and max(clean.values()) == 1), **scored}

        with args.output.open("w") as stream:
            for task in tasks:
                # Serial requests provide idle-endpoint latency rather than load-test latency.
                row = await one(task)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if len(rows) % 25 == 0:
                    print(f"Completed {len(rows)}/{len(tasks)}", flush=True)
    groups = {"all": rows}
    groups.update({kind: [r for r in rows if r["type"] == kind] for kind in {r["type"] for r in rows}})
    groups.update({tier: [r for r in rows if r["public_file"] == tier] for tier in ("easy", "original", "hard")})
    report = {}
    for name, group in groups.items():
        valid = [r for r in group if r["valid"]]
        scorable = [r for r in group if r["expected"] is not None]
        calibrated = [r for r in valid if r["expected"] is not None]
        metrics = {"rows": len(group), "valid_rows": len(valid),
                   "valid_pct": 100 * len(valid) / len(group),
                   "accuracy_all_scorable": sum(r["correct"] for r in scorable) / len(scorable) if scorable else None,
                   "collapse_pct_all": 100 * sum(r["collapse"] for r in group) / len(group),
                   "brier_valid_only": sum(r["brier"] for r in calibrated) / len(calibrated) if calibrated else None,
                   "ece_valid_only": ece_top_label([(max(r["probs"].values()), r["correct"]) for r in calibrated]),
                   "latency": latency_summary([r["latency_s"] for r in group])}
        ordinal = [r for r in calibrated if r["type"] == "score"]
        if ordinal:
            metrics["ordinal_mae_valid_only"] = sum(abs(r["ordinal_ev"] - float(r["expected"])) for r in ordinal) / len(ordinal)
        report[name] = metrics
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(evaluate(Args(underscores_to_dashes=True).parse_args()))

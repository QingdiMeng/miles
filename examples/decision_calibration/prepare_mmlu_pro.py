"""Replace MMLU with disjoint, subject-stratified MMLU-Pro questions."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from tap import Tap

from examples.decision_calibration.prepare import _example
from examples.decision_calibration.prepare_mixed import allocate, question_key
from examples.decision_calibration.report_reward import report_messages


class Args(Tap):
    previous_data_dir: Path
    mmlu_pro_parquet: Path
    source_revision: str
    output_dir: Path
    seed: int = 261002


def build(args: Args) -> dict:
    rng = random.Random(args.seed)
    splits = {}
    seen = set()
    for split in ("train", "validation", "test"):
        rows = pl.read_ndjson(args.previous_data_dir / f"{split}.jsonl").to_dicts()
        splits[split] = [r for r in rows if r["metadata"]["source"] != "mmlu"]
        for row in splits[split]:
            key = question_key(row["metadata"]["scenario"])
            if key in seen:
                raise ValueError("Duplicate retained scenario")
            seen.add(key)
    pools = defaultdict(list)
    dropped = 0
    for record in pl.read_parquet(args.mmlu_pro_parquet).to_dicts():
        key = question_key(record["question"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        choices = record["options"]
        answer = record["answer_index"]
        if not 2 <= len(choices) <= 10 or not 0 <= answer < len(choices):
            raise ValueError("Invalid MMLU-Pro options/answer")
        row = _example("mmlu_pro", key, record["question"], choices,
                       [float(i == answer) for i in range(len(choices))], rng)
        # Save the final JSON-report prompt, with keys matching this question.
        row["prompt"] = report_messages(row["prompt"], len(choices))
        row["metadata"].update(subject=record["category"], source_question_id=record["question_id"])
        pools[record["category"]].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    for split, count in (("validation", 1024), ("test", 2048), ("train", 7117)):
        splits[split].extend(allocate(pools, count))
    manifest = {
        "seed": args.seed, "thinking": False,
        "source": "TIGER-Lab/MMLU-Pro test", "source_revision": args.source_revision,
        "source_sha256": hashlib.sha256(args.mmlu_pro_parquet.read_bytes()).hexdigest(),
        "duplicates_dropped": dropped, "unused_mmlu_pro": sum(map(len, pools.values())),
        "retained_data_dir": str(args.previous_data_dir), "splits": {},
    }
    used_ids, used_questions = set(), set()
    for split, expected in (("train", 8192), ("validation", 1184), ("test", 2368)):
        rng.shuffle(splits[split])
        rows = splits[split]
        if len(rows) != expected:
            raise ValueError("Unexpected split size")
        for row in rows:
            metadata = row["metadata"]
            target = metadata["target"]
            key = question_key(metadata["scenario"])
            if metadata["id"] in used_ids or key in used_questions:
                raise ValueError("Duplicate question or cross-split leakage")
            used_ids.add(metadata["id"])
            used_questions.add(key)
            if len(target) != len(metadata["choices"]) or any(x < 0 for x in target) or not math.isclose(sum(target), 1):
                raise ValueError("Invalid target")
            if metadata["source"] != "mmlu_pro":
                row["prompt"] = report_messages(row["prompt"], len(target))
        manifest["splits"][split] = {
            "rows": len(rows), "sources": dict(Counter(r["metadata"]["source"] for r in rows)),
            "subjects": dict(Counter(r["metadata"].get("subject") for r in rows if r["metadata"]["source"] == "mmlu_pro")),
            "option_counts": dict(Counter(len(r["metadata"]["target"]) for r in rows)),
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][split]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    print(json.dumps(build(Args(underscores_to_dashes=True).parse_args()), indent=2))


if __name__ == "__main__":
    main()

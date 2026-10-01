"""Prepare a larger mixture while preserving pilot GPQA holdouts."""

import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from tap import Tap

from examples.decision_calibration.prepare import _example, _synthetic


class Args(Tap):
    pilot_data_dir: Path
    mmlu_parquet: Path
    output_dir: Path
    seed: int = 261002


def question_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def allocate(pools: dict[str, list[dict]], count: int) -> list[dict]:
    total = sum(map(len, pools.values()))
    if count > total:
        raise ValueError("Insufficient distinct MMLU questions")
    quotas = {s: len(p) * count / total for s, p in pools.items()}
    sizes = {s: math.floor(q) for s, q in quotas.items()}
    for s in sorted(pools, key=lambda s: (-(quotas[s] - sizes[s]), s))[:count - sum(sizes.values())]:
        sizes[s] += 1
    result = []
    for s in sorted(pools):
        result.extend(pools[s][:sizes[s]])
        del pools[s][:sizes[s]]
    return result


def build(args: Args) -> dict:
    rng = random.Random(args.seed)
    splits = {}
    question_keys = set()
    for split in ("train", "validation", "test"):
        rows = pl.read_ndjson(args.pilot_data_dir / f"{split}.jsonl").to_dicts()
        splits[split] = [r for r in rows if r["metadata"]["source"] == "gpqa"]
        for r in splits[split]:
            key = question_key(r["metadata"]["scenario"])
            if key in question_keys:
                raise ValueError("Duplicate GPQA question")
            question_keys.add(key)
    if [len(splits[s]) for s in ("train", "validation", "test")] != [256, 64, 128]:
        raise ValueError("Expected original pilot GPQA partition")
    pools = defaultdict(list)
    dropped = 0
    for r in pl.read_parquet(args.mmlu_parquet).to_dicts():
        key = question_key(r["question"])
        if key in question_keys:
            dropped += 1
            continue
        question_keys.add(key)
        row = _example("mmlu", key, r["question"], r["choices"], [float(i == r["answer"]) for i in range(4)], rng)
        row["metadata"]["subject"] = r["subject"]
        pools[r["subject"]].append(row)
    for p in pools.values():
        rng.shuffle(p)
    for split, count in (("validation", 1024), ("test", 2048), ("train", 7117)):
        splits[split].extend(allocate(pools, count))
    synthetic = _synthetic(819 + 96 + 192, rng)
    families = {s: [r for r in synthetic if r["metadata"]["source"] == s] for s in ("coin", "dice", "urn")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"seed": args.seed, "thinking": False, "mmlu_source": "cais/mmlu all/test", "mmlu_parquet_sha256": hashlib.sha256(args.mmlu_parquet.read_bytes()).hexdigest(), "mmlu_duplicate_questions_dropped": dropped, "mmlu_unused": sum(map(len, pools.values())), "splits": {}}
    used = set()
    for split, count in (("train", 819), ("validation", 96), ("test", 192)):
        for family, pool in families.items():
            selected = pool[:count // 3]
            del pool[:count // 3]
            for row in selected:
                text = row["prompt"][0]["content"]
                text = text.replace("Which number of heads will occur?", "Take a guess at the number of heads.")
                text = text.replace("Which outcome will occur?", "Take a guess at the outcome of the roll.")
                text = text.replace("Which color will occur?", "Take a guess at the color of the drawn ball.")
                row["prompt"][0]["content"] = text.replace("Choose one option.", "Choose one option as your guess.")
            splits[split].extend(selected)
        rng.shuffle(splits[split])
        ids = {r["metadata"]["id"] for r in splits[split]}
        if len(ids) != len(splits[split]) or ids & used:
            raise ValueError("Duplicate or cross-split scenario leakage")
        used.update(ids)
        for r in splits[split]:
            t = r["metadata"]["target"]
            if len(t) != 4 or any(x < 0 for x in t) or not math.isclose(sum(t), 1):
                raise ValueError("Invalid target distribution")
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in splits[split]))
        manifest["splits"][split] = {"rows": len(splits[split]), "sources": dict(Counter(r["metadata"]["source"] for r in splits[split])), "mmlu_subjects": dict(Counter(r["metadata"]["subject"] for r in splits[split] if r["metadata"]["source"] == "mmlu")), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if len(splits["train"]) != 8192:
        raise ValueError("Expected exactly 8192 training questions")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()

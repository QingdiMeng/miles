"""Expand calibration training with SuperGPQA while excluding public JevBench."""

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from tap import Tap

from examples.decision_calibration.prepare import _example, _synthetic
from examples.decision_calibration.prepare_mixed import allocate, question_key
from examples.decision_calibration.report_reward import REPORT_INSTRUCTION, report_messages


class Args(Tap):
    previous_data_dir: Path
    supergpqa_jsonl: Path
    source_revision: str
    jevbench_dir: Path
    jevbench_revision: str
    output_dir: Path
    seed: int = 261003


def read_rows(path: Path) -> list[dict]:
    # Preserve heterogeneous metadata without DataFrame schema coercion.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def question_text(row: dict) -> str:
    content = row["prompt"][-1]["content"]
    question, separator, _ = content.partition("\n\nA. ")
    if not separator:
        raise ValueError("Expected a question followed by lettered options")
    return question


def benchmark_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", question_key(text)))


def blocked_by_benchmark(text: str, benchmark_texts: set[str]) -> bool:
    key = benchmark_key(text)
    return key in benchmark_texts or any(
        min(len(key), len(other)) >= 80 and (key in other or other in key)
        for other in benchmark_texts
    )


def build(args: Args) -> dict:
    rng = random.Random(args.seed)
    previous = {split: read_rows(args.previous_data_dir / f"{split}.jsonl")
                for split in ("train", "validation", "test")}
    benchmark_files = [args.jevbench_dir / f"{name}.jsonl" for name in ("easy", "original", "hard")]
    benchmark = [row for path in benchmark_files for row in read_rows(path)]
    if len(benchmark) != 231 or len({row["id"] for row in benchmark}) != 231:
        raise ValueError("Expected the pinned 231-question public JevBench")
    benchmark_texts = set()
    for row in benchmark:
        state = row["state"]
        instructions = row["question"]["instructions"]
        benchmark_texts.update(benchmark_key(text) for text in (state, instructions, state + "\n\n" + instructions))
    blocked_questions = {question_key(question_text(row)) for rows in previous.values() for row in rows}
    blocked_scenarios = {(row["metadata"]["source"], question_key(row["metadata"]["scenario"]))
                         for rows in previous.values() for row in rows}
    # All previous holdouts remain excluded, even when not retained in validation.
    splits = {"train": previous["train"], "validation": []}
    retained_pools = defaultdict(list)
    for row in previous["validation"]:
        if row["metadata"]["source"] == "mmlu_pro":
            retained_pools[row["metadata"]["subject"]].append(row)
        else:
            splits["validation"].append(row)
    for pool in retained_pools.values():
        rng.shuffle(pool)
    splits["validation"].extend(allocate(retained_pools, 512))
    for rows in splits.values():
        for row in rows:
            if blocked_by_benchmark(question_text(row), benchmark_texts):
                raise ValueError(f"Existing data overlaps JevBench: {row['metadata']['id']}")
    super_rows = read_rows(args.supergpqa_jsonl)
    pools = defaultdict(list)
    dropped = Counter()
    seen = set(blocked_questions)
    for record in super_rows:
        if record["difficulty"] not in ("middle", "hard"):
            dropped["easy"] += 1
            continue
        question = record["question"].strip()
        key = question_key(question)
        if key in seen:
            dropped["duplicate_or_previous_split"] += 1
            continue
        if blocked_by_benchmark(question, benchmark_texts):
            dropped["jevbench_overlap"] += 1
            continue
        choices = record["options"]
        answer = ord(record["answer_letter"]) - ord("A")
        if not 2 <= len(choices) <= 10 or not 0 <= answer < len(choices):
            raise ValueError(f"Invalid choices or answer: {record['uuid']}")
        if question_key(choices[answer]) != question_key(record["answer"]):
            raise ValueError(f"Answer letter/text mismatch: {record['uuid']}")
        if len({question_key(choice) for choice in choices}) != len(choices):
            dropped["duplicate_options"] += 1
            continue
        row = _example("supergpqa", key, question, choices,
                       [float(i == answer) for i in range(len(choices))], rng)
        row["prompt"] = report_messages(row["prompt"], len(choices))
        row["metadata"].update(source_question_id=record["uuid"], subject=record["discipline"],
                               field=record["field"], subfield=record["subfield"],
                               difficulty=record["difficulty"], is_calculation=record["is_calculation"])
        pools[record["field"] + ":" + record["difficulty"]].append(row)
        seen.add(key)
    for pool in pools.values():
        rng.shuffle(pool)
    eligible = sum(map(len, pools.values()))
    # Fix the held-out partition before drawing additional training items.
    splits["validation"].extend(allocate(pools, 512))
    splits["train"].extend(allocate(pools, 7373))
    synthetic_pools = {family: [] for family in ("coin", "dice", "urn")}
    for row in _synthetic(2400, rng):
        metadata = row["metadata"]
        scenario = (metadata["source"], question_key(metadata["scenario"]))
        if scenario in blocked_scenarios:
            continue
        text = row["prompt"][0]["content"]
        text = text.replace("Which number of heads will occur?", "Take a guess at the number of heads.")
        text = text.replace("Which outcome will occur?", "Take a guess at the outcome of the roll.")
        text = text.replace("Which color will occur?", "Take a guess at the color of the drawn ball.")
        row["prompt"][0]["content"] = text
        row["prompt"] = report_messages(row["prompt"], 4)
        if blocked_by_benchmark(question_text(row), benchmark_texts):
            continue
        synthetic_pools[metadata["source"]].append(row)
    for family, pool in synthetic_pools.items():
        if len(pool) < 273:
            raise ValueError(f"Insufficient new distinct {family} scenarios")
        splits["train"].extend(pool[:273])
    manifest = {
        "seed": args.seed, "thinking": False, "test": "JevBench public only; never included in training or validation",
        "previous_data_dir": str(args.previous_data_dir),
        "previous_split_sha256": {split: digest(args.previous_data_dir / f"{split}.jsonl") for split in previous},
        "supergpqa": {"source": "m-a-p/SuperGPQA", "revision": args.source_revision,
                      "sha256": digest(args.supergpqa_jsonl), "raw_rows": len(super_rows),
                      "eligible_medium_hard": eligible, "dropped": dict(dropped),
                      "unused_eligible": sum(map(len, pools.values()))},
        "jevbench": {"revision": args.jevbench_revision, "rows": len(benchmark),
                     "files_sha256": {path.name: digest(path) for path in benchmark_files},
                     "overlap_check": "NFKC/case/whitespace and punctuation normalized equality; containment for texts >=80 characters",
                     "training_validation_overlap": 0}, "splits": {},
    }
    used_ids, used_questions, used_scenarios = set(), set(), set()
    for split, count in (("train", 16384), ("validation", 1184)):
        rows = splits[split]
        rng.shuffle(rows)
        if len(rows) != count:
            raise ValueError(f"Unexpected {split} size")
        for row in rows:
            metadata = row["metadata"]
            key = question_key(question_text(row))
            scenario = (metadata["source"], question_key(metadata["scenario"]))
            if metadata["id"] in used_ids or key in used_questions or scenario in used_scenarios:
                raise ValueError("Duplicate question or cross-split scenario")
            used_ids.add(metadata["id"])
            used_questions.add(key)
            used_scenarios.add(scenario)
            target = metadata["target"]
            if not 2 <= len(target) <= 10 or len(target) != len(metadata["choices"]):
                raise ValueError("Invalid target size")
            if any(not math.isfinite(x) or not 0 <= x <= 1 for x in target) or not math.isclose(sum(target), 1):
                raise ValueError("Invalid target probabilities")
            keys = ", ".join(json.dumps(chr(65 + i)) for i in range(len(target)))
            if not row["prompt"][-1]["content"].endswith(REPORT_INSTRUCTION.replace('"A", "B", "C", "D"', keys)):
                raise ValueError("Missing probability-report instructions")
            if blocked_by_benchmark(question_text(row), benchmark_texts):
                raise ValueError("JevBench overlap in final data")
        manifest["splits"][split] = {
            "rows": len(rows), "sources": dict(Counter(row["metadata"]["source"] for row in rows)),
            "supergpqa_difficulty": dict(Counter(row["metadata"]["difficulty"] for row in rows
                                               if row["metadata"]["source"] == "supergpqa")),
            "option_counts": dict(Counter(len(row["metadata"]["target"]) for row in rows)),
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["splits"][split]["sha256"] = digest(path)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(Args(underscores_to_dashes=True).parse_args()), indent=2))

"""Build disjoint GPQA and analytically labelled probability questions."""

import hashlib
import io
import json
import math
import random
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from tap import Tap

SOURCE = "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip"
SPLITS = {"train": (256, 768), "validation": (64, 192), "test": (128, 384)}


class Args(Tap):
    output_dir: Path
    seed: int = 261001
    gpqa_zip: Path | None = None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _example(
    source: str,
    scenario: str,
    question: str,
    options: Sequence[str],
    target: Sequence[float],
    rng: random.Random,
) -> dict:
    order = list(range(len(options)))
    rng.shuffle(order)
    choices = [options[i] for i in order]
    probabilities = [target[i] for i in order]
    prompt = question + "\n\n" + "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    prompt += "\n\nChoose one option. Output only its letter (A, B, C, or D)."
    metadata = {
        "id": _digest(source + ":" + scenario),
        "source": source,
        "scenario": scenario,
        "target": probabilities,
        "choices": choices,
    }
    return {"prompt": [{"role": "user", "content": prompt}], "metadata": metadata}


def _gpqa(archive: bytes, rng: random.Random) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [name for name in zipped.namelist() if name.endswith("gpqa_main.csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one gpqa_main.csv, found {names}")
        csv = zipped.read(names[0], pwd=b"deserted-untie-orchid")
    rows = pl.read_csv(io.BytesIO(csv)).to_dicts()
    rng.shuffle(rows)
    result = []
    for row in rows:
        question = row["Question"].strip()
        options = [row["Correct Answer"], *(row[f"Incorrect Answer {i}"] for i in (1, 2, 3))]
        result.append(_example("gpqa", question, question, options, [1.0, 0.0, 0.0, 0.0], rng))
    if len(result) != 448 or len({row["metadata"]["id"] for row in result}) != 448:
        raise ValueError("This pilot requires the 448 distinct original GPQA main questions")
    return result


def _canonical(weights: Sequence[int]) -> tuple[int, ...]:
    divisor = math.gcd(*weights)
    return tuple(weight // divisor for weight in weights)


def _synthetic(count: int, rng: random.Random) -> list[dict]:
    if count % 3:
        raise ValueError("Synthetic count must be divisible by three")
    per_family = count // 3
    coin_biases = rng.sample(range(1, 1000), per_family)
    result = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for numerator in coin_biases:
        p = numerator / 1000
        target = [math.comb(3, k) * p**k * (1 - p) ** (3 - k) for k in range(4)]
        question = f"A coin lands heads with probability {numerator}/1000 on each independent flip. It will be flipped three times in the future. Which number of heads will occur? No flips have yet been observed."
        result.append(_example("coin", f"three_flips:{numerator}/1000", question, ["Zero heads", "One head", "Two heads", "Three heads"], target, rng))
    for family, size in (("dice", 6), ("urn", 4)):
        while sum(row["metadata"]["source"] == family for row in result) < per_family:
            weights = _canonical([rng.randint(1, 200) for _ in range(size)])
            key = (family, weights)
            if key in seen:
                continue
            seen.add(key)
            total = sum(weights)
            if family == "dice":
                question = f"A six-sided die has faces 1 through 6. Their probabilities are proportional to the respective weights {list(weights)}. It will be rolled once in the future. Which outcome will occur? No roll has yet been observed."
                options = ["Face 1 or 2", "Face 3 or 4", "Face 5", "Face 6"]
                target = [(weights[0] + weights[1]) / total, (weights[2] + weights[3]) / total, weights[4] / total, weights[5] / total]
            else:
                question = f"An urn contains {weights[0]} red, {weights[1]} blue, {weights[2]} green, and {weights[3]} yellow balls. One ball will be drawn uniformly at random in the future. Which color will occur? No ball has yet been drawn."
                options = ["Red", "Blue", "Green", "Yellow"]
                target = [weight / total for weight in weights]
            result.append(_example(family, json.dumps(weights), question, options, target, rng))
    rng.shuffle(result)
    return result


def build(output_dir: Path, archive: bytes, seed: int) -> dict:
    rng = random.Random(seed)
    gpqa = _gpqa(archive, rng)
    # Generate globally unique scenarios before splitting; no variants cross splits.
    synthetic = _synthetic(sum(count[1] for count in SPLITS.values()), rng)
    pools = {source: [row for row in synthetic if row["metadata"]["source"] == source] for source in ("coin", "dice", "urn")}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "gpqa_source": SOURCE, "gpqa_archive_sha256": hashlib.sha256(archive).hexdigest(), "thinking": False, "splits": {}}
    used: set[str] = set()
    for split, (gpqa_count, synthetic_count) in SPLITS.items():
        rows, gpqa = gpqa[:gpqa_count], gpqa[gpqa_count:]
        for family, pool in pools.items():
            count = synthetic_count // 3
            rows.extend(pool[:count])
            pools[family] = pool[count:]
        rng.shuffle(rows)
        ids = {row["metadata"]["id"] for row in rows}
        if len(ids) != len(rows) or ids & used:
            raise ValueError("Duplicate questions or cross-split scenario leakage")
        used.update(ids)
        for row in rows:
            target = row["metadata"]["target"]
            if len(target) != 4 or any(value < 0 for value in target) or not math.isclose(sum(target), 1):
                raise ValueError(f"Invalid target: {target}")
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = output_dir / f"{split}.jsonl"
        path.write_text(text)
        manifest["splits"][split] = {"rows": len(rows), "gpqa": gpqa_count, "synthetic": synthetic_count, "sha256": _digest(text)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    if args.gpqa_zip is None:
        with urllib.request.urlopen(SOURCE, timeout=120) as response:
            archive = response.read()
    else:
        archive = args.gpqa_zip.read_bytes()
    print(json.dumps(build(args.output_dir, archive, args.seed), indent=2))


if __name__ == "__main__":
    main()

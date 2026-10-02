"""Stage verified HF exports and evaluate them on separate idle GPUs."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from tap import Tap


class Args(Tap):
    archive: Path
    benchmark: Path
    model: Path
    output: Path


def stage(source: Path, destination: Path) -> Path:
    manifest = json.loads((source / "UPLOAD_MANIFEST.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        relative = item.get("path", item.get("name", item.get("relative_path")))
        expected = item["sha256"]
        target = destination / relative
        if target.exists():
            digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
            if digest == expected:
                continue
            raise ValueError(f"Existing file checksum mismatch: {target}")
        temporary = target.with_suffix(target.suffix + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with (source / relative).open("rb") as reader, temporary.open("wb") as writer:
            while chunk := reader.read(16 << 20):
                writer.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Archive checksum mismatch: {source / relative}")
        temporary.replace(target)
    (destination / "staging-complete.json").write_text(json.dumps(manifest))
    return destination


def run_model(args: Args, label: str, gpu: int, model: Path) -> dict:
    result = args.output / f"{label}.jsonl"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    port = 32100 + gpu
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(model),
               "--host", "127.0.0.1", "--port", str(port), "--tp-size", "1",
               "--context-length", "65536", "--mem-fraction-static", "0.75",
               "--max-running-requests", "32", "--chunked-prefill-size", "4096"]
    with (args.output / f"{label}.server.log").open("w") as stream:
        server = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 1800
        with httpx.Client(timeout=10) as client:
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"{label} server exited: {server.returncode}")
                try:
                    if client.get(f"http://127.0.0.1:{port}/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(10)
            else:
                raise TimeoutError(f"{label} server readiness")
        print(f"SERVER_READY {label}", flush=True)
        with (args.output / f"{label}.eval.log").open("w") as stream:
            subprocess.run([sys.executable, "-m", "examples.decision_calibration.evaluate_jevbench",
                            "--benchmark", str(args.benchmark), "--model", str(args.model),
                            "--endpoint", f"http://127.0.0.1:{port}", "--output", str(result)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
        print(f"EVAL_COMPLETE {label}", flush=True)
        return {"label": label, "summary": json.loads(result.with_suffix(".summary.json").read_text())}
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def main(args: Args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    models = [("baseline", 0, args.model)]
    for gpu, step in enumerate((128, 256, 384, 512), 1):
        name = f"iter_{step - 1:07d}-hf"
        print(f"STAGING step{step}", flush=True)
        model = stage(args.archive / name, args.output / "models" / name)
        print(f"STAGED step{step}", flush=True)
        models.append((f"step{step}", gpu, model))
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_model, args, *model) for model in models]
        results = []
        errors = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                print(errors[-1], flush=True)
        (args.output / "results.json").write_text(json.dumps({"results": results, "errors": errors}, indent=2))
    if errors:
        raise RuntimeError(str(errors))


if __name__ == "__main__":
    main(Args(underscores_to_dashes=True).parse_args())

"""Persistently wait for training termination, then evaluate every complete save."""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import polars as pl
from ray.job_submission import JobSubmissionClient
from tap import Tap
from torch.distributed.checkpoint import FileSystemReader


class Args(Tap):
    run_id: str
    ray_job_id: str
    checkpoint_dir: Path
    model: Path
    data: Path
    result_dir: Path
    export_dir: Path
    ray_address: str = "http://127.0.0.1:8265"
    min_updates: int = 128
    poll_seconds: int = 20
    max_tokens: int = 512
    concurrency: int = 32
    include_baseline: bool = True
    dry_run: bool = False
    no_wait: bool = False


def log(message: str) -> None:
    print(datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(), message, flush=True)


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def driver_alive(run_id: str) -> bool:
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(a.endswith("/train_async.py") for a in argv) and any(run_id in a for a in argv):
            return True
    return False


def training_terminated(status: str | None, alive: bool, absent_polls: int) -> bool:
    if alive or status in {"RUNNING", "PENDING"}:
        return False
    return status in {"SUCCEEDED", "FAILED", "STOPPED"} or absent_polls >= 3


def wait_for_training(args: Args) -> str:
    client = None
    absent = 0
    while True:
        status = None
        try:
            client = client or JobSubmissionClient(args.ray_address)
            status = str(client.get_job_status(args.ray_job_id))
        except Exception as error:
            log(f"Ray status unavailable: {type(error).__name__}: {error}")
        alive = driver_alive(args.run_id)
        absent = absent + 1 if not alive else 0
        write_json(args.result_dir / "status.json", {"phase": "waiting", "training_status": status, "driver_alive": alive})
        if training_terminated(status, alive, absent):
            log(f"Training ended: {status or 'driver absent/Ray unavailable'}")
            # Allow async writers and Ray process cleanup to settle.
            time.sleep(args.poll_seconds)
            return status or "DRIVER_ABSENT"
        time.sleep(args.poll_seconds)


def checkpoint_complete(path: Path) -> bool:
    if not (path / ".metadata").is_file():
        return False
    try:
        metadata = FileSystemReader(path).read_metadata()
        locations = list(metadata.storage_data.values())
        if not locations:
            return False
        return all((path / x.relative_path).is_file() and
                   (path / x.relative_path).stat().st_size >= x.offset + x.length for x in locations)
    except Exception as error:
        log(f"Incomplete checkpoint {path.name}: {error}")
        return False


def ready_checkpoints(directory: Path, min_updates: int) -> list[Path]:
    return sorted((p for p in directory.glob("iter_*") if p.name[5:].isdigit()
                   and int(p.name[5:]) + 1 >= min_updates and checkpoint_complete(p)), key=lambda p: int(p.name[5:]))


def wait_for_gpu(timeout: int = 1800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if not output.strip():
            return
        log("Waiting for trainer GPU 0 to become free")
        time.sleep(20)
    raise TimeoutError("Trainer GPU 0 remains occupied after termination")


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=20)


def serve_and_evaluate(args: Args, model: Path, label: str) -> None:
    output = args.result_dir / f"{label}.jsonl"
    if output.is_file() and output.with_suffix(".summary.json").is_file():
        if pl.read_ndjson(output).height == pl.read_ndjson(args.data).height:
            log(f"Already evaluated {label}")
            return
    wait_for_gpu()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(model), "--host", "127.0.0.1",
               "--port", str(port), "--tp-size", "1", "--context-length", "65536", "--mem-fraction-static", "0.75",
               "--max-running-requests", "32", "--chunked-prefill-size", "4096"]
    with (args.result_dir / f"{label}.server.log").open("a") as server_log:
        server = subprocess.Popen(command, env=environment, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1800
            ready = False
            with httpx.Client(timeout=10) as client:
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited {server.returncode}; see {label}.server.log")
                    try:
                        ready = client.get(f"http://127.0.0.1:{port}/health").is_success
                    except httpx.HTTPError:
                        pass
                    if ready:
                        break
                    time.sleep(10)
            if not ready:
                raise TimeoutError("Evaluation server readiness timed out")
            with (args.result_dir / f"{label}.eval.log").open("a") as eval_log:
                subprocess.run([sys.executable, "-m", "examples.decision_calibration.evaluate_reports", "--data", str(args.data),
                                "--output", str(output), "--model", str(args.model), "--endpoint", f"http://127.0.0.1:{port}",
                                "--max-tokens", str(args.max_tokens), "--concurrency", str(args.concurrency)],
                               stdout=eval_log, stderr=subprocess.STDOUT, check=True, timeout=7200)
        finally:
            stop_server(server)
    log(f"EVALUATION_COMPLETE {label}")


def retry(label: str, operation: Callable[[], None]) -> str | None:
    last_error = None
    for attempt in range(2):
        try:
            operation()
            return None
        except Exception as error:
            last_error = f"{label}: {type(error).__name__}: {error}"
            log(f"{label} attempt {attempt + 1} failed: {type(error).__name__}: {error}")
            if attempt == 0:
                time.sleep(20)
    return last_error


def evaluate_checkpoint(args: Args, checkpoint: Path) -> None:
    label = f"step{int(checkpoint.name[5:]) + 1}"
    export = args.export_dir / checkpoint.name
    if not (export / "export-complete.json").is_file():
        export.mkdir(parents=True, exist_ok=True)
        with (args.result_dir / f"{label}.export.log").open("a") as export_log:
            subprocess.run([sys.executable, "-m", "examples.decision_calibration.export_report_checkpoint",
                            "--checkpoint", str(checkpoint), "--model", str(args.model), "--output", str(export)],
                           stdout=export_log, stderr=subprocess.STDOUT, check=True, timeout=7200)
    serve_and_evaluate(args, export, label)


def run(args: Args) -> None:
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.no_wait and not args.dry_run:
            raise ValueError("--no-wait is only available with --dry-run")
        terminal = "DRY_RUN" if args.no_wait else wait_for_training(args)
        checkpoints = ready_checkpoints(args.checkpoint_dir, args.min_updates)
        state = {"phase": "evaluating", "training_status": terminal, "checkpoints": [str(p) for p in checkpoints],
                 "completed": [], "errors": [], "dry_run": args.dry_run}
        write_json(args.result_dir / "status.json", state)
        if args.dry_run:
            log(f"DRY_RUN_READY {state['checkpoints']}")
            return
        if not checkpoints:
            state["phase"] = "no_complete_checkpoints"
            write_json(args.result_dir / "status.json", state)
            log("No complete checkpoint at or after 128 updates")
            return
        if args.include_baseline:
            error = retry("baseline", lambda: serve_and_evaluate(args, args.model, "baseline"))
            state["errors" if error else "completed"].append(error or "baseline")
            write_json(args.result_dir / "status.json", state)
        for checkpoint in checkpoints:
            label = f"step{int(checkpoint.name[5:]) + 1}"
            error = retry(label, lambda: evaluate_checkpoint(args, checkpoint))
            state["errors" if error else "completed"].append(error or label)
            write_json(args.result_dir / "status.json", state)
        state["phase"] = "complete" if not state["errors"] else "completed_with_errors"
        write_json(args.result_dir / "status.json", state)
        log(f"ALL_EVALUATIONS_FINISHED {state['phase']}")


if __name__ == "__main__":
    run(Args(underscores_to_dashes=True).parse_args())

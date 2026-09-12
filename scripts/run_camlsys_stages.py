"""One Slurm allocation, an absolute two-hour deadline, E0 gate, E1 saved first."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

START = int(os.environ["PERSONA_START"])
PYTHON = os.environ["PERSONA_PYTHON"]
CONFIG = os.environ.get("PERSONA_CONFIG", "configs/a40_two_hour.yaml")
from persona_dynamics.config import load_config
ROOT = Path(load_config(CONFIG).output_dir)
LOGS = Path(os.environ.get("PERSONA_LOGS", "logs"))
LOGS.mkdir(parents=True, exist_ok=True)
timings = []


def stage(name, cutoff_minutes, devices=""):
    remaining = START + cutoff_minutes * 60 - time.time()
    if remaining <= 0:
        raise TimeoutError(f"Deadline already reached before {name}")
    print(f"START {name} elapsed={(time.time()-START)/60:.1f} min", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices)
    begun = time.time()
    with open(LOGS / f"stage-{name}.log", "a") as log:
        process = subprocess.Popen([PYTHON, "-m", "persona_dynamics", "worker", "--config", CONFIG, "--stage", name],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"Time cutoff reached during {name}")
    timings.append(dict(stage=name, seconds=time.time()-begun, exit_code=code))
    (LOGS / "stage-timings.json").write_text(json.dumps(timings, indent=2))
    print(f"END {name}: {timings[-1]}", flush=True)
    if code:
        raise RuntimeError(f"{name} exited {code}; scientific gate will not be bypassed")
    return timings[-1]["seconds"]


def main():
    while Path("data/prepared/REVIEW_PENDING").exists():
        if time.time() - START > 25 * 60:
            raise TimeoutError("Input review was not completed within setup budget")
        time.sleep(1)
    if not Path("data/prepared/advice_review.json").exists():
        raise RuntimeError("The CaMLSys run requires the frozen semantic advice screen; broad keyword sampling is insufficient")
    if os.environ.get("PERSONA_EXTEND_FROM"):
        from extend_candidate_run import extend
        extend(load_config(CONFIG), os.environ["PERSONA_EXTEND_FROM"])
    else:
        stage("generate-gate", 30, "0,1,2,3")
        stage("gate", 35, "4,5,6,7")
    stage("generate-experiments", 90, "0,1,2,3")
    extraction_seconds = stage("extract-e1", 100, "4,5,6,7")
    stage("analyze", 105)
    snapshot = ROOT / "e1_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("results.md", "summary.json"):
        shutil.copy2(ROOT / name, snapshot / name)
    shutil.copytree(ROOT / "figures", snapshot / "figures", dirs_exist_ok=True)
    elapsed = time.time() - START
    # E2 has roughly twice as many transcripts, with up to two content copies.
    # Use a conservative four-times E1 wall time plus five minutes for reloading.
    estimate = 4 * extraction_seconds + 300
    if elapsed <= 70 * 60 and elapsed + estimate <= 100 * 60:
        try:
            stage("extract", 100, "4,5,6,7")
            stage("analyze", 110)
        except (TimeoutError, RuntimeError) as exc:
            print(f"E2 incomplete; E1 snapshot preserved: {exc}", flush=True)
    else:
        print(f"E2 DEFERRED elapsed={elapsed/60:.1f}m conservative estimate={estimate/60:.1f}m", flush=True)
    subprocess.run([PYTHON, "scripts/verify_run.py", str(ROOT)], check=True)
    print("COMPLETE: saved report and figures; human review is still required", flush=True)


if __name__ == "__main__":
    main()

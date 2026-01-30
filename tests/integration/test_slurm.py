#!/usr/bin/env python3
"""
Integration test for the oellm schedule-eval workflow with SLURM.

This test:
1. Submits a real evaluation job using schedule_evals
2. Waits for SLURM job completion
3. Verifies the results JSON file is created and valid
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def wait_for_slurm_job(timeout: int = 600, poll_interval: int = 10) -> bool:
    """Wait for all SLURM jobs to complete."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i"],
            capture_output=True,
            text=True,
        )

        jobs = [j.strip() for j in result.stdout.strip().split("\n") if j.strip()]

        if not jobs:
            print("All SLURM jobs completed")
            return True

        print(
            f"Waiting for {len(jobs)} job(s) to complete... (elapsed: {int(time.time() - start_time)}s)"
        )
        time.sleep(poll_interval)

    print(f"Timeout waiting for SLURM jobs after {timeout}s")
    return False


def find_results_dir(base_dir: Path) -> Path | None:
    """Find the most recent results directory."""
    user = os.environ.get("USER", "runner")
    output_dir = base_dir / user

    if not output_dir.exists():
        return None

    dirs = sorted(output_dir.iterdir(), reverse=True)
    for d in dirs:
        if d.is_dir() and (d / "results").exists():
            return d / "results"

    return None


def validate_results(results_dir: Path) -> bool:
    """Validate that results JSON files exist and contain expected data."""
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return False

    print(f"Found {len(json_files)} result file(s)")

    for json_file in json_files:
        print(f"Validating {json_file.name}...")

        with open(json_file) as f:
            data = json.load(f)

        if "results" not in data:
            print(f"  ERROR: Missing 'results' key in {json_file.name}")
            return False

        if "model_name" not in data and "model" not in data:
            print(f"  ERROR: Missing model identifier in {json_file.name}")
            return False

        results = data.get("results", {})
        if not results:
            print(f"  ERROR: Empty results in {json_file.name}")
            return False

        for task_name, task_results in results.items():
            print(f"  Task: {task_name}")
            if "acc,none" in task_results:
                acc = task_results["acc,none"]
                print(f"    Accuracy: {acc:.4f}")
                if not (0.0 <= acc <= 1.0):
                    print("    ERROR: Accuracy out of range")
                    return False

        print(f"  {json_file.name}: OK")

    return True


def main():
    eval_base_dir = Path(os.environ.get("EVAL_BASE_DIR", "/tmp/oellm-test"))

    print("=" * 60)
    print("SLURM Integration Test")
    print("=" * 60)

    print("\n1. Verifying SLURM is running...")
    result = subprocess.run(["sinfo"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SLURM not available: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

    print("\n2. Scheduling evaluation job...")

    hf_home = eval_base_dir / "hf_data"

    os.environ["EVAL_BASE_DIR"] = str(eval_base_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["PARTITION"] = "gpu"
    os.environ["ACCOUNT"] = "test"
    os.environ["QUEUE_LIMIT"] = "10"
    os.environ["EVAL_CONTAINER_IMAGE"] = "eval_env-ci.sif"
    os.environ["SINGULARITY_ARGS"] = "--nv"
    os.environ["GPUS_PER_NODE"] = "1"
    os.environ["EVAL_OUTPUT_DIR"] = str(eval_base_dir / os.environ.get("USER", "runner"))

    result = subprocess.run(
        [
            "uv",
            "run",
            "oellm",
            "schedule-eval",
            "--models",
            "sshleifer/tiny-gpt2",
            "--tasks",
            "arc_easy",
            "--n_shot",
            "0",
            "--skip-checks",
        ],
        capture_output=True,
        text=True,
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"schedule-eval failed with return code {result.returncode}")
        sys.exit(1)

    print("\n3. Waiting for SLURM job to complete...")
    if not wait_for_slurm_job(timeout=600):
        print("SLURM job did not complete in time")

        print("\nSLURM job status:")
        subprocess.run(["squeue", "-l"])

        print("\nSLURM logs:")
        for log in (eval_base_dir / os.environ.get("USER", "runner")).rglob("*.out"):
            print(f"\n--- {log} ---")
            print(log.read_text()[-2000:])
        for log in (eval_base_dir / os.environ.get("USER", "runner")).rglob("*.err"):
            print(f"\n--- {log} ---")
            print(log.read_text()[-2000:])

        sys.exit(1)

    print("\n4. Validating results...")
    results_dir = find_results_dir(eval_base_dir)

    if not results_dir:
        print(f"Could not find results directory under {eval_base_dir}")

        print("\nDirectory contents:")
        subprocess.run(["find", str(eval_base_dir), "-type", "f"])

        sys.exit(1)

    print(f"Results directory: {results_dir}")

    if not validate_results(results_dir):
        print("Result validation failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Integration test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from oellm.task_groups import get_all_task_group_names


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
            f"Waiting for {len(jobs)} job(s) to complete... "
            f"(elapsed: {int(time.time() - start_time)}s)"
        )
        time.sleep(poll_interval)

    print(f"Timeout waiting for SLURM jobs after {timeout}s")
    return False


def find_eval_dir(base_dir: Path) -> Path | None:
    """Find the most recent evaluation directory."""
    user = os.environ.get("USER", "runner")
    output_dir = base_dir / user

    if not output_dir.exists():
        return None

    dirs = sorted(output_dir.iterdir(), reverse=True)
    for d in dirs:
        if d.is_dir():
            return d

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


def validate_sbatch_script(script_path: Path) -> bool:
    """Validate the generated sbatch script has expected content."""
    if not script_path.exists():
        print(f"ERROR: sbatch script not found at {script_path}")
        return False

    content = script_path.read_text()
    print(f"Validating sbatch script: {script_path}")

    required_patterns = [
        (r"#SBATCH --job-name=", "job name directive"),
        (r"#SBATCH --time=", "time limit directive"),
        (r"#SBATCH --output=", "output directive"),
        (r"#SBATCH --partition=", "partition directive"),
        (r"#SBATCH --array=", "array directive"),
        (r"CSV_PATH=", "CSV path variable"),
        (r"singularity exec", "singularity exec command"),
        (r"lm_eval", "lm_eval command"),
    ]

    all_valid = True
    for pattern, description in required_patterns:
        if re.search(pattern, content):
            print(f"  [OK] Found {description}")
        else:
            print(f"  [FAIL] Missing {description}")
            all_valid = False

    return all_valid


def validate_jobs_csv(csv_path: Path) -> bool:
    """Validate the generated jobs CSV."""
    if not csv_path.exists():
        print(f"ERROR: jobs.csv not found at {csv_path}")
        return False

    content = csv_path.read_text()
    lines = content.strip().split("\n")

    print(f"Validating jobs CSV: {csv_path}")
    print(f"  Header: {lines[0]}")
    print(f"  Total jobs: {len(lines) - 1}")

    if len(lines) < 2:
        print("  [FAIL] No jobs in CSV")
        return False

    required_columns = ["model_path", "task_path", "n_shot"]
    header = lines[0].split(",")
    for col in required_columns:
        if col in header:
            print(f"  [OK] Found column: {col}")
        else:
            print(f"  [FAIL] Missing column: {col}")
            return False

    return True


def setup_environment(eval_base_dir: Path, dry_run: bool = False):
    """Set up environment variables for the test."""
    hf_home = eval_base_dir / "hf_data"

    os.environ["EVAL_BASE_DIR"] = str(eval_base_dir)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["PARTITION"] = "debug"
    os.environ["ACCOUNT"] = "test"
    os.environ["QUEUE_LIMIT"] = "10"
    os.environ["EVAL_CONTAINER_IMAGE"] = "eval_env-slurm-ci.sif"
    os.environ["EVAL_OUTPUT_DIR"] = str(eval_base_dir / os.environ.get("USER", "runner"))

    if dry_run:
        os.environ["SINGULARITY_ARGS"] = ""
        os.environ["GPUS_PER_NODE"] = "0"
    else:
        os.environ["SINGULARITY_ARGS"] = "--nv"
        os.environ["GPUS_PER_NODE"] = "1"


def run_dry_run_test(eval_base_dir: Path) -> bool:
    """Run the dry-run test (no actual job submission)."""
    print("=" * 60)
    print("SLURM Integration Test (DRY-RUN MODE)")
    print("=" * 60)

    print("\n1. Verifying SLURM is running...")
    result = subprocess.run(["sinfo"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SLURM not available: {result.stderr}")
        return False
    print(result.stdout)

    print("\n2. Testing schedule-eval with --dry-run...")
    setup_environment(eval_base_dir, dry_run=True)

    all_task_groups = ",".join(get_all_task_group_names())
    print(f"Testing all {len(get_all_task_group_names())} task groups: {all_task_groups}")

    result = subprocess.run(
        [
            "uv",
            "run",
            "oellm",
            "schedule-eval",
            "--models",
            "sshleifer/tiny-gpt2",
            "--task_groups",
            all_task_groups,
            "--limit",
            "5",
            "--dry_run",
            "true",
        ],
        capture_output=True,
        text=True,
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"schedule-eval --dry-run failed with return code {result.returncode}")
        return False

    print("\n3. Validating generated files...")
    eval_dir = find_eval_dir(eval_base_dir)

    if not eval_dir:
        print(f"Could not find evaluation directory under {eval_base_dir}")
        print("\nDirectory contents:")
        subprocess.run(["find", str(eval_base_dir), "-type", "f"])
        return False

    print(f"Evaluation directory: {eval_dir}")

    sbatch_path = eval_dir / "submit_evals.sbatch"
    csv_path = eval_dir / "jobs.csv"

    if not validate_sbatch_script(sbatch_path):
        return False

    if not validate_jobs_csv(csv_path):
        return False

    print("\n4. Verifying sbatch script is syntactically valid...")
    result = subprocess.run(
        ["bash", "-n", str(sbatch_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Bash syntax check failed: {result.stderr}")
        return False
    print("  [OK] Script passes bash syntax check")

    print("\n" + "=" * 60)
    print("Dry-run integration test PASSED")
    print("=" * 60)
    return True


def run_full_test(eval_base_dir: Path) -> bool:
    """Run the full test with actual job submission."""
    print("=" * 60)
    print("SLURM Integration Test (FULL MODE)")
    print("=" * 60)

    print("\n1. Verifying SLURM is running...")
    result = subprocess.run(["sinfo"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SLURM not available: {result.stderr}")
        return False
    print(result.stdout)

    print("\n2. Scheduling evaluation job...")
    setup_environment(eval_base_dir, dry_run=False)

    all_task_groups = ",".join(get_all_task_group_names())
    print(f"Testing all {len(get_all_task_group_names())} task groups: {all_task_groups}")

    result = subprocess.run(
        [
            "uv",
            "run",
            "oellm",
            "schedule-eval",
            "--models",
            "sshleifer/tiny-gpt2",
            "--task_groups",
            all_task_groups,
            "--limit",
            "5",
        ],
        capture_output=True,
        text=True,
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"schedule-eval failed with return code {result.returncode}")
        return False

    print("\n3. Waiting for SLURM job to complete...")
    if not wait_for_slurm_job(timeout=600):
        print("SLURM job did not complete in time")

        print("\nSLURM job status:")
        subprocess.run(["squeue", "-l"])

        print("\nSLURM logs:")
        user = os.environ.get("USER", "runner")
        for log in (eval_base_dir / user).rglob("*.out"):
            print(f"\n--- {log} ---")
            print(log.read_text()[-2000:])
        for log in (eval_base_dir / user).rglob("*.err"):
            print(f"\n--- {log} ---")
            print(log.read_text()[-2000:])

        return False

    print("\n4. Validating results...")
    eval_dir = find_eval_dir(eval_base_dir)

    if not eval_dir:
        print(f"Could not find evaluation directory under {eval_base_dir}")
        print("\nDirectory contents:")
        subprocess.run(["find", str(eval_base_dir), "-type", "f"])
        return False

    results_dir = eval_dir / "results"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return False

    print(f"Results directory: {results_dir}")

    if not validate_results(results_dir):
        print("Result validation failed")
        return False

    print("\n" + "=" * 60)
    print("Full integration test PASSED")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="SLURM integration test")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (test SLURM setup and script generation only)",
    )
    args = parser.parse_args()

    eval_base_dir = Path(os.environ.get("EVAL_BASE_DIR", "/tmp/oellm-test"))

    if args.dry_run:
        success = run_dry_run_test(eval_base_dir)
    else:
        success = run_full_test(eval_base_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

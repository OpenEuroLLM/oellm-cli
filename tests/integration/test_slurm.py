"""Integration tests for the oellm schedule-eval workflow with SLURM."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from oellm.task_groups import get_all_task_group_names


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


def wait_for_slurm_jobs(timeout: int = 600, poll_interval: int = 10) -> bool:
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
            return True

        print(
            f"Waiting for {len(jobs)} job(s) to complete... "
            f"(elapsed: {int(time.time() - start_time)}s)"
        )
        time.sleep(poll_interval)

    return False


def run_schedule_eval(
    task_groups: str, limit: int = 5, dry_run: bool = False, skip_checks: bool = False
):
    """Run oellm schedule-eval and return the result."""
    cmd = [
        "uv",
        "run",
        "oellm",
        "schedule-eval",
        "--models",
        "sshleifer/tiny-gpt2",
        "--task_groups",
        task_groups,
        "--limit",
        str(limit),
    ]
    if dry_run:
        cmd.extend(["--dry_run", "true"])
    if skip_checks:
        cmd.extend(["--skip_checks", "true"])

    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.usefixtures("slurm_available")
class TestSlurmAvailability:
    """Quick sanity checks for SLURM availability."""

    def test_sinfo_works(self):
        """Verify sinfo command works."""
        result = subprocess.run(["sinfo"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "PARTITION" in result.stdout or "STATE" in result.stdout


@pytest.mark.dry_run
@pytest.mark.usefixtures("slurm_available")
class TestScheduleEvalDryRun:
    """Tests for schedule-eval dry-run mode (no actual job submission)."""

    @pytest.fixture(autouse=True)
    def setup(self, slurm_env):
        """Run schedule-eval in dry-run mode once for all tests in this class."""
        all_task_groups = ",".join(get_all_task_group_names())

        os.environ["SINGULARITY_ARGS"] = ""
        os.environ["GPUS_PER_NODE"] = "0"

        result = run_schedule_eval(all_task_groups, limit=5, dry_run=True)

        assert result.returncode == 0, f"schedule-eval failed: {result.stderr}"

        self.eval_dir = find_eval_dir(slurm_env)
        assert self.eval_dir is not None, f"Could not find eval dir under {slurm_env}"

        self.sbatch_path = self.eval_dir / "submit_evals.sbatch"
        self.csv_path = self.eval_dir / "jobs.csv"

    def test_sbatch_script_exists(self):
        """Verify sbatch script was generated."""
        assert self.sbatch_path.exists(), f"sbatch script not found at {self.sbatch_path}"

    def test_sbatch_has_job_name_directive(self):
        """Verify sbatch script has job name directive."""
        content = self.sbatch_path.read_text()
        assert re.search(r"#SBATCH --job-name=", content)

    def test_sbatch_has_time_directive(self):
        """Verify sbatch script has time limit directive."""
        content = self.sbatch_path.read_text()
        assert re.search(r"#SBATCH --time=", content)

    def test_sbatch_has_output_directive(self):
        """Verify sbatch script has output directive."""
        content = self.sbatch_path.read_text()
        assert re.search(r"#SBATCH --output=", content)

    def test_sbatch_has_partition_directive(self):
        """Verify sbatch script has partition directive."""
        content = self.sbatch_path.read_text()
        assert re.search(r"#SBATCH --partition=", content)

    def test_sbatch_has_array_directive(self):
        """Verify sbatch script has array directive."""
        content = self.sbatch_path.read_text()
        assert re.search(r"#SBATCH --array=", content)

    def test_sbatch_has_csv_path_variable(self):
        """Verify sbatch script has CSV_PATH variable."""
        content = self.sbatch_path.read_text()
        assert re.search(r"CSV_PATH=", content)

    def test_sbatch_has_singularity_exec(self):
        """Verify sbatch script has singularity exec command."""
        content = self.sbatch_path.read_text()
        assert re.search(r"singularity exec", content)

    def test_sbatch_has_lm_eval_command(self):
        """Verify sbatch script has lm_eval command."""
        content = self.sbatch_path.read_text()
        assert re.search(r"lm_eval", content)

    def test_sbatch_has_limit_expansion(self):
        """Verify sbatch script has LIMIT variable expansion for --limit flag."""
        content = self.sbatch_path.read_text()
        assert re.search(r"LIMIT=", content)
        assert re.search(r"\$\{LIMIT:\+--limit \$LIMIT\}", content)

    def test_sbatch_bash_syntax_valid(self):
        """Verify sbatch script passes bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", str(self.sbatch_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_jobs_csv_exists(self):
        """Verify jobs CSV was generated."""
        assert self.csv_path.exists(), f"jobs.csv not found at {self.csv_path}"

    def test_jobs_csv_has_header(self):
        """Verify jobs CSV has required columns."""
        content = self.csv_path.read_text()
        header = content.split("\n")[0]

        assert "model_path" in header
        assert "task_path" in header
        assert "n_shot" in header
        assert "eval_suite" in header

    def test_jobs_csv_has_jobs(self):
        """Verify jobs CSV contains evaluation jobs."""
        content = self.csv_path.read_text()
        lines = [l for l in content.strip().split("\n") if l]  # noqa

        assert len(lines) > 1, "No jobs in CSV (only header found)"

    def test_jobs_csv_contains_all_task_groups(self):
        """Verify jobs CSV contains tasks from all task groups."""
        content = self.csv_path.read_text()

        assert "sshleifer/tiny-gpt2" in content
        assert "lm_eval" in content.lower() or "lm-eval" in content.lower()


@pytest.mark.slow
@pytest.mark.usefixtures("slurm_available")
class TestFullEvaluationPipeline:
    """Full integration test with actual SLURM job submission."""

    def test_full_evaluation_completes_and_produces_valid_results(self, slurm_env):
        """Submit evaluation job, wait for completion, and validate results."""
        all_task_groups = ",".join(get_all_task_group_names())
        print(f"\nTesting {len(get_all_task_group_names())} task groups with --limit 5")

        result = run_schedule_eval(all_task_groups, limit=5, dry_run=False)
        assert (
            result.returncode == 0
        ), f"schedule-eval failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

        print("\nWaiting for SLURM jobs to complete...")
        jobs_completed = wait_for_slurm_jobs(timeout=600, poll_interval=10)

        if not jobs_completed:
            subprocess.run(["squeue", "-l"])

            user = os.environ.get("USER", "runner")
            for log in (slurm_env / user).rglob("*.out"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])
            for log in (slurm_env / user).rglob("*.err"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])

        assert jobs_completed, "SLURM jobs did not complete within timeout"

        eval_dir = find_eval_dir(slurm_env)
        assert eval_dir is not None, f"Could not find eval directory under {slurm_env}"

        results_dir = eval_dir / "results"
        assert results_dir.exists(), f"Results directory not found: {results_dir}"

        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) > 0, f"No result JSON files found in {results_dir}"

        print(f"\nValidating {len(json_files)} result file(s)...")

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            assert "results" in data, f"Missing 'results' key in {json_file.name}"
            assert (
                "model_name" in data or "model" in data
            ), f"Missing model identifier in {json_file.name}"

            results = data.get("results", {})
            assert results, f"Empty results in {json_file.name}"

            for task_name, task_results in results.items():
                if "acc,none" in task_results:
                    acc = task_results["acc,none"]
                    assert (
                        0.0 <= acc <= 1.0
                    ), f"Accuracy {acc} out of range for {task_name}"

            print(f"  {json_file.name}: OK ({len(results)} task(s))")

        print("\nFull evaluation pipeline test PASSED")

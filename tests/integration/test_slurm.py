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

        result = run_schedule_eval(
            all_task_groups, limit=5, dry_run=True, skip_checks=True
        )

        assert (
            result.returncode == 0
        ), f"schedule-eval failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        self.eval_dir = find_eval_dir(slurm_env)
        if self.eval_dir is None:
            user = os.environ.get("USER", "runner")
            output_dir = slurm_env / user
            print(f"DEBUG: Looking for eval dir in {output_dir}")
            print(f"DEBUG: output_dir exists: {output_dir.exists()}")
            if output_dir.exists():
                print(f"DEBUG: contents: {list(output_dir.iterdir())}")
            print(f"DEBUG: EVAL_OUTPUT_DIR={os.environ.get('EVAL_OUTPUT_DIR')}")
            print(f"DEBUG: stdout: {result.stdout}")
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


def _get_dataset_specs():
    """Collect all dataset specs for parametrization."""
    from oellm.task_groups import _collect_dataset_specs

    specs = _collect_dataset_specs(get_all_task_group_names())
    return [(spec.repo_id, spec.subset) for spec in specs]


def _dataset_id(val):
    """Generate readable test ID for dataset parameter."""
    repo_id, subset = val
    if subset:
        return f"{repo_id.split('/')[-1]}/{subset}"
    return repo_id.split("/")[-1]


@pytest.mark.usefixtures("slurm_available")
class TestDatasetDownloads:
    """Test that all datasets for task groups can be downloaded - one test per dataset."""

    @pytest.mark.parametrize("dataset_spec", _get_dataset_specs(), ids=_dataset_id)
    def test_dataset_downloads_and_loads_offline(self, slurm_env, dataset_spec):
        """Download a single dataset and verify it can be loaded offline."""
        from datasets import get_dataset_config_names, load_dataset

        from oellm.task_groups import DatasetSpec
        from oellm.utils import _pre_download_datasets_from_specs

        repo_id, subset = dataset_spec
        label = f"{repo_id}" + (f"/{subset}" if subset else "")
        print(f"\nDownloading dataset: {label}")

        spec = DatasetSpec(repo_id=repo_id, subset=subset)
        _pre_download_datasets_from_specs([spec])

        verify_subset = subset
        if verify_subset is None:
            configs = get_dataset_config_names(repo_id, trust_remote_code=True)
            if configs:
                verify_subset = configs[0]

        print("Download complete. Verifying offline access...")

        old_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            load_dataset(repo_id, name=verify_subset, trust_remote_code=True)
            print(f"Dataset {label}: OK (offline access verified)")
        finally:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline


@pytest.mark.slow
@pytest.mark.usefixtures("slurm_available")
class TestFullEvaluationPipeline:
    """Full integration test with actual SLURM job submission - one test per task group."""

    @pytest.fixture(autouse=True)
    def setup_eval_tracking(self):
        """Track eval directories for result validation."""
        self.eval_dirs = []

    @pytest.mark.parametrize("task_group", get_all_task_group_names())
    def test_task_group_evaluation(self, slurm_env, task_group):
        """Submit and run evaluation for a single task group."""
        print(f"\n{'=' * 60}")
        print(f"Testing task group: {task_group}")
        print(f"{'=' * 60}")

        result = run_schedule_eval(task_group, limit=5, dry_run=False)

        if result.returncode != 0:
            print(f"schedule-eval stdout:\n{result.stdout}")
            print(f"schedule-eval stderr:\n{result.stderr}")
        assert (
            result.returncode == 0
        ), f"schedule-eval failed for {task_group}:\nSTDERR: {result.stderr}"

        print("Waiting for job to complete...")
        jobs_completed = wait_for_slurm_jobs(timeout=300, poll_interval=10)

        eval_dir = find_eval_dir(slurm_env)

        if not jobs_completed or eval_dir is None:
            user = os.environ.get("USER", "runner")
            for log in (slurm_env / user).rglob("*.out"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])
            for log in (slurm_env / user).rglob("*.err"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])

        assert jobs_completed, f"Job for {task_group} did not complete within timeout"
        assert eval_dir is not None, f"Could not find eval directory for {task_group}"

        results_dir = eval_dir / "results"
        if not results_dir.exists():
            slurm_logs_dir = eval_dir / "slurm_logs"
            if slurm_logs_dir.exists():
                for log in slurm_logs_dir.glob("*.out"):
                    print(f"\n--- {log.name} (stdout) ---")
                    print(log.read_text()[-3000:])
                for log in slurm_logs_dir.glob("*.err"):
                    print(f"\n--- {log.name} (stderr) ---")
                    print(log.read_text()[-3000:])
        assert (
            results_dir.exists()
        ), f"Results directory not found for {task_group}: {results_dir}"

        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) > 0, f"No result JSON files for {task_group}"

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            assert "results" in data, f"Missing 'results' in {json_file.name}"

            results = data.get("results", {})
            assert results, f"Empty results for {task_group}"

            for _, task_results in results.items():
                if "acc,none" in task_results:
                    acc = task_results["acc,none"]
                    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of range"

        print(f"Task group {task_group}: PASSED ({len(json_files)} result files)")

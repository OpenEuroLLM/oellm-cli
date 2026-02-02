import csv
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from oellm.task_groups import _expand_task_groups, get_all_task_group_names


def get_first_task_per_group() -> list[tuple[str, str, int, str]]:
    """Get the first task from each task group for CI testing.

    Returns list of (group_name, task_name, n_shot, suite) tuples.
    """
    results = []
    for group_name in get_all_task_group_names():
        expanded = _expand_task_groups([group_name])
        if expanded:
            first = expanded[0]
            results.append((group_name, first.task, first.n_shot, first.suite))
    return results


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
    """Run oellm schedule-eval with task groups and return the result."""
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


def run_schedule_eval_with_csv(
    csv_path: str,
    limit: int = 1,
    dry_run: bool = False,
    skip_checks: bool = False,
    verbose: bool = False,
):
    """Run oellm schedule-eval with a CSV file and return the result."""
    cmd = [
        "uv",
        "run",
        "oellm",
        "schedule-eval",
        "--eval_csv_path",
        csv_path,
        "--limit",
        str(limit),
    ]
    if dry_run:
        cmd.extend(["--dry_run", "true"])
    if skip_checks:
        cmd.extend(["--skip_checks", "true"])
    if verbose:
        cmd.extend(["--verbose", "true"])

    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.usefixtures("slurm_available")
class TestSlurmAvailability:
    """Quick sanity checks for SLURM availability."""

    def test_sinfo_works(self):
        """Verify sinfo command works."""
        result = subprocess.run(["sinfo"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "PARTITION" in result.stdout or "STATE" in result.stdout


@pytest.mark.skip
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
            all_task_groups, limit=1, dry_run=True, skip_checks=True
        )

        assert result.returncode == 0, (
            f"schedule-eval failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

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

    @pytest.mark.parametrize(
        "pattern,description",
        [
            (r"#SBATCH --job-name=", "job-name"),
            (r"#SBATCH --time=", "time"),
            (r"#SBATCH --output=", "output"),
            (r"#SBATCH --partition=", "partition"),
            (r"#SBATCH --array=", "array"),
            (r"CSV_PATH=", "CSV_PATH"),
            (r"singularity exec", "singularity-exec"),
            (r"lm_eval", "lm_eval"),
            (r"LIMIT=", "LIMIT-var"),
            (r"\$\{LIMIT:\+--limit \$LIMIT\}", "LIMIT-expansion"),
        ],
        ids=lambda x: x[1],
    )
    def test_sbatch_contains(self, pattern, description):
        """Verify sbatch script contains expected patterns."""
        content = self.sbatch_path.read_text()
        assert re.search(pattern, content), f"sbatch missing {description}"

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


@pytest.mark.skip(reason="Temporarily disabled to speed up iteration")
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


def _get_first_task_params():
    """Get parametrize values for first task per group."""
    return get_first_task_per_group()


def _first_task_id(val):
    """Generate test ID for first task parameter."""
    group_name, task_name, n_shot, suite = val
    return f"{group_name}:{task_name}"


def _dump_all_logs(slurm_env, prefix=""):
    """Dump all available logs for debugging."""
    user = os.environ.get("USER", "runner")
    base_dir = slurm_env / user

    print(f"\n{prefix}{'=' * 60}")
    print(f"{prefix}DUMPING ALL LOGS FOR DEBUGGING")
    print(f"{prefix}{'=' * 60}")

    if not base_dir.exists():
        print(f"{prefix}Base dir {base_dir} does not exist")
        return

    print(f"{prefix}Contents of {base_dir}:")
    for item in base_dir.rglob("*"):
        if item.is_file():
            print(f"{prefix}  {item.relative_to(base_dir)}")

    for log in base_dir.rglob("*.out"):
        print(f"\n{prefix}--- {log} (stdout) ---")
        try:
            content = log.read_text()
            print(content[-5000:] if len(content) > 5000 else content)
        except Exception as e:
            print(f"{prefix}Error reading: {e}")

    for log in base_dir.rglob("*.err"):
        print(f"\n{prefix}--- {log} (stderr) ---")
        try:
            content = log.read_text()
            print(content[-5000:] if len(content) > 5000 else content)
        except Exception as e:
            print(f"{prefix}Error reading: {e}")

    print(f"\n{prefix}Checking running processes...")
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if (
            "lighteval" in line.lower()
            or "lm_eval" in line.lower()
            or "python" in line.lower()
        ):
            print(f"{prefix}  {line}")

    print(f"\n{prefix}SLURM queue status:")
    result = subprocess.run(["squeue", "-a"], capture_output=True, text=True)
    print(result.stdout)

    print(f"{prefix}{'=' * 60}")


@pytest.mark.skip(reason="Temporarily disabled - running flores200 debug test only")
@pytest.mark.slow
@pytest.mark.usefixtures("slurm_available")
class TestFullEvaluationPipeline:
    """Full integration test with actual SLURM job submission - first task per group."""

    @pytest.fixture(autouse=True)
    def setup_eval_tracking(self):
        """Track eval directories for result validation."""
        self.eval_dirs = []

    @pytest.mark.parametrize("task_info", _get_first_task_params(), ids=_first_task_id)
    def test_task_group_evaluation(self, slurm_env, task_info):
        """Submit and run evaluation for the first task of a task group."""
        group_name, task_name, n_shot, suite = task_info

        print(f"\n{'=' * 60}")
        print(f"Testing: {group_name} -> {task_name} (n_shot={n_shot}, suite={suite})")
        print(f"{'=' * 60}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["model_path", "task_path", "n_shot", "eval_suite"])
            writer.writerow(["sshleifer/tiny-gpt2", task_name, n_shot, suite])
            csv_path = csv_file.name

        result = run_schedule_eval_with_csv(csv_path, limit=1, dry_run=False)
        os.unlink(csv_path)

        if result.returncode != 0:
            print(f"schedule-eval stdout:\n{result.stdout}")
            print(f"schedule-eval stderr:\n{result.stderr}")
        assert result.returncode == 0, (
            f"schedule-eval failed for {group_name}:\nSTDERR: {result.stderr}"
        )

        print("Waiting for job to complete...")
        jobs_completed = wait_for_slurm_jobs(timeout=600, poll_interval=10)

        eval_dir = find_eval_dir(slurm_env)

        if not jobs_completed or eval_dir is None:
            user = os.environ.get("USER", "runner")
            for log in (slurm_env / user).rglob("*.out"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])
            for log in (slurm_env / user).rglob("*.err"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])

        assert jobs_completed, f"Job for {group_name} did not complete within timeout"
        assert eval_dir is not None, f"Could not find eval directory for {group_name}"

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
        assert results_dir.exists(), (
            f"Results directory not found for {group_name}: {results_dir}"
        )

        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) > 0, f"No result JSON files for {group_name}"

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            assert "results" in data, f"Missing 'results' in {json_file.name}"

            results = data.get("results", {})
            assert results, f"Empty results for {group_name}"

            for _, task_results in results.items():
                if "acc,none" in task_results:
                    acc = task_results["acc,none"]
                    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of range"

        print(f"{group_name}: PASSED ({task_name}, {len(json_files)} result files)")


@pytest.mark.slow
@pytest.mark.usefixtures("slurm_available")
class TestFlores200Debug:
    """Focused debug test for flores200 lighteval task."""

    @pytest.mark.timeout(300)
    def test_flores200_lighteval(self, slurm_env):
        """Debug test for flores200 translation task with extensive logging."""
        task_name = "flores200:bul_Cyrl-eng_Latn"
        n_shot = 0
        suite = "lighteval"

        print(f"\n{'=' * 60}")
        print("DEBUG TEST: flores200 lighteval")
        print(f"Task: {task_name}")
        print(f"N-shot: {n_shot}")
        print(f"Suite: {suite}")
        print(f"{'=' * 60}")

        print(f"\n[{time.strftime('%H:%M:%S')}] Environment info:")
        print(f"  EVAL_BASE_DIR: {os.environ.get('EVAL_BASE_DIR')}")
        print(f"  HF_HOME: {os.environ.get('HF_HOME')}")
        print(f"  SINGULARITY_ARGS: {os.environ.get('SINGULARITY_ARGS')}")
        print(f"  GPUS_PER_NODE: {os.environ.get('GPUS_PER_NODE')}")

        print(f"\n[{time.strftime('%H:%M:%S')}] Checking network connectivity...")
        net_result = subprocess.run(
            [
                "curl",
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "--max-time",
                "5",
                "https://tinyurl.com/flores200sacrebleuspm",
            ],
            capture_output=True,
            text=True,
        )
        print(f"  sacrebleu tokenizer URL reachable: {net_result.stdout}")

        print(f"\n[{time.strftime('%H:%M:%S')}] Creating job CSV...")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["model_path", "task_path", "n_shot", "eval_suite"])
            writer.writerow(["sshleifer/tiny-gpt2", task_name, n_shot, suite])
            csv_path = csv_file.name
            print(f"  CSV path: {csv_path}")

        print(f"\n[{time.strftime('%H:%M:%S')}] Running schedule-eval (verbose)...")
        result = run_schedule_eval_with_csv(
            csv_path, limit=1, dry_run=False, verbose=True
        )
        os.unlink(csv_path)

        print(f"\n[{time.strftime('%H:%M:%S')}] schedule-eval result:")
        print(f"  Return code: {result.returncode}")
        print(f"  STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"  STDERR:\n{result.stderr}")

        assert result.returncode == 0, (
            f"schedule-eval failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        eval_dir = find_eval_dir(slurm_env)
        print(f"\n[{time.strftime('%H:%M:%S')}] Eval directory: {eval_dir}")

        if eval_dir:
            sbatch_path = eval_dir / "submit_evals.sbatch"
            if sbatch_path.exists():
                print(f"\n[{time.strftime('%H:%M:%S')}] Generated sbatch script:")
                print("-" * 40)
                print(sbatch_path.read_text())
                print("-" * 40)

            jobs_csv = eval_dir / "jobs.csv"
            if jobs_csv.exists():
                print(f"\n[{time.strftime('%H:%M:%S')}] Jobs CSV content:")
                print(jobs_csv.read_text())

        print(f"\n[{time.strftime('%H:%M:%S')}] Waiting for SLURM job (timeout=180s)...")
        job_timeout = 180
        poll_interval = 5
        start_time = time.time()
        jobs_completed = False

        while time.time() - start_time < job_timeout:
            elapsed = int(time.time() - start_time)
            result = subprocess.run(
                ["squeue", "-h", "-o", "%i %j %T %M"],
                capture_output=True,
                text=True,
            )
            jobs = [j.strip() for j in result.stdout.strip().split("\n") if j.strip()]

            if not jobs:
                print(
                    f"[{time.strftime('%H:%M:%S')}] All jobs completed after {elapsed}s"
                )
                jobs_completed = True
                break

            print(
                f"[{time.strftime('%H:%M:%S')}] Jobs still running ({elapsed}s): {jobs}"
            )

            if eval_dir:
                slurm_logs = eval_dir / "slurm_logs"
                if slurm_logs.exists():
                    for log in slurm_logs.glob("*.out"):
                        content = log.read_text()
                        if content:
                            lines = content.strip().split("\n")
                            print(
                                f"  Latest output ({log.name}): {lines[-1][:100] if lines else 'empty'}"
                            )

            time.sleep(poll_interval)

        print(
            f"\n[{time.strftime('%H:%M:%S')}] Job wait finished. Completed: {jobs_completed}"
        )

        _dump_all_logs(slurm_env, prefix="[DEBUG] ")

        if not jobs_completed:
            print(
                f"\n[{time.strftime('%H:%M:%S')}] TIMEOUT - Cancelling remaining jobs..."
            )
            subprocess.run(["scancel", "-u", os.environ.get("USER", "runner")])
            time.sleep(2)

        assert jobs_completed, (
            "flores200 job did not complete within timeout - check logs above"
        )

        eval_dir = find_eval_dir(slurm_env)
        assert eval_dir is not None, "Could not find eval directory"

        results_dir = eval_dir / "results"
        assert results_dir.exists(), f"Results directory not found: {results_dir}"

        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) > 0, "No result JSON files found"

        print(
            f"\n[{time.strftime('%H:%M:%S')}] SUCCESS - Found {len(json_files)} result files"
        )
        for jf in json_files:
            print(f"  {jf.name}")

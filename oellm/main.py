import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from string import Template

import pandas as pd
from jsonargparse import auto_cli

from oellm.task_groups import (
    _collect_dataset_specs,
    _expand_task_groups,
    _lookup_dataset_specs_for_tasks,
)
from oellm.utils import (
    _ensure_singularity_image,
    _expand_local_model_paths,
    _filter_warnings,
    _load_cluster_env,
    _num_jobs_in_queue,
    _pre_download_datasets_from_specs,
    _process_model_paths,
    _setup_logging,
    capture_third_party_output_from_kwarg,
)


@dataclass
class EvaluationJob:
    model_path: Path | str
    task_path: str
    n_shot: int
    eval_suite: str


@capture_third_party_output_from_kwarg("verbose")
def schedule_evals(
    models: str | None = None,
    tasks: str | None = None,
    task_groups: str | None = None,
    n_shot: int | list[int] | None = None,
    eval_csv_path: str | None = None,
    *,
    max_array_len: int = 128,
    verbose: bool = False,
    download_only: bool = False,
    dry_run: bool = False,
    skip_checks: bool = False,
    trust_remote_code: bool = True,
) -> None:
    """
    Schedule evaluation jobs for a given set of models, tasks, and number of shots.

    Args:
        models: A string of comma-separated model paths or Hugging Face model identifiers.
            Warning: does not allow passing model args such as `EleutherAI/pythia-160m,revision=step100000`
            since we split on commas. If you need to pass model args, use the `eval_csv_path` option.
            For local paths:
            - If a directory contains `.safetensors` files directly, it will be treated as a single model
            - If a directory contains subdirectories with models (e.g., converted_checkpoints/),
              all models in subdirectories will be automatically discovered
            - For each model directory, if it has an `hf/iter_XXXXX` structure, all checkpoints will be expanded
            - This allows passing a single directory containing multiple models to evaluate them all
        tasks: A string of comma-separated task names (lm_eval) or paths.
            Requires `n_shot` to be provided. Tasks here are assumed to be lm_eval unless otherwise handled via CSV.
        task_groups: A string of comma-separated task group names defined in `task-groups.yaml`.
            Each group expands into concrete (task, n_shots, suite) entries; `n_shot` is ignored for groups.
        n_shot: An integer or list of integers specifying the number of shots applied to `tasks`.
        eval_csv_path: A path to a CSV file containing evaluation data.
            Warning: exclusive argument. Cannot specify `models`, `tasks`, `task_groups`, or `n_shot` when `eval_csv_path` is provided.
        max_array_len: The maximum number of jobs to schedule to run concurrently.
            Warning: this is not the number of jobs in the array job. This is determined by the environment variable `QUEUE_LIMIT`.
        download_only: If True, only download the datasets and models and exit.
        dry_run: If True, generate the SLURM script but don't submit it to the scheduler.
        skip_checks: If True, skip container image, model validation, and dataset pre-download checks for faster execution.
        trust_remote_code: If True, trust remote code when downloading datasets. Default is True. Workflow might fail if set to False.
    """
    _setup_logging(verbose)

    # Load cluster-specific environment variables (paths, etc.)
    _load_cluster_env()

    if not skip_checks:
        _ensure_singularity_image(os.environ.get("EVAL_CONTAINER_IMAGE"))  # type: ignore
    else:
        logging.info("Skipping container image check (--skip-checks enabled)")

    if isinstance(models, str) and models is not None:
        models = [m.strip() for m in models.split(",") if m.strip()]  # type: ignore

    if isinstance(tasks, str) and tasks is not None:
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]  # type: ignore

    if isinstance(n_shot, int) and n_shot is not None:
        n_shot = [n_shot]

    eval_jobs: list[EvaluationJob] = []
    if eval_csv_path:
        if models or tasks or task_groups or n_shot:
            raise ValueError(
                "Cannot specify `models`, `tasks`, `task_groups`, or `n_shot` when `eval_csv_path` is provided."
            )
        df = pd.read_csv(eval_csv_path)
        required_cols = {"model_path", "task_path", "n_shot"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV file must contain the columns: {', '.join(required_cols)}"
            )

        if "eval_suite" not in df.columns:
            df["eval_suite"] = "lm_eval"
        else:
            df["eval_suite"] = df["eval_suite"].fillna("lm_eval")

        # Always expand local model paths, even with skip_checks
        df["model_path"].unique()
        eval_jobs.extend(
            [
                EvaluationJob(
                    model_path=row["model_path"],
                    task_path=row["task_path"],
                    n_shot=row["n_shot"],
                    eval_suite=row["eval_suite"],
                )
                for _, row in df.iterrows()
            ]
        )

    elif models:
        if task_groups is None:
            eval_jobs.extend(
                [
                    EvaluationJob(
                        model_path=model,
                        task_path=task,
                        n_shot=shot,
                        eval_suite="lm_eval",
                    )
                    for model in models
                    for task in tasks
                    for shot in n_shot
                ]
            )
        else:
            expanded = _expand_task_groups([g.strip() for g in task_groups.split(",")])
            eval_jobs.extend(
                [
                    EvaluationJob(
                        model_path=model,
                        task_path=result.task,
                        n_shot=result.n_shot,
                        eval_suite=result.suite,
                    )
                    for model in models
                    for result in expanded
                ]
            )

    expanded_eval_jobs = []
    for job in eval_jobs:
        local_model_paths = _expand_local_model_paths(job.model_path)
        if not local_model_paths:
            expanded_eval_jobs.append(job)
        else:
            for path in local_model_paths:
                expanded_eval_jobs.append(
                    EvaluationJob(
                        model_path=path,
                        task_path=job.task_path,
                        n_shot=job.n_shot,
                        eval_suite=job.eval_suite,
                    )
                )

    if not skip_checks:
        hub_models: set[str | Path] = {
            job.model_path
            for job in expanded_eval_jobs
            if not Path(job.model_path).exists()
        }
        _process_model_paths(hub_models)
    else:
        logging.info(
            "Skipping model path processing and validation (--skip-checks enabled)"
        )

    # create csv
    df = pd.DataFrame(expanded_eval_jobs)

    if df.empty:
        logging.warning("No evaluation jobs to schedule.")
        return None

    df["eval_suite"] = df["eval_suite"].str.lower()

    # Ensure that all datasets required by the tasks are cached locally to avoid
    # network access on compute nodes.
    if not skip_checks:
        dataset_specs = []
        if task_groups:
            dataset_specs = _collect_dataset_specs(
                [g.strip() for g in task_groups.split(",")]
            )
        else:
            # Look up individual tasks in task groups registry
            all_tasks = df["task_path"].unique().tolist()
            dataset_specs = _lookup_dataset_specs_for_tasks(all_tasks)
            if not dataset_specs:
                logging.info(
                    "No dataset specs found for tasks; skipping dataset pre-download"
                )

        if dataset_specs:
            _pre_download_datasets_from_specs(
                dataset_specs, trust_remote_code=trust_remote_code
            )
    else:
        logging.info("Skipping dataset pre-download (--skip-checks enabled)")

    if download_only:
        return None

    remaining_queue_capacity = (
        int(os.environ.get("QUEUE_LIMIT", 250)) - _num_jobs_in_queue()
    )

    if remaining_queue_capacity <= 0:
        logging.warning("No remaining queue capacity. Not scheduling any jobs.")
        return None

    logging.debug(
        f"Remaining capacity in the queue: {remaining_queue_capacity}. Number of "
        f"evals to schedule: {len(df)}."
    )

    evals_dir = (
        Path(os.environ["EVAL_OUTPUT_DIR"])
        / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    evals_dir.mkdir(parents=True, exist_ok=True)

    slurm_logs_dir = evals_dir / "slurm_logs"
    slurm_logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evals_dir / "jobs.csv"

    # Shuffle the dataframe to distribute fast/slow evaluations evenly across array jobs
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(
        "Shuffled evaluation jobs for even load distribution across array workers"
    )

    df.to_csv(csv_path, index=False)

    sbatch_template = (files("oellm.resources") / "template.sbatch").read_text()

    # Calculate dynamic array size and time limits
    total_evals = len(df)
    minutes_per_eval = 10  # Budget 10 minutes per eval
    total_minutes = total_evals * minutes_per_eval
    max_minutes_per_job = 18 * 60  # 18 hours
    min_array_size_for_time = max(1, int(math.ceil(total_minutes / max_minutes_per_job)))
    desired_array_size = min(128, total_evals) if total_evals >= 128 else total_evals
    if desired_array_size < min_array_size_for_time:
        desired_array_size = min_array_size_for_time
    actual_array_size = min(remaining_queue_capacity, desired_array_size, total_evals)
    evals_per_job = max(1, int(math.ceil(total_evals / actual_array_size)))
    minutes_per_job = evals_per_job * minutes_per_eval
    minutes_with_margin = int(minutes_per_job * 1.2)
    hours_with_margin = max(1, int(math.ceil(minutes_with_margin / 60)))
    hours_with_margin = max(hours_with_margin, 3)
    hours_with_margin = min(hours_with_margin, 23)
    time_limit = f"{hours_with_margin:02d}:59:00"

    # Log the calculated values
    logging.info("📊 Evaluation planning:")
    logging.info(f"   Total evaluations: {total_evals}")
    logging.info(f"   Estimated time per eval: {minutes_per_eval} minutes")
    logging.info(
        f"   Total estimated time: {total_minutes} minutes ({total_minutes / 60:.1f} hours)"
    )
    logging.info(f"   Desired array size: {desired_array_size}")
    logging.info(
        f"   Actual array size: {actual_array_size} (limited by queue capacity: {remaining_queue_capacity})"
    )
    logging.info(f"   Evaluations per job: {evals_per_job}")
    logging.info(
        f"   Time per job: {minutes_per_job} minutes ({minutes_per_job / 60:.1f} hours)"
    )
    logging.info(f"   Time limit with safety margin: {time_limit}")

    sbatch_script = sbatch_template.format(
        csv_path=csv_path,
        max_array_len=max_array_len,
        array_limit=actual_array_size - 1,  # Array is 0-indexed
        num_jobs=actual_array_size,  # This is the number of array jobs, not total evals
        total_evals=len(df),  # Pass the total number of evaluations
        log_dir=evals_dir / "slurm_logs",
        evals_dir=str(evals_dir / "results"),
        time_limit=time_limit,  # Dynamic time limit
    )

    # substitute any $ENV_VAR occurrences
    sbatch_script = Template(sbatch_script).safe_substitute(os.environ)

    sbatch_script_path = evals_dir / "submit_evals.sbatch"

    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)

    if dry_run:
        logging.info(f"Dry run mode: SLURM script generated at {sbatch_script_path}")
        logging.info(
            f"Would schedule {actual_array_size} array jobs to handle {len(df)} evaluations"
        )
        logging.info(
            f"Each array job will handle ~{(len(df) + actual_array_size - 1) // actual_array_size} evaluations"
        )
        logging.info("To submit the job, run: sbatch " + str(sbatch_script_path))
        return

    # Submit the job script to slurm by piping the script content to sbatch
    try:
        logging.info("Calling sbatch to launch the evaluations")

        # Provide helpful information about job monitoring and file locations
        logging.info(f"📁 Evaluation directory: {evals_dir}")
        logging.info(f"📄 SLURM script: {sbatch_script_path}")
        logging.info(f"📋 Job configuration: {csv_path}")
        logging.info(f"📜 SLURM logs will be stored in: {slurm_logs_dir}")
        logging.info(f"📊 Results will be stored in: {evals_dir / 'results'}")

        result = subprocess.run(
            ["sbatch"],
            input=sbatch_script,
            text=True,
            check=True,
            capture_output=True,
            env=os.environ,
        )
        logging.info("Job submitted successfully.")
        logging.info(result.stdout)
        # Extract job ID from sbatch output for monitoring commands
        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if job_id_match:
            job_id = job_id_match.group(1)
            logging.info(f"🔍 Monitor job status: squeue -j {job_id}")
            logging.info(f"📈 View job details: scontrol show job {job_id}")
            logging.info(f"❌ Cancel job if needed: scancel {job_id}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to submit job: {e}")
        logging.error(f"sbatch stderr: {e.stderr}")
    except FileNotFoundError:
        logging.error(
            "sbatch command not found. Please make sure you are on a system with SLURM installed."
        )


def collect_results(
    results_dir: str,
    output_csv: str = "eval_results.csv",
    *,
    check: bool = False,
    verbose: bool = False,
) -> None:
    """
    Collect evaluation results from JSON files and export to CSV.

    Args:
        results_dir: Path to the directory containing result JSON files
        output_csv: Output CSV filename (default: eval_results.csv)
        check: Check for missing evaluations and create a missing jobs CSV
        verbose: Enable verbose logging
    """
    import json

    import yaml

    _setup_logging(verbose)

    task_groups_yaml = files("oellm.resources") / "task-groups.yaml"
    with open(str(task_groups_yaml)) as _f:
        _tg_cfg = yaml.safe_load(_f)
    task_metrics = _tg_cfg.get("task_metrics", {})

    def _resolve_metric(task_name: str, result_dict: dict) -> float | None:
        """Return the preferred metric value for task_name from result_dict."""
        preferred = task_metrics.get(task_name)
        if preferred is not None:
            val = result_dict.get(f"{preferred},none")
            if val is None:
                val = result_dict.get(preferred)
            return val
        for metric in ["acc,none", "acc", "accuracy", "f1", "exact_match"]:
            if metric in result_dict:
                return result_dict[metric]
        return None

    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Check if we need to look in a 'results' subdirectory
    if (results_path / "results").exists() and (results_path / "results").is_dir():
        # User passed the top-level directory, look in results subdirectory
        json_files = list((results_path / "results").glob("*.json"))
    else:
        # User passed the results directory directly
        json_files = list(results_path.glob("*.json"))

    if not json_files:
        logging.warning(f"No JSON files found in {results_dir}")
        if not check:
            return

    logging.info(f"Found {len(json_files)} result files")

    # If check mode, also load the jobs.csv to compare
    if check:
        jobs_csv_path = results_path / "jobs.csv"
        if not jobs_csv_path.exists():
            logging.warning(f"No jobs.csv found in {results_dir}, cannot perform check")
            check = False
        else:
            jobs_df = pd.read_csv(jobs_csv_path)
            logging.info(f"Found {len(jobs_df)} scheduled jobs in jobs.csv")

    # Collect results
    rows = []
    completed_jobs = set()  # Track (model, task, n_shot) tuples

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Extract model name/path
        model_name = data.get("model_name", "unknown")

        # Extract results for each task
        results = data.get("results", {})
        n_shot_data = data.get("n-shot", {})

        # Infer a global n_shot if exactly one unique value exists in this JSON
        global_n_shot = None
        try:
            candidate_values = []
            for _v in n_shot_data.values():
                if isinstance(_v, (int | float)):
                    candidate_values.append(int(_v))
                elif isinstance(_v, str) and _v.isdigit():
                    candidate_values.append(int(_v))
            unique_values = set(candidate_values)
            if len(unique_values) == 1:
                global_n_shot = next(iter(unique_values))
        except Exception:
            pass

        # Aggregate groups (lm-eval harness)
        groups_map = data.get("groups", {})
        group_subtasks_map = data.get("group_subtasks", {})
        group_aggregate_names = set(groups_map.keys()) | set(group_subtasks_map.keys())
        group_subtask_names: set[str] = set()
        for _agg, _subs in group_subtasks_map.items():
            for _s in _subs:
                group_subtask_names.add(_s)

        # Prefer only the first aggregate metric from groups (simplified)
        if groups_map:
            group_name, group_results = next(iter(groups_map.items()))
            n_shot = n_shot_data.get(group_name, "unknown")
            if n_shot == "unknown":
                for subtask_name in group_subtasks_map.get(group_name, []):
                    if subtask_name in n_shot_data:
                        n_shot = n_shot_data[subtask_name]
                        break
            if n_shot == "unknown" and global_n_shot is not None:
                n_shot = global_n_shot
            performance = _resolve_metric(group_name, group_results)
            if performance is not None:
                if check:
                    completed_jobs.add((model_name, group_name, n_shot))
                rows.append(
                    {
                        "model_name": model_name,
                        "task": group_name,
                        "n_shot": n_shot,
                        "performance": performance,
                    }
                )
                # Skip per-task iteration when groups are present
                continue

        for task_name, task_results in results.items():
            # Skip entries already added from groups
            if groups_map and task_name in group_aggregate_names:
                continue
            # Skip any lm-eval group subtasks; keep only aggregates
            if task_name in group_subtask_names:
                continue

            # Skip MMLU subtasks - only keep the aggregate score
            if task_name.startswith("mmlu_") and task_name != "mmlu":
                continue

            # Skip Global MMLU subtasks - keep only aggregates like global_mmlu_full_pt
            if task_name.startswith("global_mmlu_") and task_name.count("_") >= 4:
                continue

            # Get n_shot for this task
            n_shot = n_shot_data.get(task_name, "unknown")

            # If this is a group aggregate and n_shot is missing, derive from any subtask
            if task_name in group_aggregate_names and n_shot == "unknown":
                for subtask_name in group_subtasks_map.get(task_name, []):
                    if subtask_name in n_shot_data:
                        n_shot = n_shot_data[subtask_name]
                        break
            if n_shot == "unknown" and global_n_shot is not None:
                n_shot = global_n_shot

            # Special handling for MMLU aggregate - get n_shot from any MMLU subtask
            if task_name == "mmlu" and n_shot == "unknown":
                for key, value in n_shot_data.items():
                    if key.startswith("mmlu_"):
                        n_shot = value
                        break
                if n_shot == "unknown" and global_n_shot is not None:
                    n_shot = global_n_shot

            # Special handling for Global MMLU aggregates - get n_shot from subtasks
            if task_name.startswith("global_mmlu_") and n_shot == "unknown":
                prefix = f"{task_name}_"
                for key, value in n_shot_data.items():
                    if key.startswith(prefix):
                        n_shot = value
                        break
                if n_shot == "unknown" and global_n_shot is not None:
                    n_shot = global_n_shot

            # Get the primary metric (usually acc, acc_norm)
            performance = _resolve_metric(task_name, task_results)

            if performance is not None:
                # Track completed job for check mode
                if check:
                    completed_jobs.add((model_name, task_name, n_shot))

                rows.append(
                    {
                        "model_name": model_name,
                        "task": task_name,
                        "n_shot": n_shot,
                        "performance": performance,
                    }
                )
            else:
                # Debug: log cases where we have a task but no performance metric
                if verbose:
                    logging.debug(
                        f"No performance metric found for {model_name} | {task_name} | n_shot={n_shot} in {json_file.name}"
                    )

    if not rows and not check:
        logging.warning("No results extracted from JSON files")
        return

    # Create DataFrame and save to CSV (if we have results)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logging.info(f"Results saved to {output_csv}")
        logging.info(f"Extracted {len(df)} evaluation results")

        # Print summary statistics
        if verbose:
            logging.info("Summary:")
            logging.info(f"Unique models: {df['model_name'].nunique()}")
            logging.info(f"Unique tasks: {df['task'].nunique()}")
            logging.info(
                f"N-shot values: {sorted(str(x) for x in df['n_shot'].unique())}"
            )

    # Perform check analysis if requested
    if check:
        logging.info("=== Evaluation Status Check ===")

        # Find missing jobs
        missing_jobs = []

        for _, job in jobs_df.iterrows():
            job_tuple = (job["model_path"], job["task_path"], job["n_shot"])

            # Check if this job corresponds to one of our completed results
            is_completed = False

            # Try exact matching first
            if job_tuple in completed_jobs:
                is_completed = True
            else:
                # Try fuzzy matching for model names
                for completed_job in completed_jobs:
                    completed_model, completed_task, completed_n_shot = completed_job

                    if (
                        job["n_shot"] == completed_n_shot
                        and job["task_path"] == completed_task
                        and (
                            str(job["model_path"]).endswith(completed_model)
                            or completed_model in str(job["model_path"])
                        )
                    ):
                        is_completed = True
                        break

            if not is_completed:
                missing_jobs.append(job)

        completed_count = len(jobs_df) - len(missing_jobs)

        logging.info(f"Total scheduled jobs: {len(jobs_df)}")
        logging.info(f"Completed jobs: {completed_count}")
        logging.info(f"Missing jobs: {len(missing_jobs)}")

        if len(missing_jobs) > 0:
            missing_df = pd.DataFrame(missing_jobs)
            missing_csv = output_csv.replace(".csv", "_missing.csv")
            missing_df.to_csv(missing_csv, index=False)
            logging.info(f"Missing jobs saved to: {missing_csv}")
            logging.info(
                f"You can run these with: oellm schedule-eval --eval_csv_path {missing_csv}"
            )

            # Show some examples if verbose
            if verbose and len(missing_jobs) > 0:
                logging.info("Example missing jobs:")
                for _i, (_, job) in enumerate(missing_df.head(5).iterrows()):
                    logging.info(
                        f"  - {job['model_path']} | {job['task_path']} | n_shot={job['n_shot']}"
                    )
                if len(missing_jobs) > 5:
                    logging.info(f"  ... and {len(missing_jobs) - 5} more")


def main():
    _filter_warnings()
    auto_cli(
        {
            "schedule-eval": schedule_evals,
            "collect-results": collect_results,
        },
        as_positional=False,
        description="OELLM: Multi-cluster evaluation tool for language models",
    )

import logging
import os
import re
import shutil
import subprocess
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from jsonargparse import auto_cli

from oellm.task_cache import clear_task_cache
from oellm.task_groups import _expand_task_groups
from oellm.utils import (
    _ensure_singularity_image,
    _expand_local_model_paths,
    _load_cluster_env,
    _num_jobs_in_queue,
    _pre_download_lighteval_datasets,
    _pre_download_task_datasets,
    _process_model_paths,
    _setup_logging,
    capture_third_party_output_from_kwarg,
)


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
        image_name = os.environ.get("EVAL_CONTAINER_IMAGE")
        if image_name is None:
            raise ValueError(
                "EVAL_CONTAINER_IMAGE is not set. Please set it in clusters.yaml."
            )

        _ensure_singularity_image(image_name)
    else:
        logging.info("Skipping container image check (--skip-checks enabled)")

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
        expanded_rows = []
        for _, row in df.iterrows():
            original_model_path = row["model_path"]
            local_paths = _expand_local_model_paths(original_model_path)
            if local_paths:
                # Use expanded local paths
                for expanded_path in local_paths:
                    new_row = row.copy()
                    new_row["model_path"] = expanded_path
                    expanded_rows.append(new_row)
            else:
                # Keep original path (might be HF model)
                expanded_rows.append(row)
        df = pd.DataFrame(expanded_rows)

        if "eval_suite" not in df.columns:
            df["eval_suite"] = "lm_eval"

        # Download HF models only if skip_checks is False
        if not skip_checks:
            # Process any HF models that need downloading
            hf_models = [m for m in df["model_path"].unique() if not Path(m).exists()]
            if hf_models:
                model_path_map = _process_model_paths(hf_models)
                # Update the dataframe with processed HF models
                for idx, row in df.iterrows():
                    if row["model_path"] in model_path_map:
                        # This shouldn't expand further, just update the path
                        df.at[idx, "model_path"] = model_path_map[row["model_path"]][0]
        else:
            logging.info(
                "Skipping model path processing and validation (--skip-checks enabled)"
            )

    elif models and ((tasks and n_shot is not None) or task_groups):
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        model_paths: list[Path | str] = []

        # Always expand local paths
        for model in model_list:
            local_paths = _expand_local_model_paths(model)
            if local_paths:
                model_paths.extend(local_paths)
            else:
                model_paths.append(model)

        # Download HF models only if skip_checks is False
        if not skip_checks:
            hf_models = [m for m in model_paths if not Path(m).exists()]
            if hf_models:
                model_path_map = _process_model_paths(hf_models)
                # Replace HF model identifiers with processed paths
                model_paths = [
                    model_path_map[m][0] if m in model_path_map else m
                    for m in model_paths
                ]
        else:
            logging.info(
                "Skipping model path processing and validation (--skip-checks enabled)"
            )

        rows: list[dict[str, Path | str | int]] = []

        # Handle explicit tasks (lm_eval) with provided n_shot
        if tasks:
            if n_shot is None:
                raise ValueError(
                    "When specifying `tasks`, you must also provide `n_shot`. For task groups, use `task_groups`."
                )
            tasks_list = [t.strip() for t in tasks.split(",") if t.strip()]
            shots: list[int]
            shots = n_shot if isinstance(n_shot, list) else [int(n_shot)]
            for model_path in model_paths:
                for task_name in tasks_list:
                    for s in shots:
                        rows.append(
                            {
                                "model_path": model_path,
                                "task_path": task_name,
                                "n_shot": int(s),
                                "eval_suite": "lm_eval",
                            }
                        )

        # Handle task groups
        if task_groups:
            group_names = [g.strip() for g in task_groups.split(",") if g.strip()]
            # import pdb; pdb.set_trace()
            expanded = _expand_task_groups(group_names)
            for model_path in model_paths:
                for task_name, n_shots, suite_name in expanded:
                    for s in n_shots:
                        rows.append(
                            {
                                "model_path": model_path,
                                "task_path": task_name,
                                "n_shot": int(s),
                                "eval_suite": suite_name,
                            }
                        )

        df = pd.DataFrame(
            rows, columns=["model_path", "task_path", "n_shot", "eval_suite"]
        )
    else:
        raise ValueError(
            "Provide `eval_csv_path`, or `models` with (`tasks` and `n_shot`) and/or `task_groups`."
        )

    if df.empty:
        logging.warning("No evaluation jobs to schedule.")
        return None

    # Ensure that all datasets required by the tasks are cached locally to avoid
    # network access on compute nodes.
    if not skip_checks:
        lm_eval_tasks = df[
            df["eval_suite"].str.lower().isin({"lm_eval", "lm-eval", "lm-eval-harness"})
        ]["task_path"].unique()
        if len(lm_eval_tasks) > 0:
            _pre_download_task_datasets(
                lm_eval_tasks, trust_remote_code=trust_remote_code
            )
        # Pre-download LightEval datasets (best-effort, incremental support)
        light_eval_tasks = df[
            df["eval_suite"].str.lower().isin({"lighteval", "light-eval"})
        ]["task_path"].unique()
        if len(light_eval_tasks) > 0:
            _pre_download_lighteval_datasets(light_eval_tasks)
    else:
        logging.info("Skipping dataset pre-download (--skip-checks enabled)")

    if download_only:
        return None

    queue_limit = int(os.environ.get("QUEUE_LIMIT", 250))
    remaining_queue_capacity = queue_limit - _num_jobs_in_queue()

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

    logging.debug(f"Saved evaluation dataframe to temporary CSV: {csv_path}")

    sbatch_template = (files("oellm.resources") / "template.sbatch").read_text()

    # Calculate dynamic array size and time limits
    total_evals = len(df)

    # fixed timing estimation
    minutes_per_eval = 10  # Budget 10 minutes per eval
    total_minutes = total_evals * minutes_per_eval

    # Copy LightEval benchmark files into evaluation directory if necessary
    # TODO: why do we need this?
    light_eval_paths = df[df["eval_suite"].str.lower().isin({"lighteval", "light-eval"})][
        "task_path"
    ].unique()
    benchmark_dir = evals_dir / "light_eval_tasks"
    copied_paths: dict[str, str] = {}
    if light_eval_paths.size > 0:
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        for task_path in light_eval_paths:
            candidate = Path(task_path)
            if candidate.exists() and candidate.is_file():
                destination = benchmark_dir / candidate.name
                shutil.copy(candidate, destination)
                copied_paths[str(candidate)] = str(destination)

    if copied_paths:
        df.replace({"task_path": copied_paths}, inplace=True)

    # Maximum runtime per job (18 hours with safety margin)
    max_minutes_per_job = 18 * 60  # 18 hours
    min_array_size_for_time = max(1, int(np.ceil(total_minutes / max_minutes_per_job)))
    desired_array_size = min(128, total_evals) if total_evals >= 128 else total_evals
    if desired_array_size < min_array_size_for_time:
        desired_array_size = min_array_size_for_time

    # The actual array size is limited by queue capacity and total evals
    actual_array_size = min(remaining_queue_capacity, desired_array_size, total_evals)

    # Calculate actual time per job
    evals_per_job = max(1, int(np.ceil(total_evals / actual_array_size)))
    minutes_per_job = evals_per_job * minutes_per_eval

    # Add 20% safety margin and round up to nearest hour
    minutes_with_margin = int(minutes_per_job * 1.2)
    hours_with_margin = max(1, int(np.ceil(minutes_with_margin / 60)))

    # Apply 3-hour safety minimum for array jobs
    hours_with_margin = max(hours_with_margin, 3)

    # Cap at 24 hours
    hours_with_margin = min(hours_with_margin, 23)

    # Format time limit for SLURM (HH:MM:SS)
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

    # replace the placeholders in the template with the actual values
    # First, replace python-style placeholders
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

    # substitute any $ENV_VAR occurrences (e.g., $TIME_LIMIT) since env vars are not
    # expanded in the #SBATCH directives
    sbatch_script = Template(sbatch_script).safe_substitute(os.environ)

    # Save the sbatch script to the evals directory
    sbatch_script_path = evals_dir / "submit_evals.sbatch"
    logging.debug(f"Saving sbatch script to {sbatch_script_path}")

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


def build_csv(
    output_path: str = "eval_config.csv",
    *,
    verbose: bool = False,
) -> None:
    """
    Build a CSV file for evaluation with per-task n_shot configurations using the interactive builder.

    Args:
        output_path: Path where the CSV file will be saved.
        verbose: Enable verbose logging.
    """
    _setup_logging(verbose)

    from oellm.interactive_csv_builder import build_csv_interactive

    build_csv_interactive(output_path)


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

    _setup_logging(verbose)

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

        for task_name, task_results in results.items():
            # Skip MMLU subtasks - only keep the aggregate score
            if task_name.startswith("mmlu_") and task_name != "mmlu":
                continue

            # Get n_shot for this task
            n_shot = n_shot_data.get(task_name, "unknown")

            # Special handling for MMLU aggregate - get n_shot from any MMLU subtask
            if task_name == "mmlu" and n_shot == "unknown":
                for key, value in n_shot_data.items():
                    if key.startswith("mmlu_"):
                        n_shot = value
                        break

            # Get the primary metric (usually acc,none)
            performance = task_results.get("acc,none")
            if performance is None:
                # Try other common metric names
                for metric in ["acc", "accuracy", "f1", "exact_match"]:
                    if metric in task_results:
                        performance = task_results[metric]
                        break

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
            logging.info("\nSummary:")
            logging.info(f"Unique models: {df['model_name'].nunique()}")
            logging.info(f"Unique tasks: {df['task'].nunique()}")
            logging.info(
                f"N-shot values: {sorted(str(x) for x in df['n_shot'].unique())}"
            )

    # Perform check analysis if requested
    if check:
        logging.info("\n=== Evaluation Status Check ===")

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

        logging.info(f"\nTotal scheduled jobs: {len(jobs_df)}")
        logging.info(f"Completed jobs: {completed_count}")
        logging.info(f"Missing jobs: {len(missing_jobs)}")

        if len(missing_jobs) > 0:
            missing_df = pd.DataFrame(missing_jobs)
            missing_csv = output_csv.replace(".csv", "_missing.csv")
            missing_df.to_csv(missing_csv, index=False)
            logging.info(f"\nMissing jobs saved to: {missing_csv}")
            logging.info(
                f"You can run these with: oellm schedule-eval --eval_csv_path {missing_csv}"
            )

            # Show some examples if verbose
            if verbose and len(missing_jobs) > 0:
                logging.info("\nExample missing jobs:")
                for _i, (_, job) in enumerate(missing_df.head(5).iterrows()):
                    logging.info(
                        f"  - {job['model_path']} | {job['task_path']} | n_shot={job['n_shot']}"
                    )
                if len(missing_jobs) > 5:
                    logging.info(f"  ... and {len(missing_jobs) - 5} more")


def main():
    auto_cli(
        {
            "schedule-eval": schedule_evals,
            "build-csv": build_csv,
            "collect-results": collect_results,
            "clean-cache": lambda: clear_task_cache(),
        },
        as_positional=False,
        description="OELLM: Multi-cluster evaluation tool for language models",
    )

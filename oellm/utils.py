import builtins
import fnmatch
import logging
import os
import socket
import subprocess
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from importlib.resources import files
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler

from oellm.task_cache import (
    capture_hf_dataset_calls,
    dedupe_calls,
    prewarm_from_payload,
    task_cache_get_payload,
    task_cache_lookup,
    task_cache_mark_resolved,
    task_cache_set_payload,
)


def _ensure_singularity_image(image_name: str) -> None:
    from huggingface_hub import hf_hub_download

    image_path = Path(os.getenv("EVAL_BASE_DIR")) / image_name

    try:
        hf_hub_download(
            repo_id="openeurollm/evaluation_singularity_images",
            filename=image_name,
            repo_type="dataset",
            local_dir=os.getenv("EVAL_BASE_DIR"),
        )
        logging.info("Successfully downloaded latest Singularity image from HuggingFace")
    except Exception as e:
        logging.warning(
            "Failed to fetch latest container image from HuggingFace: %s", str(e)
        )
        if image_path.exists():
            logging.info("Using existing Singularity image at %s", image_path)
        else:
            raise RuntimeError(
                f"No container image found at {image_path} and failed to download from HuggingFace. "
                f"Cannot proceed with evaluation scheduling."
            ) from e

    logging.info(
        "Singularity image ready at %s",
        Path(os.getenv("EVAL_BASE_DIR")) / os.getenv("EVAL_CONTAINER_IMAGE"),
    )


def _setup_logging(verbose: bool = False):
    rich_handler = RichHandler(
        console=Console(),
        show_time=True,
        log_time_format="%H:%M:%S",
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    class RichFormatter(logging.Formatter):
        def format(self, record):
            record.msg = f"{record.getMessage()}"
            return record.msg

    rich_handler.setFormatter(RichFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _load_cluster_env() -> None:
    """
    Loads the correct cluster environment variables from `clusters.yaml` based on the hostname.
    """
    clusters = yaml.safe_load((files("oellm.resources") / "clusters.yaml").read_text())
    hostname = socket.gethostname()

    shared_cfg = clusters.get("shared", {}) or {}

    cluster_cfg_raw: dict | None = None
    for name, cfg in clusters.items():
        if name == "shared":
            continue
        pattern = cfg.get("hostname_pattern")
        if isinstance(pattern, str) and fnmatch.fnmatch(hostname, pattern):
            cluster_cfg_raw = dict(cfg)
            break
    if cluster_cfg_raw is None:
        raise ValueError(f"No cluster found for hostname: {hostname}")

    cluster_cfg_raw.pop("hostname_pattern", None)

    class _Default(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    base_ctx = _Default({**os.environ, **{k: str(v) for k, v in cluster_cfg_raw.items()}})

    resolved_shared = {k: str(v).format_map(base_ctx) for k, v in shared_cfg.items()}

    ctx = _Default({**base_ctx, **resolved_shared})

    resolved_cluster = {k: str(v).format_map(ctx) for k, v in cluster_cfg_raw.items()}

    final_env = {**resolved_shared, **resolved_cluster}
    for k, v in final_env.items():
        os.environ[k] = v


def _num_jobs_in_queue() -> int:
    user = os.environ.get("USER")
    cmd: list[str] = ["squeue"]
    if user:
        cmd += ["-u", user]
    cmd += ["-h", "-t", "pending,running", "-r", "-o", "%i"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            logging.warning(f"squeue error: {result.stderr.strip()}")
        return 0

    output = result.stdout.strip()
    if not output:
        return 0
    return sum(1 for line in output.splitlines() if line.strip())


def _expand_local_model_paths(model: str) -> list[Path]:
    """
    Expands a local model path to include all checkpoints if it's a directory.
    Recursively searches for models in subdirectories.

    Args:
        model: Path to a model or directory containing models

    Returns:
        List of paths to model directories containing safetensors files
    """
    model_paths = []
    model_path = Path(model)

    if not model_path.exists() or not model_path.is_dir():
        return model_paths

    if any(model_path.glob("*.safetensors")):
        model_paths.append(model_path)
        return model_paths

    hf_path = model_path / "hf"
    if hf_path.exists() and hf_path.is_dir():
        for subdir in hf_path.glob("*"):
            if subdir.is_dir() and any(subdir.glob("*.safetensors")):
                model_paths.append(subdir)
        if model_paths:
            return model_paths

    subdirs = [d for d in model_path.iterdir() if d.is_dir()]

    for subdir in subdirs:
        if any(subdir.glob("*.safetensors")):
            model_paths.append(subdir)
        else:
            hf_subpath = subdir / "hf"
            if hf_subpath.exists() and hf_subpath.is_dir():
                for checkpoint_dir in hf_subpath.glob("*"):
                    if checkpoint_dir.is_dir() and any(
                        checkpoint_dir.glob("*.safetensors")
                    ):
                        model_paths.append(checkpoint_dir)

    if len(model_paths) > 1:
        logging.info(f"Expanded '{model}' to {len(model_paths)} model checkpoints")

    return model_paths


def _process_model_paths(models: Iterable[str]) -> dict[str, list[Path | str]]:
    """
    Processes model strings into a dict of model paths.

    Each model string can be a local path or a huggingface model identifier.
    This function expands directory paths that contain multiple checkpoints.
    """
    from huggingface_hub import snapshot_download

    processed_model_paths: dict[str, list[Path | str]] = {}

    for model in models:
        per_model_paths: list[Path | str] = []

        local_paths = _expand_local_model_paths(model)
        if local_paths:
            per_model_paths.extend(local_paths)
        else:
            logging.info(
                f"Model {model} not found locally, assuming it is a 🤗 hub model"
            )
            logging.debug(
                f"Downloading model {model} on the login node since the compute nodes may not have access to the internet"
            )

            if "," in model:
                model_kwargs = dict(
                    [kv.split("=") for kv in model.split(",") if "=" in kv]
                )

                repo_id = model.split(",")[0]

                snapshot_kwargs = {}
                if "revision" in model_kwargs:
                    snapshot_kwargs["revision"] = model_kwargs["revision"]

                try:
                    snapshot_download(
                        repo_id=repo_id,
                        cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                        **snapshot_kwargs,
                    )
                    per_model_paths.append(model)
                except Exception as e:
                    logging.debug(
                        f"Failed to download model {model} from Hugging Face Hub. Continuing..."
                    )
                    logging.debug(e)
            else:
                snapshot_download(
                    repo_id=model,
                    cache_dir=Path(os.getenv("HF_HOME")) / "hub",
                )
                per_model_paths.append(model)

        if not per_model_paths:
            logging.warning(
                f"Could not find any valid model for '{model}'. It will be skipped."
            )
        processed_model_paths[model] = per_model_paths

    return processed_model_paths


def _pre_download_task_datasets(
    tasks: Iterable[str], trust_remote_code: bool = True
) -> None:
    processed: set[str] = set()

    misses: list[str] = []
    for task_name in tasks:
        if not isinstance(task_name, str) or task_name in processed:
            continue
        processed.add(task_name)
        if task_cache_lookup("lm-eval", task_name):
            logging.info(
                f"Skipping dataset preparation for task '{task_name}' (cache hit within TTL)."
            )
            continue
        misses.append(task_name)

    if not misses:
        for task_name in processed:
            if task_cache_lookup("lm-eval", task_name):
                prewarm_from_payload(
                    task_cache_get_payload("lm-eval", task_name),
                    trust_remote_code=trust_remote_code,
                )
        return

    from datasets import DownloadMode  # type: ignore
    from lm_eval.tasks import TaskManager  # type: ignore

    tm = TaskManager()

    for task_name in misses:
        logging.info(
            f"Preparing dataset for task '{task_name}' (download if not cached)…"
        )

        task_config = {
            "task": task_name,
            "dataset_kwargs": {"trust_remote_code": trust_remote_code},
        }

        with capture_hf_dataset_calls() as captured_calls:
            task_objects = tm.load_config(task_config)

            stack = [task_objects]
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    stack.extend(current.values())
                    continue
                if hasattr(current, "download") and callable(current.download):
                    try:
                        current.download(
                            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
                        )  # type: ignore[arg-type]
                    except TypeError as e:
                        logging.error(
                            f"Failed to download dataset for task '{task_name}' with download_mode=REUSE_DATASET_IF_EXISTS: {e}"
                        )
                        current.download()  # type: ignore[misc]

        if captured_calls:
            payload = {"calls": dedupe_calls(captured_calls)}
            task_cache_set_payload("lm-eval", task_name, payload)
        task_cache_mark_resolved("lm-eval", task_name)
        logging.debug(f"Finished dataset preparation for task '{task_name}'.")


def _pre_download_lighteval_datasets(tasks: Iterable[str]) -> None:
    misses: list[str] = []
    processed: set[str] = set()
    for t in tasks:
        raw = str(t).strip()
        if not raw or raw in processed:
            continue
        processed.add(raw)
        if task_cache_lookup("lighteval", raw):
            logging.info(
                f"Skipping dataset preparation for LightEval task '{raw}' (cache hit within TTL)."
            )
            continue
        misses.append(raw)

    if not misses:
        for raw in processed:
            if task_cache_lookup("lighteval", raw):
                prewarm_from_payload(
                    task_cache_get_payload("lighteval", raw),
                    trust_remote_code=True,
                )
        return

    from lighteval.tasks.lighteval_task import LightevalTask  # type: ignore
    from lighteval.tasks.registry import (  # type: ignore
        TRUNCATE_FEW_SHOTS_DEFAULTS,
        Registry,
    )

    for raw in misses:
        candidate = Path(raw)
        if candidate.exists() and candidate.is_file():
            with capture_hf_dataset_calls() as captured_calls:
                reg_file = Registry()
                configs_file = reg_file.get_tasks_configs(str(candidate))
                task_dict_file = reg_file.get_tasks_from_configs(configs_file)
                LightevalTask.load_datasets(task_dict_file)
            if captured_calls:
                payload = {"calls": dedupe_calls(captured_calls)}
                task_cache_set_payload("lighteval", raw, payload)
            task_cache_mark_resolved("lighteval", raw)
            continue

        # Build single-spec string and load in isolation
        spec = raw
        truncate_default = int(TRUNCATE_FEW_SHOTS_DEFAULTS)
        if "|" not in spec:
            spec = f"lighteval|{spec}|0|{truncate_default}"
        elif spec.count("|") == 1:
            spec = f"{spec}|0|{truncate_default}"
        elif spec.count("|") == 2:
            spec = f"{spec}|{truncate_default}"

        with capture_hf_dataset_calls() as captured_calls:
            reg = Registry(custom_tasks="lighteval.tasks.multilingual.tasks")
            configs = reg.get_tasks_configs(spec)
            task_dict = reg.get_tasks_from_configs(configs)
            LightevalTask.load_datasets(task_dict)
        if captured_calls:
            payload = {"calls": dedupe_calls(captured_calls)}
            task_cache_set_payload("lighteval", raw, payload)
        task_cache_mark_resolved("lighteval", raw)


@contextmanager
def capture_third_party_output(verbose: bool = False):
    """
    Suppresses print/logging.info/logging.debug originating from non-project modules
    unless verbose=True.

    A call is considered "third-party" if its immediate caller's file path is not
    under the repository root (parent of the `oellm` package directory).
    """
    if verbose:
        yield
        return

    package_root = Path(__file__).resolve().parent

    def is_internal_stack(skip: int = 2, max_depth: int = 12) -> bool:
        f = sys._getframe(skip)
        depth = 0
        while f and depth < max_depth:
            filename = f.f_code.co_filename if f.f_code else ""
            if filename:
                p = Path(filename).resolve()
                if p.is_relative_to(package_root):
                    return True
            f = f.f_back
            depth += 1
        return False

    orig_print = builtins.print
    orig_logger_info = logging.Logger.info
    orig_logger_debug = logging.Logger.debug
    orig_module_info = logging.info
    orig_module_debug = logging.debug

    def filtered_print(*args, **kwargs):
        if is_internal_stack():
            return orig_print(*args, **kwargs)
        # third-party: drop
        return None

    def filtered_logger_info(self, msg, *args, **kwargs):
        if is_internal_stack():
            return orig_logger_info(self, msg, *args, **kwargs)
        return None

    def filtered_logger_debug(self, msg, *args, **kwargs):
        if is_internal_stack():
            return orig_logger_debug(self, msg, *args, **kwargs)
        return None

    def filtered_module_info(msg, *args, **kwargs):
        if is_internal_stack():
            return orig_module_info(msg, *args, **kwargs)
        return None

    def filtered_module_debug(msg, *args, **kwargs):
        if is_internal_stack():
            return orig_module_debug(msg, *args, **kwargs)
        return None

    builtins.print = filtered_print
    logging.Logger.info = filtered_logger_info  # type: ignore[assignment]
    logging.Logger.debug = filtered_logger_debug  # type: ignore[assignment]
    logging.info = filtered_module_info  # type: ignore[assignment]
    logging.debug = filtered_module_debug  # type: ignore[assignment]

    try:
        yield
    finally:
        builtins.print = orig_print
        logging.Logger.info = orig_logger_info  # type: ignore[assignment]
        logging.Logger.debug = orig_logger_debug  # type: ignore[assignment]
        logging.info = orig_module_info  # type: ignore[assignment]
        logging.debug = orig_module_debug  # type: ignore[assignment]


def capture_third_party_output_from_kwarg(
    verbose_kwarg: str = "verbose", default: bool = False
):
    """
    Decorator factory that wraps the function execution inside
    capture_third_party_output(verbose=kwargs.get(verbose_kwarg, default)).
    """

    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            verbose_value = bool(kwargs.get(verbose_kwarg, default))
            with capture_third_party_output(verbose=verbose_value):
                return func(*args, **kwargs)

        return _wrapper

    return _decorator

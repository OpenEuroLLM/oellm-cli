"""Pytest configuration and fixtures for integration tests."""

import os
import subprocess

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (full SLURM execution)")
    config.addinivalue_line(
        "markers", "dry_run: marks tests that only test script generation"
    )


@pytest.fixture(scope="module")
def eval_base_dir(tmp_path_factory):
    """Create a temporary evaluation directory for the test module.

    If EVAL_BASE_DIR is set in the environment, use that (e.g., for NVMe storage in CI).
    Otherwise, create a pytest temp directory.
    """
    from pathlib import Path

    env_base_dir = os.environ.get("EVAL_BASE_DIR")
    if env_base_dir:
        base_dir = Path(env_base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = tmp_path_factory.mktemp("oellm-test")

    hf_home = base_dir / "hf_data"
    hf_home.mkdir(exist_ok=True)
    (hf_home / "hub").mkdir(exist_ok=True)
    (hf_home / "datasets").mkdir(exist_ok=True)
    user = os.environ.get("USER", "runner")
    (base_dir / user).mkdir(exist_ok=True)
    return base_dir


@pytest.fixture
def slurm_env(eval_base_dir, request):
    """Set up environment variables for oellm schedule-eval.

    Use with @pytest.mark.parametrize("slurm_env", [True], indirect=True) for dry_run=True
    or @pytest.mark.parametrize("slurm_env", [False], indirect=True) for dry_run=False.
    Default is dry_run=False.
    """
    dry_run = getattr(request, "param", False)

    hf_home = eval_base_dir / "hf_data"
    user = os.environ.get("USER", "runner")

    old_env = {}
    env_vars = {
        "EVAL_BASE_DIR": str(eval_base_dir),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "PARTITION": "gpu",
        "ACCOUNT": "test",
        "QUEUE_LIMIT": "10",
        "EVAL_CONTAINER_IMAGE": "eval_env-slurm-ci.sif",
        "EVAL_OUTPUT_DIR": str(eval_base_dir / user),
        "SINGULARITY_ARGS": "" if dry_run else "--nv",
        "GPUS_PER_NODE": "0" if dry_run else "1",
    }

    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield eval_base_dir

    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def slurm_available():
    """Check if SLURM is available, skip if not."""
    result = subprocess.run(["sinfo"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("SLURM is not available on this system")
    return True

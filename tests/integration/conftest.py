import os
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--use-venv",
        action="store_true",
        default=False,
        help="Run SLURM jobs using a venv instead of container",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (full SLURM execution)")
    config.addinivalue_line(
        "markers", "dry_run: marks tests that only test script generation"
    )


@pytest.fixture(scope="module")
def eval_base_dir(tmp_path_factory):
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
def slurm_env(eval_base_dir, request, use_venv):
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
        "EVAL_OUTPUT_DIR": str(eval_base_dir / user),
        "GPUS_PER_NODE": "0" if dry_run else "1",
    }

    if not use_venv:
        env_vars["EVAL_CONTAINER_IMAGE"] = "eval_env-slurm-ci.sif"
        env_vars["SINGULARITY_ARGS"] = "" if dry_run else "--nv"
    else:
        env_vars["SINGULARITY_ARGS"] = ""

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
    result = subprocess.run(["sinfo"], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("SLURM is not available on this system")
    return True


@pytest.fixture(scope="session")
def use_venv(request):
    return request.config.getoption("--use-venv")


@pytest.fixture(scope="module")
def venv_path(eval_base_dir, use_venv):
    if use_venv:
        return str(eval_base_dir / ".venv")
    return None

import os
import sys
from importlib.resources import files

import pytest
import yaml
from datasets import get_dataset_config_names
from huggingface_hub import HfApi

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

from oellm.task_groups import DatasetSpec, TaskGroup, _parse_task_groups


def collect_all_dataset_specs() -> list[DatasetSpec]:
    """Collect all unique dataset specs from all task groups."""
    data = (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )

    all_group_names = list(data.get("task_groups", {}).keys()) + list(
        data.get("super_groups", {}).keys()
    )
    parsed = _parse_task_groups(all_group_names)

    specs: list[DatasetSpec] = []
    seen: set[tuple[str, str | None]] = set()

    def add_spec(dataset: str | None, subset: str | None):
        if dataset is None:
            return
        key = (dataset, subset)
        if key not in seen:
            seen.add(key)
            specs.append(DatasetSpec(repo_id=dataset, subset=subset))

    for _, group in parsed.items():
        if isinstance(group, TaskGroup):
            for t in group.tasks:
                add_spec(t.dataset, t.subset)
        else:
            for g in group.task_groups:
                for t in g.tasks:
                    add_spec(t.dataset, t.subset)

    return specs


def check_dataset_exists(spec: DatasetSpec) -> tuple[bool, str]:
    """Check if a dataset exists on HuggingFace."""
    label = f"{spec.repo_id}" + (f"/{spec.subset}" if spec.subset else "")
    api = HfApi()

    info = api.dataset_info(spec.repo_id)
    if info is None:
        return False, f"Dataset repo '{spec.repo_id}' not found on HuggingFace"

    if spec.subset:
        configs = get_dataset_config_names(spec.repo_id)
        if spec.subset not in configs:
            return (
                False,
                f"Subset '{spec.subset}' not found in {spec.repo_id}. Available: {configs[:10]}{'...' if len(configs) > 10 else ''}",
            )

    return True, f"OK: {label}"


ALL_SPECS = collect_all_dataset_specs()


@pytest.mark.parametrize(
    "spec",
    ALL_SPECS,
    ids=[f"{s.repo_id}/{s.subset}" if s.subset else s.repo_id for s in ALL_SPECS],
)
def test_dataset_exists(spec: DatasetSpec):
    """Test that each dataset specified in task-groups.yaml exists on HuggingFace."""
    success, message = check_dataset_exists(spec)
    assert success, message


def main():
    print("Collecting dataset specs from task-groups.yaml...")
    specs = collect_all_dataset_specs()
    print(f"Found {len(specs)} unique dataset specs\n")

    failed = []
    passed = []

    for spec in specs:
        label = f"{spec.repo_id}" + (f"/{spec.subset}" if spec.subset else "")
        print(f"Checking {label}... ", end="", flush=True)

        success, message = check_dataset_exists(spec)
        if success:
            print("✓")
            passed.append(spec)
        else:
            print(f"✗ - {message}")
            failed.append((spec, message))

    print(f"\n{'=' * 60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")

    if failed:
        print("\nFailed datasets:")
        for spec, msg in failed:
            print(f"  - {spec.repo_id}/{spec.subset}: {msg}")
        sys.exit(1)
    else:
        print("\nAll datasets verified successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""Utilities for loading and resolving task group definitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import yaml

DEFAULT_TASK_GROUPS_PATH = Path(__file__).parent / "task-groups.yaml"


def load_task_groups(path: str | Path | None = None) -> dict[str, dict]:
    """Load task group definitions from a YAML file."""
    groups_path = Path(path) if path is not None else DEFAULT_TASK_GROUPS_PATH
    if not groups_path.exists():
        raise FileNotFoundError(
            f"Task groups file not found at {groups_path}. Please create it or "
            "provide a custom path."
        )

    with groups_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    task_groups = data.get("task_groups", {})
    if not isinstance(task_groups, dict):
        raise ValueError(
            "task_groups.yaml is malformed. Expected 'task_groups' to be a mapping."
        )

    return task_groups


def resolve_task_group(
    group_name: str,
    task_groups: dict[str, dict],
    console=None,
    _chain: Sequence[str] | None = None,
) -> list[dict]:
    """Resolve a task group into its concrete task definitions."""

    if _chain is None:
        _chain = []

    if group_name not in task_groups:
        raise ValueError(
            f"Task group '{group_name}' is not defined in task-groups.yaml."
        )

    if group_name in _chain:
        cycle = " -> ".join(list(_chain) + [group_name])
        raise ValueError(f"Circular task group reference detected: {cycle}")

    group_data = task_groups.get(group_name) or {}
    chain = list(_chain) + [group_name]
    resolved_tasks: list[dict] = []

    subgroups = group_data.get("groups", [])
    if subgroups:
        if not isinstance(subgroups, list):
            raise ValueError(
                f"Task group '{group_name}' has an invalid 'groups' section; expected a list of group names."
            )
        for subgroup in subgroups:
            if not isinstance(subgroup, str):
                raise ValueError(
                    f"Task group '{group_name}' references an invalid subgroup entry: {subgroup!r}"
                )
            resolved_tasks.extend(
                resolve_task_group(subgroup, task_groups, console=console, _chain=chain)
            )

    for task_item in group_data.get("tasks", []) or []:
        if "task" not in task_item:
            message = (
                f"Skipping malformed task entry in group '{group_name}': {task_item}"
            )
            if console is not None:
                console.print(f"[yellow]{message}[/yellow]")
            else:
                logging.warning(message)
            continue
        resolved_tasks.append(
            {
                "task": task_item["task"],
                "n_shots": list(task_item.get("n_shots", [0])),
            }
        )

    return resolved_tasks


def flatten_task_groups(
    group_names: Iterable[str],
    task_groups: dict[str, dict],
    *,
    console=None,
    on_duplicate=None,
) -> list[tuple[str, int]]:
    """Flatten multiple task groups into (task, n_shot) pairs without duplicates."""

    seen: set[tuple[str, int]] = set()
    flattened: list[tuple[str, int]] = []

    for group_name in group_names:
        resolved = resolve_task_group(group_name, task_groups, console=console)
        for entry in resolved:
            task_name = entry["task"]
            for n_shot in entry.get("n_shots", [0]):
                pair = (task_name, int(n_shot))
                if pair in seen:
                    if on_duplicate is not None:
                        on_duplicate(group_name, pair)
                    continue
                seen.add(pair)
                flattened.append(pair)

    return flattened


__all__ = ["load_task_groups", "resolve_task_group", "flatten_task_groups"]

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files

import yaml


@dataclass
class _Task:
    name: str
    n_shots: list[int] | None = None


@dataclass
class TaskGroup:
    name: str
    tasks: list[_Task]
    suite: str
    description: str
    n_shots: list[int] | None = None

    def __post_init__(self):
        for task in self.tasks:
            if task.n_shots is None and self.n_shots is not None:
                task.n_shots = self.n_shots
            elif task.n_shots is None and self.n_shots is None:
                raise ValueError(
                    f"N_shots is not set for task {task.name} and no default n_shots is set for the task group: {self.name}"
                )

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "TaskGroup":
        tasks = []
        for task_data in data["tasks"]:
            task_name = task_data["task"]
            task_n_shots = task_data.get("n_shots")
            tasks.append(_Task(name=task_name, n_shots=task_n_shots))

        return cls(
            name=name,
            tasks=tasks,
            suite=data["suite"],
            description=data["description"],
            n_shots=data.get("n_shots"),
        )


@dataclass
class TaskSuperGroup:
    name: str
    task_groups: list[TaskGroup]
    description: str

    def __post_init__(self):
        resolved_groups = []
        for group in self.task_groups:
            if isinstance(group, str):
                raise ValueError(
                    f"Task group '{group}' not found in available task groups"
                )
            resolved_groups.append(group)
        self.task_groups = resolved_groups

    @classmethod
    def from_dict(
        cls, name: str, data: dict, available_task_groups: dict[str, TaskGroup]
    ) -> "TaskSuperGroup":
        task_groups = []
        for task_group_data in data["task_groups"]:
            group_name = task_group_data["task"]
            if group_name not in available_task_groups:
                raise ValueError(
                    f"Task group '{group_name}' not found in available task groups"
                )
            task_groups.append(available_task_groups[group_name])

        return cls(
            name=name,
            task_groups=task_groups,
            description=data["description"],
        )


def _parse_task_groups(
    requested_groups: list[str],
) -> dict[str, TaskSuperGroup | TaskGroup]:
    data = (
        yaml.safe_load((files("oellm.resources") / "task-groups.yaml").read_text()) or {}
    )

    task_groups: dict[str, TaskGroup] = {}

    for task_group_name, task_data in data["task_groups"].items():
        task_groups[task_group_name] = TaskGroup.from_dict(task_group_name, task_data)

    super_groups: dict[str, TaskSuperGroup] = {}
    for super_group_name, super_group_data in data.get("super_groups", {}).items():
        super_groups[super_group_name] = TaskSuperGroup.from_dict(
            super_group_name, super_group_data, task_groups
        )

    result = {**task_groups, **super_groups}
    return {
        group_name: group
        for group_name, group in result.items()
        if group_name in requested_groups
    }


def _expand_task_groups(group_names: Iterable[str]) -> list[tuple[str, list[int], str]]:
    parsed = _parse_task_groups([str(n).strip() for n in group_names if str(n).strip()])
    missing = {str(n).strip() for n in group_names if str(n).strip()} - set(parsed.keys())
    if missing:
        raise ValueError(f"Unknown task group(s): {', '.join(sorted(missing))}")

    results: list[tuple[str, list[int], str]] = []

    for _, group in parsed.items():
        if isinstance(group, TaskGroup):
            suite = group.suite
            for t in group.tasks:
                shots = [int(s) for s in (t.n_shots or [])]
                results.append((t.name, shots, suite))
        else:
            for g in group.task_groups:
                suite = g.suite
                for t in g.tasks:
                    shots = [int(s) for s in (t.n_shots or [])]
                    results.append((t.name, shots, suite))

    return results

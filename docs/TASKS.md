# Adding Tasks and Task Groups

## Overview

Tasks are defined in `oellm/resources/task-groups.yaml`. Only tasks in this file are tested and guaranteed to work. The CLI parses this via `task_groups.py` and expands groups into `(task, n_shot, suite)` tuples for scheduling.

Three evaluation suites are supported:

| Suite value | Engine | Use case |
|---|---|---|
| `lm_eval` | [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) | Text benchmarks |
| `lighteval` | [lighteval](https://github.com/huggingface/lighteval) | Translation / multilingual |
| `lmms_eval` | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | Image / VQA benchmarks |

## YAML Structure

```yaml
task_groups:
  my-group:
    description: "Short description"
    suite: lm_eval        # or lighteval, lmms_eval
    n_shots: [5]          # default for all tasks in group
    dataset: org/dataset  # default HF dataset for pre-download
    tasks:
      - task: task_name
        n_shots: [0, 5]     # overrides group default
        dataset: org/other  # overrides group default
        subset: subset_name # HF dataset config/subset
```

## Adding a Text Task Group

1. Add your group to `oellm/resources/task-groups.yaml`:

```yaml
task_groups:
  my-benchmark:
    description: "My custom benchmark"
    suite: lm_eval
    n_shots: [0]
    dataset: huggingface/dataset-name
    tasks:
      - task: task_one
        subset: split_a
      - task: task_two
        subset: split_b
```

2. Use it:

```bash
oellm schedule-eval --models "model-name" --task_groups "my-benchmark"
```

## Adding an Image Task Group

Image tasks use `suite: lmms_eval`. Each task must correspond to a task name recognized by lmms-eval.

```yaml
task_groups:
  my-image-benchmark:
    description: "My image benchmark via lmms-eval"
    suite: lmms_eval
    n_shots: [0]
    tasks:
      - task: vqav2_val_all
        dataset: HuggingFaceM4/VQAv2
      - task: mmbench_en_dev
        dataset: HuggingFaceM4/MMBench_00
```

Run with:

```bash
oellm schedule-eval --models "path/to/vlm" --task_groups "my-image-benchmark"
```

The model adapter defaults to `llava_hf`. Override via `--slurm_template_var`:

```bash
oellm schedule-eval --models "path/to/vlm" --task_groups "my-image-benchmark" \
  --slurm_template_var '{"LMMS_MODEL_TYPE":"qwen_vl_chat"}'
```

## Field Reference

| Field | Required | Level | Description |
|-------|----------|-------|-------------|
| `description` | Yes | group | Short description of the task group |
| `suite` | Yes | group | Evaluation suite: `lm_eval`, `lighteval`, or `lmms_eval` |
| `n_shots` | Yes | group or task | List of shot counts; must be set at group or task level |
| `dataset` | Yes | group or task | HuggingFace dataset repo ID (required for pre-download and testing) |
| `task` | Yes | task | Task name as recognized by the evaluation suite |
| `subset` | No | task | HuggingFace dataset config/subset name |

## Important: Dataset Requirement

**You must provide the `dataset` field** (at group or task level) for:
1. **Automatic pre-download** - Compute nodes often lack network access; datasets are cached beforehand
2. **CI testing** - The test suite validates that all datasets in `task-groups.yaml` are accessible

Tasks without a `dataset` field will not have their data pre-downloaded and are not covered by CI validation.

## Plugin Interface (Advanced)

For programmatic task registration without editing the YAML, use the `BaseTask` abstract base class from `oellm.core`:

```python
from oellm.core import BaseTask
from oellm.task_groups import DatasetSpec

class MyImageTask(BaseTask):
    @property
    def name(self) -> str:
        return "my_benchmark"        # canonical name used in CSV scheduling

    @property
    def suite(self) -> str:
        return "lmms_eval"           # or "lm_eval" / "lighteval"

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        return [DatasetSpec(repo_id="org/my-dataset")]
```

Override `engine_task_name` if the engine uses a different name than `name`:

```python
    @property
    def engine_task_name(self) -> str:
        return "my_benchmark_v2"     # passed to --tasks; defaults to self.name
```

See `oellm/core/` for the full `BaseTask`, `BaseMetric`, and `BaseModelAdapter` interfaces.

# ELLIOT Evaluation Platform

A multimodal evaluation framework for scheduling LLM and VLM evaluations across HPC clusters. Extends the original oellm-cli with image modality support and a plugin interface for adding new benchmarks and modalities.

## Features

- **Schedule evaluations** on multiple models and tasks: `oellm schedule-eval`
- **Collect results** and check for missing evaluations: `oellm collect-results`
- **Task groups** for pre-defined evaluation suites with automatic dataset pre-downloading
- **Multi-cluster support** with auto-detection (Leonardo, LUMI, JURECA)
- **Image evaluation** via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (VQAv2, MMBench, MMMU, ChartQA, DocVQA, TextVQA, OCRBench, MathVista)
- **Plugin interface** (`BaseTask` / `BaseMetric` / `BaseModelAdapter`) for adding new benchmarks without touching core scheduling logic
- **Automatic building and deployment of containers**

## Quick Start

**Prerequisites:**
- Install [uv](https://docs.astral.sh/uv/#installation)
- Set the `HF_HOME` environment variable to point to your HuggingFace cache directory (e.g. `export HF_HOME="/path/to/your/hf_home"`). This is where models and datasets will be cached. Compute nodes typically have no internet access, so all assets must be pre-downloaded into this directory.

```bash
# Install the package
uv tool install -p 3.12 git+https://github.com/OpenEuroLLM/oellm-cli.git

# Run evaluations using a task group (recommended)
oellm schedule-eval \
    --models "microsoft/DialoGPT-medium,EleutherAI/pythia-160m" \
    --task_groups "open-sci-0.01"

# Or specify individual tasks
oellm schedule-eval \
    --models "EleutherAI/pythia-160m" \
    --tasks "hellaswag,mmlu" \
    --n_shot 5
```

This will automatically:
- Detect your current HPC cluster (Leonardo, LUMI, or JURECA)
- Download and cache the specified models
- Pre-download datasets for known tasks (see warning below)
- Generate and submit a SLURM job array with appropriate cluster-specific resources and using containers built for this cluster

In case you do not want to rely on the containers provided on a given cluster or try out specific package versions, you can use a custom environment by passing `--venv_path`, see [docs/VENV.md](docs/VENV.md).

## Task Groups

Task groups are pre-defined evaluation suites in [`task-groups.yaml`](oellm/resources/task-groups.yaml). Each group specifies tasks, their n-shot settings, and HuggingFace dataset mappings.

### Text & Multilingual

| Group | Description | Engine |
|---|---|---|
| `open-sci-0.01` | COPA, MMLU, HellaSwag, ARC, etc. | lm-eval |
| `belebele-eu-5-shot` | Belebele in 23 European languages | lm-eval |
| `flores-200-eu-to-eng` | EU → English translation | lighteval |
| `flores-200-eng-to-eu` | English → EU translation | lighteval |
| `global-mmlu-eu` | Global MMLU in EU languages | lm-eval |
| `mgsm-eu` | Multilingual GSM8K | lm-eval |
| `generic-multilingual` | XWinograd, XCOPA, XStoryCloze | lm-eval |
| `include` | INCLUDE benchmarks (44 languages) | lm-eval |

Super groups:
- `oellm-multilingual` — all multilingual benchmarks combined

### Image (lmms-eval)

| Group | Benchmarks | Engine |
|---|---|---|
| `image-vqa` | VQAv2, MMBench, MMMU, ChartQA, DocVQA, TextVQA, OCRBench, MathVista | lmms-eval |

Image evaluation requires a container or venv with `lmms-eval` installed. Install the optional dependency:

```bash
pip install "oellm[image]"
```

By default, the `llava_hf` model adapter is used (suitable for most HuggingFace VLMs). Override via `--slurm_template_var`:

```bash
oellm schedule-eval --models "path/to/vlm" --task_groups "image-vqa" \
  --slurm_template_var '{"LMMS_MODEL_TYPE":"qwen_vl_chat"}'
```

```bash
# Use a task group
oellm schedule-eval --models "model-name" --task_groups "open-sci-0.01"

# Use multiple task groups
oellm schedule-eval --models "model-name" --task_groups "belebele-eu-5-shot,global-mmlu-eu"

# Use a super group
oellm schedule-eval --models "model-name" --task_groups "oellm-multilingual"

# Image evaluation
oellm schedule-eval --models "path/to/vlm" --task_groups "image-vqa"
```

## SLURM Overrides

Override cluster defaults (partition, account, time limit, etc.) with `--slurm_template_var` (JSON object):

```bash
# Use a different partition (e.g. dev-g on LUMI when small-g is crowded)
oellm schedule-eval --models "model-name" --task_groups "open-sci-0.01" \
  --slurm_template_var '{"PARTITION":"dev-g"}'

# Multiple overrides: partition, account, time limit, GPUs
oellm schedule-eval --models "model-name" --task_groups "open-sci-0.01" \
  --slurm_template_var '{"PARTITION":"dev-g","ACCOUNT":"myproject","TIME":"02:00:00","GPUS_PER_NODE":2}'
```

Use exact env var names: `PARTITION`, `ACCOUNT`, `GPUS_PER_NODE`. `TIME` (HH:MM:SS) overrides the time limit.

## ⚠️ Dataset Pre-Download Warning

**Datasets are only automatically pre-downloaded for tasks defined in [`task-groups.yaml`](oellm/resources/task-groups.yaml).**

If you use custom tasks via `--tasks` that are not in the task groups registry, the CLI will attempt to look them up but **cannot guarantee the datasets will be cached**. This may cause failures on compute nodes that don't have network access.

**Recommendation:** Use `--task_groups` when possible, or ensure your custom task datasets are already cached in `$HF_HOME` before scheduling.

## Collecting Results

After evaluations complete, collect results into a CSV:

```bash
# Basic collection
oellm collect-results /path/to/eval-output-dir

# Check for missing evaluations and create a CSV for re-running them
oellm collect-results /path/to/eval-output-dir --check --output_csv results.csv
```

The `--check` flag compares completed results against `jobs.csv` and outputs a `results_missing.csv` that can be used to re-schedule failed jobs:

```bash
oellm schedule-eval --eval_csv_path results_missing.csv
```

## CSV-Based Scheduling

For full control, provide a CSV file with columns: `model_path`, `task_path`, `n_shot`, and optionally `eval_suite`:

```bash
oellm schedule-eval --eval_csv_path custom_evals.csv
```

## Installation

### General Installation

```bash
uv tool install -p 3.12 git+https://github.com/OpenEuroLLM/oellm-cli.git
```

Update to latest:
```bash
uv tool upgrade oellm
```

### JURECA/JSC Specifics

Due to limited space in `$HOME` on JSC clusters, set these environment variables:

```bash
export UV_CACHE_DIR="/p/project1/<project>/$USER/.cache/uv-cache"
export UV_INSTALL_DIR="/p/project1/<project>/$USER/.local"
export UV_PYTHON_INSTALL_DIR="/p/project1/<project>/$USER/.local/share/uv/python"
export UV_TOOL_DIR="/p/project1/<project>/$USER/.cache/uv-tool-cache"
```

## Supported Clusters:
We support: Leonardo, Lumi, and Jureca

## CLI Options

```bash
oellm schedule-eval --help
```

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/OpenEuroLLM/oellm-cli.git
cd oellm-cli
uv sync --extra dev

# Run all unit tests
uv run pytest tests/ -v

# Run dataset validation tests (requires network access)
uv run pytest tests/test_datasets.py -v

# Download-only mode for testing
uv run oellm schedule-eval --models "EleutherAI/pythia-160m" --task_groups "open-sci-0.01" --download_only
```

## Plugin Interface

The `oellm.core` package provides abstract base classes for extending the platform without modifying core scheduling logic:

```python
from oellm.core import BaseTask, BaseMetric, BaseModelAdapter
from oellm.task_groups import DatasetSpec

# Register a new benchmark (one-liner if it's already in lmms-eval)
class MyTask(BaseTask):
    @property
    def name(self) -> str:
        return "my_benchmark"

    @property
    def suite(self) -> str:
        return "lmms_eval"  # or "lm_eval" / "lighteval"

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        return [DatasetSpec(repo_id="org/my-dataset")]
```

See `oellm/core/` for full interface documentation.

## Deploying containers

Containers are deployed manually since [PR #46](https://github.com/OpenEuroLLM/oellm-cli/pull/46) to save costs.

To build and deploy them, select run workflow in [Actions](https://github.com/OpenEuroLLM/oellm-cli/actions/workflows/build-and-push-apptainer.yml).


## Troubleshooting

**HuggingFace quota issues**: Ensure you're logged in with `HF_TOKEN` and are part of the [OpenEuroLLM](https://huggingface.co/OpenEuroLLM) organization.

**Dataset download failures on compute nodes**: Use `--task_groups` for automatic dataset caching, or pre-download datasets manually before scheduling.

# OpenEuroLLM CLI (oellm)

A lightweight CLI for scheduling LLM evaluations across multiple HPC clusters using SLURM job arrays and Singularity containers.

## Features

- **Schedule evaluations** on multiple models and tasks: `oellm schedule-eval`
- **Collect results** and check for missing evaluations: `oellm collect-results`
- **Task groups** for pre-defined evaluation suites with automatic dataset pre-downloading
- **Multi-cluster support** with auto-detection (Leonardo, LUMI, JURECA)

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
- Generate and submit a SLURM job array with appropriate cluster-specific resources

## Task Groups

Task groups are pre-defined evaluation suites in [`task-groups.yaml`](oellm/resources/task-groups.yaml). Each group specifies tasks, their n-shot settings, and HuggingFace dataset mappings.

Available task groups:
- `open-sci-0.01` - Standard benchmarks (COPA, MMLU, HellaSwag, ARC, etc.)
- `belebele-eu-5-shot` - Belebele European language tasks
- `flores-200-eu-to-eng` / `flores-200-eng-to-eu` - Translation tasks
- `global-mmlu-eu` - Global MMLU in EU languages
- `mgsm-eu` - Multilingual GSM benchmarks
- `generic-multilingual` - XWinograd, XCOPA, XStoryCloze
- `include` - INCLUDE benchmarks

Super groups combine multiple task groups:
- `oellm-multilingual` - All multilingual benchmarks combined

```bash
# Use a task group
oellm schedule-eval --models "model-name" --task_groups "open-sci-0.01"

# Use multiple task groups
oellm schedule-eval --models "model-name" --task_groups "belebele-eu-5-shot,global-mmlu-eu"

# Use a super group
oellm schedule-eval --models "model-name" --task_groups "oellm-multilingual"
```

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

# Run dataset validation tests
uv run pytest tests/test_datasets.py -v

# Download-only mode for testing
uv run oellm schedule-eval --models "EleutherAI/pythia-160m" --task_groups "open-sci-0.01" --download_only
```

## Troubleshooting

**HuggingFace quota issues**: Ensure you're logged in with `HF_TOKEN` and are part of the [OpenEuroLLM](https://huggingface.co/OpenEuroLLM) organization.

**Dataset download failures on compute nodes**: Use `--task_groups` for automatic dataset caching, or pre-download datasets manually before scheduling.

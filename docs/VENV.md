# Using Your Own Virtual Environment

## Overview

Instead of using pre-built containers, you can run evaluations with your own Python virtual environment by passing `--venv_path`.

## Setup

1. Create a venv:
   ```bash
   uv venv /path/to/your/.venv
   ```

2. Install dependencies:
   ```bash
   uv pip install --python /path/to/your/.venv/bin/python -r requirements-venv.txt
   ```

## Usage

```bash
oellm schedule-eval \
    --models HuggingFaceTB/SmolLM2-135M-Instruct \
    --task_groups multilingual \
    --venv_path /path/to/your/.venv
```

When `--venv_path` is provided, jobs run directly using that venv instead of inside a container.

## Requirements

The `requirements-venv.txt` file lists the minimum dependencies. Your venv should have:
- `lm-eval` and `lighteval[multilingual]` for evaluation
- `torch`, `transformers`, `accelerate` for model inference
- `langcodes[data]`, `pillow` for multilingual tasks

# Using Your Own Virtual Environment

## Overview

Instead of using pre-built containers, you can run evaluations with your own Python virtual environment by passing `--venv_path`.

## Setup

1. Create a venv with Python 3.12:
   ```bash
   uv venv --python 3.12 /path/to/.venv
   ```

2. Install lm-eval and lmms-eval dependencies:
   ```bash
   uv pip install --python /path/to/.venv/bin/python -r requirements-venv.txt
   ```

   This installs `lm-eval`, `torch`, `transformers`, `accelerate`, `datasets<4.0.0`, and `lmms-eval`.

3. Install lighteval as isolated tool (avoids datasets version conflict):
   ```bash
   UV_TOOL_DIR=/path/to/.uv-tools UV_TOOL_BIN_DIR=/path/to/.venv/bin \
     uv tool install --python 3.12 \
       --with "langcodes[data]" --with "pillow" \
       "lighteval[multilingual] @ git+https://github.com/huggingface/lighteval.git"
   ```

## Usage

```bash
# Text evaluation
oellm schedule-eval \
    --models HuggingFaceTB/SmolLM2-135M-Instruct \
    --task_groups open-sci-0.01 \
    --venv_path /path/to/.venv

# Image evaluation (lmms-eval)
oellm schedule-eval \
    --models path/to/vlm \
    --task_groups image-vqa \
    --venv_path /path/to/.venv
```

## Why Multiple Install Steps?

lm-eval requires `datasets<4.0.0` while lighteval requires `datasets>=4.0.0`. Installing lighteval as an isolated uv tool (like the containers do) avoids this conflict. `lmms-eval` is compatible with `datasets<4.0.0` and can be installed alongside lm-eval in the same venv.

## Dependency Summary

| Package | Install method | Reason |
|---|---|---|
| `lm-eval`, `torch`, `transformers`, `accelerate`, `datasets<4.0.0`, `lmms-eval` | `uv pip install -r requirements-venv.txt` | lm-eval + image eval, compatible dataset pin |
| `lighteval[multilingual]` | `uv tool install` (isolated) | Requires `datasets>=4.0.0` — must be isolated |

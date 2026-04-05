# Using Your Own Virtual Environment

## Overview

Instead of using pre-built containers, you can run evaluations with your own Python virtual environment by passing `--venv_path`.

## Setup

1. Create a venv with Python 3.12:
   ```bash
   uv venv --python 3.12 /path/to/.venv
   ```

2. Install lm-eval dependencies:
   ```bash
   uv pip install --python /path/to/.venv/bin/python -r requirements-venv.txt
   ```

3. Install lighteval as isolated tool (avoids datasets version conflict):
   ```bash
   UV_TOOL_DIR=/path/to/.uv-tools UV_TOOL_BIN_DIR=/path/to/.venv/bin \
     uv tool install --python 3.12 \
       --with "langcodes[data]" --with "pillow" \
       "lighteval[multilingual] @ git+https://github.com/huggingface/lighteval.git"
   ```

## Usage

```bash
oellm schedule-eval \
    --models HuggingFaceTB/SmolLM2-135M-Instruct \
    --task_groups multilingual \
    --venv_path /path/to/.venv
```

## Why Two Install Steps?

lm-eval requires `datasets<4.0.0` while lighteval requires `datasets>=4.0.0`. Installing lighteval as an isolated uv tool (like the containers do) avoids this conflict.

## DCLM-core-22

`dclm-core-22` needs `lm-eval==0.4.9.2` (v0.4.10+ breaks `agieval_lsat_ar` in few-shot). Use `requirements-venv-dclm.txt` instead of the default requirements:

```bash
uv venv --python 3.12 dclm-core-venv
uv pip install --python dclm-core-venv/bin/python -r requirements-venv-dclm.txt
```

```bash
oellm schedule-eval \
    --models Qwen/Qwen3-0.6B-Base \
    --task_groups dclm-core-22 \
    --venv_path dclm-core-venv \
    --skip_checks true
```

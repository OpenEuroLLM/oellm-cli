# Container Workflow

## Overview

Apptainer containers are built automatically via GitHub Actions and stored on HuggingFace Hub at `openeurollm/evaluation_singularity_images`.

## How It Works

1. Definition files live in `apptainer/<cluster>.def`
2. On push to `main` (when `.def` files change), GitHub Actions builds all containers
3. Built `.sif` images are uploaded to HuggingFace Hub
4. Clusters pull the image specified in `oellm/resources/clusters.yaml` via `EVAL_CONTAINER_IMAGE`

## Adding a New Cluster

1. Create `apptainer/<cluster>.def` with the appropriate base image:
   - NVIDIA: `nvcr.io/nvidia/pytorch:25.06-py3` (or newer)
   - AMD/ROCm: `rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1` (or newer)

2. Add the cluster name to the matrix in `.github/workflows/build-and-push-apptainer.yml`:
   ```yaml
   matrix:
     image: [slurm-ci, jureca, leonardo, lumi, <new-cluster>]
   ```

3. Add cluster configuration to `oellm/resources/clusters.yaml`:
   ```yaml
   <cluster>:
     hostname_pattern: "<pattern>"
     EVAL_BASE_DIR: "<path>"
     PARTITION: "<partition>"
     ACCOUNT: "<account>"
     QUEUE_LIMIT: <limit>
     EVAL_CONTAINER_IMAGE: "eval_env-<cluster>.sif"
     SINGULARITY_ARGS: "--nv"  # or "--rocm" for AMD
   ```

4. Push to `main` to trigger the build.

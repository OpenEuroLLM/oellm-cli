"""Tests for collect_results() covering both lm-eval and lmms-eval output formats."""

import json
from pathlib import Path

import pandas as pd
import pytest

from oellm.main import collect_results


# ── Helpers ───────────────────────────────────────────────────────────────────


def write_result(results_dir: Path, data: dict, filename: str = "result.json") -> None:
    (results_dir / filename).write_text(json.dumps(data))


def run_collect(tmp_path: Path, *json_payloads: dict) -> pd.DataFrame:
    """Write JSON result files and run collect_results; return CSV as DataFrame."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for i, data in enumerate(json_payloads):
        write_result(results_dir, data, filename=f"result_{i}.json")

    output_csv = str(tmp_path / "out.csv")
    collect_results(str(results_dir), output_csv=output_csv)

    csv_path = Path(output_csv)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


# ── lm-eval format (baseline — must remain unchanged) ────────────────────────


class TestCollectResultsLmEvalFormat:
    def test_standard_acc_metric(self, tmp_path):
        data = {
            "model_name": "/path/to/model",
            "results": {"mmlu": {"acc,none": 0.75}},
            "n-shot": {"mmlu": 5},
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["model_name"] == "/path/to/model"
        assert row["task"] == "mmlu"
        assert row["performance"] == pytest.approx(0.75)
        assert row["n_shot"] == 5

    def test_acc_norm_metric(self, tmp_path):
        data = {
            "model_name": "/path/to/model",
            "results": {"arc_challenge": {"acc_norm,none": 0.60}},
            "n-shot": {"arc_challenge": 10},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.60)

    def test_multiple_tasks_in_one_file(self, tmp_path):
        data = {
            "model_name": "/models/llm",
            "results": {
                "hellaswag": {"acc_norm,none": 0.80},
                "arc_easy": {"acc_norm,none": 0.85},
            },
            "n-shot": {"hellaswag": 10, "arc_easy": 10},
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 2
        assert set(df["task"].tolist()) == {"hellaswag", "arc_easy"}

    def test_multiple_json_files_aggregated(self, tmp_path):
        data1 = {
            "model_name": "/model",
            "results": {"mmlu": {"acc,none": 0.75}},
            "n-shot": {"mmlu": 5},
        }
        data2 = {
            "model_name": "/model",
            "results": {"hellaswag": {"acc_norm,none": 0.80}},
            "n-shot": {"hellaswag": 10},
        }
        df = run_collect(tmp_path, data1, data2)
        assert len(df) == 2

    def test_no_results_returns_empty(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output_csv = str(tmp_path / "out.csv")
        collect_results(str(results_dir), output_csv=output_csv)
        assert not Path(output_csv).exists()

    def test_model_name_fallback_when_no_path_field(self, tmp_path):
        """lm-eval has no model_name_or_path; must fall back to model_name."""
        data = {
            "model_name": "/path/to/model",
            "results": {"hellaswag": {"acc_norm,none": 0.70}},
            "n-shot": {"hellaswag": 10},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["model_name"] == "/path/to/model"


# ── lmms-eval format (Phase 2 additions) ─────────────────────────────────────


class TestCollectResultsLmmsEvalFormat:
    def test_model_name_or_path_takes_priority_over_model_name(self, tmp_path):
        """lmms-eval sets model_name to adapter type; real path is in model_name_or_path."""
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/checkpoints/llava-1.5-7b",
            "results": {"vqav2_val_all": {"vqav2/vqa_score,none": 0.82}},
            "n-shot": {"vqav2_val_all": 0},
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 1
        assert df.iloc[0]["model_name"] == "/checkpoints/llava-1.5-7b"

    def test_task_scoped_vqa_score_key_resolved(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"vqav2_val_all": {"vqav2/vqa_score,none": 0.82}},
            "n-shot": {"vqav2_val_all": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.82)
        assert df.iloc[0]["task"] == "vqav2_val_all"

    def test_task_scoped_acc_key_resolved(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"mmbench_en_dev": {"mmbench_en_dev/acc,none": 0.75}},
            "n-shot": {"mmbench_en_dev": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.75)
        assert df.iloc[0]["task"] == "mmbench_en_dev"

    def test_mmmu_task_scoped_key(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"mmmu_val": {"mmmu/acc,none": 0.55}},
            "n-shot": {"mmmu_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.55)
        assert df.iloc[0]["task"] == "mmmu_val"

    def test_chartqa_relaxed_accuracy(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"chartqa": {"chartqa/relaxed_accuracy,none": 0.68}},
            "n-shot": {"chartqa": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.68)

    def test_docvqa_anls_metric(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"docvqa_val": {"docvqa_val/anls,none": 0.91}},
            "n-shot": {"docvqa_val": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.91)

    def test_ocrbench_score_metric(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"ocrbench": {"ocrbench/score,none": 512.0}},
            "n-shot": {"ocrbench": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(512.0)

    def test_mathvista_acc_metric(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"mathvista_testmini": {"mathvista_testmini/acc,none": 0.49}},
            "n-shot": {"mathvista_testmini": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["performance"] == pytest.approx(0.49)

    def test_multiple_image_tasks_in_one_file(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {
                "vqav2_val_all": {"vqav2/vqa_score,none": 0.82},
                "mmbench_en_dev": {"mmbench_en_dev/acc,none": 0.75},
                "chartqa": {"chartqa/relaxed_accuracy,none": 0.68},
            },
            "n-shot": {
                "vqav2_val_all": 0,
                "mmbench_en_dev": 0,
                "chartqa": 0,
            },
        }
        df = run_collect(tmp_path, data)
        assert len(df) == 3
        assert set(df["task"].tolist()) == {"vqav2_val_all", "mmbench_en_dev", "chartqa"}

    def test_lmeval_and_lmms_eval_results_aggregated(self, tmp_path):
        """Both lm-eval and lmms-eval JSON files can coexist in the same results dir."""
        lm_eval_data = {
            "model_name": "/path/to/model",
            "results": {"mmlu": {"acc,none": 0.75}},
            "n-shot": {"mmlu": 5},
        }
        lmms_eval_data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/path/to/model",
            "results": {"vqav2_val_all": {"vqav2/vqa_score,none": 0.82}},
            "n-shot": {"vqav2_val_all": 0},
        }
        df = run_collect(tmp_path, lm_eval_data, lmms_eval_data)
        assert len(df) == 2
        assert set(df["task"].tolist()) == {"mmlu", "vqav2_val_all"}

    def test_empty_model_name_or_path_falls_back_to_model_name(self, tmp_path):
        """Empty string for model_name_or_path should fall back to model_name."""
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "",
            "results": {"vqav2_val_all": {"vqav2/vqa_score,none": 0.80}},
            "n-shot": {"vqav2_val_all": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["model_name"] == "llava_hf"

    def test_n_shot_zero_preserved(self, tmp_path):
        data = {
            "model_name": "llava_hf",
            "model_name_or_path": "/models/llava",
            "results": {"vqav2_val_all": {"vqav2/vqa_score,none": 0.82}},
            "n-shot": {"vqav2_val_all": 0},
        }
        df = run_collect(tmp_path, data)
        assert df.iloc[0]["n_shot"] == 0

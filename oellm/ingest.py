import json
import logging
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text

from oellm.utils import _setup_logging

PATH_PATTERN = re.compile(
    r"(?P<name>.*)-"
    r"(?P<params>[\d\.]+[bB])_"
    r"data-(?P<dataset>.+?)_"
    r"tokenizer-(?P<tokenizer>.+?)_"
    r".*?"
    r"machine-(?P<machine>[^_]+)"
    r".*?"
    r"/iter_(?P<iteration>\d+)"
)


def _parse_model_path(full_path: str) -> dict:
    if "converted_checkpoints/" in full_path:
        relevant_part = full_path.split("converted_checkpoints/")[1]
    else:
        relevant_part = full_path

    match = PATH_PATTERN.search(relevant_part)
    if match:
        return match.groupdict()
    return {}


def ingest_results(
    results_dir: str,
    verbose: bool = False,
) -> None:
    """
    Ingests JSON evaluation results from the given directory into the PostgreSQL database.
    Database credentials must be set in environment variables: PG_HOST, PG_USER, PG_PASS, PG_DB.
    """
    _setup_logging(verbose)

    host = os.environ.get("PG_HOST")
    user = os.environ.get("PG_USER", "postgres")
    password = os.environ.get("PG_PASS")
    dbname = os.environ.get("PG_DB", "postgres")

    if not host or not password:
        logging.warning("Sync skipped: PG_HOST or PG_PASS environment variables not set.")
        return

    safe_pass = urllib.parse.quote_plus(password)
    db_url = f"postgresql://{user}:{safe_pass}@{host}:5432/{dbname}"

    search_path = Path(results_dir)
    json_files = list(search_path.glob("*.json"))

    if not json_files:
        logging.info(f"No JSON results found in {results_dir} to sync.")
        return

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            for filepath in json_files:
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    try:
                        if "config" in data and "model_args" in data["config"]:
                            raw_path = (
                                data["config"]["model_args"]
                                .split("pretrained=")[1]
                                .split(",")[0]
                            )
                        else:
                            raw_path = data.get("model_name", "unknown")
                    except Exception:
                        logging.warning(f"Could not extract model path from {filepath.name}")
                        continue

                    meta = _parse_model_path(raw_path)
                    meta_def = {
                        "name": "unknown", "params": "0B", "dataset": "unknown",
                        "tokenizer": "unknown", "machine": "unknown", "iteration": 0
                    }
                    meta = {**meta_def, **meta}

                    res = conn.execute(
                        text("SELECT id FROM models WHERE raw_path = :p"),
                        {"p": raw_path},
                    ).fetchone()

                    if res:
                        model_id = res[0]
                    else:
                        ins = text("""
                            INSERT INTO models (raw_path, model_name, params, dataset, tokenizer, hpc, iteration) 
                            VALUES (:raw, :name, :params, :data, :tok, :mach, :iter) 
                            RETURNING id
                        """)
                        model_id = conn.execute(
                            ins,
                            {
                                "raw": raw_path,
                                "name": meta["name"],
                                "params": meta["params"],
                                "data": meta["dataset"],
                                "tok": meta["tokenizer"],
                                "mach": meta["machine"],
                                "iter": int(meta["iteration"]),
                            },
                        ).scalar()
                        conn.commit()

                    try:
                        eval_dt = datetime.fromtimestamp(data.get("date", 0))
                    except Exception:
                        eval_dt = datetime.now()

                    run_check = conn.execute(
                        text("SELECT id FROM eval_runs WHERE model_id=:mid AND eval_date=:dt"),
                        {"mid": model_id, "dt": eval_dt},
                    ).fetchone()

                    if not run_check:
                        ins_run = text("""
                            INSERT INTO eval_runs (model_id, eval_date, git_hash, framework_version) 
                            VALUES (:mid, :dt, :git, :fw) 
                            RETURNING id
                        """)
                        run_id = conn.execute(
                            ins_run,
                            {
                                "mid": model_id,
                                "dt": eval_dt,
                                "git": data.get("git_hash", "unknown"),
                                "fw": data.get("transformers_version", "unknown"),
                            },
                        ).scalar()
                        conn.commit()

                        for task, metrics in data.get("results", {}).items():
                            n_shot = data.get("n-shot", {}).get(task, 0)
                            for m_key, m_val in metrics.items():
                                if m_key == "alias":
                                    continue
                                clean_metric = m_key.split(",")[0]
                                conn.execute(
                                    text("""
                                        INSERT INTO results (run_id, task_name, n_shot, metric_name, metric_value) 
                                        VALUES (:rid, :task, :nshot, :mname, :mval)
                                    """),
                                    {
                                        "rid": run_id,
                                        "task": task,
                                        "nshot": n_shot,
                                        "mname": clean_metric,
                                        "mval": float(m_val) if m_val is not None else 0.0,
                                    },
                                )
                        conn.commit()
                        logging.info(f"Synced: {filepath.name}")

                except Exception as e:
                    logging.error(f"Failed to sync file {filepath.name}: {e}")

    except Exception as e:
        logging.error(f"Database connection failed: {e}")
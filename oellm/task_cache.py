import json
from datetime import datetime
from pathlib import Path

TASK_CACHE_TTL_DAYS = 30


def get_task_cache_file() -> Path:
    return Path(__file__).resolve().parent / "task_map_cache.json"


def load_task_cache() -> dict:
    cache_file = get_task_cache_file()
    if not cache_file.exists():
        return {}
    with open(cache_file, "r") as f:
        return json.load(f) or {}


def save_task_cache(cache: dict) -> None:
    cache_file = get_task_cache_file()
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def task_cache_key(framework: str, task_id: str) -> str:
    return f"{framework}::{task_id}"


def task_cache_is_fresh(entry: dict, ttl_days: int = TASK_CACHE_TTL_DAYS) -> bool:
    ts = float(entry.get("ts", 0))
    age_days = (datetime.now().timestamp() - ts) / 86400.0
    return age_days >= 0 and age_days < float(ttl_days)


def task_cache_lookup(
    framework: str, task_id: str, ttl_days: int = TASK_CACHE_TTL_DAYS
) -> bool:
    cache = load_task_cache()
    key = task_cache_key(framework, task_id)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return False
    return task_cache_is_fresh(entry, ttl_days)


def task_cache_mark_resolved(framework: str, task_id: str) -> None:
    cache = load_task_cache()
    key = task_cache_key(framework, task_id)
    cache[key] = {"ts": datetime.now().timestamp()}
    save_task_cache(cache)

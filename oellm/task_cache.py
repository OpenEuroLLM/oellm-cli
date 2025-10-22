import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

TASK_CACHE_TTL_DAYS = 30


_CURRENT_CAPTURE_BUFFER: ContextVar[list[dict] | None] = ContextVar(
    "_CURRENT_CAPTURE_BUFFER", default=None
)


def get_task_cache_file() -> Path:
    return Path(__file__).resolve().parent / "resources" / "task_map_cache.json"


def load_task_cache() -> dict:
    cache_file = get_task_cache_file()
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f) or {}
    return {}


def save_task_cache(cache: dict) -> None:
    cache_file = get_task_cache_file()
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def clear_task_cache() -> None:
    cache_file = get_task_cache_file()
    with open(cache_file, "w") as f:
        json.dump({}, f)


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
    entry = cache.get(key) if isinstance(cache.get(key), dict) else {}
    entry["ts"] = datetime.now().timestamp()
    cache[key] = entry
    save_task_cache(cache)


def task_cache_get_payload(framework: str, task_id: str) -> dict | None:
    cache = load_task_cache()
    key = task_cache_key(framework, task_id)
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    payload = entry.get("payload")
    return payload if isinstance(payload, dict) else None


def task_cache_set_payload(framework: str, task_id: str, payload: dict) -> None:
    cache = load_task_cache()
    key = task_cache_key(framework, task_id)
    entry: dict = cache.get(key) if isinstance(cache.get(key), dict) else {}  # type: ignore[assignment]
    entry["ts"] = datetime.now().timestamp()
    entry["payload"] = payload
    cache[key] = entry
    save_task_cache(cache)


def _canonical_key(call: dict) -> tuple:
    t = call.get("type")
    if t == "load_dataset":
        return (
            t,
            call.get("path"),
            call.get("name"),
            call.get("split"),
            call.get("revision"),
        )
    if t == "snapshot_download":
        return (
            t,
            call.get("repo_id"),
            call.get("repo_type"),
            call.get("revision"),
        )
    if t == "hf_hub_download":
        return (
            t,
            call.get("repo_id"),
            call.get("filename"),
            call.get("repo_type"),
            call.get("revision"),
        )
    return (str(t),)


def dedupe_calls(calls: list[dict]) -> list[dict]:
    if not isinstance(calls, list):
        return []
    best: dict[tuple, dict] = {}
    for c in calls:
        if not isinstance(c, dict):
            continue
        key = _canonical_key(c)
        existing = best.get(key)
        if existing is None:
            best[key] = c
            continue
        # Prefer trust_remote_code=True for load_dataset
        if c.get("type") == "load_dataset":
            if bool(c.get("trust_remote_code")) and not bool(
                existing.get("trust_remote_code")
            ):
                best[key] = c
    # Optionally drop snapshot_download if matching load_dataset exists
    filtered: list[dict] = []
    load_keys = {
        ("load_dataset", k[1], k[2], k[3], k[4])
        for k in best.keys()
        if k and k[0] == "load_dataset"
    }
    for k, v in best.items():
        if k and k[0] == "snapshot_download":
            # derive comparable key shape: (type, repo_id, None, None, revision)
            comparable = ("load_dataset", k[1], None, None, k[3])
            if comparable in load_keys:
                continue
        filtered.append(v)
    return filtered


@contextmanager
def capture_hf_dataset_calls():
    captured: list[dict] = []
    _buffer_token = _CURRENT_CAPTURE_BUFFER.set(captured)

    import datasets as _ds  # type: ignore
    import huggingface_hub as _hfh  # type: ignore

    _orig_load_dataset = _ds.load_dataset
    _orig_snapshot_download = _hfh.snapshot_download
    _orig_hf_hub_download = _hfh.hf_hub_download

    def _load_dataset_proxy(path, *args, **kwargs):  # noqa: ANN001
        name = (
            kwargs.get("name")
            if "name" in kwargs
            else (args[0] if len(args) > 0 else None)
        )
        data_files = (
            kwargs.get("data_files")
            if "data_files" in kwargs
            else (args[1] if len(args) > 1 else None)
        )
        split = (
            kwargs.get("split")
            if "split" in kwargs
            else (args[2] if len(args) > 2 else None)
        )
        trust_remote_code = kwargs.get("trust_remote_code")
        revision = kwargs.get("revision")
        buf = _CURRENT_CAPTURE_BUFFER.get()
        if isinstance(buf, list):
            buf.append(
                {
                    "type": "load_dataset",
                    "path": path,
                    "name": name,
                    "data_files": data_files,
                    "split": split,
                    "revision": revision,
                    "trust_remote_code": trust_remote_code,
                }
            )
        return _orig_load_dataset(path, *args, **kwargs)

    def _snapshot_download_proxy(*args, **kwargs):  # noqa: ANN001
        repo_id = (
            kwargs.get("repo_id")
            if "repo_id" in kwargs
            else (args[0] if len(args) > 0 else None)
        )
        repo_type = (
            kwargs.get("repo_type")
            if "repo_type" in kwargs
            else (args[1] if len(args) > 1 else None)
        )
        revision = (
            kwargs.get("revision")
            if "revision" in kwargs
            else (args[2] if len(args) > 2 else None)
        )
        buf = _CURRENT_CAPTURE_BUFFER.get()
        if isinstance(buf, list):
            buf.append(
                {
                    "type": "snapshot_download",
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                }
            )
        return _orig_snapshot_download(*args, **kwargs)

    def _hf_hub_download_proxy(*args, **kwargs):  # noqa: ANN001
        repo_id = (
            kwargs.get("repo_id")
            if "repo_id" in kwargs
            else (args[0] if len(args) > 0 else None)
        )
        filename = (
            kwargs.get("filename")
            if "filename" in kwargs
            else (args[1] if len(args) > 1 else None)
        )
        repo_type = (
            kwargs.get("repo_type")
            if "repo_type" in kwargs
            else (args[2] if len(args) > 2 else None)
        )
        revision = (
            kwargs.get("revision")
            if "revision" in kwargs
            else (args[3] if len(args) > 3 else None)
        )
        buf = _CURRENT_CAPTURE_BUFFER.get()
        if isinstance(buf, list):
            buf.append(
                {
                    "type": "hf_hub_download",
                    "repo_id": repo_id,
                    "filename": filename,
                    "repo_type": repo_type,
                    "revision": revision,
                }
            )
        return _orig_hf_hub_download(*args, **kwargs)

    _ds.load_dataset = _load_dataset_proxy  # type: ignore[assignment]
    _hfh.snapshot_download = _snapshot_download_proxy  # type: ignore[assignment]
    _hfh.hf_hub_download = _hf_hub_download_proxy  # type: ignore[assignment]

    try:
        yield captured
    finally:
        _ds.load_dataset = _orig_load_dataset  # type: ignore[assignment]
        _hfh.snapshot_download = _orig_snapshot_download  # type: ignore[assignment]
        _hfh.hf_hub_download = _orig_hf_hub_download  # type: ignore[assignment]
        _CURRENT_CAPTURE_BUFFER.reset(_buffer_token)


def prewarm_from_payload(payload: dict | None, *, trust_remote_code: bool = True) -> None:
    if not isinstance(payload, dict):
        return
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return

    from datasets import load_dataset  # type: ignore
    from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

    for call in calls:
        if not isinstance(call, dict):
            continue
        # Unified prewarm log message
        if call.get("type") == "load_dataset":
            path = call.get("path")
            name = call.get("name")
        else:
            repo_id = call.get("repo_id")
            filename = call.get("filename")

        if call.get("type") == "snapshot_download":
            repo_id = call.get("repo_id")
            if isinstance(repo_id, str) and repo_id:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=call.get("repo_type") or "dataset",
                    revision=call.get("revision"),
                )
            continue
        if call.get("type") == "hf_hub_download":
            repo_id = call.get("repo_id")
            filename = call.get("filename")
            if isinstance(repo_id, str) and isinstance(filename, str):
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=call.get("repo_type"),
                    revision=call.get("revision"),
                )
            continue
        path = call.get("path")
        name = call.get("name")
        data_files = call.get("data_files")
        split = call.get("split")
        revision = call.get("revision")
        trc = call.get("trust_remote_code", trust_remote_code)
        kwargs: dict = {}
        if name is not None:
            kwargs["name"] = name
        if data_files is not None:
            kwargs["data_files"] = data_files
        if revision is not None:
            kwargs["revision"] = revision
        kwargs["trust_remote_code"] = bool(trc)
        if split is not None:
            load_dataset(path, split=split, **kwargs)
        else:
            load_dataset(path, **kwargs)

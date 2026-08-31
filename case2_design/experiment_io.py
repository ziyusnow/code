from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import csv
import json
import os
import tempfile
import time

import numpy as np

try:
    from .experiment_types import ActionParameters, SearchConfig
except ImportError:
    from experiment_types import ActionParameters, SearchConfig


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    group: str
    method: str
    seed: int
    search_config: SearchConfig = field(default_factory=SearchConfig)
    action_parameters: ActionParameters = field(default_factory=ActionParameters)
    variant: str = "default"
    selector_options: Mapping[str, Any] = field(default_factory=dict)
    llm_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def method_id(self):  # type: () -> str
        return self.method if self.variant == "default" else "%s__%s" % (
            self.method,
            self.variant,
        )

    def to_dict(self):  # type: () -> Dict[str, Any]
        return {
            "experiment_id": self.experiment_id,
            "group": self.group,
            "method": self.method,
            "method_id": self.method_id,
            "variant": self.variant,
            "seed": self.seed,
            "search_config": asdict(self.search_config),
            "action_parameters": asdict(self.action_parameters),
            "selector_options": dict(self.selector_options),
            "llm_options": _redact_secrets(dict(self.llm_options)),
        }


@dataclass(frozen=True)
class RunPaths:
    directory: Path

    @classmethod
    def from_spec(cls, results_root, spec):  # type: (Path, RunSpec) -> "RunPaths"
        directory = (
            Path(results_root)
            / spec.experiment_id
            / spec.method_id
            / ("seed_%d" % spec.seed)
        )
        return cls(directory)

    def file(self, name):  # type: (str) -> Path
        return self.directory / name

    @property
    def config(self):  # type: () -> Path
        return self.file("config.json")

    @property
    def summary(self):  # type: () -> Path
        return self.file("summary.json")


def _redact_secrets(value):
    secret_fragments = ("api_key", "apikey", "authorization", "secret", "token")
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if any(fragment in str(key).lower() for fragment in secret_fragments)
                else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


def to_jsonable(value):
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _temporary_path(path):  # type: (Path) -> Path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    return Path(name)


def _replace_with_retry(source, destination, attempts=5):
    # Windows readers can briefly prevent replacement of an existing manifest.
    for attempt in range(attempts):
        try:
            os.replace(str(source), str(destination))
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _commit_temporary(temp_path, destination):  # type: (Path, Path) -> None
    try:
        # Windows rejects fsync on a read-only descriptor.
        with temp_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        _replace_with_retry(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_text(path, text):  # type: (Path, str) -> None
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path, value):  # type: (Path, Any) -> None
    text = json.dumps(
        to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    atomic_write_text(path, text + "\n")


def atomic_write_jsonl(path, rows):  # type: (Path, Iterable[Mapping[str, Any]]) -> None
    lines = [
        json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
        for row in rows
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def atomic_write_csv(path, fieldnames, rows):
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(to_jsonable(row))
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_via(path, writer):
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        writer(temporary)
        _commit_temporary(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path):  # type: (Path) -> Dict[str, Any]
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def ensure_run_config(paths, spec):  # type: (RunPaths, RunSpec) -> None
    expected = spec.to_dict()
    paths.directory.mkdir(parents=True, exist_ok=True)
    if paths.config.exists():
        existing = read_json(paths.config)
        if existing != expected:
            raise ValueError(
                "Run directory already contains a different config: %s"
                % paths.directory
            )
        return
    atomic_write_json(paths.config, expected)


def completed_summary(paths):  # type: (RunPaths) -> Optional[Dict[str, Any]]
    if not paths.summary.exists():
        return None
    summary = read_json(paths.summary)
    return summary if summary.get("status") == "complete" else None


def append_attempt(paths, record):  # type: (RunPaths, Mapping[str, Any]) -> None
    destination = paths.file("attempts.jsonl")
    rows = []
    if destination.exists():
        with destination.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    rows.append(dict(record))
    atomic_write_jsonl(destination, rows)


def initialize_experiment_manifest(results_root, specs, timestamp, resume=False):
    # type: (Path, Sequence[RunSpec], str, bool) -> Path
    if not specs:
        raise ValueError("Cannot create a manifest without run specifications")
    experiment_ids = {spec.experiment_id for spec in specs}
    groups = {spec.group for spec in specs}
    if len(experiment_ids) != 1 or len(groups) != 1:
        raise ValueError("All manifest tasks must belong to one experiment and group")

    root = Path(results_root)
    experiment_id = specs[0].experiment_id
    path = root / experiment_id / "manifest.json"
    tasks = []
    for spec in specs:
        run_paths = RunPaths.from_spec(root, spec)
        tasks.append(
            {
                "task_id": "%s/seed_%d" % (spec.method_id, spec.seed),
                "method": spec.method,
                "method_id": spec.method_id,
                "variant": spec.variant,
                "seed": spec.seed,
                "relative_path": str(run_paths.directory.relative_to(path.parent)),
                "status": "pending",
                "config": spec.to_dict(),
                "updated_at": timestamp,
            }
        )

    if path.exists():
        existing = read_json(path)
        existing_configs = [task.get("config") for task in existing.get("tasks", [])]
        requested_configs = [task["config"] for task in tasks]
        if existing_configs != requested_configs:
            raise ValueError(
                "Experiment manifest already contains a different task matrix: %s" % path
            )
        prior = {task["task_id"]: task for task in existing["tasks"]}
        for task in tasks:
            previous = prior[task["task_id"]]
            if previous.get("status") in ("complete", "skipped_complete"):
                task.update(previous)
            elif not resume:
                task.update(previous)
            else:
                task["status"] = "pending"
                task.pop("error", None)
                task.pop("outcome", None)
        created_at = existing.get("created_at", timestamp)
    else:
        created_at = timestamp

    method_ids = {spec.method_id for spec in specs}
    seeds = {spec.seed for spec in specs}
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "group": specs[0].group,
        "created_at": created_at,
        "updated_at": timestamp,
        "expected_task_count": len(tasks),
        "expected_method_count": len(method_ids),
        "expected_seed_count": len(seeds),
        "tasks": tasks,
    }
    atomic_write_json(path, manifest)
    return path


def update_manifest_task(path, task_id, status, timestamp, outcome=None, error=None):
    # type: (Path, str, str, str, Optional[str], Optional[str]) -> None
    manifest = read_json(path)
    matching = [task for task in manifest["tasks"] if task["task_id"] == task_id]
    if len(matching) != 1:
        raise KeyError("Manifest task not found: %s" % task_id)
    task = matching[0]
    task["status"] = status
    task["updated_at"] = timestamp
    if outcome is None:
        task.pop("outcome", None)
    else:
        task["outcome"] = outcome
    if error is None:
        task.pop("error", None)
    else:
        task["error"] = error
    manifest["updated_at"] = timestamp
    atomic_write_json(path, manifest)


def commit_run_artifacts(
    paths,
    *,
    history,
    history_fields,
    actions,
    schedule_writer,
    validation,
    summary,
    llm_calls=None
):
    """Write a complete run, committing summary.json last."""
    atomic_write_csv(paths.file("history.csv"), history_fields, history)
    atomic_write_jsonl(paths.file("actions.jsonl"), actions)
    atomic_write_via(paths.file("schedule.csv"), schedule_writer)
    atomic_write_json(paths.file("validation.json"), validation)
    if llm_calls is not None:
        atomic_write_jsonl(paths.file("llm_calls.jsonl"), llm_calls)
    atomic_write_json(paths.summary, summary)

"""Run manifest: everything needed to trace a published number to its inputs.

Recorded in every report next to the traces, so a run can be re-scored
offline (`membench rescore`) and compared only against runs made with the
same pack, judge model and judge prompt version.
"""

from __future__ import annotations

import platform
import random
import re
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import membench
from membench.core.adapter import MemoryAdapter
from membench.core.reader import READER_PROMPT_VERSION
from membench.core.scoring.judge import JUDGE_PROMPT_VERSION
from membench.packs.schema import Pack

REPORT_SCHEMA_VERSION = "2"
_SECRET_KEY = re.compile(r"(key|secret|password|passwd|credential|token|auth)$")


def redact(value: Any) -> Any:
    """Copy of a config with secret-looking keys (`api_key`, `token`, …) masked."""
    if isinstance(value, dict):
        return {
            k: "<redacted>" if v is not None and _SECRET_KEY.search(str(k).lower()) else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def git_info() -> dict[str, Any]:
    """Commit + dirty flag of the membench checkout, or None outside one."""
    root = Path(membench.__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True,
            timeout=10,
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            raise ValueError("membench is not the repository root")
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def seed_everything(seed: int) -> None:
    """Seed Python's RNG, and numpy's if a backend already imported it."""
    random.seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(seed)


def _name(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"


def build_manifest(
    *,
    pack: Pack,
    adapter: MemoryAdapter,
    backend_name: str,
    judge: Any,
    tokenizer: Callable[[str], int],
    seed: int,
    repeat: int,
    started_at: str,
    finished_at: str,
    reader: Any = None,
) -> dict[str, Any]:
    judge_info = None
    if judge is not None:
        judge_info = {
            "model": judge.model,
            "base_url": getattr(judge, "base_url", None),
            "prompt_version": JUDGE_PROMPT_VERSION,
        }
    reader_info = None
    if reader is not None:
        reader_info = {
            "model": reader.model,
            "base_url": reader.base_url,
            "prompt_version": READER_PROMPT_VERSION,
        }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "membench": {"version": membench.__version__, **git_info()},
        "python": platform.python_version(),
        "os": platform.platform(),
        "packages": package_versions(adapter.packages),
        "backend": {
            "name": backend_name,
            "config": redact(adapter.config),
            "models": adapter.describe(),
            "deployment": adapter.deployment,
            "region": adapter.config.get("region"),
        },
        "pack": {
            "name": pack.name, "sha256": pack.sha256, "subset": pack.subset,
            "qa_style": pack.qa_style,
        },
        "dataset": pack.dataset,  # how membench.datasets built it (None: a YAML pack)
        "judge": judge_info,
        "reader": reader_info,
        "tokenizer": getattr(tokenizer, "name", None) or _name(tokenizer),
        "seed": seed,
        "repeat": repeat,
        "started_at": started_at,
        "finished_at": finished_at,
    }

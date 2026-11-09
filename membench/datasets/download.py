"""Download a pinned dataset file once, check its sha256, cache it.

Files land under `~/.cache/membench/datasets/<sha256>/<filename>`
(`MEMBENCH_CACHE_DIR` overrides `~/.cache/membench`). A cached file is
re-hashed on every load, so a corrupted or swapped file is never used.
Dataset files are never committed to this repository.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

CACHE_ENV = "MEMBENCH_CACHE_DIR"


class DatasetError(ValueError):
    """A dataset could not be fetched, failed its checksum, or failed to convert."""


@dataclass(frozen=True)
class Source:
    """One pinned upstream file: `url` names an immutable revision."""

    id: str
    url: str
    revision: str
    sha256: str
    license: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "membench")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)


def fetch(source: Source) -> Path:
    """Path to the verified local copy of `source`, downloading it if needed."""
    path = cache_dir() / "datasets" / source.sha256 / source.filename
    if path.exists() and sha256_of(path) == source.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        _download(source.url, partial)
    except httpx.HTTPError as exc:
        raise DatasetError(f"could not download {source.id} from {source.url}: {exc}") from exc
    actual = sha256_of(partial)
    if actual != source.sha256:
        partial.unlink()
        raise DatasetError(
            f"{source.id}: sha256 {actual} does not match the pinned {source.sha256} "
            f"({source.url})"
        )
    partial.replace(path)
    return path

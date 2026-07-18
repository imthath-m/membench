"""Transcript loader.

A "transcript" is just free-form conversation text — one file, or a
directory of files (e.g. one per session) concatenated in sorted order.
No structure is assumed beyond plain text; the LLM prompt does the work
of pulling facts, updates, and contradictions out of it.
"""

from __future__ import annotations

from pathlib import Path


def load_transcripts(path: str | Path) -> str:
    """Read a transcript file, or concatenate every file in a directory."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        if not files:
            raise ValueError(f"no transcript files found in {path}")
        return "\n\n".join(f.read_text() for f in files)
    return path.read_text()

"""Packs beyond the bundled YAML: public datasets and synthetic generators.

`resolve_pack(spec)` is what the CLI's `--pack` goes through. `spec` is a
YAML path, a public dataset (`longmemeval-s`, `longmemeval-m`,
`longmemeval-oracle`, `locomo`: downloaded at a pinned revision, sha256
checked, cached; see `download.py`), a synthetic pack (`tool-call-memory`,
`document-chain`, `multi-step-decisions`, `customer-support-xl`,
`coding-assistant-xl`: generated in memory from `--pack-seed`), or a bundled
pack name. Every pack is built in memory; nothing is written to disk.

A loaded pack records how it was built in `Pack.dataset` (→ the manifest's
`dataset`: id, revision, sha256, options) and how it was cut down in
`Pack.subset` (`--limit` / `--sample`), so `membench rescore` rebuilds the
exact same pack from the manifest alone.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from membench.datasets import locomo, longmemeval
from membench.datasets.download import DatasetError, Source, fetch
from membench.generate.synthetic import GENERATOR_VERSION, generate
from membench.packs.schema import Pack, QueryStep, builtin_pack_path, load_pack, parse_pack

DATASETS = ("longmemeval-s", "longmemeval-m", "longmemeval-oracle", "locomo")
SYNTHETIC = ("tool-call-memory", "document-chain", "multi-step-decisions",
             "customer-support-xl", "coding-assistant-xl")

__all__ = ["DATASETS", "SYNTHETIC", "DatasetError", "resolve_pack", "select_queries"]


def _load_json(source: Source) -> list:
    with fetch(source).open() as f:
        return json.load(f)


def load_dataset(name: str) -> Pack:
    """Build a public dataset's full pack (downloads on first use)."""
    if name.startswith("longmemeval-"):
        variant = name.removeprefix("longmemeval-")
        source = longmemeval.SOURCES[variant]
        pack = parse_pack(longmemeval.build(_load_json(source), variant))
        pack.qa_style = "longmemeval"
    else:
        source = locomo.SOURCE
        raw, qa_ids = locomo.build(_load_json(source))
        pack = parse_pack(raw)
        excluded = locomo.exclusions()
        queries = [i for i, s in enumerate(pack.steps) if isinstance(s, QueryStep)]
        pack.flagged = [i for i, qa_id in zip(queries, qa_ids) if qa_id in excluded]
    pack.dataset = {
        "id": name, "revision": source.revision, "sha256": source.sha256,
        "url": source.url, "license": source.license, "options": {},
    }
    return pack


def select_queries(pack: Pack, keep: set[int]) -> tuple[Pack, list[int]]:
    """Pack with only the query steps in `keep` (step indices).

    Writes and forgets stay, except in episodes left with no query, which are
    dropped. Returns the new pack and the kept steps' original indices.
    """
    episodes = sorted({pack.steps[i].episode for i in keep})
    renumber = {old: new for new, old in enumerate(episodes)}
    kept = [
        i for i, step in enumerate(pack.steps)
        if (i in keep if isinstance(step, QueryStep) else step.episode in renumber)
    ]
    position = {old: new for new, old in enumerate(kept)}
    digest = hashlib.sha256(f"{pack.sha256}:{sorted(keep)}".encode()).hexdigest()
    subset = Pack(
        pack.name, pack.description,
        [replace(pack.steps[i], episode=renumber[pack.steps[i].episode]) for i in kept],
        digest, dataset=pack.dataset, subset=pack.subset, qa_style=pack.qa_style,
        flagged=[position[i] for i in pack.flagged if i in position],
    )
    return subset, kept


def _stratified(queries: list[int], categories: list[str], n: int, seed: int) -> set[int]:
    """`n` queries, each category's share proportional (largest remainder)."""
    if n >= len(queries):
        return set(queries)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, category in zip(queries, categories):
        groups[category].append(index)
    exact = {c: n * len(group) / len(queries) for c, group in groups.items()}
    quota = {c: int(share) for c, share in exact.items()}
    by_remainder = sorted(groups, key=lambda c: (quota[c] - exact[c], c))
    for c in by_remainder[: n - sum(quota.values())]:
        quota[c] += 1
    rng = random.Random(seed)
    return {i for c in sorted(groups) for i in rng.sample(groups[c], quota[c])}


def resolve_pack(
    spec: str,
    *,
    pack_seed: int = 0,
    exclude_flagged: bool = False,
    limit: int | None = None,
    sample: int | None = None,
    sample_seed: int = 0,
) -> Pack:
    """Load `spec` (see module docstring), then apply the options in order:
    drop flagged queries, then keep the first `limit` queries or a stratified
    (by category) random `sample` of them, seeded by `sample_seed`."""
    name = spec.replace("_", "-")
    if pack_seed and name not in SYNTHETIC:
        raise ValueError(f"--pack-seed applies to synthetic packs only, not {spec!r}")
    if limit is not None and sample is not None:
        raise ValueError("use --limit or --sample, not both")
    if Path(spec).exists():
        pack = load_pack(spec)
    elif name in DATASETS:
        pack = load_dataset(name)
    elif name in SYNTHETIC:
        pack = parse_pack(generate(name.replace("-", "_"), seed=pack_seed))
        pack.dataset = {
            "id": name, "revision": f"synthetic-v{GENERATOR_VERSION}", "sha256": pack.sha256,
            "options": {"pack_seed": pack_seed},
        }
    else:
        pack = load_pack(builtin_pack_path(spec))

    queries = [i for i, s in enumerate(pack.steps) if isinstance(s, QueryStep)]
    if exclude_flagged:
        if pack.dataset is None:
            raise ValueError(f"{spec!r} has no exclusion list (only dataset packs do)")
        pack, _ = select_queries(pack, set(queries) - set(pack.flagged))
        pack.dataset = {**pack.dataset, "options": {**pack.dataset["options"],
                                                    "exclude_flagged": True}}
        queries = [i for i, s in enumerate(pack.steps) if isinstance(s, QueryStep)]
    if limit is not None:
        pack, _ = select_queries(pack, set(queries[:limit]))
        pack.subset = {"limit": limit}
    elif sample is not None:
        categories = [pack.steps[i].category or "uncategorized" for i in queries]
        pack, _ = select_queries(pack, _stratified(queries, categories, sample, sample_seed))
        pack.subset = {"sample": sample, "sample_seed": sample_seed}
    return pack

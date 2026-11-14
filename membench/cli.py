"""membench CLI (Typer app)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import typer
import yaml

from membench.core.publish import RecordingCache, result_dir, sha256_file, trim, write_result
from membench.core.reader import READER_API_KEY_ENV, Reader
from membench.core.registry import get_adapter_class, list_backends
from membench.core.report.compare import compare_reports, render_compare_markdown
from membench.core.report.failures import render_failures
from membench.core.report.html import render_html
from membench.core.report.json import write_json
from membench.core.report.leaderboard import collect, render_leaderboard
from membench.core.report.markdown import render_markdown
from membench.core.runner import (
    DEFAULT_STEP_TIMEOUT_S,
    MAX_ERROR_RATE,
    StepTrace,
    merge_repeats,
    run_pack,
)
from membench.core.scoring import score_detailed
from membench.core.scoring.judge import (
    API_KEY_ENV,
    DEFAULT_JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    PROMPT_BUILDERS,
    CacheMissError,
    JudgeCache,
    build_judge,
    resolve_client,
)
from membench.core.stats import summarize_repeats
from membench.core.tokens import default_tokenizer
from membench.core.validation import JUDGE_METRICS
from membench.core.validation.agreement import agreement_report, render_agreement_markdown
from membench.core.validation.cross import (
    build_panel_judge,
    load_panel,
    run_judges,
    share_cache,
)
from membench.core.validation.label_server import LabelStore, make_server
from membench.core.validation.labels import (
    gold,
    load_labels,
    merge_annotations,
    read_jsonl,
    write_jsonl,
)
from membench.core.validation.sample import (
    DEFAULT_N_PER_METRIC,
    judge_items,
    sample_summary,
    stratified_sample,
)
from membench.core.validation.status import agreement_path
from membench.core.validation.tune import (
    build_version_judges,
    join_sample,
    render_tuning_markdown,
    run_versions,
    tuning_report,
)
from membench.datasets import DATASETS, SYNTHETIC, resolve_pack, select_queries
from membench.generate.generator import GenerationError, generate_pack
from membench.generate.synthetic import paraphrase_queries
from membench.packs.schema import Pack, PackValidationError, QueryStep, pack_to_raw

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """membench: eval harness for agent memory backends."""


_PACK_HELP = (
    "Pack: a YAML path, a bundled pack (e.g. 'mini'), a dataset "
    f"({', '.join(DATASETS)}) or a synthetic pack ({', '.join(SYNTHETIC)})."
)


# `--pack` selection options shared by `run` and `convert`.
_OPT = {
    "pack_seed": typer.Option(0, "--pack-seed", help="Seed of a synthetic pack."),
    "exclude_flagged": typer.Option(
        False, "--exclude-flagged",
        help="Drop queries on the dataset's exclusion list (e.g. LoCoMo).",
    ),
    "limit": typer.Option(None, "--limit", min=1, help="Keep only the first N queries."),
    "sample": typer.Option(
        None, "--sample", min=1,
        help="Keep a random N queries, stratified by category (seeded by --seed).",
    ),
}


def _load_backend_config(path: Path | None) -> dict:
    """Read a JSON or YAML mapping of adapter config (YAML is a JSON superset)."""
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"backend config {path} must be a JSON/YAML mapping")
    return data


@app.command()
def run(
    backend: str = typer.Option(
        ..., "--backend", help=f"Adapter to run against. One of: {list_backends()}"
    ),
    pack: str = typer.Option(..., "--pack", help=_PACK_HELP),
    out: Path = typer.Option(
        Path("report.json"), "--out", help="Path to write the JSON report to."
    ),
    persist_dir: str = typer.Option(
        ".membench_chroma", "--persist-dir", help="Local persistence dir (chroma backend only)."
    ),
    backend_config: Path = typer.Option(
        None,
        "--backend-config",
        help="JSON/YAML file of adapter config (e.g. mem0 user_id, top_k, mem0_config).",
    ),
    reset: bool = typer.Option(
        True,
        "--reset/--no-reset",
        help="Clear the backend's stored state (its collection / user_id) before the run.",
    ),
    judge: bool = typer.Option(
        True, "--judge/--no-judge", help="Enable the LLM judge (semantic recall + contradiction rate)."
    ),
    judge_model: str = typer.Option(
        DEFAULT_JUDGE_MODEL, "--judge-model", help="Judge model id (default: Claude)."
    ),
    judge_base_url: str = typer.Option(
        None,
        "--judge-base-url",
        help="OpenAI-compatible endpoint base URL; omit for the Anthropic/Claude default.",
    ),
    judge_cache: Path = typer.Option(
        Path(".membench_judge_cache.json"), "--judge-cache", help="On-disk judge response cache path."
    ),
    step_timeout: float = typer.Option(
        DEFAULT_STEP_TIMEOUT_S,
        "--step-timeout",
        help="Seconds each adapter call may take (retries included) before it is recorded "
        "as an error.",
    ),
    strict_cache: bool = typer.Option(
        False,
        "--strict-cache",
        help="Judge from the cache only (no API key needed); a cache miss is an error. "
        "Use to replay published results.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for every seedable RNG (recorded)."),
    pack_seed: int = _OPT["pack_seed"],
    exclude_flagged: bool = _OPT["exclude_flagged"],
    limit: int = _OPT["limit"],
    sample: int = _OPT["sample"],
    repeats: int = typer.Option(
        None, "--repeats", min=1,
        help="Run the pack N times, each on a fresh, reset backend namespace; the report holds "
        "every repeat plus mean and SD. Default: 3 when the backend, judge or reader uses an "
        "LLM, else 1.",
    ),
    repeat_index: int = typer.Option(
        0, "--repeat-index", help="Index of the (first) run among repeats of the same config "
        "(recorded)."
    ),
    reader_model: str = typer.Option(
        None,
        "--reader-model",
        help="Enable the reader stage: this model answers each query from the recalled "
        f"context (qa_accuracy, abstention). Key: ${READER_API_KEY_ENV} or ${API_KEY_ENV}.",
    ),
    reader_base_url: str = typer.Option(
        None,
        "--reader-base-url",
        help="OpenAI-compatible endpoint for the reader; omit for Anthropic/Claude.",
    ),
) -> None:
    """Replay a scenario pack against a backend adapter and write a JSON report."""
    try:
        loaded_pack = resolve_pack(
            pack, pack_seed=pack_seed, exclude_flagged=exclude_flagged, limit=limit,
            sample=sample, sample_seed=seed,
        )
        adapter_cls = get_adapter_class(backend)
        adapter_config = {"persist_dir": persist_dir, **_load_backend_config(backend_config)}
    except (
        PackValidationError, ValueError, TypeError, ImportError, OSError, yaml.YAMLError
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    judge_obj = None
    if judge:
        judge_obj = build_judge(
            model=judge_model, base_url=judge_base_url, cache_path=judge_cache,
            strict=strict_cache,
        )
        if judge_obj is None:
            typer.echo(
                f"Judge enabled but no API key in ${API_KEY_ENV}; "
                "semantic recall + contradiction rate will be null.",
                err=True,
            )

    reader = None
    if reader_model:
        client = resolve_client(
            model=reader_model, base_url=reader_base_url,
            api_key=os.environ.get(READER_API_KEY_ENV),
        )
        if client is None:
            typer.echo(
                f"Reader enabled but no API key in ${READER_API_KEY_ENV} or ${API_KEY_ENV}; "
                "qa_accuracy + abstention will be null.",
                err=True,
            )
        else:
            reader = Reader(client)

    if repeats is None:
        uses_llm = adapter_cls.uses_llm or judge_obj is not None or reader is not None
        repeats = 3 if uses_llm else 1
    results = []
    for i in range(repeats):
        for llm in (judge_obj, reader):  # each repeat's cost counts its own calls only
            if llm is not None:
                llm.usage = dict.fromkeys(llm.usage, 0)
        config = dict(adapter_config)
        if repeats > 1:  # a new namespace per repeat
            config["namespace"] = f"{config.get('namespace', 'membench')}-r{repeat_index + i}"
        try:
            adapter = adapter_cls(config)
            if reset:
                adapter.reset()
        except Exception as exc:  # missing extra, bad config, unreachable backend
            typer.echo(f"Error: could not set up backend {backend!r}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        try:
            results.append(run_pack(
                loaded_pack, adapter, backend_name=backend, judge=judge_obj,
                step_timeout=step_timeout, seed=seed, repeat=repeat_index + i,
                tokenizer=default_tokenizer(), reader=reader,
            ))
        except CacheMissError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    result = merge_repeats(results)
    write_json(result, out)
    typer.echo(f"Wrote report for pack={loaded_pack.name!r} backend={backend!r} to {out}")
    for r in results:
        if not r.valid:
            label = "run" if len(results) == 1 else f"repeat {r.manifest['repeat']}"
            typer.echo(
                f"Warning: {label} is invalid: error_rate "
                f"{r.scores['error_rate']:.0%} is above {MAX_ERROR_RATE:.0%} "
                "(see trace `error` fields).",
                err=True,
            )


@app.command()
def generate(
    transcripts: Path = typer.Option(
        ..., "--transcripts", help="Transcript file or directory to generate a pack from."
    ),
    out: Path = typer.Option(..., "--out", help="Path to write the generated pack YAML to."),
    pack_name: str = typer.Option(
        "generated", "--pack-name", help="Name for the generated pack."
    ),
    judge_model: str = typer.Option(
        DEFAULT_JUDGE_MODEL, "--judge-model", help="Generation model id (default: Claude)."
    ),
    judge_base_url: str = typer.Option(
        None,
        "--judge-base-url",
        help="OpenAI-compatible endpoint base URL; omit for the Anthropic/Claude default.",
    ),
) -> None:
    """Generate a schema-valid scenario pack YAML from a transcript via an LLM.

    Reuses the judge's provider-agnostic client config (model, base URL,
    `MEMBENCH_JUDGE_API_KEY`) from `core.scoring.judge`. Generation needs an
    API key; running the resulting pack afterwards does not (`--no-judge`).
    """
    client = resolve_client(model=judge_model, base_url=judge_base_url)
    if client is None:
        typer.echo(
            f"Error: pack generation needs an LLM; set ${API_KEY_ENV} "
            "(running the generated pack afterwards does not need a key: "
            "membench run --no-judge ...).",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        raw_pack = generate_pack(transcripts, client, pack_name=pack_name)
    except (GenerationError, OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    out.write_text(yaml.safe_dump(raw_pack, sort_keys=False))
    typer.echo(f"Wrote generated pack {raw_pack.get('name', pack_name)!r} to {out}")


_RENDERERS = {"md": render_markdown, "html": render_html}


@app.command()
def report(
    in_paths: list[Path] = typer.Option(
        ..., "--in", help="JSON report file(s) to render. Repeat --in for a comparison."
    ),
    format: str = typer.Option(
        "md", "--format", help="Output format: 'md' or 'html'."
    ),
    out: Path = typer.Option(..., "--out", help="Path to write the rendered report to."),
    failures: bool = typer.Option(
        False, "--failures", help="Per-query failure listing of one report (Markdown): not "
        "retrieved / stale surfaced / answered wrong / false abstention / forget violation.",
    ),
    excerpt_chars: int = typer.Option(
        200, "--excerpt-chars", min=0,
        help="Context excerpt length in the failure listing (0 = ids only, e.g. for LoCoMo).",
    ),
) -> None:
    """Render one or more JSON reports as a Markdown or HTML scorecard."""
    if failures:
        try:
            if len(in_paths) != 1 or format != "md":
                raise ValueError("--failures takes exactly one --in and --format md")
            report_data = json.loads(in_paths[0].read_text())
            text = render_failures(report_data, _report_pack(report_data, in_paths[0]),
                                   excerpt_chars)
        except (PackValidationError, ValueError, KeyError, OSError, yaml.YAMLError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        out.write_text(text)
        typer.echo(f"Wrote failure listing to {out}")
        return
    renderer = _RENDERERS.get(format)
    if renderer is None:
        typer.echo(
            f"Error: unknown format {format!r} (expected one of {sorted(_RENDERERS)})",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        reports = [json.loads(p.read_text()) for p in in_paths]
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    out.write_text(renderer(reports))
    typer.echo(f"Wrote {format} report from {len(reports)} run(s) to {out}")


_RESULTS_ARG = typer.Argument(Path("results"), help="Results root (published runs).")


@app.command()
def leaderboard(
    results: Path = _RESULTS_ARG,
    out: Path = typer.Option(
        None, "--out", help="Write the Markdown table here (default stdout)."
    ),
) -> None:
    """Backend × metric tables (value, 95% CI, n, cost per 1k queries, recall p95) from every
    committed result under RESULTS, one table per dataset, judge and reader."""
    try:
        entries = collect(results)
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    text = render_leaderboard(entries, (out or Path("x")).resolve().parent)
    if out is None:
        typer.echo(text)
        return
    out.write_text(text)
    typer.echo(f"Wrote leaderboard of {len(entries)} result(s) to {out}")


@app.command()
def compare(
    a_path: Path = typer.Option(..., "--a", help="JSON report of run A (the baseline)."),
    b_path: Path = typer.Option(..., "--b", help="JSON report of run B, on the same pack."),
    out: Path = typer.Option(
        None, "--out", help="Write here (.json for the raw rows, else Markdown); default stdout."
    ),
) -> None:
    """Paired comparison of two runs on the same queries: B − A with a paired bootstrap CI,
    McNemar's test for pass/fail metrics and Holm's correction across metrics × categories."""
    try:
        comparison = compare_reports(
            json.loads(a_path.read_text()), json.loads(b_path.read_text())
        )
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if out is not None and out.suffix == ".json":
        out.write_text(json.dumps(comparison, indent=2))
    else:
        text = render_compare_markdown(comparison)
        if out is None:
            typer.echo(text)
            return
        out.write_text(text)
    typer.echo(f"Wrote comparison of {comparison['a']} vs {comparison['b']} to {out}")


@app.command()
def rescore(
    in_path: Path = typer.Option(..., "--in", help="JSON report to re-score."),
    pack: str = typer.Option(
        None, "--pack", help="Pack name or YAML path (default: the manifest's pack name)."
    ),
    judge_cache: Path = typer.Option(
        Path(".membench_judge_cache.json"), "--judge-cache", help="Judge cache to replay."
    ),
    out: Path = typer.Option(None, "--out", help="Write the re-scored report here."),
    check: bool = typer.Option(
        False, "--check", help="Exit 1 if the re-scored scores differ from the report's."
    ),
    exclude_flagged: bool = typer.Option(
        False, "--exclude-flagged",
        help="Score without the dataset's flagged queries (shown as changes from the report).",
    ),
) -> None:
    """Recompute scores and CIs from a report's traces and the judge cache (no adapter, no
    network). Repeats in the report are re-scored too.

    A dataset pack is rebuilt from the manifest (`dataset` + `pack.subset`);
    its cached download is needed, or network access to fetch it again.
    """
    try:
        report_data = json.loads(in_path.read_text())
        loaded_pack = _report_pack(report_data, in_path, pack)
        judge_info = report_data["manifest"].get("judge")
        judge_obj = None
        if judge_info is not None:
            if judge_info["prompt_version"] != JUDGE_PROMPT_VERSION:
                raise ValueError(
                    f"run used judge prompt v{judge_info['prompt_version']}, this membench has "
                    f"v{JUDGE_PROMPT_VERSION}; re-run the judge instead"
                )
            judge_obj = build_judge(
                model=judge_info["model"], base_url=judge_info["base_url"],
                cache_path=judge_cache, strict=True,
            )
        kept = None
        if exclude_flagged:  # score the same run without the flagged queries
            if loaded_pack.dataset is None or loaded_pack.subset is not None:
                raise ValueError("--exclude-flagged needs a full (not --limit/--sample) "
                                 "dataset run")
            queries = {i for i, s in enumerate(loaded_pack.steps) if isinstance(s, QueryStep)}
            loaded_pack, kept = select_queries(loaded_pack, queries - set(loaded_pack.flagged))
        runs = [report_data, *report_data.get("repeats", [])]
        rescored = [_rescore_run(run, loaded_pack, judge_obj, kept) for run in runs]
    except (
        PackValidationError, ValueError, TypeError, KeyError, OSError, CacheMissError,
        yaml.YAMLError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    diffs = {}
    for run, new in zip(runs, rescored):
        prefix = f"repeat {run['manifest']['repeat']}: " if run is not report_data else ""
        for name, value in new["scores"].items():
            if run["scores"].get(name) != value:
                diffs[prefix + name] = (run["scores"].get(name), value)
        for key in ("by_category", "outcomes", "stats"):  # outcomes/stats: Phase 13+ reports
            if (key == "by_category" or key in run) and run.get(key) != new[key]:
                diffs[prefix + key] = (run.get(key), new[key])
    if out is not None:
        for run, new in zip(runs, rescored):
            if exclude_flagged:
                run["manifest"]["pack"]["sha256"] = loaded_pack.sha256
                run["manifest"]["dataset"]["options"]["exclude_flagged"] = True
            run.update(new)
        if len(runs) > 1:
            report_data["repeat_summary"] = summarize_repeats([new["scores"] for new in rescored])
        out.write_text(json.dumps(report_data, indent=2))
        typer.echo(f"Wrote re-scored report to {out}")
    if not diffs:
        typer.echo("Re-scored: identical to the report's scores.")
        return
    for name, (old, new) in diffs.items():
        typer.echo(f"Re-scored: {name} {old} -> {new}", err=True)
    if check:
        raise typer.Exit(code=1)


def _report_pack(report_data: dict, path: Path, pack: str | None = None) -> Pack:
    """The pack a report ran on (from `--pack` or its manifest), sha256-checked."""
    manifest = report_data.get("manifest") or {}
    if "pack" not in manifest:
        raise ValueError(f"{path} has no run manifest (made before membench 0.2.x)")
    dataset = manifest.get("dataset") or {}
    loaded_pack = resolve_pack(
        pack or dataset.get("id") or manifest["pack"]["name"],
        **dataset.get("options", {}), **(manifest["pack"].get("subset") or {}),
    )
    if loaded_pack.sha256 != manifest["pack"]["sha256"]:
        raise ValueError(
            f"pack {loaded_pack.name!r} content (sha256 {loaded_pack.sha256[:12]}…) differs "
            f"from the run's ({manifest['pack']['sha256'][:12]}…)"
        )
    return loaded_pack


def _rescore_run(run: dict, pack: Pack, judge: object, kept: list[int] | None) -> dict:
    """Re-scored fields of one run (the report or one of its repeats)."""
    traces = [StepTrace(**t) for t in run["traces"]]
    if kept is not None:
        traces = [traces[i] for i in kept]
    scoring = score_detailed(pack, traces, judge, seed=run["manifest"].get("seed", 0))
    error_rate = scoring.scores["error_rate"]
    return {
        "traces": [asdict(t) for t in traces],
        "scores": scoring.scores,
        "by_category": scoring.by_category,
        "outcomes": scoring.outcomes,
        "stats": scoring.stats,
        "valid": error_rate is None or error_rate <= MAX_ERROR_RATE,
    }


@app.command()
def publish(
    in_path: Path = typer.Option(..., "--in", help="JSON report of the run to publish."),
    results: Path = typer.Option(Path("results"), "--results", help="Results root."),
    judge_cache: Path = typer.Option(
        Path(".membench_judge_cache.json"), "--judge-cache",
        help="Judge cache the run wrote (only the entries it uses are published).",
    ),
    config: str = typer.Option(
        "default", "--config", help="Backend config label: 'default' or 'vendor-tuned' "
        "(its own leaderboard row)."
    ),
    lock: Path = typer.Option(Path("uv.lock"), "--lock", help="Lock file whose sha256 to record."),
) -> None:
    """Copy a run into `results/<date>/<dataset>/<backend>/` (report.json, judge_cache.json,
    meta.json), after checking that it re-scores identically offline."""
    try:
        report_data = json.loads(in_path.read_text())
        directory = result_dir(report_data, results, config)
        loaded_pack = _report_pack(report_data, in_path)
        judge_info = report_data["manifest"].get("judge")
        judge_obj = None
        if judge_info is not None:
            judge_obj = build_judge(
                model=judge_info["model"], base_url=judge_info["base_url"],
                cache_path=judge_cache, strict=True,
            )
            judge_obj.cache = RecordingCache(judge_cache)
        for run in [report_data, *report_data.get("repeats", [])]:
            new = _rescore_run(run, loaded_pack, judge_obj, None)
            if any(run.get(k) != new[k] for k in ("scores", "by_category", "outcomes", "stats")):
                raise ValueError(f"{in_path} does not re-score identically; run "
                                 f"`membench rescore --in {in_path} --check` to see the diff")
    except (
        PackValidationError, ValueError, TypeError, KeyError, OSError, CacheMissError,
        yaml.YAMLError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    meta = {"config": config, "uv_lock_sha256": sha256_file(lock), "trimmed": trim(report_data)}
    write_result(directory, report_data, judge_obj.cache.used if judge_obj else {}, meta)
    typer.echo(f"Published {in_path} to {directory}")


def _builtin_pack_names() -> list[str]:
    packs_dir = Path(__file__).parent / "packs"
    return sorted(p.stem for p in packs_dir.glob("*.yaml"))


@app.command()
def convert(
    pack: str = typer.Option(..., "--pack", help=_PACK_HELP),
    out: Path = typer.Option(..., "--out", help="Path to write the pack YAML to."),
    seed: int = typer.Option(0, "--seed", help="Seed for --sample."),
    pack_seed: int = _OPT["pack_seed"],
    exclude_flagged: bool = _OPT["exclude_flagged"],
    limit: int = _OPT["limit"],
    sample: int = _OPT["sample"],
    paraphrase_model: str = typer.Option(
        None, "--paraphrase-model",
        help=f"Rewrite each query with this LLM to cut its word overlap with the answer "
        f"(key: ${API_KEY_ENV}).",
    ),
    paraphrase_base_url: str = typer.Option(
        None, "--paraphrase-base-url", help="OpenAI-compatible endpoint for --paraphrase-model."
    ),
) -> None:
    """Write any pack (dataset, synthetic, bundled) as YAML, for local inspection.

    Dataset licenses still apply: never commit or redistribute converted
    dataset files (LoCoMo is CC BY-NC 4.0).
    """
    try:
        loaded_pack = resolve_pack(
            pack, pack_seed=pack_seed, exclude_flagged=exclude_flagged, limit=limit,
            sample=sample, sample_seed=seed,
        )
    except (PackValidationError, ValueError, OSError, yaml.YAMLError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    raw = pack_to_raw(loaded_pack)
    if paraphrase_model:
        client = resolve_client(model=paraphrase_model, base_url=paraphrase_base_url)
        if client is None:
            typer.echo(f"Error: --paraphrase-model needs ${API_KEY_ENV}.", err=True)
            raise typer.Exit(code=1)
        raw = paraphrase_queries(raw, client)
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    typer.echo(f"Wrote pack {loaded_pack.name!r} to {out} (local inspection only)")


_VALIDATION = Path("validation")


@app.command(name="judge-label")
def judge_label(
    annotator: str = typer.Option(..., "--annotator", help="Your annotator name."),
    sample: Path = typer.Option(_VALIDATION / "sample.jsonl", "--sample", help="Sample JSONL."),
    out: Path = typer.Option(
        None, "--out", help="Your annotator JSONL (default validation/annotations/NAME.jsonl)."
    ),
    metric: list[str] = typer.Option(None, "--metric", help="Only label this metric (repeat)."),
    seed: int = typer.Option(0, "--seed", help="Shuffle seed (combined with --annotator)."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", help="Port (0 = any free port)."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open a browser."),
) -> None:
    """Label judge items blind in a local browser page; saves every label immediately."""
    out = out or _VALIDATION / "annotations" / f"{annotator}.jsonl"
    try:
        store = LabelStore(read_jsonl(sample), annotator, out, metric or None, seed)
        server = make_server(store, host, port)
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    url = f"http://{host}:{server.server_address[1]}/"
    typer.echo(f"Labelling {len(store.items)} items as {annotator!r} ({len(store.labels)} done); "
               f"saving to {out}\nOpen {url}  (Ctrl-C to stop)")
    if not no_open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo(f"\nStopped. {len(store.labels)} labels in {out}")
    finally:
        server.server_close()


@app.command(name="judge-sample")
def judge_sample(
    in_paths: list[Path] = typer.Option(
        ..., "--in", help="Run report(s) to sample judge items from (one per backend/pack)."
    ),
    out: Path = typer.Option(_VALIDATION / "sample.jsonl", "--out", help="Sample JSONL."),
    n: int = typer.Option(
        DEFAULT_N_PER_METRIC, "--n", min=1, help="Items per judge metric (fewer if the pool "
        "is smaller)."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
    judge_cache: Path = typer.Option(
        Path(".membench_judge_cache.json"), "--judge-cache",
        help="Judge cache for each item's verdict (stratum `unjudged` when absent).",
    ),
    judge_model: str = typer.Option(
        DEFAULT_JUDGE_MODEL, "--judge-model",
        help="Judge whose cached verdicts to use when a report ran without a judge.",
    ),
) -> None:
    """Stratified sample of judge items (rubric or question+answer, evidence) for human
    labelling: per metric, round-robin over backend × verdict × context-length bucket."""
    cache = JudgeCache(judge_cache) if judge_cache.exists() else None
    items = []
    try:
        for path in in_paths:
            report_data = json.loads(path.read_text())
            loaded_pack = _report_pack(report_data, path)
            model = (report_data["manifest"].get("judge") or {}).get("model") or judge_model
            items += judge_items(report_data, loaded_pack, cache, model)
    except (PackValidationError, ValueError, TypeError, KeyError, OSError,
            yaml.YAMLError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    sample = stratified_sample(items, n, seed)
    write_jsonl(out, sample)
    typer.echo(f"Wrote {len(sample)} items from {len(items)} judge calls to {out}")
    summary = sample_summary(sample)
    for metric, strata in summary.items():
        typer.echo(f"  {metric}: {sum(strata.values())} items, {len(strata)} strata")
        if sum(strata.values()) < n:
            typer.echo(f"  Warning: {metric} has fewer than {n} distinct items", err=True)
    for metric in JUDGE_METRICS:
        if not any(m.split(":")[0] == metric for m in summary):
            typer.echo(f"  Warning: no {metric} items (qa_accuracy needs a run with "
                       "--reader-model)", err=True)


@app.command(name="judge-labels")
def judge_labels(
    sample: Path = typer.Option(_VALIDATION / "sample.jsonl", "--sample", help="Sample JSONL."),
    annotator: list[str] = typer.Option(
        ..., "--annotator", help="NAME=PATH of one annotator's JSONL ({item_id, label, note?}); "
        "repeat for each (>= 2).",
    ),
    adjudication: Path = typer.Option(
        None, "--adjudication", help="JSONL of adjudicated labels for disagreements."
    ),
    out: Path = typer.Option(_VALIDATION / "labels.jsonl", "--out", help="Labels JSONL."),
) -> None:
    """Merge independent annotators (and adjudication) into a validated labels file."""
    try:
        files = dict(spec.split("=", 1) for spec in annotator)
        lines = merge_annotations(
            read_jsonl(sample), {name: read_jsonl(path) for name, path in files.items()},
            read_jsonl(adjudication) if adjudication else None,
        )
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    write_jsonl(out, lines)
    pending = sum(gold(line) is None for line in lines)
    typer.echo(f"Wrote {len(lines)} labelled items to {out} ({pending} need adjudication)")


@app.command(name="judge-cross")
def judge_cross(
    sample: Path = typer.Option(_VALIDATION / "sample.jsonl", "--sample", help="Sample JSONL."),
    judges: Path = typer.Option(
        Path("configs/judges.yaml"), "--judges", help="Judge panel YAML (name, model, "
        "base_url, api_key_env)."
    ),
    judge_cache: Path = typer.Option(
        _VALIDATION / "judge_cache.json", "--judge-cache", help="Judge cache (all judges)."
    ),
    strict_cache: bool = typer.Option(
        False, "--strict-cache", help="Replay from the cache only; a miss is an error."
    ),
    out: Path = typer.Option(
        None, "--out", help=f"Verdicts JSONL (default validation/verdicts-v"
        f"{JUDGE_PROMPT_VERSION}.jsonl).",
    ),
) -> None:
    """Run every judge of the panel on the same sampled items (one verdict row each)."""
    out = out or _VALIDATION / f"verdicts-v{JUDGE_PROMPT_VERSION}.jsonl"
    try:
        items = read_jsonl(sample)
        panel = {c.name: build_panel_judge(c, judge_cache, strict_cache)
                 for c in load_panel(judges)}
        share_cache(panel, judge_cache)
        rows = run_judges(items, panel)
    except (ValueError, TypeError, OSError, CacheMissError, yaml.YAMLError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    write_jsonl(out, rows)
    errors = sum(r["passed"] is None for r in rows)
    typer.echo(f"Wrote {len(rows)} verdicts ({len(panel)} judges x {len(items)} items, "
               f"{errors} errors) to {out}")


@app.command(name="judge-agreement")
def judge_agreement(
    labels: Path = typer.Option(None, "--labels", help="Human labels JSONL."),
    verdicts: Path = typer.Option(None, "--verdicts", help="Judge verdicts JSONL (judge-cross)."),
    out: Path = typer.Option(
        None, "--out", help=f"Agreement JSON (default validation/agreement-v"
        f"{JUDGE_PROMPT_VERSION}.json); a Markdown copy is written next to it.",
    ),
) -> None:
    """Human–human κ, judge–human κ / accuracy / false-pass / false-fail (also by context
    length), judge–judge κ and disagreements; marks each judge metric validated or not."""
    if labels is None and verdicts is None:
        typer.echo("Error: give --labels and/or --verdicts", err=True)
        raise typer.Exit(code=1)
    try:
        result = agreement_report(load_labels(labels) if labels else None,
                                  read_jsonl(verdicts) if verdicts else None)
    except (ValueError, KeyError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    out = out or agreement_path(result["prompt_version"], _VALIDATION)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    out.with_suffix(".md").write_text(render_agreement_markdown(result))
    typer.echo(f"Wrote judge agreement (prompt v{result['prompt_version']}) to {out}")
    for name, info in sorted(result["judges"].items()):
        for metric, m in info["metrics"].items():
            typer.echo(f"  {name} {metric}: kappa {m['kappa']} n={m['n']} {m['status']}")


@app.command(name="judge-tune")
def judge_tune(
    labels: Path = typer.Option(_VALIDATION / "labels.jsonl", "--labels",
                                help="Human labels JSONL."),
    sample: Path = typer.Option(_VALIDATION / "sample.jsonl", "--sample", help="Sample JSONL."),
    prompt_version: list[str] = typer.Option(
        ..., "--prompt-version", help="Judge prompt version (repeat; first = baseline, "
        "last = candidate)."
    ),
    seed: int = typer.Option(0, "--seed", help="Dev / held-out split seed."),
    held_out: float = typer.Option(0.5, "--held-out", help="Held-out fraction per metric."),
    judge_model: str = typer.Option(DEFAULT_JUDGE_MODEL, "--judge-model", help="Judge model."),
    judge_base_url: str = typer.Option(None, "--judge-base-url",
                                       help="OpenAI-compatible base URL."),
    judge_cache: Path = typer.Option(_VALIDATION / "judge_cache.json", "--judge-cache",
                                     help="Judge cache (shared by all versions)."),
    strict: bool = typer.Option(False, "--strict/--no-strict",
                                help="Cached verdicts only, no API calls."),
    out: Path = typer.Option(None, "--out", help="Tuning JSON (default validation/tuning-v"
                             "<candidate>.json); a Markdown copy is written next to it."),
) -> None:
    """Compare judge prompt versions against human labels on a dev / held-out split."""
    versions = list(dict.fromkeys(prompt_version))
    unknown = [v for v in versions if v not in PROMPT_BUILDERS]
    if unknown or len(versions) < 2:
        problem = (f"unknown prompt version(s) {', '.join(unknown)}" if unknown
                   else "give at least two distinct --prompt-version values")
        typer.echo(f"Error: {problem} (known: {', '.join(sorted(PROMPT_BUILDERS))})", err=True)
        raise typer.Exit(code=1)
    if not 0 <= held_out <= 1:
        typer.echo(f"Error: --held-out must be in [0, 1], got {held_out}", err=True)
        raise typer.Exit(code=1)
    if not strict and resolve_client(model=judge_model, base_url=judge_base_url) is None:
        typer.echo(f"No ${API_KEY_ENV}: using cached verdicts only (as --strict).")
        strict = True
    try:
        label_lines = load_labels(labels)
        items = join_sample(label_lines, read_jsonl(sample))
        judges = build_version_judges(versions, judge_cache, strict=strict, model=judge_model,
                                      base_url=judge_base_url)
        verdicts, misses = run_versions(items, judges)
    except (ValueError, KeyError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if misses:
        for v, ids in misses.items():
            typer.echo(f"Error: v{v} lacks cached verdicts for {len(ids)} items (first: "
                       f"{', '.join(ids[:5])})", err=True)
        typer.echo(f"Set ${API_KEY_ENV} (without --strict) to judge them.", err=True)
        raise typer.Exit(code=1)
    report_data = tuning_report(label_lines, items, verdicts, seed=seed,
                                held_out_fraction=held_out, judge_model=judge_model,
                                base_url=judge_base_url,
                                usage={v: j.usage for v, j in judges.items()})
    out = out or _VALIDATION / f"tuning-v{report_data['candidate']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_data, indent=2) + "\n")
    out.with_suffix(".md").write_text(render_tuning_markdown(report_data))
    typer.echo(f"Wrote judge tuning ({len(versions)} versions, {report_data['n_items']} items) "
               f"to {out}")
    for v in versions:
        m = report_data["results"][v]["dev"]["overall"]
        typer.echo(f"  v{v} dev: kappa {m['kappa']} accuracy {m['accuracy']} n={m['n']} "
                   f"errors={m['errors']}")
    typer.echo(f"  held-out delta kappa (v{report_data['candidate']} - "
               f"v{report_data['baseline']}): "
               f"{report_data['deltas']['held_out']['overall']['kappa']}")


@app.command(name="list")
def list_() -> None:
    """List available packs (bundled, datasets, synthetic) and backend adapters."""
    typer.echo("Packs:")
    for name in _builtin_pack_names():
        typer.echo(f"  {name}")
    typer.echo("Datasets (downloaded on first use):")
    for name in DATASETS:
        typer.echo(f"  {name}")
    typer.echo("Synthetic (generated, --pack-seed):")
    for name in SYNTHETIC:
        typer.echo(f"  {name}")
    typer.echo("Backends:")
    for name in list_backends():
        typer.echo(f"  {name}")


if __name__ == "__main__":
    app()

"""membench CLI (Typer app)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import typer
import yaml

from membench.core.reader import READER_API_KEY_ENV, Reader
from membench.core.registry import get_adapter_class, list_backends
from membench.core.report.html import render_html
from membench.core.report.json import write_json
from membench.core.report.markdown import render_markdown
from membench.core.runner import (
    DEFAULT_STEP_TIMEOUT_S,
    MAX_ERROR_RATE,
    StepTrace,
    run_pack,
)
from membench.core.scoring import score_all
from membench.core.scoring.judge import (
    API_KEY_ENV,
    DEFAULT_JUDGE_MODEL,
    JUDGE_PROMPT_VERSION,
    CacheMissError,
    build_judge,
    resolve_client,
)
from membench.core.tokens import default_tokenizer
from membench.generate.generator import GenerationError, generate_pack
from membench.packs.schema import PackValidationError, builtin_pack_path, load_pack

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """membench: eval harness for agent memory backends."""


def _resolve_pack_path(pack: str) -> Path:
    candidate = Path(pack)
    if candidate.exists():
        return candidate
    return builtin_pack_path(pack)


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
    pack: str = typer.Option(
        ..., "--pack", help="Builtin pack name (e.g. 'mini') or path to a pack YAML file."
    ),
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
    repeat_index: int = typer.Option(
        0, "--repeat-index", help="Index of this run among repeats of the same config (recorded)."
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
        pack_path = _resolve_pack_path(pack)
        loaded_pack = load_pack(pack_path)
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

    try:
        adapter = adapter_cls(adapter_config)
        if reset:
            adapter.reset()
    except Exception as exc:  # missing extra, bad config, unreachable backend
        typer.echo(f"Error: could not set up backend {backend!r}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        result = run_pack(
            loaded_pack, adapter, backend_name=backend, judge=judge_obj,
            step_timeout=step_timeout, seed=seed, repeat=repeat_index,
            tokenizer=default_tokenizer(), reader=reader,
        )
    except CacheMissError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    write_json(result, out)
    typer.echo(f"Wrote report for pack={loaded_pack.name!r} backend={backend!r} to {out}")
    if not result.valid:
        typer.echo(
            f"Warning: run is invalid: error_rate {result.scores['error_rate']:.0%} "
            f"is above {MAX_ERROR_RATE:.0%} (see trace `error` fields).",
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
) -> None:
    """Render one or more JSON reports as a Markdown or HTML scorecard."""
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
) -> None:
    """Recompute scores from a report's traces and the judge cache (no adapter, no network)."""
    try:
        report_data = json.loads(in_path.read_text())
        manifest = report_data.get("manifest") or {}
        if "pack" not in manifest:
            raise ValueError(f"{in_path} has no run manifest (made before membench 0.2.x)")
        loaded_pack = load_pack(_resolve_pack_path(pack or manifest["pack"]["name"]))
        if loaded_pack.sha256 != manifest["pack"]["sha256"]:
            raise ValueError(
                f"pack {loaded_pack.name!r} content (sha256 {loaded_pack.sha256[:12]}…) differs "
                f"from the run's ({manifest['pack']['sha256'][:12]}…)"
            )
        judge_info = manifest.get("judge")
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
        traces = [StepTrace(**t) for t in report_data["traces"]]
        scores, by_category = score_all(loaded_pack, traces, judge_obj)
    except (
        PackValidationError, ValueError, TypeError, KeyError, OSError, CacheMissError,
        yaml.YAMLError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    diffs = {
        name: (report_data["scores"].get(name), value)
        for name, value in scores.items()
        if report_data["scores"].get(name) != value
    }
    if report_data.get("by_category") != by_category:
        diffs["by_category"] = (report_data.get("by_category"), by_category)
    if out is not None:
        error_rate = scores["error_rate"]
        report_data.update(
            traces=[asdict(t) for t in traces],
            scores=scores,
            by_category=by_category,
            valid=error_rate is None or error_rate <= MAX_ERROR_RATE,
        )
        out.write_text(json.dumps(report_data, indent=2))
        typer.echo(f"Wrote re-scored report to {out}")
    if not diffs:
        typer.echo("Re-scored: identical to the report's scores.")
        return
    for name, (old, new) in diffs.items():
        typer.echo(f"Re-scored: {name} {old} -> {new}", err=True)
    if check:
        raise typer.Exit(code=1)


def _builtin_pack_names() -> list[str]:
    packs_dir = Path(__file__).parent / "packs"
    return sorted(p.stem for p in packs_dir.glob("*.yaml"))


@app.command(name="list")
def list_() -> None:
    """List available builtin packs and backend adapters."""
    typer.echo("Packs:")
    for name in _builtin_pack_names():
        typer.echo(f"  {name}")
    typer.echo("Backends:")
    for name in list_backends():
        typer.echo(f"  {name}")


if __name__ == "__main__":
    app()

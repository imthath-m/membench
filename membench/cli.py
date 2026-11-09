"""membench CLI (Typer app)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from membench.core.registry import get_adapter_class, list_backends
from membench.core.report.html import render_html
from membench.core.report.json import write_json
from membench.core.report.markdown import render_markdown
from membench.core.runner import DEFAULT_STEP_TIMEOUT_S, MAX_ERROR_RATE, run_pack
from membench.core.scoring.judge import (
    API_KEY_ENV,
    DEFAULT_JUDGE_MODEL,
    build_judge,
    resolve_client,
)
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
        help="Seconds each write/recall may take (retries included) before it is recorded "
        "as an error.",
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
            model=judge_model, base_url=judge_base_url, cache_path=judge_cache
        )
        if judge_obj is None:
            typer.echo(
                f"Judge enabled but no API key in ${API_KEY_ENV}; "
                "semantic recall + contradiction rate will be null.",
                err=True,
            )

    try:
        adapter = adapter_cls(adapter_config)
        if reset:
            adapter.reset()
    except Exception as exc:  # missing extra, bad config, unreachable backend
        typer.echo(f"Error: could not set up backend {backend!r}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    result = run_pack(
        loaded_pack, adapter, backend_name=backend, judge=judge_obj, step_timeout=step_timeout
    )
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

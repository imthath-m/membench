"""membench CLI (Typer app)."""

from __future__ import annotations

from pathlib import Path

import typer

from membench.core.registry import get_adapter_class, list_backends
from membench.core.report.json import write_json
from membench.core.runner import run_pack
from membench.core.scoring.judge import API_KEY_ENV, DEFAULT_JUDGE_MODEL, build_judge
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
) -> None:
    """Replay a scenario pack against a backend adapter and write a JSON report."""
    try:
        pack_path = _resolve_pack_path(pack)
        loaded_pack = load_pack(pack_path)
        adapter_cls = get_adapter_class(backend)
    except (PackValidationError, ValueError, ImportError) as exc:
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

    adapter = adapter_cls({"persist_dir": persist_dir})
    result = run_pack(loaded_pack, adapter, backend_name=backend, judge=judge_obj)
    write_json(result, out)
    typer.echo(f"Wrote report for pack={loaded_pack.name!r} backend={backend!r} to {out}")


if __name__ == "__main__":
    app()

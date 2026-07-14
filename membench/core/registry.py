"""Resolve a `--backend` name to its adapter class."""

from __future__ import annotations

from membench.core.adapter import MemoryAdapter

_REGISTRY: dict[str, str] = {
    "full-context": "membench.adapters.full_context.FullContextAdapter",
    "chroma": "membench.adapters.chroma.ChromaAdapter",
}


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter_class(name: str) -> type[MemoryAdapter]:
    """Import and return the adapter class registered under `name`.

    Raises ValueError for an unknown name, and re-raises ImportError
    (with the adapter's own message) if the backend's dependency isn't
    installed.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend {name!r}. Available backends: {list_backends()}"
        )
    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)

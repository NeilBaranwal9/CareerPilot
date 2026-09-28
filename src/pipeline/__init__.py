from typing import Any

__all__ = [
    "PipelineRunner",
]


def __getattr__(name: str) -> Any:
    # Lazy import keeps `src.pipeline.schemas` importable from provider/source modules without import cycles.
    if name == "PipelineRunner":
        from src.pipeline.runner import PipelineRunner

        return PipelineRunner
    raise AttributeError(f"module 'src.pipeline' has no attribute {name!r}")

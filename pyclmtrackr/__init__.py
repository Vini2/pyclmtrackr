"""Python utilities for fitting clmtrackr facial models to still images."""

from .model import load_model

__all__ = ["CLMTracker", "FitResult", "load_model"]


def __getattr__(name):
    if name in {"CLMTracker", "FitResult"}:
        from .tracker import CLMTracker, FitResult

        return {"CLMTracker": CLMTracker, "FitResult": FitResult}[name]
    raise AttributeError(name)

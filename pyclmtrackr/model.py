"""Load packaged clmtrackr model data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union


DEFAULT_MODEL = Path(__file__).resolve().parent / "data" / "model_pca_20_svm.json"


def load_model(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load a clmtrackr model JSON file."""

    model_path = Path(path) if path is not None else DEFAULT_MODEL
    return json.loads(model_path.read_text(encoding="utf-8"))

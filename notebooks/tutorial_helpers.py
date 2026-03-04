from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def find_repo_root(start: Path | None = None) -> Path:
    """Locate repository root by searching for examples/synthetic_binary_workflow.py."""
    if start is None:
        start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "examples" / "synthetic_binary_workflow.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate examples/synthetic_binary_workflow.py"
    )


def load_synthetic_workflow_module(repo_root: Path | None = None):
    """Load examples/synthetic_binary_workflow.py as a module object."""
    if repo_root is None:
        repo_root = find_repo_root()
    module_path = repo_root / "examples" / "synthetic_binary_workflow.py"
    spec = spec_from_file_location("synthetic_binary_workflow", module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

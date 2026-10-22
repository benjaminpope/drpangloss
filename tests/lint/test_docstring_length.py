"""No docstring in ``src/virgil`` may exceed 60 lines.

Derivations belong in ``docs/`` and design history in ``design/``. See
"Docstring style" in AGENTS.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

MAX_LINES = 60
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def docstring_lengths() -> dict[str, int]:
    """Line count of every module, class and function docstring."""
    lengths = {}

    def visit(node, module, qualname):
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                name = [*qualname, child.name]
                doc = ast.get_docstring(child, clean=False)
                if doc:
                    lengths[f"{module}:{'.'.join(name)}"] = len(
                        doc.splitlines()
                    )
                visit(child, module, name)
            else:
                visit(child, module, qualname)

    for path in sorted(SRC.glob("virgil/**/*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree, clean=False)
        if doc:
            lengths[f"{module}:<module>"] = len(doc.splitlines())
        visit(tree, module, [])
    return lengths


def test_no_docstring_is_over_the_limit():
    over = sorted(
        f"{key} ({n} lines)"
        for key, n in docstring_lengths().items()
        if n > MAX_LINES
    )
    assert not over, (
        f"docstrings over {MAX_LINES} lines: {over}. Shorten them (derivations "
        "go in docs/, history in design/)."
    )

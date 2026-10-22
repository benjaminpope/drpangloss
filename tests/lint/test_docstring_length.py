"""No docstring in ``src/virgil`` may exceed 60 lines.

Docstrings already over the limit are listed in ``docstring_allowlist.txt``
(one ``module:qualname`` per line, sorted by module); the list may only
shrink. See "Docstring style" in AGENTS.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

MAX_LINES = 60
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
ALLOWLIST = Path(__file__).with_name("docstring_allowlist.txt")


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


def allowlist() -> list[str]:
    lines = ALLOWLIST.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def test_no_new_docstring_is_over_the_limit():
    allowed = set(allowlist())
    over = sorted(
        f"{key} ({n} lines)"
        for key, n in docstring_lengths().items()
        if n > MAX_LINES and key not in allowed
    )
    assert not over, (
        f"docstrings over {MAX_LINES} lines: {over}. Shorten them (derivations "
        "go in docs/, history in design/); new code may not join "
        "tests/lint/docstring_allowlist.txt."
    )


def test_the_allowlist_is_sorted_and_has_no_stale_entries():
    entries = allowlist()
    assert entries == sorted(entries, key=lambda e: e.split(":")[0]), (
        "docstring_allowlist.txt must be sorted by module"
    )
    assert len(entries) == len(set(entries)), "duplicate allowlist entries"
    lengths = docstring_lengths()
    stale = [e for e in entries if lengths.get(e, 0) <= MAX_LINES]
    assert not stale, (
        f"{stale} no longer exceed {MAX_LINES} lines (or no longer exist): "
        "delete them from tests/lint/docstring_allowlist.txt, which may "
        "only shrink."
    )

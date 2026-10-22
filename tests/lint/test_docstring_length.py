"""No docstring in ``src/virgil`` may exceed 60 lines.

Docstrings already over the limit are listed in ``docstring_allowlist.txt``
(one ``module:qualname N`` per line, sorted by that key; ``#`` starts a
comment). ``N`` is the docstring's current length, so a docstring can neither grow
nor shrink without its line being edited; the list may only shrink. See "Docstring style" in AGENTS.md.
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


def parse_lines(text: str) -> list[str]:
    """The non-blank, non-comment lines, as written."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_caps(text: str) -> dict[str, int]:
    """``{module:qualname: cap}``."""
    caps = {}
    for line in parse_lines(text):
        key, _, cap = line.rpartition(" ")
        assert key and cap.isdigit(), (
            f"bad allowlist line {line!r}: expected 'module:qualname N'"
        )
        caps[key] = int(cap)
    return caps


def allowlist_lines() -> list[str]:
    return parse_lines(ALLOWLIST.read_text(encoding="utf-8"))


def allowlist() -> dict[str, int]:
    return parse_caps(ALLOWLIST.read_text(encoding="utf-8"))


def test_no_new_docstring_is_over_the_limit():
    allowed = allowlist()
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


def test_an_allowlisted_docstring_does_not_grow():
    lengths = docstring_lengths()
    off = sorted(
        f"{key} ({lengths[key]} lines, cap {cap})"
        for key, cap in allowlist().items()
        if key in lengths and lengths[key] > MAX_LINES and lengths[key] != cap
    )
    assert not off, (
        f"allowlisted docstrings whose length is not their cap: {off}. A "
        "docstring may not grow; when one shrinks, set its cap to the new "
        "length in the same PR (or delete the line once it is within the limit)."
    )


def test_the_allowlist_is_sorted_and_has_no_stale_entries():
    entries = allowlist()
    keys = [line.rpartition(" ")[0] for line in allowlist_lines()]
    assert keys == sorted(keys), (
        "docstring_allowlist.txt must be sorted by key"
    )
    assert len(keys) == len(set(keys)), "duplicate allowlist entries"
    lengths = docstring_lengths()
    stale = [k for k in entries if lengths.get(k, 0) <= MAX_LINES]
    assert not stale, (
        f"{stale} no longer exceed {MAX_LINES} lines (or no longer exist): "
        "delete them from tests/lint/docstring_allowlist.txt, which may "
        "only shrink."
    )


def test_the_allowlist_parser_reads_caps_and_comments():
    text = "# why\n\na.b:C 70\n  # more\na.b:D.e 61\n"
    assert parse_caps(text) == {"a.b:C": 70, "a.b:D.e": 61}

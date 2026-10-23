"""Compare code-shape metrics of two source trees.

Usage: ``python scripts/code_shape_metrics.py OLD_DIR NEW_DIR [OLD_LABEL NEW_LABEL]``
(each directory holds ``.py`` files, e.g. ``src/virgil``). Prints, per tree:
single-use private defs (private names referenced at most once besides the
def), docstrings in the 25-40 line band, over 60 lines, and mean docstring
length. Reads only.
"""

import ast
import collections
import pathlib
import re
import sys


def stats(root):
    """Metrics of the ``.py`` files under ``root``."""
    docs, private, uses = [], [], collections.Counter()
    for f in sorted(pathlib.Path(root).rglob("*.py")):
        text = f.read_text()
        uses.update(re.findall(r"\b_\w+", text))
        for n in ast.walk(ast.parse(text)):
            if isinstance(
                n,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Module,
                ),
            ):
                d = ast.get_docstring(n, clean=False)
                if d is not None:
                    docs.append(d.count("\n") + 1)
            if (
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name.startswith("_")
                and not n.name.startswith("__")
            ):
                private.append(n.name)
    return dict(
        single_use=sum(uses[n] <= 2 for n in private),
        private_defs=len(private),
        docstrings_25_40=sum(25 <= k <= 40 for k in docs),
        docstrings_over_60=sum(k > 60 for k in docs),
        mean_docstring_lines=round(sum(docs) / max(1, len(docs)), 2),
    )


def main(argv):
    old, new = argv[0], argv[1]
    labels = argv[2:4] if len(argv) >= 4 else ("before", "after")
    a, b = stats(old), stats(new)
    print(f"| metric | {labels[0]} | {labels[1]} |\n|---|---:|---:|")
    for k in a:
        print(f"| {k.replace('_', ' ')} | {a[k]} | {b[k]} |")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Measure the shape of ``src/virgil``: private helpers, wrappers, duplicates.

Usage: ``python scripts/code_shape.py [--json out.json] [--extra DIR ...]``.
Reads only; prints a per-module table and the candidate lists used by
``design/code_shape_audit.md``. ``--extra`` directories (analysis scripts,
notebooks) are grepped for each private name.
"""

import argparse
import ast
import collections
import difflib
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "virgil"


def _strip_doc(body):
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _count(stmts):
    return sum(isinstance(n, ast.stmt) for s in stmts for n in ast.walk(s))


class _Norm(ast.NodeTransformer):
    """Rename arguments and locals by first use so the hash ignores names."""

    def __init__(self, names):
        self.map = {n: f"v{i}" for i, n in enumerate(names)}

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(self.map.get(node.id, node.id), node.ctx), node)

    def visit_arg(self, node):
        return ast.arg(self.map.get(node.arg, node.arg), None)


def _norm_hash(fn):
    names = []
    for n in ast.walk(fn):
        if isinstance(n, ast.arg) and n.arg not in names:
            names.append(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in names:
            names.append(n.id)
    clone = ast.FunctionDef(fn.name, fn.args, _strip_doc(fn.body), [], None, lineno=0, col_offset=0)
    clone = _Norm(names).visit(ast.fix_missing_locations(clone))
    clone.name = "f"
    clone.args.defaults = []
    return hashlib.md5(ast.dump(clone).encode()).hexdigest()[:10]


def _is_wrapper(fn):
    body = _strip_doc(fn.body)
    if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Call):
        return True
    return False


def collect():
    defs, uses, modules = [], collections.Counter(), {}
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        text = path.read_text()
        tree = ast.parse(text)
        modules[rel] = dict(lines=len(text.splitlines()), pub=0, priv=0, methods_pub=0, methods_priv=0)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                uses[node.id] += 1
            elif isinstance(node, ast.Attribute):
                uses[node.attr] += 1
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                private = node.name.startswith("_") and not node.name.startswith("__")
                modules[rel]["priv" if private else "pub"] += 1
                if isinstance(node, ast.ClassDef):
                    for m in node.body:
                        if isinstance(m, ast.FunctionDef):
                            mp = m.name.startswith("_") and not m.name.startswith("__")
                            modules[rel]["methods_priv" if mp else "methods_pub"] += 1
                elif private:
                    body = _strip_doc(node.body)
                    defs.append(dict(
                        module=rel, name=node.name, line=node.lineno, end=node.end_lineno,
                        stmts=_count(body), wrapper=_is_wrapper(node),
                        decorators=[ast.unparse(d) for d in node.decorator_list],
                        hash=_norm_hash(node), body=ast.unparse(ast.Module(body, [])),
                    ))
    return defs, uses, modules


def grep_extra(names, dirs):
    """Count whole-word occurrences of each name in non-src files."""
    pats = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    counts = collections.Counter()
    for d in dirs:
        for p in pathlib.Path(d).expanduser().rglob("*"):
            if p.suffix not in {".py", ".ipynb", ".md", ".toml", ".sh"} or not p.is_file():
                continue
            try:
                t = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for n, pat in pats.items():
                counts[n] += len(pat.findall(t))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    defs, uses, modules = collect()
    names = collections.Counter(d["name"] for d in defs)
    for d in defs:
        # one def = one occurrence in `uses` (its own FunctionDef is not a Name), so uses is call sites
        d["sites"] = uses[d["name"]] or len(d["decorators"])  # decorated: registered, e.g. defjvp
    extra = grep_extra({d["name"] for d in defs}, a.extra or [ROOT / "tests", ROOT / "docs", ROOT / "examples"])
    for d in defs:
        d["extra"] = extra[d["name"]]
    print(f"{'module':32s} {'lines':>6s} {'pub':>4s} {'priv':>5s} {'m_pub':>5s} {'m_priv':>6s}")
    for m, v in sorted(modules.items(), key=lambda kv: -kv[1]["priv"]):
        print(f"{m:32s} {v['lines']:6d} {v['pub']:4d} {v['priv']:5d} {v['methods_pub']:5d} {v['methods_priv']:6d}")
    print("\n(a) single-use, <=3 statements:")
    for d in defs:
        if d["sites"] == 1 and d["stmts"] <= 3:
            print(f"  {d['module']}:{d['line']} {d['name']} stmts={d['stmts']} wrapper={d['wrapper']}")
    print("\n(d) zero src uses:")
    for d in defs:
        if d["sites"] == 0:
            print(f"  {d['module']}:{d['line']} {d['name']} extra={d['extra']}")
    print("\n(b) identical normalized bodies (>=3 statements):")
    by_hash = collections.defaultdict(list)
    for d in defs:
        if d["stmts"] >= 3:
            by_hash[d["hash"]].append(d)
    for g in by_hash.values():
        if len({x["module"] for x in g}) > 1:
            print("  " + "; ".join(f"{x['module']}:{x['name']}" for x in g))
    print("\n(b) same name, several modules (similarity of bodies):")
    for n, c in names.items():
        if c > 1:
            g = [d for d in defs if d["name"] == n]
            sims = [difflib.SequenceMatcher(None, x["body"], y["body"]).ratio()
                    for i, x in enumerate(g) for y in g[i + 1:]]
            print(f"  {n}: {len(g)} defs, max sim {max(sims):.2f}, in {', '.join(d['module'] for d in g)}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(defs=defs, modules=modules), indent=1))


if __name__ == "__main__":
    main()

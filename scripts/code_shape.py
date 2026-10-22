"""Measure the shape of ``src/virgil``: private helpers, wrappers, duplicates.

Usage: ``python scripts/code_shape.py [--json out.json] [--extra DIR ...]
[--max-stmts 3] [--since REF]``.

Reads only. Prints a per-module table and the candidate lists of
``design/code_shape_audit.md``:

- (a) inline candidates (one call site, at most ``--max-stmts`` statements,
  undecorated, not mentioned in tests/docs/examples), and the keepers
  (decorated, or mentioned there);
- (b) identical normalized bodies, same-named helpers with their body
  similarity, and near-duplicates by line-shingle overlap;
- (d) private functions with no use.

Call sites are counted per module: the module's own ``Name`` uses, plus
``from x import _name`` and ``mod._name`` in other modules. A decorator that
registers the function (``@f.defjvp``, ``@x.register``) counts as a use.
``--extra`` adds directories (default tests, docs, examples) grepped for each
private name. ``--since REF`` lists only the private functions added since
``REF`` that meet the inline rule: the guard of the audit.
"""

import argparse
import ast
import collections
import copy
import difflib
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "virgil"
_REGISTERING = re.compile(r"\.(def\w*|register)$")


def _strip_doc(body):
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _count(stmts):
    return sum(isinstance(n, ast.stmt) for s in stmts for n in ast.walk(s))


def _private(name):
    return name.startswith("_") and not name.startswith("__")


class _Norm(ast.NodeTransformer):
    """Rename arguments and locals by first use so the hash ignores names."""

    def __init__(self, names):
        self.map = {n: f"v{i}" for i, n in enumerate(names)}

    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(self.map.get(node.id, node.id), node.ctx), node
        )

    def visit_arg(self, node):
        return ast.arg(self.map.get(node.arg, node.arg), None)


def _norm_hash(fn):
    fn = copy.deepcopy(fn)  # the transformer rewrites in place
    names = []
    for n in ast.walk(fn):
        if isinstance(n, ast.arg) and n.arg not in names:
            names.append(n.arg)
        elif (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id not in names
        ):
            names.append(n.id)
    fn.body = _strip_doc(fn.body)
    fn.decorator_list = []
    fn.args.defaults = []
    fn.name = "f"
    fn = _Norm(names).visit(fn)
    return hashlib.md5(ast.dump(fn).encode()).hexdigest()[:10]


def _is_wrapper(fn):
    body = _strip_doc(fn.body)
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Call)
    )


def collect():
    """Private functions (with call-site counts) and sizes of ``src/virgil``."""
    defs, modules = [], {}
    own = collections.Counter()  # (module, name): Name loads in that module
    imported = collections.Counter()  # name: ``from x import name`` anywhere
    attrs = collections.defaultdict(
        collections.Counter
    )  # module: attribute names
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        text = path.read_text()
        tree = ast.parse(text)
        modules[rel] = dict(
            lines=len(text.splitlines()),
            pub=0,
            priv=0,
            methods_pub=0,
            methods_priv=0,
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                own[rel, node.id] += 1
            elif isinstance(node, ast.Attribute):
                attrs[rel][node.attr] += 1
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported[alias.name] += 1
        for node in tree.body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            private = _private(node.name)
            modules[rel]["priv" if private else "pub"] += 1
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef):
                        key = (
                            "methods_priv"
                            if _private(m.name)
                            else "methods_pub"
                        )
                        modules[rel][key] += 1
            elif private:
                body = _strip_doc(node.body)
                defs.append(
                    dict(
                        module=rel,
                        name=node.name,
                        line=node.lineno,
                        end=node.end_lineno,
                        stmts=_count(body),
                        wrapper=_is_wrapper(node),
                        decorators=[
                            ast.unparse(d) for d in node.decorator_list
                        ],
                        hash=_norm_hash(node),
                        body=ast.unparse(ast.Module(body, [])),
                    )
                )
    for d in defs:
        d["sites"] = (
            own[d["module"], d["name"]]
            + imported[d["name"]]
            + sum(c[d["name"]] for m, c in attrs.items() if m != d["module"])
            + sum(bool(_REGISTERING.search(x)) for x in d["decorators"])
        )
    return defs, modules


def grep_extra(names, dirs):
    """Count whole-word occurrences of each name in non-src files."""
    pats = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    counts = collections.Counter()
    for d in dirs:
        for p in pathlib.Path(d).expanduser().rglob("*"):
            if (
                p.suffix not in {".py", ".ipynb", ".md", ".toml", ".sh"}
                or not p.is_file()
            ):
                continue
            try:
                t = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for n, pat in pats.items():
                counts[n] += len(pat.findall(t))
    return counts


def _shingles(body):
    lines = [
        re.sub(r"[A-Za-z_]\w*", "v", s.strip()) for s in body.splitlines()
    ]
    return set(zip(lines, lines[1:]))


def _functions_at(ref):
    """Names of the module-level functions of each module at git ``ref``."""
    out = {}
    for p in sorted(SRC.rglob("*.py")):
        rel = str(p.relative_to(SRC))
        git = subprocess.run(
            ["git", "show", f"{ref}:src/virgil/{rel}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        tree = (
            ast.parse(git.stdout)
            if git.returncode == 0
            else ast.Module([], [])
        )
        out[rel] = {
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--max-stmts", type=int, default=3)
    ap.add_argument("--since")
    a = ap.parse_args()
    defs, modules = collect()
    extra = grep_extra(
        {d["name"] for d in defs},
        a.extra or [ROOT / "tests", ROOT / "docs", ROOT / "examples"],
    )
    for d in defs:
        d["extra"] = extra[d["name"]]
    small = [d for d in defs if d["sites"] == 1 and d["stmts"] <= a.max_stmts]
    inline = [d for d in small if not d["decorators"] and not d["extra"]]
    if a.since:
        before = _functions_at(a.since)
        print(f"single-use private functions added since {a.since}:")
        for d in inline:
            if d["name"] not in before.get(d["module"], set()):
                print(f"  {d['module']}:{d['line']} {d['name']}")
        return
    print(
        f"{'module':32s} {'lines':>6s} {'pub':>4s} {'priv':>5s} {'m_pub':>5s} {'m_priv':>6s}"
    )
    for m, v in sorted(modules.items(), key=lambda kv: -kv[1]["priv"]):
        print(
            f"{m:32s} {v['lines']:6d} {v['pub']:4d} {v['priv']:5d} "
            f"{v['methods_pub']:5d} {v['methods_priv']:6d}"
        )
    print(
        f"\n(a) inline: one use, <={a.max_stmts} statements, undecorated, "
        f"unmentioned in tests/docs/examples ({len(inline)}):"
    )
    for d in inline:
        print(
            f"  {d['module']}:{d['line']} {d['name']} "
            f"stmts={d['stmts']} wrapper={d['wrapper']}"
        )
    print(
        "\n(a) keep: the same, but decorated (jit) or mentioned in tests/docs/examples:"
    )
    for d in small:
        if d not in inline:
            why = ", ".join(
                w
                for w, on in (("jit", d["decorators"]), ("tests", d["extra"]))
                if on
            )
            print(f"  {d['module']}:{d['line']} {d['name']} ({why})")
    print("\n(d) no use in src:")
    for d in defs:
        if d["sites"] == 0:
            print(
                f"  {d['module']}:{d['line']} {d['name']} extra={d['extra']}"
            )
    print("\n(b) identical normalized bodies (>=3 statements):")
    by_hash = collections.defaultdict(list)
    for d in defs:
        if d["stmts"] >= 3:
            by_hash[d["hash"]].append(d)
    for g in by_hash.values():
        if len({x["module"] for x in g}) > 1:
            print("  " + "; ".join(f"{x['module']}:{x['name']}" for x in g))
    print("\n(b) same name, several modules (body similarity):")
    names = collections.Counter(d["name"] for d in defs)
    for n, c in names.items():
        if c > 1:
            g = [d for d in defs if d["name"] == n]
            sims = [
                difflib.SequenceMatcher(None, x["body"], y["body"]).ratio()
                for i, x in enumerate(g)
                for y in g[i + 1 :]
            ]
            print(
                f"  {n}: {len(g)} defs, max sim {max(sims):.2f}, "
                f"in {', '.join(d['module'] for d in g)}"
            )
    print(
        "\n(b) near-duplicates by line-shingle overlap (>=8 statements, >=0.3):"
    )
    big = [d for d in defs if d["stmts"] >= 8]
    for d in big:
        d["sh"] = _shingles(d["body"])
    pairs = []
    for i, x in enumerate(big):
        for y in big[i + 1 :]:
            j = len(x["sh"] & y["sh"]) / max(1, len(x["sh"] | y["sh"]))
            if j >= 0.3:
                pairs.append((j, x, y))
    for j, x, y in sorted(pairs, key=lambda t: -t[0]):
        print(
            f"  {j:.2f} {x['module']}:{x['name']} ~ {y['module']}:{y['name']}"
        )
    if a.json:
        for d in defs:
            d.pop("sh", None)
        pathlib.Path(a.json).write_text(
            json.dumps(dict(defs=defs, modules=modules), indent=1)
        )


if __name__ == "__main__":
    main()

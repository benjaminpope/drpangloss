#!/usr/bin/env python3
"""Generate ``docs/references.md`` from ``references/provenance.toml``.

The TOML file is the single source of truth for where each idea or piece of
code in virgil came from. This script renders it as the public references
page: the narrative in ``references/references_intro.md``, one table per
module (symbol, source, ported or "after"), the sources that were compared
only, and an alphabetical list with one anchor per bibkey, so a docstring can
link ``../references.md#<bibkey>``. It also runs as a census::

    python3 scripts/sync_references_page.py              # write the page
    python3 scripts/sync_references_page.py --check      # fail if it is stale
    python3 scripts/sync_references_page.py --by-module  # census per module
    python3 scripts/sync_references_page.py --unverified # entries to verify

An entry is *unverified* when it has no arXiv ID or DOI, or ``verified`` is
not true. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "references" / "provenance.toml"
INTRO = ROOT / "references" / "references_intro.md"
PAGE = ROOT / "docs" / "references.md"

# Roles that are not a source of virgil's code or methods.
ROLES = ("compared", "context", "bibliography")

HEADER = (
    "<!-- AUTO-GENERATED from references/provenance.toml and "
    "references/references_intro.md by scripts/sync_references_page.py. "
    "Edit those, not this file. -->\n"
)


def load_refs(path: Path = REFS) -> dict[str, dict]:
    with open(path, "rb") as f:
        return tomllib.load(f)["ref"]


def is_verified(entry: dict) -> bool:
    """True when an arXiv ID or DOI is on record and was checked."""
    return bool(entry.get("arxiv") or entry.get("doi")) and bool(
        entry.get("verified")
    )


def relation(entry: dict) -> str:
    """``ported``, ``after``, ``compared``, ``context`` or ``bibliography``."""
    role = entry.get("role")
    if role in ROLES:
        return role
    return entry.get("how", "after")


def citation(entry: dict) -> str:
    """``Authors (year)`` as a short label."""
    year = entry.get("year")
    return f"{entry['authors']} ({year})" if year else entry["authors"]


def links(entry: dict) -> list[str]:
    """Markdown links to arXiv, ADS, the DOI and the repository."""
    out = []
    if entry.get("arxiv"):
        out.append(
            f"[arXiv:{entry['arxiv']}](https://arxiv.org/abs/{entry['arxiv']})"
        )
    if entry.get("bibcode"):
        out.append(
            f"[ADS](https://ui.adsabs.harvard.edu/abs/{entry['bibcode']}/abstract)"
        )
    if entry.get("doi"):
        out.append(f"[doi:{entry['doi']}](https://doi.org/{entry['doi']})")
    for field, label in (("repo", "GitHub"), ("url", "link")):
        url = entry.get(field)
        if url and url not in "".join(out):
            if field == "url" and "github.com" in url:
                label = "GitHub"
            out.append(f"[{label}]({url})")
    return out


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def split_used_in(item: str) -> tuple[str, str]:
    """``'virgil.models:A.b'`` -> ``('virgil.models', 'A.b')``."""
    module, _, symbol = item.partition(":")
    return module, symbol or "(module)"


def by_module(refs: dict[str, dict]) -> dict[str, list[tuple[str, str]]]:
    """``{module: [(symbol, bibkey), ...]}`` for entries with code."""
    rows = defaultdict(list)
    for key, entry in refs.items():
        if entry.get("role") in ROLES:
            continue
        for item in entry.get("used_in", []):
            module, symbol = split_used_in(item)
            rows[module].append((symbol, key))
    return {m: sorted(rows[m]) for m in sorted(rows)}


def source_cell(key: str, entry: dict) -> str:
    parts = [f"[{citation(entry)}](#{key})", *links(entry)]
    return _cell(", ".join(parts))


def render_module_tables(refs: dict[str, dict]) -> str:
    out = []
    for module, rows in by_module(refs).items():
        out += [
            f"### `{module}`",
            "",
            "| Symbol | Source | Relation |",
            "|---|---|---|",
        ]
        for symbol, key in rows:
            entry = refs[key]
            out.append(
                f"| `{symbol}` | {source_cell(key, entry)} | {relation(entry)} |"
            )
        out.append("")
    return "\n".join(out)


def render_role_section(refs: dict[str, dict], role: str) -> str:
    out = ["| Source | Note |", "|---|---|"]
    for key, entry in sorted(
        refs.items(), key=lambda kv: kv[1]["authors"].lower()
    ):
        if entry.get("role") == role:
            out.append(
                f"| {source_cell(key, entry)} | {_cell(entry.get('note', ''))} |"
            )
    return "\n".join(out) + "\n"


def render_heritage(refs: dict[str, dict]) -> str:
    """Sources with no code symbol that are neither compared nor bibliography."""
    out = ["| Source | Note |", "|---|---|"]
    for key, entry in sorted(
        refs.items(), key=lambda kv: kv[1]["authors"].lower()
    ):
        if entry.get("role") is None and not entry.get("used_in"):
            out.append(
                f"| {source_cell(key, entry)} | {_cell(entry.get('note', ''))} |"
            )
    return "\n".join(out) + "\n"


def render_all_sources(refs: dict[str, dict]) -> str:
    out = []
    for key, entry in sorted(
        refs.items(),
        key=lambda kv: (kv[1]["authors"].lower(), kv[1].get("year", 0)),
    ):
        line = f'- <a id="{key}"></a>**{citation(entry)}**. {entry["title"]}.'
        extra = links(entry)
        if extra:
            line += " " + ", ".join(extra) + "."
        if entry.get("note"):
            line += f" {entry['note'].rstrip('.')}."
        if not is_verified(entry):
            line += " *Not yet verified against arXiv, Crossref or ADS.*"
        out.append(line)
    return "\n".join(out) + "\n"


def render_page(refs: dict[str, dict], intro: str) -> str:
    sections = [
        HEADER,
        intro.rstrip() + "\n",
        "## Sources by module\n",
        "Each row is a place where code in virgil follows a paper or package. "
        "*ported*: code was adapted. *after*: the idea and equations were "
        "reimplemented from the paper or the package's documentation, with "
        "no code copied.\n",
        render_module_tables(refs),
        "## Compared, not ported\n",
        "Studied for comparison; nothing was taken from them.\n",
        render_role_section(refs, "compared"),
        "## Background literature\n",
        "Cited in the documentation as motivation or terminology; not a "
        "source of virgil's methods.\n",
        render_role_section(refs, "context"),
        "## Other background sources\n",
        "Heritage and context without a code symbol of their own.\n",
        render_heritage(refs),
        "## In the bibliography only\n",
        "Cited by the paper, not a source of virgil's methods.\n",
        render_role_section(refs, "bibliography"),
        "## All sources\n",
        "Alphabetical, one anchor per bibkey. Docstrings link here as "
        "`../references.md#<bibkey>`.\n",
        render_all_sources(refs),
    ]
    return "\n".join(sections)


def render_default() -> str:
    return render_page(load_refs(), INTRO.read_text(encoding="utf-8"))


def census_by_module(refs: dict[str, dict]) -> str:
    counts = defaultdict(lambda: defaultdict(set))
    for key, entry in refs.items():
        for item in entry.get("used_in", []):
            module, _ = split_used_in(item)
            counts[module]["all"].add(key)
            counts[module][relation(entry)].add(key)
            if not is_verified(entry):
                counts[module]["unverified"].add(key)
    cols = ["all", "ported", "after", *ROLES, "unverified"]
    width = max(len(m) for m in counts) + 2
    lines = [f"{'module':<{width}}" + "".join(f"{c:>13}" for c in cols)]
    for module in sorted(counts):
        lines.append(
            f"{module:<{width}}"
            + "".join(f"{len(counts[module][c]):>13}" for c in cols)
        )
    return "\n".join(lines)


def census_unverified(refs: dict[str, dict]) -> str:
    lines = []
    for key, entry in sorted(refs.items()):
        if is_verified(entry):
            continue
        why = (
            "no arXiv ID or DOI"
            if not (entry.get("arxiv") or entry.get("doi"))
            else "verified is not true"
        )
        lines.append(f"{key:<24} {citation(entry)[:40]:<42} {why}")
    lines.append(f"{len(lines)} of {len(refs)} entries unverified")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if stale")
    parser.add_argument("--by-module", action="store_true")
    parser.add_argument("--unverified", action="store_true")
    args = parser.parse_args(argv)
    if args.by_module or args.unverified:
        refs = load_refs()
        if args.by_module:
            print(census_by_module(refs))
        if args.unverified:
            print(census_unverified(refs))
        return 0
    page = render_default()
    current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else None
    if args.check:
        if current != page:
            print("docs/references.md is stale: run sync_references_page.py")
            return 1
        return 0
    if current == page:
        print("unchanged docs/references.md")
    else:
        PAGE.write_text(page, encoding="utf-8")
        print("synced docs/references.md <- references/provenance.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())

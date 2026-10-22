"""Every source cited in ``src/`` or ``docs/`` is in ``references/provenance.toml``.

Checks arXiv IDs, GitHub repositories and "Author et al. (YYYY)" or
"Author & Author (YYYY)" citations against the TOML (the single list behind
``docs/references.md``), and that each ``used_in`` symbol imports. A source
that cannot be checked on the arXiv abstract page stays ``verified = false``.
"""

from __future__ import annotations

import importlib
import re
import tomllib
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFS = tomllib.loads(
    (ROOT / "references" / "provenance.toml").read_text(encoding="utf-8")
)["ref"]

# Links that are not sources of ideas: the project's own repositories and the
# libraries virgil is built on (credited on the Contributions page).
SELF_AND_DEPENDENCY_REPOS = {
    "benjaminpope/virgil",
    "benjaminpope/virgil-validation",
    "benjaminpope/virgil-vlti",
    "benjaminpope/claude-code-carbon-dashboard",
    "louisdesdoigts/zodiax_tutorials",
    "jax-ml/jax",
    "google-deepmind/optax",
    "patrick-kidger/equinox",
    "patrick-kidger/lineax",
    "patrick-kidger/optimistix",
    "pyro-ppl/numpyro",
}

ARXIV = re.compile(
    r"(?:arXiv[:/ ]\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})", re.I
)
GITHUB = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)")
NAME = r"[A-ZÀ-ÿ][\w'’\-ß]+"
CITATION = re.compile(
    rf"\b({NAME})(?: et al\.?| (?:&|and) {NAME}),? \(?(\d{{4}})[a-z]?\)?"
)
# The generated page restates the TOML, so it cannot check it.
EXCLUDED = {"docs/references.md"}


def _files() -> list[Path]:
    paths = [*ROOT.glob("src/virgil/**/*.py"), *ROOT.glob("docs/*.md")]
    return sorted(p for p in paths if str(p.relative_to(ROOT)) not in EXCLUDED)


def _repo(url: str) -> str | None:
    m = GITHUB.search(url)
    return f"{m.group(1)}/{m.group(2)}".rstrip(".").lower() if m else None


def _ascii(text: str) -> str:
    text = text.replace("ß", "ss").replace("Enßlin", "Ensslin")
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )


def _surnames(authors: str) -> set[str]:
    authors = re.sub(r"\(.*?\)", "", authors)
    parts = re.split(r",|&| and |et al\.?| group", authors)
    return {_ascii(p.split()[-1]) for p in parts if p.strip()}


def _years(entry: dict) -> set[str]:
    years = {str(entry.get("year"))}
    if entry.get("bibcode"):
        years.add(entry["bibcode"][:4])
    years |= set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", entry.get("note", "")))
    return years


def _known_arxiv() -> set[str]:
    return {e["arxiv"] for e in REFS.values() if "arxiv" in e}


def _known_repos() -> set[str]:
    repos = {
        _repo(e[field])
        for e in REFS.values()
        for field in ("url", "repo")
        if field in e
    }
    return (repos - {None}) | SELF_AND_DEPENDENCY_REPOS


def _matches(surname: str, year: str) -> bool:
    return any(
        year in _years(e) and _ascii(surname) in _surnames(e["authors"])
        for e in REFS.values()
    )


def _scan(pattern: re.Pattern) -> list[tuple[str, tuple[str, ...]]]:
    hits = []
    for path in _files():
        rel = str(path.relative_to(ROOT))
        for m in pattern.finditer(path.read_text(encoding="utf-8")):
            hits.append((rel, m.groups()))
    return hits


def test_every_arxiv_id_is_in_the_toml():
    known = _known_arxiv()
    missing = sorted(
        {f"{rel}: {g[0]}" for rel, g in _scan(ARXIV) if g[0] not in known}
    )
    assert not missing, f"add these to references/provenance.toml: {missing}"


def test_every_github_repository_is_in_the_toml():
    known = _known_repos()
    missing = sorted(
        {
            f"{rel}: {g[0]}/{g[1]}"
            for rel, g in _scan(GITHUB)
            if f"{g[0]}/{g[1]}".rstrip(".").lower() not in known
        }
    )
    assert not missing, (
        f"add these to references/provenance.toml (url or repo) or, for the "
        f"project's own links, to SELF_AND_DEPENDENCY_REPOS: {missing}"
    )


def test_every_author_year_citation_is_in_the_toml():
    missing = sorted(
        {
            f"{rel}: {g[0]} ({g[1]})"
            for rel, g in _scan(CITATION)
            if not _matches(*g)
        }
    )
    assert not missing, f"add these to references/provenance.toml: {missing}"


def test_the_citation_matcher_reads_both_forms():
    for text in ("Kammerer et al. (2020)", "Lucy & Smith 2014, A&A"):
        assert CITATION.search(text)
    assert _matches("Kammerer", "2020") and not _matches("Kammerer", "1999")
    assert _matches("Ensslin", "2009") and _matches("Enßlin", "2009")


def _used_in_symbols():
    for key, entry in REFS.items():
        for item in entry.get("used_in", []):
            yield key, item


@pytest.mark.parametrize(
    ("key", "item"), sorted(_used_in_symbols()), ids=lambda v: str(v)
)
def test_used_in_symbols_import(key, item):
    module, _, symbol = item.partition(":")
    obj = importlib.import_module(module)
    for part in filter(None, symbol.split(".")):
        assert hasattr(obj, part), f"{key}: {item} does not import"
        obj = getattr(obj, part)

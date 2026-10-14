#!/usr/bin/env python3
"""Resolve ``references/provenance.toml`` against NASA ADS and export BibTeX.

Each entry is looked up by arXiv ID, then DOI, then bibcode, then an
author/year/title query. The script writes ``references/virgil.bib`` (ADS
BibTeX export), reports the bibcode and arXiv ID that ADS returns, and flags
every entry whose stored arXiv ID or DOI disagrees with ADS, or whose stored
first author does not match the record found. With ``--write-back`` the
bibcode and arXiv ID that ADS found are written into the TOML file; entries
that resolved cleanly also get ``verified = true``.

The ADS token is read from ``ADS_API_TOKEN`` or ``ADS_DEV_KEY``, else from
``~/.ads/dev_key``. Only ``requests`` and the standard library are used.

Usage::

    python3 scripts/fetch_references.py              # report and write the .bib
    python3 scripts/fetch_references.py --write-back # also update the TOML
"""

import argparse
import os
import re
import sys
import tomllib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "references" / "provenance.toml"
BIB = ROOT / "references" / "virgil.bib"
API = "https://api.adsabs.harvard.edu/v1"
FIELDS = "bibcode,identifier,doi,title,author,year"


def find_token():
    """Return the ADS token from the environment or ``~/.ads/dev_key``."""
    for name in ("ADS_API_TOKEN", "ADS_DEV_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    key = Path.home() / ".ads" / "dev_key"
    if key.exists():
        return key.read_text().strip()
    return None


def load_refs(path=REFS):
    """Read the provenance file and return ``{bibkey: entry}``."""
    with open(path, "rb") as f:
        return tomllib.load(f)["ref"]


def _quote(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _search(session, query):
    r = session.get(
        f"{API}/search/query",
        params={"q": query, "fl": FIELDS, "rows": 3},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["response"]["docs"]


def queries(entry):
    """Yield ``(how, query)`` pairs in order of reliability."""
    if entry.get("arxiv"):
        yield "arxiv", f"identifier:{_quote('arXiv:' + entry['arxiv'])}"
    if entry.get("doi"):
        yield "doi", f"doi:{_quote(entry['doi'])}"
    if entry.get("bibcode"):
        yield "bibcode", f"bibcode:{_quote(entry['bibcode'])}"
    first = re.split(r"[,&]| et al| and ", entry.get("authors", ""))[0].strip()
    surname = first.split()[-1] if first else ""
    if surname and entry.get("year") and entry.get("title"):
        title = re.sub(r'["\\]', " ", entry["title"])
        yield (
            "title",
            f'author:"^{surname}" year:{entry["year"]} title:{_quote(title)}',
        )


def _surname(authors):
    first = re.split(r"[,&]| et al| and ", authors)[0].strip()
    return first.split()[-1].lower() if first else ""


def resolve(entry, session):
    """Look an entry up in ADS.

    Returns ``(doc, how, problems)``: the first ADS record found (or None),
    the kind of query that found it, and a list of mismatch messages.
    """
    for how, query in queries(entry):
        docs = _search(session, query)
        if not docs:
            continue
        doc = docs[0]
        problems = []
        ids = [i.replace("arXiv:", "") for i in doc.get("identifier", [])]
        if entry.get("arxiv") and entry["arxiv"] not in ids:
            problems.append(f"stored arXiv {entry['arxiv']} not in ADS ids")
        if entry.get("doi") and entry["doi"].lower() not in [
            d.lower() for d in doc.get("doi", [])
        ]:
            problems.append(f"stored DOI {entry['doi']} not in ADS record")
        sur = _surname(entry.get("authors", ""))
        ads_first = (doc.get("author") or [""])[0].split(",")[0].lower()
        if (
            sur
            and ads_first
            and sur.replace("é", "e") not in ads_first.replace("é", "e")
        ):
            problems.append(f"first author {sur!r} vs ADS {ads_first!r}")
        return doc, how, problems
    return None, None, ["not found in ADS"]


def ads_arxiv_id(doc):
    """Return the arXiv ID in an ADS record, or None."""
    for i in doc.get("identifier", []):
        if i.startswith("arXiv:"):
            return i[len("arXiv:") :]
    return None


def export_bibtex(session, bibcodes):
    """Return ADS BibTeX for ``bibcodes`` as one string."""
    r = session.post(
        f"{API}/export/bibtex", json={"bibcode": list(bibcodes)}, timeout=120
    )
    r.raise_for_status()
    return r.json()["export"]


def write_back(path, found):
    """Insert ``bibcode``/``arxiv`` found by ADS into the TOML text, in place.

    ``found`` maps bibkey to ``{"bibcode", "arxiv", "verified"}``.
    """
    lines = Path(path).read_text().split("\n")
    out, key, buf = [], None, []

    def flush():
        if key in found:
            f = found[key]
            keep = [
                ln
                for ln in buf
                if not re.match(r"(bibcode|verified|verified_via)\s*=", ln)
                and not (f.get("arxiv") and re.match(r"arxiv\s*=", ln))
            ]
            while keep and not keep[-1].strip():
                keep.pop()
            keep.append(f"bibcode = {_quote(f['bibcode'])}")
            if f.get("arxiv"):
                keep.append(f"arxiv = {_quote(f['arxiv'])}")
            keep.append(f"verified = {'true' if f['verified'] else 'false'}")
            keep.append(
                f"verified_via = {_quote('ads' if f['verified'] else 'none')}"
            )
            keep.append("")
            out.extend(keep)
        else:
            out.extend(buf)

    for ln in lines:
        m = re.match(r"\[ref\.([^\]]+)\]", ln)
        if m:
            flush()
            key, buf = m.group(1).strip('"'), [ln]
        elif key is None:
            out.append(ln)
        else:
            buf.append(ln)
    flush()
    Path(path).write_text("\n".join(out))


def main(argv=None, session=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", default=str(REFS))
    ap.add_argument("--bib", default=str(BIB))
    ap.add_argument("--write-back", action="store_true")
    args = ap.parse_args(argv)
    if session is None:
        token = find_token()
        if not token:
            sys.exit(
                "No ADS token: set ADS_API_TOKEN or ADS_DEV_KEY, or write ~/.ads/dev_key"
            )
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {token}"
    refs = load_refs(args.refs)
    found, bibcodes, bad = {}, [], 0
    for key, entry in refs.items():
        doc, how, problems = resolve(entry, session)
        if doc is None:
            print(f"{key:24s} NOT FOUND ({entry.get('url', 'no url')})")
            bad += 1
            continue
        arxiv = ads_arxiv_id(doc)
        # a title-only hit is a guess, so it is not marked verified
        clean = not problems and how != "title"
        found[key] = {
            "bibcode": doc["bibcode"],
            "arxiv": arxiv,
            "verified": clean,
        }
        bibcodes.append(doc["bibcode"])
        flag = "" if not problems else "  MISMATCH: " + "; ".join(problems)
        bad += bool(problems)
        print(
            f"{key:24s} {doc['bibcode']:22s} arXiv:{arxiv or '-':16s} via {how}{flag}"
        )
    if bibcodes:
        Path(args.bib).write_text(export_bibtex(session, bibcodes))
        print(f"wrote {args.bib} ({len(bibcodes)} records)")
    if args.write_back:
        write_back(args.refs, found)
    print(f"{bad} entries need attention")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests of scripts/fetch_references.py with the ADS HTTP layer mocked."""

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fetch_references", ROOT / "scripts" / "fetch_references.py"
)
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

TOML = """[ref.good]
authors = "Dholakia & Pope"
year = 2025
title = "Analytic Interferometry of Rotating Stellar Surfaces"
arxiv = "2509.25433"
verified = false
used_in = []

[ref.wrong]
authors = "Kuramochi et al."
year = 2018
title = "TSV"
arxiv = "1802.07859"
verified = false
used_in = []
"""

DOCS = {
    "arXiv:2509.25433": {
        "bibcode": "2025arXiv250925433D",
        "identifier": ["arXiv:2509.25433"],
        "author": ["Dholakia, Shashank", "Pope, Benjamin"],
    },
    "arXiv:1802.07859": {
        "bibcode": "2018arXiv180207859S",
        "identifier": ["arXiv:1802.07859"],
        "author": ["Shah, Zubair"],
    },
}


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params["q"])
        for k, doc in DOCS.items():
            if k in params["q"]:
                return _Resp({"response": {"docs": [doc]}})
        return _Resp({"response": {"docs": []}})

    def post(self, url, json=None, timeout=None):
        return _Resp(
            {"export": "".join(f"@ARTICLE{{{b},}}\n" for b in json["bibcode"])}
        )


def test_resolve_flags_author_mismatch(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text(TOML)
    refs = fr.load_refs(p)
    s = FakeSession()
    doc, how, problems = fr.resolve(refs["good"], s)
    assert (
        how == "arxiv"
        and not problems
        and doc["bibcode"] == "2025arXiv250925433D"
    )
    doc, how, problems = fr.resolve(refs["wrong"], s)
    assert any("first author" in m for m in problems)


def test_main_writes_bib_and_write_back(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text(TOML)
    bib = tmp_path / "v.bib"
    rc = fr.main(
        ["--refs", str(p), "--bib", str(bib), "--write-back"],
        session=FakeSession(),
    )
    assert rc == 1  # the wrong entry needs attention
    assert "@ARTICLE{good," in bib.read_text()  # keyed by bibkey
    assert "{wrong," not in bib.read_text()  # mismatch kept out
    out = tomllib.loads(p.read_text())["ref"]
    assert (
        out["good"]["bibcode"] == "2025arXiv250925433D"
        and out["good"]["verified"]
    )
    assert out["wrong"]["verified"] is False and "bibcode" not in out["wrong"]


def test_provenance_file_is_well_formed():
    refs = fr.load_refs()
    for key, e in refs.items():
        assert e["authors"] and e["title"], key
        assert (
            e.get("arxiv") or e.get("doi") or e.get("url") or e.get("note")
        ), key
        if e["verified"]:
            assert e.get("arxiv") or e.get("doi") or e.get("bibcode"), key


def test_used_in_symbols_exist():
    """Each ``module:symbol`` named in used_in appears in that module's source."""
    src = ROOT / "src"
    for key, e in fr.load_refs().items():
        for ref in e["used_in"]:
            mod, _, sym = ref.partition(":")
            base = src / Path(*mod.split("."))
            path = (
                base.with_suffix(".py")
                if base.with_suffix(".py").exists()
                else base / "__init__.py"
            )
            assert path.exists(), f"{key}: no module {mod}"
            if sym:
                assert sym.split(".")[-1] in path.read_text(), f"{key}: {ref}"

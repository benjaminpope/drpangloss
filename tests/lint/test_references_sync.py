from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "sync_references_page", ROOT / "scripts" / "sync_references_page.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_references_page_is_synced_with_the_toml():
    module = _load_script()
    assert (ROOT / "docs" / "references.md").read_text(
        encoding="utf-8"
    ) == module.render_default(), (
        "docs/references.md is out of sync with references/provenance.toml: "
        "run scripts/sync_references_page.py."
    )


def test_every_bibkey_has_an_anchor_and_every_module_a_table():
    module = _load_script()
    refs = module.load_refs()
    page = module.render_default()
    for key in refs:
        assert f'<a id="{key}"></a>' in page, key
    for mod in module.by_module(refs):
        assert f"### `{mod}`" in page, mod


def test_census_modes_run(capsys):
    module = _load_script()
    assert module.main(["--by-module"]) == 0
    assert module.main(["--unverified"]) == 0
    out = capsys.readouterr().out
    assert "virgil.models" in out and "entries unverified" in out


def test_unverified_means_no_id_or_not_checked():
    module = _load_script()
    assert module.is_verified({"arxiv": "1", "verified": True})
    assert not module.is_verified({"arxiv": "1", "verified": False})
    assert not module.is_verified({"url": "x", "verified": True})

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sync_module(repo_root: Path):
    script_path = repo_root / "scripts" / "sync_tutorial_docs.py"
    spec = importlib.util.spec_from_file_location(
        "sync_tutorial_docs", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_markdown_is_synced_with_notebooks():
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_sync_module(repo_root)

    for nb_rel, doc_rel in module.MAPPINGS.items():
        nb_path = repo_root / nb_rel
        doc_path = repo_root / doc_rel

        expected = module.render_notebook_markdown(nb_path)
        actual = doc_path.read_text(encoding="utf-8")

        assert actual == expected, (
            f"{doc_rel} is out of sync with {nb_rel}. "
            "Run scripts/sync_tutorial_docs.py to regenerate tutorial docs."
        )

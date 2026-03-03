from __future__ import annotations

import json
from pathlib import Path

MAPPINGS = {
    "notebooks/binary_recovery_grid_hmc.ipynb": "docs/binary_recovery_grid_hmc.md",
    "notebooks/synthetic_data_file_roundtrip.ipynb": "docs/synthetic_data_file_roundtrip.md",
    "notebooks/contrast_limits_ruffio.ipynb": "docs/contrast_limits_ruffio.md",
}


def _to_md_source(cell_source: list[str] | str) -> str:
    if isinstance(cell_source, list):
        return "".join(cell_source).rstrip()
    return str(cell_source).rstrip()


def render_notebook_markdown(nb_path: Path) -> str:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    lines: list[str] = []
    lines.append(
        f"<!-- AUTO-GENERATED FROM {nb_path.as_posix()} by scripts/sync_tutorial_docs.py. -->"
    )
    lines.append("<!-- Edit the notebook, then re-run the sync script. -->")
    lines.append("")

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type")
        source = _to_md_source(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            lines.append(source)
            lines.append("")
        elif cell_type == "code":
            lines.append("```python")
            lines.append(source)
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for nb_rel, doc_rel in MAPPINGS.items():
        nb_path = repo_root / nb_rel
        doc_path = repo_root / doc_rel
        rendered = render_notebook_markdown(nb_path)
        doc_path.write_text(rendered, encoding="utf-8")
        print(f"synced {doc_rel} <- {nb_rel}")


if __name__ == "__main__":
    main()

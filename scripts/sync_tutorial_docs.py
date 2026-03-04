from __future__ import annotations

import base64
import json
from pathlib import Path

MAPPINGS = {
    "notebooks/binary_search.ipynb": "docs/binary_search.md",
    "notebooks/data_io.ipynb": "docs/data_io.md",
    "notebooks/contrast_limits.ipynb": "docs/contrast_limits.md",
}


def _to_md_source(cell_source: list[str] | str) -> str:
    if isinstance(cell_source, list):
        return "".join(cell_source).rstrip()
    return str(cell_source).rstrip()


def _to_text(value) -> str:
    if isinstance(value, list):
        return "".join(str(v) for v in value)
    return str(value)


def _output_text(output: dict) -> str:
    out_type = output.get("output_type")
    if out_type == "stream":
        return _to_text(output.get("text", ""))
    if out_type in {"execute_result", "display_data"}:
        data = output.get("data", {})
        if "text/plain" in data:
            return _to_text(data["text/plain"])
    if out_type == "error":
        return _to_text(output.get("traceback", ""))
    return ""


def _output_png_bytes(output: dict) -> bytes | None:
    if output.get("output_type") not in {"execute_result", "display_data"}:
        return None
    data = output.get("data", {})
    image_b64 = data.get("image/png")
    if image_b64 is None:
        return None
    image_b64 = _to_text(image_b64).replace("\n", "")
    return base64.b64decode(image_b64)


def render_notebook_markdown(nb_path: Path) -> str:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    repo_root = nb_path.resolve().parents[1]
    generated_dir = repo_root / "docs" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for old_img in generated_dir.glob(f"{nb_path.stem}_cell*_out*.png"):
        old_img.unlink()

    lines: list[str] = []
    lines.append(
        f"<!-- AUTO-GENERATED FROM {nb_path.as_posix()} by scripts/sync_tutorial_docs.py. -->"
    )
    lines.append("<!-- Edit the notebook, then re-run the sync script. -->")
    lines.append("")

    for cell_index, cell in enumerate(nb.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = _to_md_source(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            lines.append(source.strip())
            lines.append("")
        elif cell_type == "code":
            lines.append("```python")
            lines.append(source)
            lines.append("```")
            lines.append("")

            for output_index, output in enumerate(
                cell.get("outputs", []), start=1
            ):
                png_bytes = _output_png_bytes(output)
                if png_bytes is not None:
                    image_name = (
                        f"{nb_path.stem}_cell{cell_index:03d}"
                        f"_out{output_index:02d}.png"
                    )
                    image_path = generated_dir / image_name
                    image_path.write_bytes(png_bytes)
                    lines.append(
                        f"![{nb_path.stem} output {cell_index}.{output_index}]"
                        f"(generated/{image_name})"
                    )
                    lines.append("")
                    continue

                text_out = _output_text(output).rstrip()
                if text_out:
                    lines.append("```text")
                    lines.append(text_out)
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

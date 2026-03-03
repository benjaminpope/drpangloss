from __future__ import annotations

import base64
import json
from pathlib import Path

MAPPINGS = {
    "notebooks/binary_recovery_grid_hmc.ipynb": "docs/binary_recovery_grid_hmc.md",
    "notebooks/synthetic_data_file_roundtrip.ipynb": "docs/synthetic_data_file_roundtrip.md",
    "notebooks/contrast_limits_ruffio.ipynb": "docs/contrast_limits_ruffio.md",
}

INTRO_PARAGRAPHS = {
    "notebooks/binary_recovery_grid_hmc.ipynb": (
        "This tutorial walks through end-to-end binary recovery on synthetic "
        "interferometric observables, starting from a coarse likelihood grid and "
        "continuing through vanilla HMC and Fisher-reparameterized HMC. It "
        "emphasizes practical initialization, posterior diagnostics in Cartesian and "
        "polar coordinates, and posterior-predictive correlation checks so you can "
        "verify both numerical stability and physical consistency in one workflow."
    ),
    "notebooks/synthetic_data_file_roundtrip.ipynb": (
        "This tutorial demonstrates a full synthetic-data roundtrip: generate a "
        "realistic OIFITS product, reload it through `OIData`, and recover companion "
        "parameters with grid, HMC, and Fisher-HMC approaches under shared "
        "diagnostics. It focuses on reproducibility of file I/O, uncertainty "
        "conventions, and cross-checks that confirm recovered parameters remain "
        "consistent with the known truth model within expected posterior spreads."
    ),
    "notebooks/contrast_limits_ruffio.ipynb": (
        "This tutorial covers Ruffio-style contrast-limit estimation on interferometry "
        "data, including likelihood evaluation on a spatial grid, local uncertainty "
        "estimation via Laplace approximation, and conversion to practical upper-limit "
        "maps and radial summaries. It highlights finite-value diagnostics and "
        "visualization choices in both flux-ratio and Δmag units so sensitivity "
        "structure is interpretable for scientific comparison."
    ),
}


def _to_md_source(cell_source: list[str] | str) -> str:
    if isinstance(cell_source, list):
        return "".join(cell_source).rstrip()
    return str(cell_source).rstrip()


def _clean_markdown_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("This notebook mirrors"):
            continue
        if stripped.startswith("Notebook version of"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


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
    nb_rel = nb_path.relative_to(repo_root).as_posix()
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

    intro_inserted = False
    intro_text = INTRO_PARAGRAPHS.get(nb_rel)

    for cell_index, cell in enumerate(nb.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = _to_md_source(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            cleaned = _clean_markdown_text(source)
            if not cleaned:
                continue
            lines.append(cleaned)
            if (
                not intro_inserted
                and intro_text is not None
                and cleaned.lstrip().startswith("# ")
            ):
                lines.append("")
                lines.append(intro_text)
                intro_inserted = True
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

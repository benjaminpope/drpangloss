# AGENTS.md

Guidance for AI coding agents (Copilot, Claude Code, and similar) working in this
repository. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

## Project overview

`drpangloss` fits interferometric (AMI/OIFITS) data with JAX- and zodiax-based models.
Library code lives in `src/drpangloss/`, tutorials in `notebooks/`, reusable end-to-end
scripts in `examples/`, and tests in `tests/`.

## Setup

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
pre-commit install
```

Cloud agents get this environment from `.github/workflows/copilot-setup-steps.yml`.

## Commands

| Task | Command |
| --- | --- |
| Run tests | `uv run --python .venv/bin/python pytest` |
| Run one test | `uv run --python .venv/bin/python pytest tests/test_models_core.py::test_name` |
| Lint (check only) | `bash scripts/lint_local.sh` |
| Lint (apply fixes) | `bash scripts/lint_local.sh --fix` |
| Lint changed files only | `bash scripts/lint_local.sh --changed --fix` |
| Full pre-commit pass | `uv run --python .venv/bin/python pre-commit run --all-files` |
| Regenerate tutorial docs | `uv run --python .venv/bin/python scripts/sync_tutorial_docs.py` |
| Build docs | `uv run --python .venv/bin/python mkdocs build --strict` |
| Build docs (zensical) | `uv run --python .venv/bin/python zensical build --clean` |

## Definition of done

Before committing:

1. `bash scripts/lint_local.sh --fix`
2. `uv run --python .venv/bin/python pytest`
3. If you touched a tutorial notebook: rerun `scripts/sync_tutorial_docs.py` and
   `pytest tests/test_tutorial_docs_sync.py`.
4. If you touched docstrings or `docs/`: `mkdocs build --strict`.

CI also autofixes formatting on same-repo PRs (`.github/workflows/lint.yml`), but do not
rely on it — a clean diff keeps review focused on the actual change.

## Conventions

- Ruff is pinned to **0.11.0**; `[tool.ruff] required-version`, the `.pre-commit-config.yaml`
  rev, and `RUFF_VERSION` in the workflows must always match. A different ruff version will
  reformat files differently and fail CI.
- Line length 79, double quotes, rules `E` + `F` (see `pyproject.toml` for ignores).
- JAX runs in float64 (`jax.config.update("jax_enable_x64", True)`); keep it that way in
  new notebooks and tests.
- New model code goes in `src/drpangloss/models.py`.

## Image coordinate convention

**This convention must never be violated.** It has been the direct cause of real
bugs (see below), and any new coordinate, rendering, or plotting code must be
verified against it with a direct orientation test — not just a
finiteness/normalization check.

- Rendered images and `(dra, ddec)` grids are 2D arrays using standard array
  indexing: the first index (row) runs top-to-bottom, the second index (column)
  runs left-to-right. That part is generic.
- The **physical sky coordinates** assigned to those array positions follow
  astronomical convention, not a generic image convention: **East points left,
  North points up**.
  - `x` (a right-ascension-like offset, e.g. `dra`) **increases to the left**
    (East is positive `x`).
  - `y` (a declination-like offset, e.g. `ddec`) **increases upward** (North
    is positive `y`).
  - Consequently: as the **row index increases, `y` decreases** (row 0 =
    North, the top row). As the **column index increases, `x` decreases**
    (column 0 = East, the leftmost column).
- The coordinate origin `(x, y) = (0, 0)` sits at the **geometric center of
  the pixel grid** — the center of the center pixel(s), not a pixel edge.
- **Position angles** (binary separations, an ellipse's projected major axis,
  azimuthal modulation phases) are measured **counter-clockwise from the top**,
  i.e. **North-to-East**: PA=0° points North, PA=90° points East.

The canonical, tested reference implementation is `_image_coordinates` in
`src/drpangloss/models.py` (image-plane pixel coordinates) and the elliptical
rotation/stretch helpers in `src/drpangloss/_utils.py` (`undo_`/`apply_elliptical_transf_coord`
and `..._spat_freq`). Every geometric `SourceModel`'s `model()`/`render()` must
be dimensionally consistent with these, and should have a direct regression
test analogous to `test_gaussian_disk_render_uses_interferometric_image_orientation`,
`test_uniform_disk_render_uses_interferometric_image_orientation`, or
`test_modulated_gaussian_rim_render_follows_north_to_east_pa_convention` in
`tests/test_models_sources.py` — i.e. one that asserts flux/argmax lands at the
*correct* pixel for a known offset or PA, not just that the output is finite.

**Plotting rule:** any `imshow`-based display of a sky-coordinate image or grid
must end up with the x-axis increasing toward the left (East) and the y-axis
increasing toward the top (North) *regardless of how the underlying
array/extent/origin was constructed*. Do not assume a particular `dra`/`ddec`
axis ordering (ascending vs. descending) — different parts of this codebase
build these axes both ways. Use `_enforce_sky_orientation` in
`src/drpangloss/plotting.py`, which corrects an `Axes`' final displayed limits
regardless of the plotted array's construction, rather than hand-tuning
`origin`/`extent` per call site.

Historical motivation: this exact convention was violated in three independent
places found in one audit — `BinaryModelAngular`'s position angle was mirrored
about the North-South axis (a real bug in `model()`/`render()`, not just
display); `notebooks/source_models.ipynb`'s render plots used `origin="lower"`
with a non-reversed extent; and two of `plotting.py`'s five grid-plotting
functions displayed `dra` backwards. None of these were caught by existing
tests because none of them asserted orientation directly — see
`add_modulated_gaussian_rim_and_uniform_disk`'s commit history for the fixes.

## Do not modify

- `site/` — committed build output, regenerated by the docs tooling.
- `docs/generated/` and `data/*.npy` — generated or fixture data.
- `docs/*.md` pages listed in `scripts/sync_tutorial_docs.py::MAPPINGS` — generated from
  notebooks (see below).
- `notebooks/ami_exploration.ipynb`, `notebooks/louis_visibilities.ipynb` — excluded from
  linting on purpose.
- `.venv/`, `.lint-logs/`.

## Notebooks and docs are coupled

The notebooks listed in `MAPPINGS` in `scripts/sync_tutorial_docs.py` are the source of
truth for the corresponding `docs/*.md` pages. Edit the notebook, execute it so outputs
are current (the sync embeds text and PNG outputs), then run the sync script.
`tests/test_tutorial_docs_sync.py` fails if the markdown is stale.

## Testing notes

`pytest` is preconfigured with `-q` and `testpaths = ["tests"]`. The JAX suites are slow to
warm up, so iterate with a single test id and run the full suite once at the end.

## Pull requests

- Keep diffs small and focused on the request.
- Do not commit notebook output churn unrelated to your change.
- Do not add runtime dependencies to `[project].dependencies` without asking.
- Never use `--no-verify`, never rewrite published history, never commit secrets.

# DrPangloss Review Notes (Summary)

Date: 2026-03-03

This file tracks implementation progress for low-risk modernization work and links to detailed notes in `docs/review_2026-03-03.md`.

## Scope
- Uncontroversial documentation, comments, tests, CI hygiene, and correctness bug fixes.
- Defer architecture-heavy refactors (composable models, generalized observables, kernel phase/DISCO operators) to staged follow-up PRs.

## Initial Findings (from full repo audit)
- Phase/closure handling correctness issues in `src/drpangloss/models.py`.
- Likelihood semantics and naming ambiguity (`loglike` currently returns only residual term).
- Fragile broad exception handling in `OIData.__init__`.
- Test suite is monolithic and includes plotting-heavy integration paths.
- Docs and contribution guidance are stale/placeholder in parts.
- CI workflows use older GitHub Action versions and minimal trigger matrix.

## Execution Log
- 2026-03-03: Started low-risk implementation batch.
- 2026-03-03: Added persistent review notes files.
- 2026-03-03: Applied low-risk docs/metadata fixes (`README.md`, `docs/index.md`, `CONTRIBUTING.md`, `mkdocs.yml`, `pyproject.toml`).
- 2026-03-03: Modernized CI workflow basics in `.github/workflows/tests.yml` and `.github/workflows/documentation.yml`.
- 2026-03-03: Applied correctness fixes in `src/drpangloss/models.py` for `cp_flag`, phase conversion consistency, and safer dict-vs-OIFITS initialization.
- 2026-03-03: Added targeted regression tests in `tests/test_regressions.py`.
- 2026-03-03: Switched local execution setup to `uv` with Python 3.11 (`.venv`), and validated targeted regressions via `uv run --python .venv/bin/python pytest -q tests/test_regressions.py`.
- 2026-03-03: Added `.python-version` (`3.11`) and updated setup docs for `uv` workflow.
- 2026-03-03: Reworked `chi2ppf` in `src/drpangloss/models.py` to use a JAX-native closed-form path for `df=1` (used by `nsigma`), avoiding brittle `tensorflow_probability` import failures in Python 3.11/JAX 0.9 environments.
- 2026-03-03: Added regression test coverage for `chi2ppf(..., df=1)` finite behavior.
- 2026-03-03: Validation run in uv/Python 3.11: `15 passed` (`tests/test_regressions.py` + `tests/tests.py`).
- 2026-03-03: Added pytest configuration in `pyproject.toml` so default test discovery includes both `test_*.py` and legacy `tests.py`.
- 2026-03-03: Updated CI workflows to use `uv` + Python 3.11 environments for reproducible test/docs execution.
- 2026-03-03: Split legacy `tests/tests.py` into focused test modules with shared data fixture helper (`tests/_test_data.py`, `tests/test_models_core.py`, `tests/test_grid_and_limits.py`) and removed the monolithic file.
- 2026-03-03: Added/cleaned docstrings across source modules (including legacy `models_old.py`) and verified zero missing docstrings via AST audit.
- 2026-03-03: Added API docs page (`docs/api.md`) and updated MkDocs nav/config to render API docstrings from source.
- 2026-03-03: Local validation (uv/Python 3.11): `pytest` -> 15 passed; `mkdocs build --strict` -> pass.
- 2026-03-03: Added lightweight, notebook-derived docs tutorials using synthetic data (no large uploads required):
	- `docs/binary_recovery_grid_hmc.md`
	- `docs/contrast_limits_ruffio.md`
- 2026-03-03: Updated docs navigation in `mkdocs.yml` and validated with strict build.
- 2026-03-03: Added synthetic OIFITS roundtrip docs and executable workflow:
	- `docs/synthetic_data_file_roundtrip.md`
	- `docs/examples/synthetic_binary_workflow.py`
	- output location scaffold: `docs/generated/.gitkeep`
- 2026-03-03: Added CI-enforced synthetic docs recovery test (`tests/test_docs_synthetic_recovery.py`) that fails on runtime errors or >2σ inconsistency vs truth.
- 2026-03-03: Added `termcolor` dependency required by `oifits_implaneia.save` import path.
- 2026-03-03: Clarified and reduced warning noise:
	- restricted `mkdocs-jupyter` conversion scope,
	- moved docs helper script to `examples/synthetic_binary_workflow.py` to avoid conversion warnings,
	- fixed `oifits_implaneia` invalid escape-sequence warnings,
	- fixed Simbad field-name deprecation (`sp_type`),
	- filtered external Optax/JAX config deprecation in pytest output.

## Deferred Major Work
- General observable schema replacing `cp_flag`/`v2_flag` branching.
- Kernel-phase and DISCO matrix-operator support.
- Composable source/system model framework and new source types (Gaussian disk, rings).
- Parameter-axis decoupling in grid search APIs.

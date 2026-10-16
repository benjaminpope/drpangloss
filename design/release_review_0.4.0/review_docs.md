# virgil 0.4.0 documentation review (origin/main 420d1b7)

Scope: README, docs/ (pages, nav, api), CHANGELOG, CONTRIBUTING, CITATION.cff, MIGRATION_VIRGIL.md, pyproject metadata, tutorial notebooks rendered into docs. Read-only; nothing in the repo was edited (the checkout stayed clean after the build and the two read-only scripts I ran).

## What was checked and passed

- `zensical build --clean` and `--strict` (the Pages CI uses `--strict`): "No issues found". zensical reads `mkdocs.yml`. Every `docs/**/*.md` except `docs/misc/README.md` is in the nav (see D18). No nav page is missing.
- All internal relative links and all `page.md#anchor` links in docs resolve (own script against the built `site/`). Rendered HTML has no unresolved `[`x`][virgil.y]` autorefs except D7.
- Deprecated names: AST check of every code cell in `docs/*.md`, all `notebooks/*.ipynb` (incl. `mwe/`), `examples/*.py` against the live signatures (no bad keywords anywhere). Grep for `samples_dict`, `data_obj=`, `model_object`, `model_fn=`, `observations=`, `ud=`, `drpangloss`, `regularis*` keyword uses: none in docs calls. Every call to the 22 reordered functions (`loglike`, `joint_loglike`, `likelihood_grid`, `numpyro_model`, `laplace_cov`, `posterior_predictive_summary`, ...) in docs, notebooks and examples is already in model-before-data order. The only `data_obj` left is a user-defined function argument in `examples/synthetic_binary_workflow.py` (D19).
- Spot-checked about 20 prose claims against source, all correct: pipeline stage names and tuple order, `BinaryPipeline`/`StarPipeline` defaults, CLI subcommands and flags, check thresholds (`r_hat` 1.01/1.05, `ess` 400/100), 8192-coordinate Gauss-Newton limit, `fit_orbit_aliases` new defaults (`sign_free=True`, `primary_diam="auto"`, `vis_amp=True`, `flip_flux=True`, `n_bands_max=200`), `image_field`/`field_of_view` signatures, `EnsembleSpec.field_factors=(0.75, 1.0, 1.5)` (page), `scales="quoted"` default with FutureWarning, `StarPipeline` registry names, "As a pipeline" sections in the three tutorials, the named regression test in conventions.md, the `OIData.epochs`/`Attached`/`orientation_priors` references, README's jaxoplanet lazy-import claim.
- Every name in backticks in the Unreleased section exists in `src` (apart from deliberately removed ones). No design/*.md is linked from docs, README or CONTRIBUTING.
- External links: 154 checked with curl; only one genuinely dead (D6). 403/405 are publisher bot blocks (doi.org, ADS).

## Findings

### D1. Version strings still 0.3.0; Unreleased heading
- severity: should-fix-before-0.4.0 (release checklist)
- `pyproject.toml:3` `version = "0.3.0"`; `CITATION.cff:6` `version: 0.3.0`; `CHANGELOG.md:8` `## Unreleased` (0.3.0 entry is `## 0.3.0 (2026-10-06)`).
- fix: bump pyproject and CITATION.cff, rename heading to `## 0.4.0 (YYYY-MM-DD)`, add `date-released:` to CITATION.cff.
- effort: 5 min.

### D2. CITATION.cff `doi: "TBD"` is invalid
- severity: should-fix-before-0.4.0
- `CITATION.cff:9` `doi: "TBD"  # ... to be added at release`. cffconvert and GitHub's "Cite this repository" validate `doi` against a DOI pattern; a non-DOI string makes the file fail validation. Authors order also differs from README (`README.md:7` Blakely, Pope, Desdoigts, Dholakia, De Prins, Goldfine, Charles) and pyproject (Pope, Blakely, Dholakia, Charles, De Prins, Desdoigts, Goldfine): three different orders; decide one.
- fix: delete the `doi` line until a Zenodo DOI exists (or mint it for the release tag); run `cffconvert --validate`; align author order or say deliberately that they differ.
- effort: 10 min.

### D3. CHANGELOG Unreleased is incomplete for user-visible changes
- severity: should-fix-before-0.4.0
- Merged since v0.3.0 and absent from the changelog (grep of `CHANGELOG.md` Unreleased, 0 hits each):
  - #370: `imaging.chi2_by_observable`, `short_baseline_residual`, `residual_image` (per-observable chi2, short-baseline residual, residual image).
  - #371: `ensemble.evidence_grid`, `gp_members`, `GPFit`, `EvidenceGrid`, and the new `EnsembleSpec` fields `gp_sigmas`, `gp_lengths`, `gp_sigma_limits`, `gp_length_limits`, `gp_growths`, `gp_window_nats`, `gp_axis_ratio`; `combine` taking GP members.
  - `metrics` entry lists four functions (`CHANGELOG.md` "virgil.metrics" bullet) but the module and docs also expose `score`, `resample`, `align` (docs/imaging_ensemble.md uses `metrics.score`, `metrics.resample`, `metrics.align`).
  - `aliases.alias_bands`, `position_profile`, `spectral_window` are public (documented in `docs/api/aliases.md`) but unmentioned; `plotting.plot_uv_coverage` is new and unmentioned.
  - Docs-side changes (Zensical replaces MkDocs, `docs` extra, new Comparison page, Development carbon page, Contributions page) and the new `virgil-pipeline` console script are only partly covered (pipelines yes; Zensical and comparison page not).
- Stale or contradictory claims inside Unreleased:
  - The `virgil.ensemble` bullet (`CHANGELOG.md` ~line 330) still says "The default `field_factors` are (1, 2, 4) times `field_of_view(data)`", contradicted by the Changed bullet "The ensemble sweeps the data's field" (default `(0.75, 1.0, 1.5)`, verified in `ensemble.py:219`). Rewrite the Added bullet.
  - The Changed bullet on Oxford spelling (`CHANGELOG.md:538`) says the spelling "in docstrings, comments and the hand-written docs follow"; ~55 British -ise forms remain in tutorial pages (D9).
- fix: add one Added bullet for #370/#371 names, extend metrics bullet, fix the field_factors sentence.
- effort: 30 min.

### D4. CHANGELOG structure: duplicate headings, no summary, 660 lines
- severity: should-fix-before-0.4.0
- `CHANGELOG.md:10 ### Added`, `:57 ### Changed`, `:128 ### Migration from 0.3.0`, `:160 ### Added` (again), `:468 ### Changed` (again), `:586 ### Fixed`. A reader sees two Added and two Changed lists, with the migration guide sandwiched between them (breaking changes are in the middle, not on top). Entries are developer-level (checkpoint file names, private `_Surface.weights`, private helper renames). 0.3.0 is the same style but 0.4.0 has no user-level "Highlights".
- fix: merge into one Added/Changed/Fixed, move "Migration from 0.3.0" to the top (it is the thing a 0.3 user needs), add a 8-10 line Highlights paragraph (argument-order change, pipelines, PolarDisk, orbit aliases/search, variational inference, ensembles), move private-only items to one "Internal" bullet.
- effort: 1 hour (mechanical).

### D5. Landing page has no quickstart and no "where to start"
- severity: should-fix-before-0.4.0
- `README.md` / `docs/index.md` (generated from README by `sync_tutorial_docs.py`): sections are What is virgil (2 sentences), Installation, "Use & Documentation" (one sentence: "Documentation is published at benjaminpope.github.io/virgil", a self-link on the docs site itself, `docs/index.md:62`), Independent validation, Collaboration, Name, Formerly drpangloss. The 0.3.0 "Using these docs" map was removed (#334, #364); nothing replaced it. The page has no code, no link to any tutorial. Reachability test: from the landing page a student can reach binary search, contrast limits and the orbit tutorial in one click only through the left sidebar (collapsed behind a menu button on a phone); there is no content link. "Make an image" lands on "1. Simulating AMI data", which simulates data and fits nothing; the first reconstruction is part 2. There is no page titled for "I have an OIFITS file, what do I do": the one-call route (`BinaryPipeline(data, output=...).run()`) lives at "Pipelines and their stability".
- fix: replace "Use & Documentation" with a "Where to start" table (task -> page): fit a binary / grid search -> Binary search; contrast curve -> Contrast limits; orbit -> Orbits from interferometric data; image -> Imaging part 2 (and part 1 for simulating); own OIFITS file -> Reading data; one-command run -> Pipelines. Add a 10-line quickstart (`OIData("x.oifits")`, `fit`, `likelihood_grid`). Retitle the nav entry "Pipelines and their stability" to "Pipelines: one-command analyses" and "1. Simulating AMI data" to "1. Simulating AMI data (start here for imaging)".
- effort: 1 hour. Needs README edit then `sync_tutorial_docs.py` regeneration.

### D6. Dead external link in binary_search tutorial
- severity: should-fix-before-0.4.0
- `docs/binary_search.md:213` (source: `notebooks/binary_search.ipynb:360`): `https://github.com/LouisDesdoigts/zodiax_tutorials/blob/main/optimisation_tools.ipynb` returns 404; the notebook now lives at `.../blob/main/tutorials/optimisation_tools.ipynb` (verified via the GitHub API).
- fix: edit the markdown cell, resync (a markdown-only edit does not change the source hash).
- effort: 5 min.

### D7. Broken Attributes row on the aliases API page
- severity: should-fix-before-0.4.0 (cosmetic but on the headline new module's page)
- `src/virgil/aliases.py:1480-1483`: the sentence "Both are [`UNRECORDED`][virgil.aliases.UNRECORDED] on a band loaded from a file..." follows `labels_free : bool` in the numpydoc Attributes block, so it is parsed as an attribute named `Both are [...] on a band loaded`; `site/api/aliases/index.html:5451` shows the raw `[`UNRECORDED`][...]` markup in a table cell and the two autorefs never resolve (strict build does not catch it).
- fix: move that paragraph above the `Attributes` heading or into a Notes section.
- effort: 10 min.

### D8. Orbit alias/search workflows and several new features have no prose page
- severity: should-fix-before-0.4.0 (or state in release notes that they are reference-only)
- `fit_orbit_aliases`, `period_window`, `NightFlag`/`epoch_positions` flags, `score_orbits`, `PolarField`, `ensemble.gp_members`/`evidence_grid`, `chi2_by_observable`/`residual_image`: zero hits in `docs/*.md` (grep). `fit_orbit_aliases` is the largest feature of the release (`aliases.py` +2947 lines, about a third of the changelog) and is documented only by the API page and CHANGELOG. `docs/orbit_fitting.md` does not mention the alias search or tell a student what to do when the period is undersampled. `docs/imaging_ensemble.md` does not mention GP members although `EnsembleSpec` now carries seven `gp_*` fields. `PolarField` appears only in the API page.
- fix: add a section "When the period is undersampled: alias bands" to the orbit tutorial (a five-cell example, as for PolarDisk), plus two short paragraphs for the ensemble page and the Image/GP imaging page. If not done for 0.4.0, state in the changelog that these are reference-only.
- effort: 0.5-1 day for the orbit section; 30 min for the stubs.

### D9. Provisional status of new features is not stated anywhere
- severity: should-fix-before-0.4.0
- grep of `docs`, `src`, CHANGELOG for provisional/experimental/may change finds only the pipeline tier table and "anything before 1.0 may change" (`CHANGELOG.md:6`). Nothing flags PolarDisk (`RadialGP`, `PolarKernel`, `Harmonics` form change), `fit_orbit_aliases`/`score_orbits`/`Epochs` (the `scales` default is announced to flip from `"quoted"` to `"marginal"`, `epochs.py:707`), `virgil.ensemble`, `virgil.svi`. Conversely `docs/pipeline.md:171-175` declares both `BinaryPipeline` and `StarPipeline` "stable" with a frozen contract (stage names, every summary key, check names and thresholds, file names) although both landed on 2026-10-08 (#305, #307) and #323 (pre-0.4.0 review) was needed for StarPipeline model choice. The docs also say "Further pipelines (orbit and imaging, both provisional) will be added", but no DiskPipeline/ImagePipeline is on main (the design note is private); that sentence promises something unreleased.
- fix: add a stability table to `docs/api/index.md` or a "Stability" section on the index page (stable core: OIData, models, fit, grids, limits; provisional: the list above) and a `.. note:: provisional` in each module docstring. Decide whether the two pipelines ship as "provisional" for 0.4.x (they already support the tier via `STABILITY`, `FutureWarning` once per session) and only graduate in 0.5. Drop or soften the "will be added" sentence.
- effort: 1 hour for docs; decision for Ben on the pipelines' tier.

### D10. Tutorials needing extras do not say so
- severity: should-fix-before-0.4.0
- `docs/orbit_fitting.md` needs the `orbits` extra (jaxoplanet); the old README sentence "(needs the `[orbits]` extra)" was removed with the docs map and the tutorial never mentions it (`grep extra docs/orbit_fitting.md` finds only `get_extra_fields`). `docs/binary_search.md`, `hierarchical_inference.md` and `orbit_fitting.md` use ChainConsumer (`plots` extra); the three "As a pipeline" sections need `pipeline`. Only `harmonix.md` and `pipeline.md` say what to install. README does not mention the `notebooks` extra (the one-line install for all tutorials) nor that the notebooks are not in the wheel (they are in the GitHub repo).
- fix: an install admonition at the top of each affected tutorial (or one README sentence: `pip install "virgil-astro[notebooks,pipeline]"` runs every tutorial; notebooks are in the repo under notebooks/).
- effort: 30 min.

### D11. British -ise spelling in prose (OUP -ize required)
- severity: should-fix-before-0.4.0 (cheap), or 0.4.x
- About 55 occurrences in the tutorial pages, all inherited from notebook markdown cells: `docs/imaging_rml.md` (12: "regularised", "regulariser", "regularisation", lines 2, 10, 108, 111, 131, 147, 163), `imaging_clean.md` (5), `imaging_gp.md` (5), `orbit_fitting.md` (6: "marginalised", "initialisation", lines 430, 500, 873), `hierarchical_inference.md` (4), `binary_search.md`, `detection_roc.md`, `limb_darkening.md`, `imaging_sampling.md`, `imaging_ami.md` (2 each), `composition.md`, `gravity_darkened_star.md`, `harmonix.md`, `wr_chromatic_search.md`; plus `docs/gp_and_ift.md:3` "recognise". The Changed bullet at `CHANGELOG.md:538` claims the hand-written docs follow -ize. (Generated pages must be fixed in the notebooks, then resynced.) `docs/dev_carbon.md` -ise is quoted PR titles, leave.
- fix: sed over notebook markdown cells only (code cells and so the output hash are untouched), run `scripts/sync_tutorial_docs.py`, fix `gp_and_ift.md:3` by hand.
- effort: 20 min.

### D12. Drevon et al. year inconsistent
- severity: should-fix-before-0.4.0 (trivial)
- `docs/imaging_ensemble.md:5`, `:121` and `docs/api/ensemble.md:4` say "Drevon et al. 2025" with arXiv:2609.15365; `docs/contributors.md:81`, `src/virgil/ensemble.py:5,1058` say 2026 (and 2609 is September 2026).
- fix: 2025 -> 2026 in the three places.
- effort: 2 min.

### D13. pyproject metadata thin
- severity: should-fix-before-0.4.0
- `pyproject.toml:19` `urls = {repository = ...}` only (PyPI page shows one link; confirmed against pypi.org/pypi/virgil-astro/json). No Homepage/Documentation/Changelog/Issues. Classifiers lack `Development Status :: 4 - Beta`, `Intended Audience :: Science/Research`, `Topic :: Scientific/Engineering :: Astronomy`, `License :: ...` (not needed with the license expression), 3.14 absent. Keywords (`optics, diffraction, simulation, telescope`) are inherited from dLux-style boilerplate and omit `interferometry`, `oifits`, `aperture-masking`.
- fix: add `Homepage`, `Documentation` (https://benjaminpope.github.io/virgil/), `Changelog`, `Issues` URLs, the three classifiers, better keywords.
- effort: 10 min.

### D14. README install advice
- severity: 0.4.x
- `README.md:22-25`: nothing warns that `pip install virgil` is an unrelated package (stated only in the project's CLAUDE.md). `README.md:51-58` recommends `uv python install 3.11`, but `pyproject.toml` comments that JAX 0.11 needs Python 3.12, so the recommended setup gets the older locked JAX. 
- fix: one sentence "the PyPI name is virgil-astro; `pip install virgil` is a different project"; recommend Python 3.12.
- effort: 5 min.

### D15. CONTRIBUTING partly stale and contradictory
- severity: 0.4.x
- `CONTRIBUTING.md:30-35` extras list omits `pipeline` and `orbits` (both now in `dev`/`test`) and says `integrations` is for `HarmonixModel`. `:96` "Do not run the full suite locally ... Call the venv interpreter directly rather than `uv run`" is the maintainer's laptop policy (JAX tests are heavy), stated to outside contributors as a rule; say "run the tests for the files you change; CI runs the full suite". The `docs` extra still pins `mkdocs>=1.6,<2` and `mkdocs-material` (`pyproject.toml` docs extra) although CONTRIBUTING and CI use Zensical, which reads `mkdocs.yml`; say why or drop.
- fix: update the extras list; reword the local-test sentence for contributors.
- effort: 15 min.

### D16. `MIGRATION_VIRGIL.md` in the repo root is an agent coordination script
- severity: 0.4.x (note)
- Root-level file is a how-to for coordinating agents during the 0.2.0 rename ("Hard rules (all agents)", "Do not edit that script or this file", "The refactor is done by one coordinator"). It is not user migration documentation, is linked from nowhere, and its "Status: the rename has landed" text sits beside live instructions. A 0.2 -> 0.4 user would be better served by the README's "Formerly drpangloss" paragraph.
- fix: move to `design/` (or delete, history keeps it) before the public release; keep `scripts/rename_to_virgil.py` only if still wanted.
- effort: 10 min.

### D17. No upgrade guide outside the changelog
- severity: 0.4.x
- The 0.3 -> 0.4 argument-order migration (the one breaking change, with removal in 0.5) lives only in `CHANGELOG.md` "Migration from 0.3.0" (line 128). README, docs index and API index do not link it. A user whose 0.3 code now emits FutureWarnings searches the docs site, which does not contain the changelog.
- fix: add a "Upgrading from 0.3" page (or include the changelog in the site) and link from the index page; copy the `filterwarnings` recipe there.
- effort: 30 min.

### D18. Nav organisation
- severity: 0.4.x
- `mkdocs.yml` nav: "Variational inference" sits under Binaries (it is generic inference, used by imaging as well); "Development carbon" and "Contributions" are under Background beside conventions and GP/IFT, mixing project-process pages with science background; "Pipelines" is a one-page top-level section; the PolarDisk tutorial is under Sources while the API page is under Models > Classes; API Reference is a flat list of 32 entries including "Linear marginalization (internal)" and `api/imaging.md` renders private `virgil._geometry`. Single-page-per-feature count is otherwise reasonable (22 tutorials, no duplicates); `docs/misc/README.md` and `export_claude_chats.py` are copied into the published site (`site/misc/`) although outside the nav.
- fix: move VI under an "Inference" or "Fitting" group, put Contributions/Development carbon in a "Project" section last, drop `docs/misc` from the build (move to `scripts/`), hide the internal linear page.
- effort: 30 min.

### D19. `site_url` missing: empty sitemap and no canonical links
- severity: 0.4.x
- `mkdocs.yml` has `repo_url` but no `site_url`; the built `site/sitemap.xml` is an empty `<urlset>` and `site/index.html` has no canonical link, so search engines get no sitemap. Also no `site_description`.
- fix: `site_url: https://benjaminpope.github.io/virgil/`, a description.
- effort: 2 min.

### D20. Stale executed outputs in tutorials
- severity: 0.4.x
- Fifteen docs-mapped notebooks carry outputs but no `virgil.source_hash` stamp (`scripts/check_notebook_outputs.py` warnings: amigo_disco, composition, data_io, detection_roc, gravity_darkened_star, harmonix, hierarchical_inference, imaging_ami/clean/composite/gp/rml, model_syntax, source_models, wr_chromatic_search), so staleness cannot be checked; `check_notebook_outputs.py` reports no mismatch. `docs/contrast_limits.md:101,146` shows warnings attributed to `virgil/_deprecate.py:150`, which is what pre-#315 code produced (the current `_grid.py` uses `external_stacklevel`), so the published output predates the source. `docs/harmonix.md:265` publishes a "did not converge in 20000 steps" RuntimeWarning without comment. `docs/imaging_ami.md:160` and the notebook still use `data_object` as a lambda argument (harmless).
- fix: re-execute and stamp the tutorials on OzSTAR at the release commit; add a sentence in harmonix.md about the warning or tighten the fit.
- effort: cluster time; Ben runs.

### D21. Comparison page and dev-carbon page are dated snapshots
- severity: 0.4.x
- `docs/comparison.md:3` virgil column "checked ... commit 0771bcc" (12 Oct), 217 commits before main; PolarDisk, orbit alias search, pipelines, GP ensembles landed after. `docs/dev_carbon.md:8` period ends 2026-10-06 and README badge says 76.5 kg (the page says it lags).
- fix: refresh both at the release commit (`python3 scripts/dev_carbon.py`), or add "as of" wording on the README badge.
- effort: 20 min plus running the collector.

### D22. CHANGELOG mentions private design notes
- severity: note
- `CHANGELOG.md:412, 425, 536` (Unreleased) cite `design/visibility_orbits.md` and `design/sampler_flat_coordinates.md`; `src/virgil/_flat.py:13`, `_gauss_newton.py:71`, `fitting.py:418` do the same in docstrings. Not links, but the design notes are private by policy and these paths are dead for PyPI users; the changelog also names specific targets (AL Dor, HD 41255, HD 188088, "2024 contest's Obj1 GRAVITY") as evidence.
- fix: replace with the issue or PR number, or drop the path.
- effort: 15 min.

### D23. Minor
- `examples/synthetic_binary_workflow.py:265,296,339,364,375` and `examples/hierarchical_binary_workflow.py`: numpyro model argument still `data_obj`/`observations`; works, but the changelog says these names are retired everywhere. Rename to `data` for consistency (note).
- `docs/api` exception classes `NoLeastSquaresForm`, `SingularCurvature`, `NoStartingValue`, `GaussNewtonMass`, `LinearFluxGrid` are public in `fitting`/`grid_fit` but not rendered on the API pages (checked against `site/api` ids); `fitting.py` has no `__all__` so the API pages rely on module-level scanning. Add to the members list since the changelog tells users to catch them (note).
- `docs/api/aliases.md`, `epochs.md`, `observables.md`, `pipeline.md`: the constants `LABELS_FREE_NATS`, `RESOLVE_NATS`, `KINDS`, `SCHEMA`, `Stage` are in `__all__` but not rendered (note).
- Landing page repeats the dev-carbon badge on PyPI (loads from shields.io); fine, but the PyPI long description links `dev_carbon/` on the docs site, which is a project-process page (note).

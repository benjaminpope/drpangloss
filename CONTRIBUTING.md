# Contributing Guide

drpangloss is an open-source package and welcomes contributions via pull requests.

---

## Getting Started

Firstly, you will need to fork the repository to your own GitHub account. This will allow you to make changes to the code and then submit a pull request to the main repository. To do this, click the fork button in the top right of the repository page. This will create a copy of the repository in your own account that you can make changes to and then request to merge with the main repository.

Next, you will need to clone the repository to your local machine. To do this, open a terminal and navigate to the directory you would like to clone the repository to. Then run the following command:

```bash
git clone https://github.com/your-username-here/drpangloss.git
cd drpangloss
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . pytest ruff pre-commit mkdocs mkdocstrings mkdocstrings[python] mkdocs-material mkdocs-jupyter ipywidgets
```

Then you will need to install the pre-commit hooks. This will ensure that the code is formatted correctly and that the unit tests pass before you can commit your changes. To do this, run the following command:

```bash
pre-commit install
```

This will ensure that any changes you make will adhere to the code style and formatting guidelines of the rest of the package!

You can also run linting and formatting manually at any time:

```bash
ruff check . --fix
ruff format .
```

For lower-noise local runs (especially when notebooks are involved), use:

```bash
bash scripts/lint_local.sh
```

This command writes full diagnostics to `.lint-logs/` and keeps terminal output compact.
Useful flags:

```bash
bash scripts/lint_local.sh --fix
bash scripts/lint_local.sh --changed
bash scripts/lint_local.sh --no-notebooks
```

---

## Making Changes

Next you can start to make any changes you desire!

**Unit Tests**

It is important that any changes you make are tested to ensure that they work as intended and do not break any existing functionality. If you are creating _new_ functionality you will need to create some new unit tests, otherwise you should be able to modify the existing tests.

To ensure that everything is working as expected, you can run the unit tests by running the following command:

```bash
uv run --python .venv/bin/python pytest tests
```

This will run all tests in the `tests` directory. If you would like to run a specific test, you can run:

```bash
uv run --python .venv/bin/python pytest tests/test_file.py
```

Note that passing locally does not guarantee cross-platform compatibility. GitHub Actions runs CI checks for consistency across environments.

**Documentation**

Any changes you make should also be appropriately documented! For small API changes this shouldn't require any changes, however if you are adding new functionality you will need to add some documentation. This can be done by modifying the appropriates files in the `docs` directory.

Tutorial pages are notebook-synced. For these tutorials:

- `notebooks/binary_search.ipynb`
- `notebooks/data_io.ipynb`
- `notebooks/contrast_limits.ipynb`
- `notebooks/model_syntax.ipynb`

edit the notebook first, then regenerate the corresponding docs markdown with:

```bash
uv run --python .venv/bin/python scripts/sync_tutorial_docs.py
```

Notebook style conventions for tutorials:

- Keep explanatory markdown between major code blocks (zodiax sandbox style).
- Keep imports in the top import cell; avoid repeated imports in later cells.
- Prefer shared plotting helpers from `src/drpangloss/plotting.py` over long notebook-local plotting scripts.
- When a plotting helper is missing, add/extend it in `src/drpangloss/plotting.py` first, then call it from the notebook.
- Keep notebook plotting cells short and declarative (prepare inputs, call helper, show figure).

Typical helper usage patterns:

- Grid and map visuals: `plot_likelihood_grid`, `plot_contrast_limit_map`
- Posterior diagnostics: `plot_chainconsumer_diagnostics`, `plot_trace_panels`
- Recovery/correlation summaries: `plot_recovery_residuals`, `plot_data_model_correlation`, `plot_radial_limit_summary`

To build the documentation locally and make sure everything is working correctly, you can run the following command:

```bash
mkdocs serve
```

This will build the documentation and serve it on a local server. You can then navigate to `localhost:8000` in your browser to view the documentation.

---

## Contributing the Changes

After these steps have been completed, you can commit your changes and push them to your forked repository. These changes should have its formatting and linting checked by the pre-commit hooks. If there are any issues, you will need to fix them before you can commit your changes. Once you have pushed your changes to your forked repository, you can submit a pull request to the main repository. This will allow the maintainers to review your changes and merge them into the main repository!

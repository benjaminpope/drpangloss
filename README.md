# drpangloss
[![PyPI version](https://badge.fury.io/py/drpangloss.svg)](https://badge.fury.io/py/drpangloss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![integration](https://github.com/benjaminpope/drpangloss/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpope/drpangloss/actions/workflows/tests.yml)
[![Documentation](https://github.com/benjaminpope/drpangloss/actions/workflows/documentation.yml/badge.svg)](https://benjaminpope.github.io/drpangloss/)

The best of all possible interferometry models.

Contributors: [Dori Blakely](https://github.com/blakelyd), [Benjamin Pope](https://github.com/benjaminpope)

## What is drpangloss?

drpangloss is a package for modelling optical interferometry data in JAX.

## Installation

drpangloss is hosted on PyPI; the easiest way to install it is:

```
pip install drpangloss
```

You can also build from source. To do so, clone the git repo, enter the directory, and run

```
pip install .
```

We recommend using a virtual environment to avoid dependency conflicts.

Using `uv` (recommended):

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . pytest
uv run --python .venv/bin/python pytest -q
```


## Use & Documentation

Documentation is published at [benjaminpope.github.io/drpangloss](https://benjaminpope.github.io/drpangloss/).

## Collaboration & Development

We welcome collaboration and development contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and pull request workflow.

Current modernization priorities are tracked in [REVIEW_NOTES.md](REVIEW_NOTES.md) and [docs/review_2026-03-03.md](docs/review_2026-03-03.md).

## Name

Why is it called drpangloss?

The leading optical interferometry model fitting code is [CANDID](https://github.com/amerand/CANDID). In Voltaire's *Candide*, Dr Pangloss' belief that we live in the best of all possible worlds is a satire of Leibniz' theodicy. But we *do* live in a world with Jax, so that if we can't optimize the world, at least we can optimize our fits to VLTI data.  
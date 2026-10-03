# virgil
[![PyPI version](https://badge.fury.io/py/virgil.svg)](https://badge.fury.io/py/virgil)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![integration](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml)
[![Documentation](https://github.com/benjaminpope/virgil/actions/workflows/zensical-pages.yml/badge.svg)](https://benjaminpope.github.io/virgil/)

The best of all possible interferometry models.

Contributors: [Dori Blakely](https://github.com/blakelyd), [Benjamin Pope](https://github.com/benjaminpope)

## What is virgil?

virgil is a package for modelling optical interferometry data in JAX.

## Installation

virgil is hosted on PyPI; the easiest way to install it is:

```
pip install virgil-astro
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

Documentation is published at [benjaminpope.github.io/virgil](https://benjaminpope.github.io/virgil/).

Documentation tooling is currently migrating from MkDocs to Zensical. During this transition, both builders are supported from the same configuration file.

Local docs checks:

```bash
uv run --python .venv/bin/python mkdocs build --strict
uv run --python .venv/bin/python zensical build --clean
```

## Collaboration & Development

We welcome collaboration and development contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and pull request workflow.

## Name

Why is it called virgil?

The leading optical interferometry model fitting code is [CANDID](https://github.com/amerand/CANDID). In Voltaire's *Candide*, Dr Pangloss' belief that we live in the best of all possible worlds is a satire of Leibniz' theodicy. But we *do* live in a world with Jax, so that if we can't optimize the world, at least we can optimize our fits to VLTI data.  
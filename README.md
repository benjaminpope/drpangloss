# virgil
[![PyPI version](https://badge.fury.io/py/virgil-astro.svg)](https://badge.fury.io/py/virgil-astro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![integration](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml)
[![Documentation](https://github.com/benjaminpope/virgil/actions/workflows/zensical-pages.yml/badge.svg)](https://benjaminpope.github.io/virgil/)

Versatile Interferometric Reconstruction and Gradient-based Inference Library.

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

VIRGIL is the **V**ersatile **I**nterferometric **R**econstruction and **G**radient-based **I**nference **L**ibrary. In Dante's *Divine Comedy*, the poet Virgil is Dante's guide through the Inferno and Purgatory. In Virgil's own *Aeneid*, when Aeneas enters the underworld he draws his sword on the monsters crowding its threshold. His guide, the Cumaean Sibyl, warns him that they are only thin, bodiless lives flitting in a hollow semblance of form (*Aeneid* VI.292–294). Image reconstruction from sparse interferometric data is full of false visions like these: artefacts that look like structure but have no substance in the data. VIRGIL aims to help you tell the difference. The acronym is Jonah Goldfine's.

Until version 0.1.1 the package was named after Voltaire's Dr Pangloss, a nod to Antoine Mérand's [CANDID](https://github.com/amerand/CANDID); it is now distributed on PyPI as `virgil-astro`.


# virgil
[![PyPI version](https://badge.fury.io/py/virgil-astro.svg)](https://badge.fury.io/py/virgil-astro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![integration](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml)
[![Documentation](https://github.com/benjaminpope/virgil/actions/workflows/zensical-pages.yml/badge.svg)](https://benjaminpope.github.io/virgil/)

Versatile Interferometric Reconstruction and Gradient-based Inference Library.

Contributors: [Dori Blakely](https://github.com/blakelyd), [Benjamin Pope](https://github.com/benjaminpope), [Louis Desdoigts](https://github.com/LouisDesdoigts), [Shashank Dholakia](https://github.com/shashankdholakia), [Toon De Prins](https://github.com/DePrinsT), [Jonah Goldfine](https://github.com/JonahDG), [Max Charles](https://github.com/maxecharles), [Anand Sivaramakrishnan](https://github.com/anand0xff), [Ian Czekala](https://github.com/iancze) and [Jens Kammerer](https://github.com/kammerje). The [Contributors](https://benjaminpope.github.io/virgil/contributors/) page says who did what.

## What is virgil?

virgil is a package for modelling optical interferometry data in JAX.

## Installation

virgil is hosted on PyPI; the easiest way to install it is:

```
pip install virgil-astro
```

You can also build from source. To do so, clone the git repo and enter the directory:

```
git clone --filter=blob:none https://github.com/benjaminpope/virgil
cd virgil
pip install .
```

`--filter=blob:none` makes a partial clone: you get the full history, but old
versions of files are fetched only if you ask for them. It skips large data
files that are no longer used, so the download is about 15 MB rather than
about 280 MB.

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

### Using these docs

The sections in the sidebar hold worked examples on simulated and bundled data:
- **Background:** [who contributed what](https://benjaminpope.github.io/virgil/contributors/), and [Gaussian-process priors and information field theory](https://benjaminpope.github.io/virgil/gp_and_ift/).
- **Data Handling:** [reading OIFITS files into `OIData`](https://benjaminpope.github.io/virgil/data_io/), and [AMIGO's DISCO data from JWST aperture masking](https://benjaminpope.github.io/virgil/amigo_disco/).
- **Binaries:** [searching for companions](https://benjaminpope.github.io/virgil/binary_search/), [detection limits](https://benjaminpope.github.io/virgil/contrast_limits/), and [fitting several datasets together](https://benjaminpope.github.io/virgil/hierarchical_inference/).
- **Sources:** [visibility models](https://benjaminpope.github.io/virgil/model_syntax/), [extended sources](https://benjaminpope.github.io/virgil/source_models/), [composing scenes](https://benjaminpope.github.io/virgil/composition/), [spotted stars](https://benjaminpope.github.io/virgil/harmonix/) and [gravity-darkened stars](https://benjaminpope.github.io/virgil/gravity_darkened_star/).
- **Imaging:** image reconstruction in five parts: [simulating data](https://benjaminpope.github.io/virgil/imaging_ami/), [regularised maximum likelihood](https://benjaminpope.github.io/virgil/imaging_rml/), [Gaussian-process priors](https://benjaminpope.github.io/virgil/imaging_gp/), [a ring around a binary](https://benjaminpope.github.io/virgil/imaging_composite/) and [sampling the posterior](https://benjaminpope.github.io/virgil/imaging_sampling/).
- **[API Reference](https://benjaminpope.github.io/virgil/api/)** documents every public class and function.

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

### Formerly drpangloss

Until version 0.1.1 this package was called **drpangloss**, after Voltaire's Dr Pangloss and as a nod to Antoine Mérand's [CANDID](https://github.com/amerand/CANDID). From version 0.2.0 it is **virgil**: `import virgil`, installed with `pip install virgil-astro`. A final release of `drpangloss` under its own name will depend on `virgil-astro` and point here, so old installs find the new package.


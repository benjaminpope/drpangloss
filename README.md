# drpangloss
[![PyPI version](https://badge.fury.io/py/drpangloss.svg)](https://badge.fury.io/py/drpangloss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![integration](https://github.com/benjaminpope/drpangloss/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpope/drpangloss/actions/workflows/tests.yml)
[![Documentation](https://github.com/benjaminpope/drpangloss/actions/workflows/documentation.yml/badge.svg)](https://benjaminpope.github.io/drpangloss/)

the best of all possible interferometry models

Contributors: Dori Blakely, [Benjamin Pope](https://github.com/benjaminpope)

## What is drpangloss?

drpangloss is a package for modelling optical interferometry data in Jax.

## Installation

drpangloss is hosted on PyPI (though this is currently a placeholder): the easiest way to install this is with 

```
pip install drpangloss
```

You can also build from source. To do so, clone the git repo, enter the directory, and run

```
pip install .
```

We encourage the creation of a virtual enironment to run drpangloss to prevent software conflicts as we keep the software up to date with the lastest version of the core packages.


## Use & Documentation

Documentation will be found [here](https://benjaminpope.github.io/drpangloss/), though this is currently a placeholder. 

## Collaboration & Development

We are always looking to collaborate and further develop this software! We have focused on flexibility and ease of development, so if you have a project you want to use drpangloss for, but it currently does not have the required capabilities, don't hesitate to [email me](b.pope@uq.edu.au) and we can discuss how to implement and merge it! Similarly you can take a look at the `CONTRIBUTING.md` file.

## Name

Why is it called drpangloss?

The leading optical interferometry model fitting code is [CANDID](https://github.com/amerand/CANDID). In Voltaire's *Candide*, Dr Pangloss' belief that we live in the best of all possible worlds is a satire of Leibniz' theodicy. But we *do* live in a world with Jax, so that if we can't optimize the world, at least we can optimize our fits to VLTI data.  
"""Scriptable pipelines on top of virgil (provisional in 0.4).

A pipeline runs a standard analysis in named stages with simple defaults
that keywords override, and writes a standard run folder: ``run.json``
(configuration, inputs, provenance, status), ``summary.json`` (key numbers
and quality checks), HDF5 grids and samples, plots, the processed data as
OIFITS, the best-fit model, and an executed quicklook notebook. The folder
reloads into real virgil objects with [`load`][virgil.pipeline.load].

    from virgil.pipeline import BinaryPipeline, load

    res = BinaryPipeline(data, output="runs/hd1234").run()
    res = load("runs/hd1234")          # later, in a fresh session
    res.summary["companion"]["sep_mas"], res.checks

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

The whole of ``virgil.pipeline`` is provisional in 0.4: class names,
constructor keywords, stage names, settings, summary keys and checks may
change in any 0.4.x release. The run folder is versioned
(``virgil-pipeline-run-v1``) and ``load`` refuses schemas it does not know.
The pipelines are promoted to stable after a release of use. They need
the ``pipeline`` extra (``pip install 'virgil-astro[pipeline]'``).
"""

import importlib

from ._checks import Check
from ._core import ConfigMismatchError, Stage
from ._io import SCHEMA, Result, SkyImage, load

_LAZY = {
    "BinaryPipeline": ".binary",
    "DiskPipeline": ".disk",
    "ImagingPipeline": ".image",
    "OrbitPipeline": ".orbit",
    "StarPipeline": ".star",
}

__all__ = [
    "SCHEMA",
    "BinaryPipeline",
    "Check",
    "ConfigMismatchError",
    "DiskPipeline",
    "ImagingPipeline",
    "OrbitPipeline",
    "Result",
    "SkyImage",
    "Stage",
    "StarPipeline",
    "load",
]

__provisional__ = ("*",)


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

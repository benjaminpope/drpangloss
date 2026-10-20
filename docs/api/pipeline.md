# `virgil.pipeline`

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

The classes and functions of the pipelines. For what a pipeline does, the run
folder, the checks and the stability policy, see
[Pipelines and their stability](../pipeline.md).

::: virgil.pipeline.binary.BinaryPipeline

::: virgil.pipeline.star.StarPipeline

::: virgil.pipeline.disk.DiskPipeline

::: virgil.pipeline.orbit.OrbitPipeline

The result of `OrbitPipeline.run()` is a [`Result`][virgil.pipeline.Result]: `Result.aliases()` returns the engine's [`AliasResult`][virgil.aliases.AliasResult] (every alias band, its evidence and the best orbit), `Result.samples()` the posterior draws of the winning band, and `Result.summary` the numbers and checks. The [orbit tutorial](../orbit_fitting.md) reads one.

::: virgil.pipeline.disk.DiskModel

::: virgil.pipeline.disk.models

::: virgil.pipeline.disk.plot_summary

::: virgil.pipeline.load

::: virgil.pipeline.Result

::: virgil.pipeline.Check

::: virgil.pipeline.ConfigMismatchError

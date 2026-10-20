# `virgil.epochs`

Datasets grouped into named epochs, with one snapshot of a moving scene
per dataset, for orbit fits to visibilities and closure phases across
epochs, and the tools that start and sample those fits: ranking trial
orbits by the data, choosing distinct starts for each chain, and starting
from a fit of positions.

!!! note "`start_from_positions` is superseded"
    For an orbit with an unknown period use
    [`fit_orbit_aliases`](aliases.md) or
    [`OrbitPipeline`](pipeline.md) (see [Orbits from interferometric
    data](../orbit_fitting.md)). `start_from_positions` remains for starting a
    fit of a model of your own from per-night positions.

::: virgil.epochs
    options:
      members:
        - Epochs
        - rank_orbits
        - RankedOrbits
        - chain_starts
        - epoch_positions
        - estimate_primary_diameter
        - PrimaryDiameter
        - marginal_loglike
        - broad_loglike
        - sidelobe_flags
        - EpochPositions
        - EpochPeaks
        - NightFlag
        - start_from_positions
        - OrbitStart

# `virgil.orbit_search`

Scoring candidate orbits on multi-epoch data with the nuisances an orbit
shares across epochs shared in the score: one companion flux per band
(with an optional chromatic slope) integrated out on a log-uniform grid,
each dataset's error scales marginalized, calibration gains and
closure-phase offsets profiled, and extra terms such as radial velocities
added. Positions are evaluated at every sample's own time.

## Details

For a scene linear in the companion's flux f, the complex visibility is
`V(f) = (A + f B) / (T₀ + f ΔT)`, with `A` and the total fluxes computed
once per dataset and `B` from the visibilities of a unit companion.

The coarse flux grid only finds the peak. Newton steps in ln f (and the
slope β) profile it, and Gauss–Legendre nodes on each side of it, out to
a 12-nat drop or the prior's bounds, integrate it: a well-measured flux
(σ_ln f ≈ 0.02) is far narrower than any affordable grid step.

With `s_max=None` a block's −(ν/2) ln χ² is unbounded as χ² → 0, so a
block with few degrees of freedom (one triangle's closure phases, say)
that a flux or slope fits almost exactly makes a narrow spike in the
score, which can dominate the profile (`profiled`, `flux`, `slope`). The
default `s_max=5` leaves about twice the error inflation of typical
interferometric data and bounds each block's gain at about ν ln `s_max`.

::: virgil.orbit_search
    options:
      members:
        - score_orbits
        - SharedFlux
        - OrbitScores
        - rank_scores

# Stability

virgil has two tiers.

- **Stable**: a change needs a deprecation cycle. A release keeps accepting the old form and warns before the new one replaces it.
- **Provisional**: the name, signature, defaults and output may change in any 0.4.x release without a deprecation. Provisional features are new, and are still settling.

Everything public that is not listed below is stable. A provisional page or API entry carries this notice:

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Nothing warns at runtime: the tier is in the documentation and in each provisional module's `__provisional__` tuple, which a test checks. A new public feature starts provisional; promotion to stable is decided at a release.

## Provisional in 0.4

| Area | Provisional names |
| --- | --- |
| Orbit fitting | the modules [`virgil.epochs`](api/epochs.md), [`virgil.aliases`](api/aliases.md), [`virgil.orbit_search`](api/orbit_search.md) and [`virgil.tempering`](api/tempering.md) (the whole of each) |
| `PolarDisk` family ([`virgil.models`](api/models/classes/polar_disk.md)) | `PolarDisk`, `ThinRing`, `GaussianRing`, `RadialPowerLaw`, `ExpTail`, `Gap`, `RadialSum`, `RadialLinear`, `RadialPower`, `RadialGP`, `PolarKernel`, `Harmonics`, `VonMises`, `Cone`, `PowerLawHeight`, `polar_disk_cvis`, `gp_priors` |
| Polar fields ([`virgil.fields`](api/fields.md)) | `PolarField`, `polar_field_priors` |
| Ensembles' Gaussian-process members ([`virgil.ensemble`](api/ensemble.md)) | `evidence_grid`, `gp_members`, `GPFit`, `EvidenceGrid` |
| Variational inference | the module [`virgil.svi`](api/svi.md) (`variational`, `VariationalResult`) |
| Pipelines | the whole of [`virgil.pipeline`](pipeline.md): `BinaryPipeline`, `StarPipeline`, `Result`, `load` and the rest. Promoted to stable after a release of use |
| Sparse imaging ([`virgil.imaging`](api/imaging.md)) | `clean`, `CleanResult` |
| Image-recovery scores | the module [`virgil.metrics`](api/metrics.md) |

The user-guide pages about these features carry the same notice: [orbits](orbit_fitting.md), [`PolarDisk` rings](polar_disk_rings.md), [CLEAN](imaging_clean.md), [ensembles](imaging_ensemble.md), [variational inference](variational_inference.md) and [pipelines](pipeline.md).

The authoritative list is the `__provisional__` tuple in each module (`("*",)` means the whole module). It is checked by `tests/test_stability.py`.

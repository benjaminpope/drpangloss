# Stability

virgil has two tiers.

- **Stable**: a change needs a deprecation cycle. A release keeps accepting the old form and warns before the new one replaces it.
- **Provisional**: new features still settling. The name, signature, defaults and output may change in any 0.4.x release without a deprecation.

Everything public not listed below is stable. Provisional pages and API entries carry this notice:

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Nothing warns at runtime. The tier lives in the documentation and in each module's `__provisional__` tuple, which a test checks. New public features start provisional and are promoted at a release.

## Provisional in 0.4

| Area | Provisional names |
| --- | --- |
| Orbit fitting | the modules [`virgil.epochs`](api/epochs.md), [`virgil.aliases`](api/aliases.md), [`virgil.arcs`](api/arcs.md), [`virgil.orbit_search`](api/orbit_search.md) and [`virgil.tempering`](api/tempering.md) (the whole of each) |
| `MultipoleDisk` family ([`virgil.models`](api/models/classes/multipole_disk.md)) | `MultipoleDisk`, `ThinRing`, `GaussianRing`, `RadialPowerLaw`, `ExpTail`, `Gap`, `RadialSum`, `RadialLinear`, `RadialPower`, `RadialGP`, `RadialGPFunction`, `MultipoleKernel`, `Harmonics`, `VonMises`, `Cone`, `PowerLawHeight`, `multipole_disk_cvis`, `gp_priors` |
| Polar fields ([`virgil.fields`](api/fields.md)) | `MultipoleField`, `multipole_field_priors` |
| Ensembles' Gaussian-process members ([`virgil.ensemble`](api/ensemble.md)) | `evidence_grid`, `gp_members`, `GPFit`, `EvidenceGrid` |
| Variational inference | the module [`virgil.svi`](api/svi.md) (`variational`, `VariationalResult`) |
| Pipelines | the whole of [`virgil.pipeline`](pipeline.md): `BinaryPipeline`, `StarPipeline`, `Result`, `load` and the rest. Promoted to stable after a release of use |
| Sparse imaging ([`virgil.imaging`](api/imaging.md)) | `clean`, `CleanResult` |
| Image-recovery scores | the module [`virgil.metrics`](api/metrics.md) |

The user-guide pages about these features carry the same notice: [orbits](orbit_fitting.md), [`MultipoleDisk` rings](multipole_disk_rings.md), [CLEAN](imaging_clean.md), [ensembles](imaging_ensemble.md), [variational inference](variational_inference.md) and [pipelines](pipeline.md).

The authoritative list is each module's `__provisional__` tuple (`("*",)` means the whole module), checked by `tests/test_stability.py`.

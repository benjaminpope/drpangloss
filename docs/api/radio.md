# `virgil.radio`

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Complex visibilities from radio interferometers  (CASA measurement sets), with per-observation calibration nuisances and a
likelihood term for [`fit`](fitting.md). See the
[radio interferometry guide](../radio_interferometry.md).

::: virgil.radio
    options:
      members:
        - ComplexVisData
        - CalibrationNuisance
        - primary_beam
        - dartboard_folds

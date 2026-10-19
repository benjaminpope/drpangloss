# `virgil.radio`

Complex visibilities from radio interferometers (for example ALMA
measurement sets), with per-execution-block calibration nuisances and a
likelihood term for [`fit`](fitting.md). See the
[radio interferometry guide](../radio_interferometry.md).

::: virgil.radio
    options:
      members:
        - ComplexVisData
        - CalibrationNuisance
        - primary_beam
        - dartboard_folds

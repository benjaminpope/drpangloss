# `drpangloss.imaging`

Regularisers, priors and helpers for image reconstruction with
[`Image`](models/classes/image.md). How to choose a regularisation weight is
discussed in `design/regulariser_weight_selection.md`: the L-curve's corner, the
discrepancy principle, classic MaxEnt (`LCurve.classic_maxent`) and, for
Gaussian-field images, the Laplace evidence (`log_evidence`).

::: drpangloss.imaging
    options:
      members:
        - TSV
        - TV
        - MaxEntropy
        - Centroid
        - starting_image
        - dirty_image
        - image_priors
        - nyquist_pixel_scale
        - field_of_view
        - beam
        - Beam
        - l_curve
        - LCurve
        - log_evidence
        - error_scale
        - diagnose
        - Diagnosis

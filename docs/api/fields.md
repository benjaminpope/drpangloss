# `virgil.fields`

Gaussian-process priors for pixel images. A `GaussianField` can take the place
of an `Image`'s log-brightness array: the log-brightness is then a stationary
Gaussian process about an optional template, parameterized by whitened
coefficients on the image's cosine basis, which
`imaging.image_priors` gives standard-normal priors. A `PolarField` is the
same idea on a polar grid in a disk's frame, a residual aligned with a
`PolarDisk`.

::: virgil.fields
    options:
      members:
        - GaussianField
        - PolarField
        - polar_field_priors
        - polar_interpolate
        - field_spectrum

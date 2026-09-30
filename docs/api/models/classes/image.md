# `Image`: pixelised brightness

A pixel image is one more [component](composition.md) of a `System`, for
image reconstruction. Its visibilities are the Fourier transform of
the pixels: exact by default, or approximate but faster with the optional
NUFFT backend. The design is recorded in `design/image_reconstruction.md`.

::: drpangloss.models.Image
    options:
      show_root_heading: true
      heading_level: 2
      show_attributes: false
      members:
        - brightness
        - from_brightness
        - from_model

::: drpangloss.models.circular_support
    options:
      show_root_heading: true
      heading_level: 2

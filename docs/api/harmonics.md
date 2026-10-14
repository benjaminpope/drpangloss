# `virgil.harmonics`

Azimuthal harmonics of a [`PolarDisk`](models/classes/polar_disk.md): the image
harmonics $I_m$ that every form maps to and from (the linear, positive and von
Mises forms of the azimuthal profile and the legacy `az_amps`/`az_pas` of
`ModulatedGaussianRim` and `TruncatedCone`), and the reflection-coefficient
machinery of the positive form.

::: virgil.harmonics
    options:
      members:
        - to_image_harmonics
        - from_image_harmonics
        - from_legacy
        - to_legacy
        - rim_coefficients
        - cone_coefficients
        - reflection_coefficients
        - step_up
        - step_down
        - latents_from_reflection
        - spectral_factor
        - image_harmonics_from_h
        - image_harmonics_from_latents

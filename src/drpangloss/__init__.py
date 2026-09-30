"""drpangloss: the best of all possible interferometry models.

The everyday names are available at the top level::

    from drpangloss import OIData, System, PointSource, likelihood_grid

Modules:

* [`oidata`][drpangloss.oidata] and [`oifits`][drpangloss.oifits]: data and
  OIFITS files; [`amigo`][drpangloss.amigo]: AMIGO mixed-DISCO products.
* [`models`][drpangloss.models]: source models and their visibilities.
* [`likelihood`][drpangloss.likelihood]: likelihoods and numpyro models.
* [`fitting`][drpangloss.fitting]: `Problem` (model, data, priors and
  regularisers) and `fit` (Levenberg–Marquardt, L-BFGS, Adam).
* [`imaging`][drpangloss.imaging]: regularisers and helpers for image
  reconstruction; [`scenes`][drpangloss.scenes]: synthetic truth images.
* [`inference`][drpangloss.inference]: Laplace and Fisher curvature.
* [`grid_fit`][drpangloss.grid_fit]: grid searches.
* [`limits`][drpangloss.limits]: contrast limits and flux/contrast/Δmag
  conversions.
* [`spectra`][drpangloss.spectra]: wavelength-dependent fluxes.
* [`plotting`][drpangloss.plotting]: figures.
* [`bessel`][drpangloss.bessel]: Bessel functions in JAX.

The legacy ImPlaneIA tools in ``drpangloss.legacy`` are not imported here.
"""

name = "drpangloss"

from . import (  # noqa: E402
    amigo,
    bessel,
    fitting,
    grid_fit,
    imaging,
    inference,
    likelihood,
    limits,
    models,
    oidata,
    oifits,
    plotting,
    scenes,
    spectra,
)
from .amigo import load_oi_data  # noqa: E402
from .fitting import Problem, fit  # noqa: E402
from .grid_fit import (  # noqa: E402
    best_grid_point,
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    optimized_flux_grid,
    optimized_likelihood_grid,
)
from .inference import fisher, laplace_cov  # noqa: E402
from .likelihood import (  # noqa: E402
    build_model,
    inflated_errors,
    loglike,
    model_loglike,
    numpyro_model,
    whitened_residuals,
)
from .limits import (  # noqa: E402
    absil_limits,
    contrast_to_flux,
    delta_mag_to_flux,
    flux_to_contrast,
    flux_to_delta_mag,
    radial_profile,
    ruffio_upperlimit,
)
from .models import (  # noqa: E402
    BinaryModelAngular,
    BinaryModelCartesian,
    GaussianDisk,
    GaussianDiskModel,
    Image,
    ModulatedGaussianRim,
    PointSource,
    Resolved,
    SourceModel,
    System,
    UniformDisk,
    circular_support,
)
from .oidata import OIData  # noqa: E402
from .oifits import read_oifits, write_oifits  # noqa: E402
from .spectra import PowerLaw  # noqa: E402


__all__ = [
    "BinaryModelAngular",
    "BinaryModelCartesian",
    "GaussianDisk",
    "GaussianDiskModel",
    "Image",
    "ModulatedGaussianRim",
    "OIData",
    "PointSource",
    "Problem",
    "PowerLaw",
    "Resolved",
    "SourceModel",
    "System",
    "UniformDisk",
    "absil_limits",
    "circular_support",
    "best_grid_point",
    "build_model",
    "contrast_to_flux",
    "delta_mag_to_flux",
    "fit",
    "fisher",
    "inflated_errors",
    "flux_to_contrast",
    "flux_to_delta_mag",
    "laplace_cov",
    "laplace_flux_uncertainty_grid",
    "likelihood_grid",
    "load_oi_data",
    "loglike",
    "model_loglike",
    "numpyro_model",
    "optimized_flux_grid",
    "optimized_likelihood_grid",
    "radial_profile",
    "read_oifits",
    "ruffio_upperlimit",
    "whitened_residuals",
    "write_oifits",
]

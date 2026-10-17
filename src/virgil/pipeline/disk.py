"""`DiskPipeline`: rings and haloes fitted with `MultipoleDisk` (provisional)."""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import warnings
from typing import Callable

import numpy as onp

from .._precision import in_float64
from . import _checks, _io
from ._common import (
    _SAMPLER_SETTINGS,
    _bound_samples,
    _chi2,
    _m2loglike,
    _diagnostics,
    _geometry,
    _interior_start,
    _load_report,
    _run_chains,
    _noise_quantiles,
    _quantiles,
    _sample_stats,
    _save,
)
from ._core import Stage, _Pipeline

# The quadrature of a profile's ring is cut at this many σ; a ring's prior
# floor on σ and the radial grid below are stated in half-fringes.
_WIDTH_FLOOR = 0.1
# Prior ranges (group-invariant: log-uniform for scales, uniform for
# locations and angles).
_FLUX_RANGE = (1e-3, 10.0)
_PLANET_FLUX_RANGE = (1e-4, 1.0)
_INDEX_RANGE = (-10.0, 10.0)
_TEMPERATURE_RANGE = (300.0, 3000.0)
_TEMPERATURE_START = 1000.0
# The rings' harmonics are fractions of the ring's light (|w| <= 1 keeps the
# intensity positive); an isotropic Gaussian on re and im has a uniform phase,
# where a box would favour the diagonals.
_HARMONIC_SCALE = 0.4
# Fluxes (relative to the star at the reference wavelength) tried by the
# grid searches; the geometry grid profiles over them.
_FLUX_GRID = onp.geomspace(0.003, 3.0, 8)
_SEARCH_FLUX_GRID = onp.geomspace(1e-4, 0.5, 12)
# Fractional bounds of the GP's σ from the single ring's dynamic range.
_GP_SIGMA_BOUNDS = (0.5, 5.0)
_ALPHA_SIGMA = 3.0
_HARMONIC_SIGMA = 0.5
_HARMONIC_NODES = 10
_ALPHA_NODES = 22
# Squared-exponential equivalent of the Matérn ν = 3/2 length ℓ: ℓ/√3.
_ELL_PER_EQUIVALENT = math.sqrt(3.0)
# Δχ² within which another cell of the geometry grid is a rival minimum.
_RIVAL_DELTA_CHI2 = _checks.RIVAL_DELTA_CHI2
_BACKFIT_PASSES = 2
_BACKFIT_MIN_GAIN = 2.0  # BIC units: below this a re-placement is noise
# Coarse starts for the Gaussian process's geometry, beside the single ring's.
_GEOMETRY_LATTICE_INC = (30.0, 50.0, 70.0)
_GEOMETRY_LATTICE_PA = (0.0, 45.0, 90.0, 135.0)
_SCREEN_STEPS = 25


# === THE MODEL REGISTRY ===


@dataclasses.dataclass(frozen=True)
class DiskModel:
    """A model ``DiskPipeline`` can fit, registered under a short name.

    The registry holds parallel options, not a chain: each entry runs the
    ``profile`` and ``compare`` stages start to finish on its own, with its
    own settings and checks. Adding an entry (say a cone) is adding a row
    to ``MODELS``, with no new stage.

    Attributes
    ----------
    name : str
        The short name accepted by ``DiskPipeline(model=...)``.
    label : str
        Human-readable name, for plots and messages.
    disk : str
        What the disk is, in one line.
    profile : callable
        ``profile(pipeline) -> dict``: the ``profile`` stage.
    compare : callable
        ``compare(pipeline) -> dict``: the ``compare`` stage; it saves the
        preferred fit as ``models/best``.
    checks : tuple of str
        The names of the checks that apply (from ``CHECKS``).
    candidate : str
        Prefix of the names of the candidate models under ``models/``.
    """

    name: str
    label: str
    disk: str
    profile: Callable
    compare: Callable
    checks: tuple
    candidate: str


# Every check of the pipeline, in the order they are reported.
CHECKS = (
    "chi2_reduced",
    "length_scale",
    "ring_count",
    "inner_edge",
    "outer_edge",
    "inclination",
    "absorbed_companion",
)

_MODELS = None


def models():
    """The registry of short names to [`DiskModel`][virgil.pipeline.disk.DiskModel]."""
    global _MODELS
    if _MODELS is None:
        _MODELS = {
            e.name: e
            for e in (
                DiskModel(
                    name="rings",
                    label="Gaussian rings",
                    disk="N GaussianRing profiles sharing inc and pa",
                    profile=_profile_rings,
                    compare=_compare_rings,
                    checks=(
                        "chi2_reduced",
                        "ring_count",
                        "outer_edge",
                        "inclination",
                        "absorbed_companion",
                    ),
                    candidate="rings",
                ),
                DiskModel(
                    name="gp",
                    label="Gaussian-process disk",
                    disk="a RadialGP profile with low-degree harmonics",
                    profile=_profile_gp,
                    compare=_compare_gp,
                    checks=(
                        "chi2_reduced",
                        "length_scale",
                        "inner_edge",
                        "outer_edge",
                        "inclination",
                        "absorbed_companion",
                    ),
                    candidate="gp",
                ),
            )
        }
    return _MODELS


# === THE PIPELINE ===


class DiskPipeline(_Pipeline):
    """Fit a disk, ring or halo with a [`MultipoleDisk`][virgil.models.MultipoleDisk].

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The disk is always a ``MultipoleDisk`` and the star a single point
    source; a companion search is an opt-in expert setting.

    Stages: ``load → overview → geometry → profile → compare → posterior →
    quicklook``.

    * ``load`` applies ``wavel_range`` and ``error_floor`` and writes the
      data that are fitted to ``data/processed.oifits``.
    * ``overview`` plots the data and their uv coverage and sets the
      **resolution scale**, the half-fringe λ/2B of the longest baseline.
      The inner radius, the ring-width floor and the length-scale grid are
      stated in half-fringes.
    * ``geometry`` maps χ² over a grid of the inclination (uniform in
      cos ``inc``) and the position angle on [0°, 180°) (an axisymmetric
      disk without height is the same under ``pa`` → ``pa`` + 180°), with a
      single ring at each point, then fits that single ring from the best
      cell. The inclination and position angle are multimodal, so they are
      never left to a local optimizer from one start. The grid map goes to
      ``grids.h5``, the single-ring fit is ``models/geometry``, and its
      profile's dynamic range sets the Gaussian process's σ.
    * ``profile`` and ``compare`` depend on the model; see below.
    * ``posterior`` (``posterior=True``) samples every free parameter with
      NUTS in flat coordinates, with the geometry released.
    * ``quicklook`` writes and executes ``quicklook.ipynb``.

    **Models** (``model=``) are separate, parallel options:

    * ``"rings"``: *N* Gaussian rings share the inclination and position
      angle. Rings are added one at a time, each starting at the radius of a
      one-ring scan (the others held at their fit) with a width of one
      half-fringe, and fitted by Levenberg–Marquardt to the MAP; adding
      stops when the BIC no longer improves, or at ``max_rings``.
      ``compare`` takes the Laplace log evidence and the BIC of every
      count and keeps the one of least BIC.
    * ``"gp"`` (the default): one ``RadialGP`` profile, with low-degree
      ``Harmonics`` and, with several channels, a ``RadialGP`` spectral
      index. At the geometry's inclination and position angle (held, so
      each refit is a matrix-vector product) the profile is fitted at each
      length scale of a grid (ℓ/√3 of 1, 2 and 4 half-fringes by default),
      then refitted with the inclination and position angle released, since
      a single ring's geometry is a few degrees off for a lopsided,
      multi-ring disk;
      ``compare`` keeps the length of largest Laplace evidence. σ comes
      from the dynamic range of the single-ring profile, not from a known
      truth, and the Matérn smoothness is fixed at ν = 3/2.

    The spectral index is relative to the star, which is a grey reference
    source unless ``star_temperature_k`` is given. The fit is on the quoted
    errors unless ``error_scale="fit"`` (rings only, since the Gaussian
    process's evidence needs the quoted errors), and χ²/N is always
    reported on the quoted errors. Priors are group-invariant: uniform in
    ``cos inc``, uniform angles (an ``AngleVector``, with no wrap),
    log-uniform in ring radii, widths and fluxes, and an isotropic Gaussian
    (scale 0.4) on the real and imaginary parts of each ring's harmonics, so
    that the phase of the lopsidedness is uniform.

    Parameters
    ----------
    data : OIData
        The data.
    model : {"gp", "rings"}, optional
        What to fit (default ``"gp"``, the option that assumes least about
        the disk).
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of the defaults (see ``DiskPipeline.defaults()``):

        - ``r_in_mas`` (None), ``r_out_mas`` (None): the radial range; by
          default one half-fringe and half of
          [`image_field`][virgil.imaging.image_field].
        - ``n_r`` (None): quadrature nodes of a profile; by default
          [`suggest_n_r`][virgil.models.MultipoleDisk.suggest_n_r] (for the
          Gaussian process, at least enough to resolve the shortest ℓ).
        - ``n_inc`` (10), ``n_pa`` (12), ``inc_max_deg`` (85): the geometry
          grid.
        - ``max_rings`` (4): the most rings fitted (``"rings"``).
        - ``ell_halffringes`` ([1, 2, 4]): the grid of ℓ/√3, in
          half-fringes (``"gp"``).
        - ``harmonic_degree`` (1): the degree of the harmonics (0 for none).
        - ``spectral_index`` (None): ``True`` or ``False`` to force the
          colour on or off; by default it is on with more than
          ``spectral_index_min_channels`` (3) channels.
        - ``ring_colour`` (``"power_law"``): the colour of each ring and
          of the planet when the colour is on (``"rings"`` only):
          ``"power_law"`` fits a spectral index, ``"blackbody"`` a
          temperature (300 to 3000 K, log-uniform) with the flux at the
          reference wavelength, which represents blackbody-like spectra over
          a broad band that a power law misses.
        - ``wavel0_m`` (None), ``star_temperature_k`` (None): the
          reference wavelength (default the mean wavelength) and the star's
          blackbody (default a grey reference).
        - ``companion`` (False): search the residuals for a point source and
          fit any detection (``companion_min_delta_chi2``, 25) jointly;
          ``companion_max_sep_mas`` (None, i.e. ``r_out_mas``) and
          ``companion_step_mas`` (None, i.e. half a half-fringe) set the
          search grid.
        - ``length_scale_nats`` (3), ``ring_count_nats`` (3),
          ``inner_edge_fraction`` (0.1), ``inner_edge_correlation`` (0.9),
          ``outer_edge_fraction`` (0.01), ``inclination_min_deg`` (15):
          thresholds of the checks.
        - ``wavel_range`` (None), ``error_floor`` (None), ``error_scale``
          ("quoted"), ``batch_size`` (None): as for the other pipelines.
        - ``posterior`` (False), ``num_warmup``, ``num_samples``,
          ``num_chains`` (1000, 1000, 4), ``chain_method`` ("vectorized"),
          ``seed`` (0): NUTS.
        - ``quicklook_execute`` (True): execute the quicklook notebook.

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import DiskPipeline  # doctest: +SKIP
    >>> pipeline = DiskPipeline(vg.OIData("disk.oifits"), "rings", output="runs/disk")  # doctest: +SKIP
    >>> res = pipeline.run()  # doctest: +SKIP
    >>> res.summary["disk"]["rings"]  # doctest: +SKIP
    """

    NAME = "disk"
    STABILITY = "provisional"
    STAGES = (
        "load",
        "overview",
        "geometry",
        "profile",
        "compare",
        "posterior",
        "quicklook",
    )
    _DEFAULTS = {
        "r_in_mas": None,
        "r_out_mas": None,
        "n_r": None,
        "n_inc": 10,
        "n_pa": 12,
        "inc_max_deg": 85.0,
        "max_rings": 4,
        "ell_halffringes": [1.0, 2.0, 4.0],
        "harmonic_degree": 1,
        "spectral_index": None,
        "spectral_index_min_channels": 3,
        "ring_colour": "power_law",
        "wavel0_m": None,
        "star_temperature_k": None,
        "companion": False,
        "companion_min_delta_chi2": 25.0,
        "companion_max_sep_mas": None,
        "companion_step_mas": None,
        "length_scale_nats": 3.0,
        "ring_count_nats": 3.0,
        "inner_edge_fraction": 0.1,
        "inner_edge_correlation": 0.9,
        "outer_edge_fraction": 0.01,
        "inclination_min_deg": 15.0,
        "wavel_range": None,
        "error_floor": None,
        "error_scale": "quoted",
        "batch_size": None,
        "posterior": False,
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "chain_method": "vectorized",
        "seed": 0,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        registry = models()
        name = self._default_model() if model is None else model
        if not isinstance(name, str) or name not in registry:
            raise ValueError(
                f"model must be one of {sorted(registry)}; got {model!r}."
            )
        self.name = name
        super().__init__(data, name, output=output, **settings)
        if name == "gp" and self.settings["ring_colour"] != "power_law":
            raise ValueError(
                'ring_colour="blackbody" is for model="rings": the Gaussian '
                "process's colour is its own spectral index."
            )
        if name == "gp" and self.settings["error_scale"] == "fit":
            raise ValueError(
                'error_scale="fit" is not available for model="gp": the '
                "Laplace evidence that chooses ℓ needs the quoted errors."
            )

    def _default_model(self):
        return "gp"

    def __repr__(self):
        return (
            f"DiskPipeline(output={str(self.output)!r}, model={self.name!r})"
        )

    @classmethod
    def _validate(cls, s):
        if s["r_in_mas"] is not None and not s["r_in_mas"] > 0:
            raise ValueError("r_in_mas must be positive.")
        if s["r_out_mas"] is not None and not s["r_out_mas"] > (
            s["r_in_mas"] or 0.0
        ):
            raise ValueError("r_out_mas must exceed r_in_mas.")
        if s["n_r"] is not None and not (
            isinstance(s["n_r"], int) and 8 <= s["n_r"] <= 144
        ):
            raise ValueError("n_r must be an integer from 8 to 144.")
        for name in ("n_inc", "n_pa"):
            if not (isinstance(s[name], int) and s[name] >= 2):
                raise ValueError(f"{name} must be an integer >= 2.")
        if not 0.0 < s["inc_max_deg"] < 90.0:
            raise ValueError("inc_max_deg must be between 0 and 90.")
        if not (isinstance(s["max_rings"], int) and s["max_rings"] >= 1):
            raise ValueError("max_rings must be an integer >= 1.")
        ell = s["ell_halffringes"]
        if not ell or any(not v > 0 for v in ell):
            raise ValueError("ell_halffringes must be positive numbers.")
        if not (
            isinstance(s["harmonic_degree"], int) and s["harmonic_degree"] >= 0
        ):
            raise ValueError("harmonic_degree must be an integer >= 0.")
        if s["spectral_index"] not in (None, True, False):
            raise ValueError("spectral_index must be None, True or False.")
        if s["ring_colour"] not in ("power_law", "blackbody"):
            raise ValueError('ring_colour must be "power_law" or "blackbody".')
        if s["error_scale"] not in ("quoted", "fit"):
            raise ValueError('error_scale must be "quoted" or "fit".')
        for name in ("num_samples", "num_chains"):
            if not (isinstance(s[name], int) and s[name] >= 1):
                raise ValueError(f"{name} must be an integer >= 1.")
        if not (isinstance(s["num_warmup"], int) and s["num_warmup"] >= 0):
            raise ValueError("num_warmup must be an integer >= 0.")
        if s["chain_method"] not in ("vectorized", "sequential", "parallel"):
            raise ValueError(
                'chain_method must be "vectorized", "sequential" or "parallel".'
            )
        if s["wavel_range"] is not None and len(s["wavel_range"]) != 2:
            raise ValueError("wavel_range must be [min, max] in metres.")
        if s["error_floor"] is not None and not isinstance(
            s["error_floor"], dict
        ):
            raise ValueError("error_floor must be a dict of absolute floors.")

    def _model_digest(self):
        return {
            "class": "MultipoleDisk",
            "digest": None,
            "models": [self.name],
        }

    def _ctx(self):
        """The resolution scale and the defaults derived from it."""
        return _Context.from_report(self.settings, self._report("overview"))

    def _stages(self):
        return (
            Stage("load", _load_disk, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage(
                "geometry",
                _geometry_stage,
                (
                    "grids.h5",
                    "plots/geometry_grid.png",
                    *_model_files("geometry"),
                ),
            ),
            Stage(
                "profile",
                _profile,
                ("stages/profile/report.json", "plots/profile_fits.png"),
            ),
            Stage(
                "compare",
                _compare,
                ("plots/disk_summary.png", *_model_files("best")),
            ),
            Stage("posterior", _posterior, ("stages/posterior/report.json",)),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        return _summarize(self.name, self.settings, reports)


def _model_files(name):
    """The files ``_save_fit`` writes for the model ``models/<name>``."""
    return tuple(
        f"models/{name}/{f}"
        for f in ("manifest.json", "values.npz", "info.json")
    )


# === CONTEXT: THE RESOLUTION SCALE AND THE DEFAULTS THAT FOLLOW ===


@dataclasses.dataclass(frozen=True)
class _Context:
    """Numbers every stage needs, from the overview report and settings."""

    halffringe: float
    r_in: float
    r_out: float
    n_ring: int
    n_gp: int
    wavel0: float
    colour: bool
    law: str
    degree: int
    star_teff: float | None
    companion: bool
    max_sep: float
    step: float
    inc_max: float
    ells: tuple

    @classmethod
    def from_report(cls, settings, overview):
        return cls(
            halffringe=overview["resolution_mas"],
            r_in=overview["r_in_mas"],
            r_out=overview["r_out_mas"],
            n_ring=overview["n_r_rings"],
            n_gp=overview["n_r_gp"],
            wavel0=overview["wavel0_m"],
            colour=overview["colour"],
            law=settings["ring_colour"],
            degree=int(settings["harmonic_degree"]),
            star_teff=settings["star_temperature_k"],
            companion=bool(settings["companion"]),
            max_sep=overview["companion_max_sep_mas"],
            step=overview["companion_step_mas"],
            inc_max=float(settings["inc_max_deg"]),
            ells=tuple(overview["ell_mas"]),
        )

    @property
    def ell_mid(self):
        return self.ells[len(self.ells) // 2]


def _channels(data):
    wavel = onp.asarray(data.wavel, dtype=float)
    return int(onp.unique(onp.round(wavel, 12)).size)


def _wavel0(data, settings):
    if settings["wavel0_m"] is not None:
        return float(settings["wavel0_m"])
    return float(onp.mean(onp.unique(onp.asarray(data.wavel, dtype=float))))


def _n_nodes(settings, data, r_in, r_out, halffringe, ell_min):
    """Quadrature nodes for the ring family and for the Gaussian process."""
    from ..models import GaussianRing, MultipoleDisk, RadialGP

    if settings["n_r"] is not None:
        return int(settings["n_r"]), int(settings["n_r"])
    ring = MultipoleDisk(
        GaussianRing(0.5 * r_out, halffringe), inc=40.0, pa=0.0
    ).suggest_n_r(data)
    gp = MultipoleDisk(RadialGP(r_in, r_out), n_r=16).suggest_n_r(data)
    # Four nodes per length scale resolve the shortest ℓ.
    per_ell = int(8 * math.ceil(4.0 * (r_out - r_in) / ell_min / 8.0))
    return int(ring), int(min(max(gp, per_ell, 16), 96))


# === STAGES: LOAD AND OVERVIEW ===


def _load_disk(p):
    return _load_report(p)


def _overview(p):
    from ..imaging import image_field
    from ..plotting import plot_oidata_overview, plot_uv_coverage

    s, data = p.settings, p.processed
    fig, _ = plot_oidata_overview(data)
    _save(fig, p.output / "plots" / "overview_data.png")
    fig, _ = plot_uv_coverage(data)
    _save(fig, p.output / "plots" / "overview_uv.png")

    halffringe = _geometry(data)["resolution_mas"]
    r_in = float(s["r_in_mas"] or halffringe)
    r_out = float(s["r_out_mas"] or 0.5 * image_field(data))
    if not r_out > r_in:
        raise ValueError(
            f"The default outer radius {r_out:.3g} mas does not exceed the "
            f"inner radius {r_in:.3g} mas; give r_out_mas."
        )
    ell = [_ELL_PER_EQUIVALENT * k * halffringe for k in s["ell_halffringes"]]
    n_rings, n_gp = _n_nodes(s, data, r_in, r_out, halffringe, min(ell))
    channels = _channels(data)
    colour = (
        channels > s["spectral_index_min_channels"]
        if s["spectral_index"] is None
        else bool(s["spectral_index"])
    )
    return {
        "resolution_mas": halffringe,
        "r_in_mas": r_in,
        "r_out_mas": r_out,
        "n_r_rings": n_rings,
        "n_r_gp": n_gp,
        "n_channels": channels,
        "colour": bool(colour),
        "wavel0_m": _wavel0(data, s),
        "ell_mas": ell,
        "companion_max_sep_mas": float(s["companion_max_sep_mas"] or r_out),
        "companion_step_mas": float(
            s["companion_step_mas"] or 0.5 * halffringe
        ),
    }


# === SCENES, PRIORS AND FITS ===


@dataclasses.dataclass(frozen=True)
class _Shape:
    """What a ring scene contains: rings, colour, harmonics and a planet."""

    n: int
    colour: bool
    degree: int
    planet: bool


def _star(ctx):
    from ..models import PointSource
    from ..spectra import BlackBody

    if ctx.star_teff is None:
        return PointSource(flux=1.0)
    return PointSource(flux=BlackBody(1.0, ctx.star_teff, ctx.wavel0))


def _flux(ctx, ratio, colour_value, colour):
    """A source's flux: ``ratio``, or with colour a power law of index
    ``colour_value`` or a blackbody of that temperature (``ring_colour``)."""
    from ..spectra import BlackBody, PowerLaw

    if not colour:
        return ratio
    if ctx.law == "blackbody":
        return BlackBody(ratio, colour_value, ctx.wavel0)
    return PowerLaw(ratio, colour_value, ctx.wavel0)


def _colour_start(ctx):
    """The starting value of a colour parameter."""
    return _TEMPERATURE_START if ctx.law == "blackbody" else 0.0


def _colour_prior(ctx):
    import numpyro.distributions as dist

    if ctx.law == "blackbody":
        return dist.LogUniform(*_TEMPERATURE_RANGE)
    return dist.Uniform(*_INDEX_RANGE)


def _planet(ctx, v, colour):
    from ..models import PointSource

    return PointSource(
        dra=v["dra"],
        ddec=v["ddec"],
        flux=_flux(ctx, v["fp"], v["ap"] if colour else 0.0, colour),
    )


def _ring_scene(ctx, shape, v):
    """Star, ``shape.n`` coplanar rings and optionally a planet.

    One ``inc`` and ``pa`` go to every ring.
    """
    from ..models import GaussianRing, Harmonics, MultipoleDisk, System

    parts = {"star": _star(ctx)}
    for i in range(1, shape.n + 1):
        harmonics = None
        if shape.degree:
            harmonics = Harmonics(
                re=tuple(v[f"h{i}_re{j}"] for j in range(1, shape.degree + 1)),
                im=tuple(v[f"h{i}_im{j}"] for j in range(1, shape.degree + 1)),
            )
        parts[f"ring{i}"] = MultipoleDisk(
            GaussianRing(v[f"r{i}"], v[f"s{i}"]),
            inc=v["inc"],
            pa=v["pa"],
            n_r=ctx.n_ring,
            flux=_flux(
                ctx,
                v[f"f{i}"],
                v[f"a{i}"] if shape.colour else 0.0,
                shape.colour,
            ),
            harmonics=harmonics,
        )
    if shape.planet:
        parts["planet"] = _planet(ctx, v, shape.colour)
    return System(**parts)


def _builder(ctx, shape):
    """The scene as a function of its parameters (``fit`` calls it by name)."""

    def build(**v):
        return _ring_scene(ctx, shape, v)

    return build


def _geometry_priors(ctx):
    from ..angles import AngleVector
    from ..priors import IsotropicInclination

    return {
        "inc": IsotropicInclination(0.0, ctx.inc_max),
        "pa": AngleVector(),
    }


def _planet_priors(ctx, colour):
    import numpyro.distributions as dist

    priors = {
        "dra": dist.Uniform(-ctx.max_sep, ctx.max_sep),
        "ddec": dist.Uniform(-ctx.max_sep, ctx.max_sep),
        "fp": dist.LogUniform(*_PLANET_FLUX_RANGE),
    }
    if colour:
        priors["ap"] = _colour_prior(ctx)
    return priors


def _ring_priors(ctx, shape):
    import numpyro.distributions as dist

    priors = _geometry_priors(ctx)
    for i in range(1, shape.n + 1):
        priors[f"r{i}"] = dist.LogUniform(ctx.r_in, ctx.r_out)
        priors[f"s{i}"] = dist.LogUniform(
            _WIDTH_FLOOR * ctx.halffringe, 0.5 * (ctx.r_out - ctx.r_in)
        )
        priors[f"f{i}"] = dist.LogUniform(*_FLUX_RANGE)
        if shape.colour:
            priors[f"a{i}"] = _colour_prior(ctx)
        for j in range(1, shape.degree + 1):
            priors[f"h{i}_re{j}"] = dist.Normal(0.0, _HARMONIC_SCALE)
            priors[f"h{i}_im{j}"] = dist.Normal(0.0, _HARMONIC_SCALE)
    if shape.planet:
        priors.update(_planet_priors(ctx, shape.colour))
    return priors


def _scalars(values):
    """Fitted values as floats (the angle's vector and noise terms apart)."""
    return {
        k: float(onp.asarray(v))
        for k, v in values.items()
        if not k.endswith("_vec") and onp.size(v) == 1
    }


@dataclasses.dataclass(frozen=True)
class _DiskFit:
    """A fitted model with its χ² on the quoted errors.

    ``score`` is the −2 log L the BIC uses: at the fitted error scales (with
    their normalization) under ``error_scale="fit"``, where χ² on the quoted
    errors would reward every extra ring by the unmodelled noise; None means
    the quoted-error χ².
    """

    model: object
    values: dict
    info: dict
    chi2: float
    scales: dict
    n_free: int
    score: float | None = None


def _scales_of(values):
    return {
        k.split(".", 1)[1]: float(v)
        for k, v in values.items()
        if k.startswith("noise.")
    }


@in_float64
def _chi2_64(model, data):
    return _chi2(model, data)


@in_float64
def _m2loglike_64(model, data, scales):
    return _m2loglike(model, data, **scales)


def _n_free(values, priors):
    return int(
        sum(onp.size(values[k]) for k in priors if k in values)
        + sum(1 for k in values if k.startswith("noise."))
    )


def _finish(model, values, info, priors, data):
    """A ``_DiskFit`` from a ``fit`` result's pieces."""
    chi2, _ = _chi2_64(model, data)
    scales = _scales_of(values)
    return _DiskFit(
        model=model,
        values={k: onp.asarray(v) for k, v in values.items()},
        info=_io.clean_json(dict(info)),
        chi2=float(chi2),
        scales=scales,
        n_free=_n_free(values, priors),
        score=_m2loglike_64(model, data, scales) if scales else None,
    )


def _inside_priors(priors, start):
    """``start`` with any value outside its prior's support pulled just inside.

    A ring's starting width is one resolution element, which exceeds the
    width prior's upper end (half the radial range) when the field holds
    less than two of them, and a radius saved in single precision can fall a
    rounding error below the radial range. A start outside the support has
    no unconstrained value, and the fit returns NaN. Values on or inside a
    bound stay as they are.
    """
    from .._flat import _base
    from ..fitting import _bound_fraction

    outside = {}
    for name, value in start.items():
        prior = priors.get(name)
        if prior is None or _bound_fraction(prior, value) is None:
            continue  # no prior, or none with finite edges
        if not onp.all(_base(prior).support(onp.asarray(value, float))):
            outside[name] = value
    return {**start, **_interior_start(priors, outside)}


def _fit_rings(p, shape, init):
    """MAP fit of a ring scene from ``init``: a ``_DiskFit``."""
    from ..fitting import fit

    ctx, data = p._ctx(), p.processed
    priors = _ring_priors(ctx, shape)
    result = fit(
        _builder(ctx, shape),
        priors,
        data,
        noise=p._noise(),
        init=_inside_priors(priors, init),
    )
    model, values = _fold_fit(ctx, shape, priors, result.model, result.values)
    return _finish(model, values, result.info, priors, data)


def _fold_fit(ctx, shape, priors, model, values):
    """The fit with ``pa`` in [0°, 180°), the harmonics turned to match.

    ``AngleVector`` returns [0°, 360°), and ``pa`` and ``pa`` + 180° are the
    same disk once the order-*m* harmonics are multiplied by ``(-1)^m``. The
    grid and the posterior use [0°, 180°); the MAP is folded to match before
    it is saved or summarized.
    """
    pa = float(onp.asarray(values["pa"]))
    if 0.0 <= pa < 180.0:
        return model, values
    out = dict(values)
    folded = pa % 180.0
    out["pa"] = onp.asarray(folded)
    if "pa_vec" in out:
        angle = math.radians(folded)
        out["pa_vec"] = onp.asarray([math.cos(angle), math.sin(angle)])
    if (pa % 360.0) >= 180.0:
        for i in range(1, shape.n + 1):
            for j in range(1, shape.degree + 1):
                for part in ("re", "im"):
                    key = f"h{i}_{part}{j}"
                    out[key] = onp.asarray(out[key]) * (-1.0) ** j
    scene = _builder(ctx, shape)(
        **{k: onp.asarray(out[k]) for k in priors if k in out}
    )
    return scene, out


# === STAGE: GEOMETRY ===


def _single_ring(ctx, *, cos_inc, pa, r1, f1):
    """One ring of width one half-fringe, for the geometry grid."""
    import jax.numpy as np

    v = {
        "inc": np.degrees(np.arccos(cos_inc)),
        "pa": pa,
        "r1": r1,
        "s1": ctx.halffringe,
        "f1": f1,
    }
    return _ring_scene(ctx, _Shape(1, False, 0, False), v)


@in_float64
def _geometry_grid(p, ctx, cos_axis, pa_axis, r_axis):
    import jax.numpy as np

    from ..grid_fit import likelihood_grid

    ll = onp.asarray(
        likelihood_grid(
            functools.partial(_single_ring, ctx),
            data=p.processed,
            grid={
                "cos_inc": np.asarray(cos_axis),
                "pa": np.asarray(pa_axis),
                "r1": np.asarray(r_axis),
                "f1": np.asarray(_FLUX_GRID),
            },
            batch_size=p.settings["batch_size"],
        )
    )
    return ll


def _grid_minima(delta, inc_deg):
    """Local minima of the geometry map ``delta[inc, pa]``.

    The position angle is periodic (the map spans [0°, 180°)); the
    inclination is not. Cells within 5° of face-on are skipped: their
    position angle is undefined, so they form a flat ridge.
    """
    n_inc, n_pa = delta.shape
    minima = []
    for i in range(n_inc):
        if inc_deg[i] < 5.0:
            continue
        for j in range(n_pa):
            value = delta[i, j]
            around = [
                delta[a, (j + b) % n_pa]
                for a in (i - 1, i, i + 1)
                for b in (-1, 0, 1)
                if (a, b) != (i, 0) and 0 <= a < n_inc
            ]
            if all(value <= v for v in around):
                minima.append((float(value), i, j))
    return sorted(minima)


def _dynamic_range_sigma(ctx, r0, sigma):
    """The Gaussian process's σ: ln of the single ring's peak-to-mean contrast.

    The contrast is the peak intensity over the intensity averaged over the
    radial range by area, ``(r_out² − r_in²) / (2 √(2π) σ r0)`` for a
    Gaussian ring, so ``exp(±σ)`` spans the profile's dynamic range.
    """
    contrast = (ctx.r_out**2 - ctx.r_in**2) / (
        2.0 * math.sqrt(2.0 * math.pi) * sigma * r0
    )
    lo, hi = _GP_SIGMA_BOUNDS
    return float(min(max(math.log(max(contrast, 1.0)), lo), hi))


def _plot_geometry(p, cos_axis, pa_axis, delta):
    import matplotlib.pyplot as plt

    inc = onp.degrees(onp.arccos(cos_axis))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    shown = onp.clip(delta, 0.0, onp.nanmin(delta) + 400.0)
    mesh = ax.pcolormesh(pa_axis, inc, shown, shading="auto", cmap="viridis_r")
    fig.colorbar(mesh, ax=ax, label="Δχ² (clipped)")
    ax.set(xlabel="position angle (deg)", ylabel="inclination (deg)")
    ax.set_title("Single-ring geometry grid")
    _save(fig, p.output / "plots" / "geometry_grid.png")


def _geometry_stage(p):
    ctx, data = p._ctx(), p.processed
    s = p.settings
    inc_max = math.radians(ctx.inc_max)
    cos_axis = onp.linspace(1.0, math.cos(inc_max), s["n_inc"])
    pa_axis = onp.linspace(0.0, 180.0, s["n_pa"], endpoint=False)
    r_axis = onp.arange(ctx.r_in, ctx.r_out + 1e-9, ctx.halffringe)
    if r_axis.size < 2:
        r_axis = onp.linspace(ctx.r_in, ctx.r_out, 2)

    ll = _geometry_grid(p, ctx, cos_axis, pa_axis, r_axis)
    ll = onp.where(onp.isfinite(ll), ll, -onp.inf)
    best_cell = ll.reshape(ll.shape[:2] + (-1,)).max(axis=-1)
    flat = ll.reshape(ll.shape[:2] + (-1,)).argmax(axis=-1)
    delta = 2.0 * (best_cell.max() - best_cell)
    i, j = onp.unravel_index(int(onp.argmax(best_cell)), best_cell.shape)
    k_r, k_f = onp.unravel_index(
        int(flat[i, j]), (r_axis.size, _FLUX_GRID.size)
    )
    inc_deg = onp.degrees(onp.arccos(cos_axis))
    minima = _grid_minima(delta, inc_deg)
    rivals = [m for m in minima[1:] if m[0] < _RIVAL_DELTA_CHI2]

    _io.write_h5(
        p.output / "grids.h5",
        {
            "axes": {
                "cos_inc": (cos_axis, {"units": ""}),
                "pa": (pa_axis, {"units": "deg"}),
            },
            "geometry": {
                "delta_chi2": delta,
                "r1_mas": r_axis[
                    ll.reshape(ll.shape[:2] + (-1,)).argmax(axis=-1)
                    // _FLUX_GRID.size
                ],
            },
        },
        attrs={
            "/": {"axis_order": ["cos_inc", "pa"], "schema": _io.SCHEMA},
            "axes": {"inc_deg": inc_deg},
        },
    )
    _plot_geometry(p, cos_axis, pa_axis, delta)

    shape = _Shape(1, False, 0, False)
    start = {
        "inc": float(inc_deg[i]),
        "pa": float(pa_axis[j]),
        "r1": float(r_axis[k_r]),
        "s1": ctx.halffringe,
        "f1": float(_FLUX_GRID[k_f]),
    }
    # Face-on has no position angle: start a little away from it.
    start["inc"] = max(start["inc"], 5.0)
    fit = _fit_rings(p, shape, start)
    values = _scalars(fit.values)
    names = list(_ring_priors(ctx, shape))
    sd = _laplace_sd(ctx, shape, names, values, data)
    n = int(data.n_independent)
    _save_fit(p.output / "models" / "geometry", fit, names)
    return {
        "inc_deg": values["inc"],
        "pa_deg": values["pa"],
        "r_mas": values["r1"],
        "sigma_mas": values["s1"],
        "flux": values["f1"],
        "chi2": fit.chi2,
        "chi2_reduced": fit.chi2 / n,
        "n_independent": n,
        "converged": fit.info.get("converged"),
        "laplace_sd": sd,
        "gp_sigma": _dynamic_range_sigma(ctx, values["r1"], values["s1"]),
        "grid": {
            "shape": list(delta.shape),
            "best_index": [int(i), int(j)],
            "n_minima": 1 + len(rivals),
            "rivals": [
                {
                    "inc_deg": float(inc_deg[a]),
                    "pa_deg": float(pa_axis[b]),
                    "delta_chi2": d,
                }
                for d, a, b in rivals
            ],
        },
    }


def _save_fit(folder, fit, params):
    _io.save_model(folder, fit.model, fit.values, params)
    _io.write_json(folder / "info.json", fit.info)


def _laplace_sd(ctx, shape, names, values, data):
    """Laplace standard deviations of the ring scene's scalar parameters."""
    from ..inference import laplace_cov

    try:
        x = onp.array([values[k] for k in names], dtype=float)
        cov = onp.asarray(laplace_cov(x, names, _builder(ctx, shape), data))
        sd = onp.sqrt(onp.clip(onp.diag(cov), 0.0, None))
    except Exception as err:  # noqa: BLE001 - a Laplace failure is a result
        warnings.warn(
            f"The Laplace covariance could not be computed ({err}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return {k: None for k in names}
    return {k: float(v) for k, v in zip(names, sd)}


# === POINT-SOURCE SEARCH ON RESIDUALS ===


@in_float64
def _point_source_search(model, data, ctx, batch_size=None):
    """The best point source to add to ``model``, on a grid of positions.

    Adds a grey ``PointSource`` and maximizes the likelihood over its flux
    at every position of a grid; the Δχ² is against ``model`` itself.

    Returns
    -------
    dict
        ``delta_chi2``, ``dra_mas``, ``ddec_mas`` and ``flux`` of the best
        position.
    """
    import jax.numpy as np

    from .._grid import batch_size_or_default, resolve_grid_keys
    from ..grid_fit import _optimize_flux_grid
    from ..likelihood import model_loglike
    from ..models import PointSource, System

    probe = System({**model.components, "probe": PointSource(flux=0.01)})
    steps = int(math.floor(ctx.max_sep / ctx.step))
    axis = ctx.step * onp.arange(-steps, steps + 1)
    grid = {
        "probe.dra": np.asarray(axis),
        "probe.ddec": np.asarray(axis),
        "probe.flux": np.asarray(_SEARCH_FLUX_GRID),
    }
    params, coord_keys, flux_key = resolve_grid_keys(grid, "probe.flux")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unconverged cells lose, not warn
        flux, loglike, _ = _optimize_flux_grid(
            data,
            probe,
            grid,
            params=params,
            coord_keys=coord_keys,
            flux_key=flux_key,
            batch_size=batch_size_or_default(batch_size, data),
        )
    flux, loglike = onp.asarray(flux), onp.asarray(loglike)
    null = float(model_loglike(model, data))
    delta = onp.where(
        (flux > 0) & onp.isfinite(loglike),
        onp.maximum(2.0 * (loglike - null), 0.0),
        0.0,
    )
    a, b = onp.unravel_index(int(onp.argmax(delta)), delta.shape)
    return {
        "delta_chi2": float(delta[a, b]),
        "dra_mas": float(axis[a]),
        "ddec_mas": float(axis[b]),
        "flux": float(flux[a, b]),
    }


# === STAGE: PROFILE ===


def _profile(p):
    return models()[p.name].profile(p)


def _geometry_values(p):
    """The single-ring fit of the ``geometry`` stage, as floats."""
    return _scalars(_io.load_model_values(p.output / "models" / "geometry"))


def _planet_start(search, ctx):
    start = {
        "dra": search["dra_mas"],
        "ddec": search["ddec_mas"],
        "fp": min(max(search["flux"], _PLANET_FLUX_RANGE[0] * 2), 0.5),
    }
    if ctx.colour:
        start["ap"] = _colour_start(ctx)
    return start


def _ring_start(ctx, held, n, new, degree, colour):
    """Starting values of an ``n``-ring scene.

    ``held`` has the earlier rings' values; ``new`` is the scan's radius and
    flux for ring ``n`` (None for the first ring, which ``held`` has).
    """
    start = {k: v for k, v in held.items() if k != "pa_vec"}
    start["inc"] = max(float(start["inc"]), 5.0)
    if new is not None:
        start[f"r{n}"], start[f"s{n}"], start[f"f{n}"] = (
            new["r"],
            ctx.halffringe,
            new["f"],
        )
    for ring in range(1, n + 1):
        if colour:
            start.setdefault(f"a{ring}", _colour_start(ctx))
        for j in range(1, degree + 1):
            start.setdefault(f"h{ring}_re{j}", 0.0)
            start.setdefault(f"h{ring}_im{j}", 0.0)
    return start


def _scan_model(ctx, shape, base, *, rnew, fnew):
    v = dict(base)
    i = shape.n
    v[f"r{i}"], v[f"s{i}"], v[f"f{i}"] = rnew, ctx.halffringe, fnew
    if shape.colour:
        v[f"a{i}"] = _colour_start(ctx)
    for j in range(1, shape.degree + 1):
        v[f"h{i}_re{j}"] = 0.0
        v[f"h{i}_im{j}"] = 0.0
    return _ring_scene(ctx, shape, v)


@in_float64
def _scan_new_ring(p, ctx, shape, base, step=None):
    """The radius and flux of one more ring, scanned with the rest held.

    The radii are ``step`` apart (default one half-fringe).
    """
    import jax.numpy as np

    from ..grid_fit import likelihood_grid

    r_axis = onp.arange(ctx.r_in, ctx.r_out + 1e-9, step or ctx.halffringe)
    ll = onp.asarray(
        likelihood_grid(
            functools.partial(_scan_model, ctx, shape, base),
            data=p.processed,
            grid={"rnew": np.asarray(r_axis), "fnew": np.asarray(_FLUX_GRID)},
            batch_size=p.settings["batch_size"],
        )
    )
    ll = onp.where(onp.isfinite(ll), ll, -onp.inf)
    a, b = onp.unravel_index(int(onp.argmax(ll)), ll.shape)
    return {"r": float(r_axis[a]), "f": float(_FLUX_GRID[b])}


def _bic(fit, n):
    score = fit.chi2 if fit.score is None else fit.score
    return score + fit.n_free * math.log(n)


def _fit_with_planet(p, shape, start, backfit=False):
    """Fit ``shape`` (without a planet), search the residuals, refit with one.

    With ``backfit`` the rings are then re-placed (``_backfit``) in the scene
    that is returned, planet included: the objective is the one reported, so
    the back-fit can only lower it. Returns ``(fit, shape, search, steps)``:
    the search result is None unless ``companion`` is on, the returned shape
    has the planet if the search found one, and ``steps`` records each
    back-fit step (empty without ``backfit``).
    """
    ctx = p._ctx()
    fit = _fit_rings(p, shape, start)
    search = None
    if ctx.companion:
        search = _point_source_search(
            fit.model, p.processed, ctx, p.settings["batch_size"]
        )
        if search["delta_chi2"] >= p.settings["companion_min_delta_chi2"]:
            shape = dataclasses.replace(shape, planet=True)
            values = {**_scalars(fit.values), **_planet_start(search, ctx)}
            fit = _fit_rings(p, shape, values)
    steps = []
    if backfit:
        fit, steps = _backfit(p, shape, fit)
    return fit, shape, search, steps


_RING_KEY = re.compile(r"^(?:[rsfa]|h)(\d+)(?:_(?:re|im)\d+)?$")


def _without_ring(values, k):
    """``values`` of a ring scene without ring ``k``, the later rings
    renumbered down (geometry, planet and noise terms kept)."""
    out = {}
    for key, v in values.items():
        m = _RING_KEY.match(key)
        if m is None:
            out[key] = v
            continue
        i = int(m.group(1))
        if i == k:
            continue
        j = i - (i > k)
        out[key[: m.start(1)] + str(j) + key[m.end(1) :]] = v
    return out


def _unresolved_pairs(ctx, values, n):
    """Pairs of the ``n`` rings closer than the data can tell apart.

    Two rings are one feature, fitted as two, when their radii differ by
    less than twice the narrower one's width (two equal Gaussians are
    unimodal within 2σ; a narrow ring on a broad one is a second feature
    however broad the other) or by less than a half-fringe (the
    visibilities hold nothing finer). Both limits are lengths in the same
    units as the radii, so the rule has no preferred scale beyond the data's
    own. Returns ``(i, j, separation, limit)`` for each such pair, ``i < j``,
    in mas.
    """
    pairs = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            sep = abs(float(values[f"r{i}"]) - float(values[f"r{j}"]))
            limit = max(
                2.0 * min(float(values[f"s{i}"]), float(values[f"s{j}"])),
                ctx.halffringe,
            )
            if sep < limit:
                pairs.append((i, j, sep, limit))
    return pairs


def _backfit(p, shape, fit):
    """Re-place each ring in turn and keep what improves the fit.

    Rings are added one at a time at the radius of a scan, so a ring can be
    left away from where the others would put it once they are all there. For
    each ring in turn this drops it, scans for the best radius of a ring added
    to the rest on a grid twice as fine as the first scan's, refits every
    parameter (geometry included) from there and keeps the result if the BIC
    falls by more than ``_BACKFIT_MIN_GAIN`` and it leaves no more rings
    unresolved from each other (``_unresolved_pairs``) than before. A pass
    that changes nothing ends it, as does ``_BACKFIT_PASSES``.

    Returns ``(fit, steps)``; each step records the pass, the ring dropped and
    its radius, the scan's best radius and flux, the refit's χ², geometry and
    radii, the BIC change (negative is better), the number of unresolved pairs
    of rings and whether it was kept.
    """
    ctx, n_data = p._ctx(), int(p.processed.n_independent)
    steps = []
    if shape.n < 2:
        return fit, steps
    for sweep in range(1, _BACKFIT_PASSES + 1):
        improved = False
        for k in range(1, shape.n + 1):
            held = _without_ring(_scalars(fit.values), k)
            base = _fill_new(ctx, held, shape.n)
            new = _scan_new_ring(
                p, ctx, shape, base, step=0.5 * ctx.halffringe
            )
            start = _ring_start(
                ctx, held, shape.n, new, ctx.degree, ctx.colour
            )
            trial = _fit_rings(p, shape, start)
            d_bic = _bic(trial, n_data) - _bic(fit, n_data)
            pairs = _unresolved_pairs(ctx, trial.values, shape.n)
            accepted = d_bic < -_BACKFIT_MIN_GAIN and len(pairs) <= len(
                _unresolved_pairs(ctx, fit.values, shape.n)
            )
            steps.append(
                {
                    "pass": sweep,
                    "ring": k,
                    "dropped_r": float(fit.values[f"r{k}"]),
                    "scan_r": new["r"],
                    "scan_f": new["f"],
                    "chi2_reduced": trial.chi2 / n_data,
                    "inc": float(trial.values["inc"]),
                    "pa": float(trial.values["pa"]),
                    "radii": [
                        float(trial.values[f"r{i}"])
                        for i in range(1, shape.n + 1)
                    ],
                    "delta_bic": d_bic,
                    "n_unresolved": len(pairs),
                    "converged": trial.info.get("converged"),
                    "accepted": accepted,
                }
            )
            if accepted:
                fit, improved = trial, True
        if not improved:
            break
    return fit, steps


def _prune(p, shape, fit):
    """Drop every ring whose removal lowers the BIC, one at a time.

    A greedy fit can end with a ring that carries no light (parked at the
    inner radius, at the width floor and the flux floor): the data do not
    want it, and a fit without it has the same χ² and fewer parameters. Each
    ring in turn is dropped and the rest refitted from where they are; the
    drop that lowers the BIC most is kept, and this repeats until none does.

    Returns ``(fit, shape, dropped)``; each dropped ring is recorded with its
    radius, width and flux and the BIC change (negative is better).
    """
    ctx, n_data = p._ctx(), int(p.processed.n_independent)
    dropped = []
    while shape.n > 1:
        smaller = dataclasses.replace(shape, n=shape.n - 1)
        best = None
        for k in range(1, shape.n + 1):
            held = _without_ring(_scalars(fit.values), k)
            start = _ring_start(
                ctx, held, smaller.n, None, ctx.degree, ctx.colour
            )
            trial = _fit_rings(p, smaller, start)
            d_bic = _bic(trial, n_data) - _bic(fit, n_data)
            if d_bic < 0 and (best is None or d_bic < best[2]):
                best = (k, trial, d_bic)
        if best is None:
            break
        k, trial, d_bic = best
        dropped.append(
            {
                "r": float(fit.values[f"r{k}"]),
                "s": float(fit.values[f"s{k}"]),
                "f": float(fit.values[f"f{k}"]),
                "delta_bic": d_bic,
            }
        )
        fit, shape = trial, smaller
    return fit, shape, dropped


def _profile_rings(p):
    s, ctx, data = p.settings, p._ctx(), p.processed
    n_data = int(data.n_independent)
    geo = _geometry_values(p)
    candidates, searches = [], []
    previous = {k: geo[k] for k in ("inc", "pa", "r1", "s1", "f1")}
    for n in range(1, s["max_rings"] + 1):
        # Each count is fitted on its own without a companion, searched for
        # one, and refitted with it if one is found: a companion found with
        # too few rings (the next ring's light) does not carry over.
        shape = _Shape(n, ctx.colour, ctx.degree, False)
        held = {
            k: v
            for k, v in previous.items()
            if not k.startswith(("dra", "ddec", "fp", "ap"))
        }
        if n == 1:
            start = _ring_start(ctx, held, 1, None, ctx.degree, ctx.colour)
        else:
            new = _scan_new_ring(p, ctx, shape, _fill_new(ctx, held, n))
            start = _ring_start(ctx, held, n, new, ctx.degree, ctx.colour)
        fit, shape, search, steps = _fit_with_planet(
            p, shape, start, backfit=True
        )
        if search is not None:
            searches.append({"rings": n, **search})
        fit, shape, dropped = _prune(p, shape, fit)
        previous = _scalars(fit.values)
        # A ring that had to be dropped was one too many: the rings are used
        # up. The pruned fit is the candidate for its own count (and replaces
        # that count's earlier fit if it beats it).
        count = shape.n
        name = f"rings_{count}"
        unresolved = _unresolved_pairs(ctx, fit.values, count)
        earlier = next((c for c in candidates if c["n"] == count), None)
        if earlier is not None and earlier["bic"] <= _bic(fit, n_data):
            break
        candidates = [c for c in candidates if c["n"] != count]
        _save_fit(
            p.output / "models" / name, fit, list(_ring_priors(ctx, shape))
        )
        candidates.append(
            {
                "n": count,
                "name": name,
                "chi2": fit.chi2,
                "chi2_reduced": fit.chi2 / n_data,
                "n_free": fit.n_free,
                "bic": _bic(fit, n_data),
                "converged": fit.info.get("converged"),
                "planet": shape.planet,
                "unresolved": [list(pair) for pair in unresolved],
                "pruned": dropped,
                "backfit": steps,
            }
        )
        candidates.sort(key=lambda c: c["n"])
        if dropped:
            break
        # A count with two rings too close to tell apart is one ring split
        # in two (a profile that is not Gaussian) or a ring still to be
        # found; `compare` leaves it out, and the search goes on from it. It
        # stops when a count does not beat the last resolved one.
        resolved = [c for c in candidates[:-1] if not c["unresolved"]]
        if (
            not unresolved
            and resolved
            and candidates[-1]["bic"] >= resolved[-1]["bic"]
        ):
            break
    _plot_profile_fits(p, candidates, "n", "rings")
    return {"candidates": candidates, "searches": searches}


def _fill_new(ctx, held, n):
    """``held`` values with placeholders for ring ``n`` (the scan replaces them)."""
    v = dict(held)
    v[f"r{n}"], v[f"s{n}"], v[f"f{n}"] = ctx.r_in, ctx.halffringe, 0.01
    for ring in range(1, n + 1):
        if ctx.colour:
            v.setdefault(f"a{ring}", _colour_start(ctx))
        for j in range(1, ctx.degree + 1):
            v.setdefault(f"h{ring}_re{j}", 0.0)
            v.setdefault(f"h{ring}_im{j}", 0.0)
    return v


def _plot_profile_fits(p, candidates, key, label):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = [c[key] for c in candidates]
    ax.plot(x, [c["chi2_reduced"] for c in candidates], "o-")
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set(xlabel=label, ylabel="χ²/N (quoted errors)", yscale="log")
    _save(fig, p.output / "plots" / "profile_fits.png")


# --- the Gaussian-process disk ---


def _gp_template(ctx, geo, ell, planet, degree=None):
    """A star, a ``RadialGP`` disk and optionally a planet, with its priors."""
    import numpyro.distributions as dist

    from ..models import (
        Harmonics,
        MultipoleDisk,
        MultipoleKernel,
        RadialGP,
        RadialGPFunction,
        System,
        gp_priors,
    )

    degree = ctx.degree if degree is None else degree
    r_in, r_out = ctx.r_in, ctx.r_out
    harmonics = None
    if degree:
        harmonics = Harmonics(
            degree=degree,
            prior=MultipoleKernel(
                ell_r=ell,
                ell_theta=0.5 * (r_in + r_out),
                sigma=_HARMONIC_SIGMA,
                n=_HARMONIC_NODES,
                r_in=r_in,
                r_out=r_out,
            ),
        )
    index = (
        RadialGPFunction(
            r_in, r_out, sigma=_ALPHA_SIGMA, ell=ell, n=_ALPHA_NODES
        )
        if ctx.colour
        else None
    )
    disk = MultipoleDisk(
        RadialGP(r_in, r_out, sigma=geo["gp_sigma"], ell=ell),
        harmonics=harmonics,
        spectral_index=index,
        wavel0=ctx.wavel0,
        inc=geo["inc"],
        pa=geo["pa"],
        n_r=ctx.n_gp,
        flux=geo["f1"],
    )
    parts = {"star": _star(ctx), "disk": disk}
    if planet is not None:
        parts["planet"] = _planet(ctx, planet, ctx.colour)
    model = System(**parts)
    priors = gp_priors(model) | {"disk.flux": dist.LogUniform(*_FLUX_RANGE)}
    if planet is not None:
        paths = {
            "dra": "planet.dra",
            "ddec": "planet.ddec",
            "fp": "planet.flux.ratio" if ctx.colour else "planet.flux",
            "ap": "planet.flux.index",
        }
        priors |= {
            paths[k]: prior
            for k, prior in _planet_priors(ctx, ctx.colour).items()
        }
    return model, priors


def _apply(template, values, priors):
    """The unfrozen ``template`` with the fitted values of ``priors`` set."""
    paths = [k for k in priors if k in values]
    return template.set(paths, [values[k] for k in paths])


def _gp_geometry(p):
    geo = _geometry_values(p)
    report = p._report("geometry")
    return {**geo, "gp_sigma": report["gp_sigma"]}


def _released_geometry(p, values):
    """``_gp_geometry`` with the inclination and position angle of ``values``."""
    return {
        **_gp_geometry(p),
        "inc": float(onp.asarray(values["disk.inc"])),
        "pa": float(onp.asarray(values["disk.pa"])),
    }


def _planet_values(values, colour):
    """A fit's planet as the pipeline's ``dra, ddec, fp, ap`` values."""
    out = {
        "dra": float(values["planet.dra"]),
        "ddec": float(values["planet.ddec"]),
        "fp": float(values["planet.flux.ratio" if colour else "planet.flux"]),
    }
    if colour:
        out["ap"] = float(values["planet.flux.index"])
    return out


def _companion_report(p, planet):
    """The fitted companion for the summary, or None without one.

    ``planet`` is ``dra, ddec, fp[, ap]`` of the preferred model;
    ``detection_delta_chi2`` is the best Δχ² of the searches that found it.
    """
    if planet is None:
        return None
    searches = p._report("profile")["searches"]
    found = [
        x["delta_chi2"]
        for x in searches
        if x["delta_chi2"] >= p.settings["companion_min_delta_chi2"]
    ]
    return {
        "dra_mas": planet["dra"],
        "ddec_mas": planet["ddec"],
        "flux": planet["fp"],
        "spectral_index": planet.get("ap"),
        "detection_delta_chi2": max(found) if found else None,
    }


def _gp_fit(
    p, ell, planet, degree=None, geo=None, release=False, max_steps=None
):
    """MAP fit of the Gaussian-process disk at one ℓ.

    The geometry is frozen at ``geo`` (the single-ring fit's by default),
    which makes the fit fast. A single ring is only an approximation to a
    lopsided, multi-ring disk, so its inclination and position angle are a
    few degrees off, and a frozen error is paid for by the profile and the
    companion. With ``release`` a second fit frees them, starting from the
    first fit's values, and the returned values and priors include them.
    ``max_steps`` bounds the frozen fit (a screening fit).
    """
    import equinox as eqx

    from ..fitting import fit
    from ..models import System

    ctx, data = p._ctx(), p.processed
    geo = geo or _gp_geometry(p)
    template, priors = _gp_template(ctx, geo, ell, planet, degree)
    frozen = eqx.tree_at(
        lambda m: m.parts[1], template, template.disk.freeze_geometry(data)
    )
    assert isinstance(frozen, System)
    result = fit(frozen, priors, data, max_steps=max_steps)
    values, info = result.values, result.info
    if release:
        priors = {
            **priors,
            **{f"disk.{k}": v for k, v in _geometry_priors(ctx).items()},
        }
        start = {k: v for k, v in values.items() if k in priors}
        start["disk.inc"] = max(float(geo["inc"]), 5.0)
        result = fit(template, priors, data, init=start)
        values = _fold_gp_values(ctx, result.values, degree)
        info = result.info
    model = _apply(template, values, priors)
    return _finish(model, values, info, priors, data), priors


def _fold_gp_values(ctx, values, degree=None):
    """``values`` with the disk's ``pa`` in [0°, 180°), the harmonics turned.

    The Gaussian process's version of ``_fold_fit``: the harmonics are a
    ``MultipoleKernel``'s latents, of shape ``(2, order, node)``, and the order-*m*
    latents are multiplied by ``(-1)^m`` when ``pa`` is moved by 180°.
    """
    pa = float(onp.asarray(values["disk.pa"]))
    if 0.0 <= pa < 180.0:
        return values
    out = dict(values)
    folded = pa % 180.0
    out["disk.pa"] = onp.asarray(folded)
    if "disk.pa_vec" in out:
        angle = math.radians(folded)
        out["disk.pa_vec"] = onp.asarray([math.cos(angle), math.sin(angle)])
    degree = ctx.degree if degree is None else degree
    site = "disk.harmonics.z"
    if degree and (pa % 360.0) >= 180.0 and site in out:
        orders = onp.arange(1, degree + 1, dtype=float)
        out[site] = onp.asarray(out[site]) * ((-1.0) ** orders)[None, :, None]
    return out


def _geometry_starts(p, geo):
    """Candidate ``(inc, pa)`` starts for the Gaussian process.

    The single-ring fit's, the grid's rival minima and a coarse lattice: a
    single ring is the wrong model for a multi-ring, lopsided disk and can
    settle in a mode far from the disk's geometry, which the released fit
    then cannot leave.
    """
    ctx = p._ctx()
    starts = [(geo["inc"], geo["pa"])]
    starts += [
        (r["inc_deg"], r["pa_deg"])
        for r in p._report("geometry")["grid"]["rivals"]
    ]
    starts += [
        (inc, pa)
        for inc in _GEOMETRY_LATTICE_INC
        for pa in _GEOMETRY_LATTICE_PA
        if inc <= ctx.inc_max
    ]
    return starts


def _geometry_search(p, geo, ell):
    """The start whose frozen-geometry Gaussian-process fit has the lowest χ².

    Returns ``(geometry, fit, rows)``: the winning ``geo``, its converged fit
    (with no planet) and every start's χ²/N after a short screening fit. The
    geometry is frozen, and a wrong geometry costs far more χ² than a right
    one, so a few steps rank the starts.
    """
    n_data = int(p.processed.n_independent)
    best, rows = None, []
    for inc, pa in _geometry_starts(p, geo):
        trial = {**geo, "inc": float(inc), "pa": float(pa)}
        screen, _ = _gp_fit(p, ell, None, geo=trial, max_steps=_SCREEN_STEPS)
        rows.append(
            {
                "inc_deg": trial["inc"],
                "pa_deg": trial["pa"],
                "chi2_reduced": screen.chi2 / n_data,
            }
        )
        if best is None or screen.chi2 < best[0]:
            best = (screen.chi2, trial)
    fit, _ = _gp_fit(p, ell, None, geo=best[1])
    return best[1], fit, rows


def _profile_gp(p):
    ctx, data = p._ctx(), p.processed
    n_data = int(data.n_independent)
    geo = _gp_geometry(p)
    geo, fit, starts = _geometry_search(p, geo, ctx.ell_mid)
    planet, searches = None, []
    if ctx.companion:
        # Search the residuals of the middle ℓ with no planet, and add any
        # detection to every fit.
        search = _point_source_search(
            fit.model, data, ctx, p.settings["batch_size"]
        )
        searches.append(search)
        if search["delta_chi2"] >= p.settings["companion_min_delta_chi2"]:
            planet = _planet_start(search, ctx)
    candidates = []
    for k, ell in enumerate(ctx.ells):
        fit, priors = _gp_fit(p, ell, planet, geo=geo, release=True)
        name = f"gp_ell{k}"
        _save_fit(p.output / "models" / name, fit, list(priors))
        candidates.append(
            {
                "name": name,
                "ell_mas": ell,
                "ell_equivalent_mas": ell / _ELL_PER_EQUIVALENT,
                "ell_halffringes": ell / _ELL_PER_EQUIVALENT / ctx.halffringe,
                "chi2": fit.chi2,
                "chi2_reduced": fit.chi2 / n_data,
                "n_free": fit.n_free,
                "converged": fit.info.get("converged"),
                "steps": fit.info.get("steps"),
                "planet": planet is not None,
            }
        )
    for c in candidates:
        c["ell_index"] = candidates.index(c)
    _plot_profile_fits(p, candidates, "ell_mas", "ℓ (mas)")
    return {
        "candidates": candidates,
        "searches": searches,
        "geometry_search": starts,
    }


# === STAGE: COMPARE ===


def _compare(p):
    return models()[p.name].compare(p)


def _log_prior(priors, values):
    total = 0.0
    for k, prior in priors.items():
        if k == "pa":  # an AngleVector's density per degree
            total += -math.log(360.0)
        else:
            total += float(prior.log_prob(values[k]))
    return total


def _light_fraction(model, lo, hi):
    """Fraction of the disks' light (at the reference wavelength) in [lo, hi]."""
    from ..models import MultipoleDisk

    num = den = 0.0
    for part in model.components.values():
        if not isinstance(part, MultipoleDisk):
            continue
        r, w = (onp.asarray(x, dtype=float) for x in part.profile.quadrature())
        flux = float(onp.asarray(part._weight()))
        total = float(w.sum())
        if total <= 0:
            continue
        num += flux * float(w[(r >= lo) & (r <= hi)].sum()) / total
        den += flux
    return num / den if den > 0 else float("nan")


def _residual_search(p, model):
    """The point-source search on ``model``'s residuals, as a check."""
    return _point_source_search(
        model, p.processed, p._ctx(), p.settings["batch_size"]
    )


def _common_compare(p, model, fit, extra):
    """Numbers every ``compare`` reports, for the preferred model."""
    ctx, data = p._ctx(), p.processed
    n = int(data.n_independent)
    return {
        "n_independent": n,
        "chi2": fit.chi2,
        "chi2_reduced": fit.chi2 / n,
        "error_scales": fit.scales or None,
        "degree": ctx.degree,
        "outer_fraction": _light_fraction(
            model, ctx.r_out - ctx.halffringe, onp.inf
        ),
        **extra,
    }


def _load_candidate(p, name):
    folder = p.output / "models" / name
    manifest = _io.read_json(folder / "manifest.json")
    values = _io.load_model_values(folder)
    return folder, manifest, values, _io.load_model(folder)


def _save_best(p, name):
    folder, manifest, values, model = _load_candidate(p, name)
    _io.save_model(
        p.output / "models" / "best", model, values, manifest["params"]
    )
    _io.write_json(
        p.output / "models" / "best" / "info.json",
        _io.read_json(folder / "info.json"),
    )


def _compare_rings(p):
    from ..inference import laplace_cov
    from ..likelihood import whitened_residuals

    ctx, data = p._ctx(), p.processed
    rows = []
    for c in p._report("profile")["candidates"]:
        n = c["n"]
        folder, manifest, values, model = _load_candidate(p, c["name"])
        shape = _Shape(n, ctx.colour, ctx.degree, c["planet"])
        priors = _ring_priors(ctx, shape)
        names = list(priors)
        flat = _scalars(values)
        sd = _laplace_sd(ctx, shape, names, flat, data)
        log_z = None
        if p.settings["error_scale"] == "quoted" and all(
            sd[k] is not None for k in names
        ):
            x = onp.array([flat[k] for k in names])
            cov = onp.asarray(
                laplace_cov(x, names, _builder(ctx, shape), data)
            )
            sign, logdet = onp.linalg.slogdet(cov)
            if sign > 0:
                r = onp.asarray(whitened_residuals(model, data), dtype=float)
                log_z = float(
                    -0.5 * onp.sum(r**2)
                    + _log_prior(priors, flat)
                    + 0.5 * len(names) * math.log(2.0 * math.pi)
                    + 0.5 * logdet
                )
        rows.append(
            {
                **c,
                "log_evidence": log_z,
                "radii": [flat[f"r{i}"] for i in range(1, n + 1)],
                "widths": [flat[f"s{i}"] for i in range(1, n + 1)],
                "fluxes": [flat[f"f{i}"] for i in range(1, n + 1)],
                "sd": sd,
                "params": {k: flat[k] for k in names},
            }
        )
    # A count with two rings too close to tell apart is one ring fitted as
    # two (see ``_unresolved_pairs``): it does not compete. The first count
    # is one ring and never has such a pair. Those with a lower BIC than the
    # winner are reported: the data want a profile the rings cannot give.
    split = [r for r in rows if r.get("unresolved")]
    rows = [r for r in rows if not r.get("unresolved")]
    n_best = min(rows, key=lambda r: r["bic"])["n"]
    split = [
        r["n"]
        for r in split
        if r["bic"] < next(q for q in rows if q["n"] == n_best)["bic"]
    ]
    evidences = {
        r["n"]: r["log_evidence"]
        for r in rows
        if r["log_evidence"] is not None
    }
    by_evidence = max(evidences, key=evidences.get) if evidences else None
    best = next(r for r in rows if r["n"] == n_best)
    _save_best(p, best["name"])
    _, _, _, model = _load_candidate(p, best["name"])
    fit = _DiskFit(model, {}, {}, best["chi2"], {}, best["n_free"])
    search = _residual_search(p, model)
    pa_sd = best["sd"].get("pa")
    report = _common_compare(
        p,
        model,
        fit,
        {
            "best": best["name"],
            "n_best": n_best,
            "preferred_by_bic": n_best,
            "preferred_by_evidence": by_evidence,
            "split_counts": split,
            "rows": rows,
            "params": best["params"],
            "inc_deg": best["params"]["inc"],
            "pa_deg": best["params"]["pa"],
            "pa_sd_deg": pa_sd,
            "residual_search": search,
            "raised_search": None,
            "companion": _companion_report(
                p,
                {
                    k: best["params"][k]
                    for k in ("dra", "ddec", "fp", "ap")
                    if k in best["params"]
                }
                if best["planet"]
                else None,
            ),
        },
    )
    _plot_summary_file(p, model)
    return report


def _inner_stats(model, data, ctx):
    """Light near the inner edge, and its Laplace correlation with the flux.

    The fraction of the disk's light within one half-fringe of ``r_in``,
    and the correlation between ``disk.flux`` and that fraction under the
    Gauss-Newton posterior of the flux and the profile's latents (a flat
    prior on the flux and the standard-normal prior of the latents).
    """
    import jax
    import jax.numpy as np

    from ..likelihood import whitened_residuals

    z0 = np.asarray(model.get("disk.profile.z"))
    flux0 = np.asarray(model.get("disk.flux"))

    def build(x):
        return model.set(["disk.flux", "disk.profile.z"], [x[0], x[1:]])

    def residuals(x):
        return whitened_residuals(build(x), data)

    def fraction(x):
        r, w = build(x).disk.profile.quadrature()
        return np.sum(
            np.where(r <= ctx.r_in + ctx.halffringe, w, 0.0)
        ) / np.sum(w)

    x0 = np.concatenate([flux0.reshape(1), z0])
    jac = onp.asarray(jax.jacfwd(residuals)(x0), dtype=float)
    prior = onp.diag(onp.r_[1e-9, onp.ones(z0.size)])
    cov = onp.linalg.inv(jac.T @ jac + prior)
    grad = onp.asarray(jax.grad(fraction)(x0), dtype=float)
    value = float(fraction(x0))
    var_flux, var_frac = cov[0, 0], float(grad @ cov @ grad)
    cross = float(cov[0] @ grad)
    denom = math.sqrt(max(var_flux, 0.0) * max(var_frac, 0.0))
    return value, (cross / denom if denom > 0 else None)


def _compare_gp(p):
    from ..imaging import log_evidence

    ctx, data = p._ctx(), p.processed
    rows = []
    for c in p._report("profile")["candidates"]:
        _, _, _, model = _load_candidate(p, c["name"])
        try:
            log_z = float(log_evidence(model, data, path="disk"))
        except Exception as err:  # noqa: BLE001 - reported as a missing evidence
            warnings.warn(
                f"The evidence at ℓ = {c['ell_mas']:.3g} mas could not be "
                f"computed ({err}).",
                RuntimeWarning,
                stacklevel=2,
            )
            log_z = None
        rows.append({**c, "log_evidence": log_z})
    scored = [r for r in rows if r["log_evidence"] is not None]
    best = max(scored or rows, key=lambda r: r["log_evidence"] or -onp.inf)
    _save_best(p, best["name"])
    _, _, values, model = _load_candidate(p, best["name"])
    fit = _DiskFit(model, {}, {}, best["chi2"], {}, best["n_free"])
    inner_fraction, inner_corr = _inner_stats(model, data, ctx)
    geometry = p._report("geometry")
    planet = best["planet"]
    search = _residual_search(p, model)
    raised = None
    if search["delta_chi2"] >= p.settings["companion_min_delta_chi2"]:
        # Does a higher harmonic degree remove it?
        planet_values = (
            _planet_values(
                _io.load_model_values(p.output / "models" / "best"), ctx.colour
            )
            if planet
            else None
        )
        higher, _ = _gp_fit(
            p,
            best["ell_mas"],
            planet_values,
            degree=ctx.degree + 1,
            geo=_released_geometry(p, values),
        )
        raised = _residual_search(p, higher.model)
    report = _common_compare(
        p,
        model,
        fit,
        {
            "best": best["name"],
            "ell_mas": best["ell_mas"],
            "rows": rows,
            "inner_fraction": inner_fraction,
            "inner_correlation": inner_corr,
            # The released geometry; the position angle's spread is the
            # single-ring fit's, an approximation used only to gate the
            # inclination check.
            "inc_deg": float(onp.asarray(values["disk.inc"])),
            "pa_deg": float(onp.asarray(values["disk.pa"])),
            "pa_sd_deg": geometry["laplace_sd"].get("pa"),
            "flux": float(onp.asarray(values["disk.flux"])),
            "residual_search": search,
            "raised_search": raised,
            "companion": _companion_report(
                p, _planet_values(values, ctx.colour) if planet else None
            ),
        },
    )
    _plot_summary_file(p, model)
    return report


# === PLOTS ===


def _profile_curves(model, r):
    """Radial profile, spectral index and lopsidedness of a model's disks.

    Each is evaluated at the radii ``r`` and combined over the disks by
    their share of the light at that radius.
    """
    from ..models import MultipoleDisk

    total = onp.zeros_like(r)
    alpha = onp.zeros_like(r)
    lop = onp.zeros_like(r)
    has_alpha = has_lop = False
    for part in model.components.values():
        if not isinstance(part, MultipoleDisk):
            continue
        i0 = onp.asarray(part.profile.intensity(r, 0.0), dtype=float)
        norm = onp.trapezoid(i0 * r, r)
        if not norm > 0:
            continue
        i0 = float(onp.asarray(part._weight())) * i0 / norm
        total += i0
        a = onp.zeros_like(r)
        if part.spectral_index is not None:
            a = onp.asarray(part._alpha(r), dtype=float)
            has_alpha = True
        elif hasattr(part.flux, "index"):
            a = a + float(onp.asarray(part.flux.index))
            has_alpha = True
        alpha += i0 * a
        if part.harmonics is not None and part.harmonics.degree:
            try:
                h = onp.asarray(part.harmonics.image_harmonics(r))
            except TypeError:
                h = onp.asarray(part.harmonics.image_harmonics())
            first = onp.abs(h[1]) if h.ndim == 1 else onp.abs(h[1])
            lop += i0 * onp.broadcast_to(first, r.shape)
            has_lop = True
    safe = onp.where(total > 0, total, 1.0)
    return (
        total,
        alpha / safe if has_alpha else None,
        lop / safe if has_lop else None,
    )


def plot_summary(model, data, *, r_out_mas, r_in_mas=0.0, npix=200):
    """The fitted disk: images, profile, colour, lopsidedness and residuals.

    Parameters
    ----------
    model : System
        The fitted model (``Result.model()``).
    data : OIData
        The data (``Result.data()``).
    r_out_mas : float
        Outer radius of the radial range (mas); the images span ±1.2 times
        this.
    r_in_mas : float, optional
        Inner radius of the radial range (mas); the curves start there.
    npix : int, optional
        Pixels of each image.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from ..models import MultipoleDisk, System
    from ..plotting import plot_model

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fov = 2.4 * r_out_mas
    plot_model(
        model, fov, npix=npix, ax=axes[0, 0], saturate=0.99, title="Model"
    )
    disks = {
        k: v
        for k, v in model.components.items()
        if isinstance(v, MultipoleDisk)
    }
    plot_model(
        System(disks),
        fov,
        npix=npix,
        ax=axes[0, 1],
        title="Disk without the point sources",
    )
    r = onp.linspace(max(r_in_mas, 1e-3), r_out_mas, 300)
    i0, alpha, lop = _profile_curves(model, r)
    ax = axes[0, 2]
    ax.plot(r, i0)
    ax.set(
        yscale="log",
        xlabel="radius in the disk plane (mas)",
        ylabel="I₀(r)",
        ylim=(max(i0.max() * 1e-5, 1e-30), i0.max() * 3),
    )
    for ax, curve, label in (
        (axes[1, 0], alpha, "spectral index α(r)"),
        (axes[1, 1], lop, "lopsidedness |I₁|/I₀"),
    ):
        if curve is None:
            ax.text(
                0.5, 0.5, "not fitted", ha="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            continue
        ax.plot(r, onp.where(i0 > 1e-3 * i0.max(), curve, onp.nan))
        ax.set(xlabel="radius in the disk plane (mas)", ylabel=label)
    ax = axes[1, 2]
    _, err = data.flatten_data()
    z = onp.asarray(data.residuals(data.model(model))) / err
    n_vis = onp.asarray(data.vis).size
    for sl, label in (
        (slice(0, n_vis), "V²"),
        (slice(n_vis, None), "closure phase"),
    ):
        if z[sl].size:
            ax.plot(
                z[sl],
                ".",
                ms=3,
                label=f"{label}: rms z = {onp.sqrt(onp.mean(z[sl] ** 2)):.2f}",
            )
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="data index", ylabel="residual / quoted error", ylim=(-5, 5))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_summary_file(p, model):
    ctx = p._ctx()
    fig = plot_summary(
        model, p.processed, r_out_mas=ctx.r_out, r_in_mas=ctx.r_in
    )
    _save(fig, p.output / "plots" / "disk_summary.png")


# === STAGE: POSTERIOR ===


def _fold_pa(samples, pa_site, parity, centre):
    """``pa`` folded to within 90° of ``centre``, the harmonics rotated to match.

    A disk without height is the same under ``pa`` → ``pa`` + 180° with the
    order-*m* harmonic turned by 180°, i.e. multiplied by ``(-1)^m``, so
    harmonics let a sampler move between the two. Folding every sample to
    the half-circle around the MAP makes the summary unimodal wherever the
    MAP lies; a cut at a fixed angle would split a posterior near it.

    Parameters
    ----------
    samples : dict
        Site to array of shape ``(chain, draw, ...)``.
    pa_site : str
        The position-angle site (degrees).
    parity : dict
        Harmonic site to its sign ``(-1)^m``, an array broadcast against the
        site's trailing dimensions.
    centre : float
        The position angle (degrees) to fold about, normally the MAP's.
    """
    pa = onp.asarray(samples[pa_site], dtype=float)
    turns = onp.floor((pa - centre + 90.0) / 180.0)
    flip = (turns % 2.0) == 1.0
    out = dict(samples)
    out[pa_site] = pa - 180.0 * turns
    for site, sign in parity.items():
        x = onp.asarray(samples[site], dtype=float)
        f = flip.reshape(flip.shape + (1,) * (x.ndim - flip.ndim))
        out[site] = onp.where(f, x * onp.asarray(sign, dtype=float), x)
    return out


def _posterior_setup(p):
    """Model, priors, start, position-angle site and parity of the NUTS run.

    The geometry is released: the rings' ``inc`` and ``pa`` are free
    already; the Gaussian process's are added, starting at the geometry
    stage's values (a Gaussian process's NUTS is run on the unfrozen model).
    """
    ctx = p._ctx()
    compare = p._report("compare")
    values = _io.load_model_values(p.output / "models" / "best")
    if p.name == "rings":
        row = next(r for r in compare["rows"] if r["n"] == compare["n_best"])
        shape = _Shape(row["n"], ctx.colour, ctx.degree, row["planet"])
        priors = _ring_priors(ctx, shape)
        start = {k: onp.asarray(values[k]) for k in priors}
        start["pa_vec"] = onp.asarray(values["pa_vec"])
        parity = {
            f"h{i}_{part}{j}": (-1.0) ** j
            for i in range(1, shape.n + 1)
            for j in range(1, ctx.degree + 1)
            for part in ("re", "im")
        }
        return (
            _builder(ctx, shape),
            priors,
            start,
            "pa",
            parity,
            float(values["pa"]),
        )
    geo = _released_geometry(p, values)
    row = next(r for r in compare["rows"] if r["name"] == compare["best"])
    planet = _planet_values(values, ctx.colour) if row["planet"] else None
    template, priors = _gp_template(ctx, geo, compare["ell_mas"], planet)
    model = _apply(template, values, priors)
    priors = {
        **priors,
        **{f"disk.{k}": v for k, v in _geometry_priors(ctx).items()},
    }
    start = {k: onp.asarray(values[k]) for k in priors if k in values}
    angle = math.radians(float(geo["pa"]))
    start["disk.inc"] = onp.asarray(max(float(geo["inc"]), 5.0))
    start["disk.pa_vec"] = onp.asarray([math.cos(angle), math.sin(angle)])
    orders = onp.arange(1, ctx.degree + 1, dtype=float)
    parity = (
        {"disk.harmonics.z": ((-1.0) ** orders)[None, :, None]}
        if ctx.degree
        else {}
    )
    return model, priors, start, "disk.pa", parity, float(geo["pa"])


@in_float64
def _nuts(p, model, priors, start):
    from ..likelihood import numpyro_model

    posterior = numpyro_model(
        model, priors, p.processed, noise=p._noise(), flat_coordinates=True
    )
    mcmc = _run_chains(
        posterior, p.settings, start, dense_mass=True, priors=priors
    )
    samples = {
        k: onp.asarray(v)
        for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    return samples, _sample_stats(mcmc)


def _posterior(p):
    s = p.settings
    if not s["posterior"]:
        (p.output / "samples.h5").unlink(missing_ok=True)
        return {"skipped": True, "reason": "posterior=False"}
    model, priors, start, pa_site, parity, centre = _posterior_setup(p)
    samples, stats = _nuts(p, model, priors, start)
    samples = _fold_pa(
        samples,
        pa_site,
        {k: v for k, v in parity.items() if k in samples},
        centre,
    )
    floats = {
        k: x
        for k, x in samples.items()
        if onp.issubdtype(x.dtype, onp.floating)
    }
    diagnostics = {k: _diagnostics([x]) for k, x in floats.items()}
    summary = {
        k: {
            **_quantiles(x, angle=k == pa_site, period=180.0),
            "std": float(onp.std(x)),
            **diagnostics[k],
        }
        for k, x in floats.items()
        if x.ndim == 2 and not k.endswith("_vec")
    }
    bound = {
        k: _bound_samples(priors[k], samples[k])
        for k in priors
        if k in samples and samples[k].ndim == 2
    }
    _io.write_h5(
        p.output / "samples.h5",
        {"posterior": samples, "sample_stats": stats},
        attrs={
            "/": {
                "schema": _io.SCHEMA,
                "pa_site": pa_site,
                "pa_folded": "pa within 90 degrees of the MAP, harmonics of order m times (-1)^m",
                "sampler": {k: s[k] for k in _SAMPLER_SETTINGS},
            }
        },
    )
    return {
        "skipped": False,
        "best": p._report("compare")["best"],
        "params": summary,
        "noise": _noise_quantiles(samples),
        "num_chains": s["num_chains"],
        "num_samples": s["num_samples"],
        "num_warmup": s["num_warmup"],
        "r_hat_max": max(d["r_hat"] for d in diagnostics.values()),
        "ess_bulk_min": min(d["ess_bulk"] for d in diagnostics.values()),
        "divergence_fraction": float(onp.mean(stats["diverging"])),
        "prior_bound_fraction": {
            k: v for k, v in bound.items() if v is not None
        },
    }


def _quicklook(p):
    from . import _quicklook as ql

    ql.write_quicklook(p.output, execute=p.settings["quicklook_execute"])
    return {"executed": bool(p.settings["quicklook_execute"])}


# === SUMMARY AND CHECKS ===

_DATA_KEYS = (
    "n_vis",
    "n_phi",
    "n_independent",
    "closure_phases",
    "wavel_min_m",
    "wavel_max_m",
    "baseline_min_m",
    "baseline_max_m",
    "resolution_mas",
    "fov_mas",
)
_OVERVIEW_KEYS = (
    "resolution_mas",
    "r_in_mas",
    "r_out_mas",
    "n_r_rings",
    "n_r_gp",
    "n_channels",
    "colour",
    "wavel0_m",
    "ell_mas",
)
_GEOMETRY_KEYS = (
    "inc_deg",
    "pa_deg",
    "r_mas",
    "sigma_mas",
    "flux",
    "chi2_reduced",
    "converged",
    "gp_sigma",
    "laplace_sd",
    "grid",
)
_POSTERIOR_KEYS = (
    "best",
    "params",
    "noise",
    "num_chains",
    "num_samples",
    "num_warmup",
    "r_hat_max",
    "ess_bulk_min",
    "divergence_fraction",
)


def _disk_section(name, compare):
    section = {"model": name, "best": compare["best"]}
    if name == "rings":
        section.update(
            n_rings=compare["n_best"],
            rings=[
                {"r_mas": r, "sigma_mas": w, "flux": f}
                for r, w, f in zip(
                    *(
                        next(
                            row
                            for row in compare["rows"]
                            if row["n"] == compare["n_best"]
                        )[key]
                        for key in ("radii", "widths", "fluxes")
                    )
                )
            ],
        )
    else:
        section.update(ell_mas=compare["ell_mas"], flux=compare["flux"])
    section["inc_deg"] = compare["inc_deg"]
    section["pa_deg"] = compare["pa_deg"]
    section["companion"] = compare["companion"]
    return section


def _chi2_checks(entry, compare, settings):
    checks = [
        _checks.chi2_reduced(
            "chi2_reduced",
            compare["chi2_reduced"],
            label=f"the {entry.label} fit",
        )
    ]
    if compare["error_scales"]:
        checks.append(_checks.error_scale(compare["error_scales"]))
    return checks


def _fit_checks(entry, settings, overview, geometry, compare):
    hf = overview["resolution_mas"]
    by = {}
    rows = compare["rows"]
    if entry.name == "gp":
        by["length_scale"] = _checks.length_scale(
            [r["log_evidence"] for r in rows],
            [r["ell_mas"] for r in rows],
            hf,
            nats=settings["length_scale_nats"],
        )
        by["inner_edge"] = _checks.inner_edge(
            compare["inner_fraction"],
            compare["inner_correlation"],
            hf,
            max_fraction=settings["inner_edge_fraction"],
            max_correlation=settings["inner_edge_correlation"],
        )
    else:
        by["ring_count"] = _checks.ring_count(
            compare["n_best"],
            rows,
            hf,
            nats=settings["ring_count_nats"],
            split=compare["split_counts"],
        )
    by["outer_edge"] = _checks.outer_edge(
        compare["outer_fraction"], hf, warn=settings["outer_edge_fraction"]
    )
    by["inclination"] = _checks.inclination(
        compare["inc_deg"],
        compare["pa_sd_deg"],
        geometry["grid"]["n_minima"],
        minimum=settings["inclination_min_deg"],
    )
    raised = compare["raised_search"]
    by["absorbed_companion"] = _checks.absorbed_companion(
        compare["residual_search"]["delta_chi2"],
        raised["delta_chi2"] if raised else None,
        compare["degree"],
        threshold=settings["companion_min_delta_chi2"],
    )
    return [by[name] for name in CHECKS if name in by and name in entry.checks]


def _posterior_checks(post):
    return [
        _checks.prior_bound(post["prior_bound_fraction"]),
        _checks.r_hat(post["r_hat_max"]),
        _checks.ess(post["ess_bulk_min"]),
        _checks.divergences(post["divergence_fraction"]),
    ]


def _summarize(name, settings, reports):
    entry = models()[name]
    sections, checks = {}, []
    load, over, geo, prof, comp, post = (
        reports.get(k)
        for k in (
            "load",
            "overview",
            "geometry",
            "profile",
            "compare",
            "posterior",
        )
    )
    if load:
        sections["data"] = {k: load[k] for k in _DATA_KEYS}
    if over:
        sections["model"] = {
            "name": name,
            "label": entry.label,
            **{k: over[k] for k in _OVERVIEW_KEYS},
        }
    if geo:
        sections["geometry"] = {k: geo[k] for k in _GEOMETRY_KEYS}
    if prof:
        sections["profile"] = {"candidates": prof["candidates"]}
    if not comp:
        return sections, checks
    sections["chi2"] = {
        "n_independent": comp["n_independent"],
        "error_scale": settings["error_scale"],
        "reduced": {comp["best"]: comp["chi2_reduced"]},
        "error_scales": comp["error_scales"],
    }
    sections["compare"] = {
        "best": comp["best"],
        "rows": comp["rows"],
        "residual_search": comp["residual_search"],
        "raised_search": comp["raised_search"],
    }
    sections["disk"] = _disk_section(name, comp)
    checks += _chi2_checks(entry, comp, settings)
    checks += _fit_checks(entry, settings, over, geo, comp)
    if post and not post.get("skipped"):
        sections["posterior"] = {k: post[k] for k in _POSTERIOR_KEYS}
        checks += _posterior_checks(post)
    return sections, checks

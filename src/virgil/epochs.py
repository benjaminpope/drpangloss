"""Named epochs of data, with one snapshot of a moving scene per dataset.

An orbit fit to interferometric data across epochs predicts each epoch's
visibilities and closure phases from the scene at that epoch's time. When
nothing moves appreciably during an observation (a binary whose period is
much longer than a night), one snapshot of the scene per dataset is
enough: the orbit is solved once per dataset, and the snapshot is a static
model, evaluated on all of the dataset's samples at once on the fast path
(e.g. [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian] for
[`OrbitalBinary`][virgil.models.OrbitalBinary]). That is much cheaper
than [`OIData.model`][virgil.oidata.OIData.model]'s evaluation of a
time-dependent scene at every sample's own time, which remains the choice
when the scene moves within an observation.

[`Epochs`][virgil.epochs.Epochs] keys epochs and datasets **by name**,
so that per-dataset nuisance terms (error scales, wavelength scales,
North angles) are attached to the right data however the list is
ordered, and it hands [`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model] what they take: a
model function returning one snapshot per dataset, the datasets, and one
``noise`` dict per dataset.

Fringe aliases make the likelihood of visibilities multimodal on the
scale of the resolution λ/B, so an orbit fit needs a start in the right
basin. The tools for starting and sampling such a fit are:

* [`rank_orbits`][virgil.epochs.rank_orbits]: the log likelihood of all
  the data for each of many trial orbits (e.g. from
  [`starting_orbits`][virgil.orbits.starting_orbits]), best first;
* [`chain_starts`][virgil.epochs.chain_starts]: the best *distinct* orbits
  of a ranking, one per chain;
* [`epoch_positions`][virgil.epochs.epoch_positions]: the companion's
  position in each dataset, from a grid and a binary fit;
* [`start_from_positions`][virgil.epochs.start_from_positions]: the whole
  start: positions, starting orbits, ranking on the visibilities and
  refinement with [`fit`][virgil.fitting.fit] from several distinct
  orbits, with [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values]
  giving one start per chain for
  [`chain_init_params`][virgil.likelihood.chain_init_params].
"""

import dataclasses
import functools
import math
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import i0e

from . import _gauss_newton
from ._geometry import fringe_scales
from ._grid import batch_size_or_default, batch_visibilities
from ._precision import cast_tree, run_in
from .fitting import fit
from .likelihood import (
    _gaussian_loglike,
    _whitened_and_errors,
    model_loglike,
    whitened_residuals,
)
from .models import BinaryModelCartesian, PointSource, System, UniformDisk
from .orbits import (
    _PHASE_COHERENCE,
    PositionData,
    _phase_drift,
    starting_orbits,
)
from ._deprecate import external_stacklevel


__all__ = [
    "Epochs",
    "EpochPeaks",
    "EpochPositions",
    "NightFlag",
    "OrbitStart",
    "PrimaryDiameter",
    "RESOLVE_NATS",
    "RankedOrbits",
    "chain_starts",
    "epoch_positions",
    "estimate_primary_diameter",
    "marginal_loglike",
    "rank_orbits",
    "start_from_positions",
]

_AT = ("dataset", "epoch")


class Epochs:
    """Datasets grouped into named epochs, each seen as one snapshot.

    Parameters
    ----------
    epochs : dict
        ``{epoch name: data}``, where ``data`` is one
        [`OIData`][virgil.oidata.OIData], a list of them, or a dict
        ``{dataset name: OIData}``. One epoch may hold several datasets,
        e.g. several files of one night, or two filters a day apart. A
        lone dataset is named after its epoch; datasets in a list are
        named ``"<epoch>[k]"``. Every name must be unique.
    at : {"dataset", "epoch"}, optional
        Where each snapshot is taken: at the mean time of each dataset's
        samples (default), or at the mean time of the whole epoch, shared
        by its datasets.
    times : dict, optional
        ``{epoch or dataset name: MJD}``: snapshot times to use instead of
        the data's own, e.g. for data without per-sample times (AMIGO
        DISCOs). A dataset's entry overrides its epoch's.

    Attributes
    ----------
    names : tuple of str
        The epochs, in the order given.
    dataset_names : tuple of str
        The datasets, epoch by epoch.
    epoch_of : tuple of str
        The epoch of each dataset.
    data : tuple of OIData
        The datasets, in the order of ``dataset_names``: pass this as
        ``data`` to ``fit`` and ``numpyro_model``.
    times : numpy.ndarray
        The snapshot time of each dataset (MJD, float64).
    spread_days : numpy.ndarray
        The largest distance in time of each dataset's samples from its
        snapshot (days). The scene should move by much less than the
        resolution (λ/B) over this time; if it does not, evaluate the
        dataset per sample instead (pass it to ``fit`` directly with the
        time-dependent scene).

    Examples
    --------
    >>> import numpy as onp
    >>> from virgil.epochs import Epochs
    >>> from virgil.oidata import OIData
    >>> def night(mjd):
    ...     return OIData({"u": [10.0, 20.0], "v": [5.0, -3.0],
    ...                    "wavel": 2.2e-6, "vis": [1.0, 1.0],
    ...                    "d_vis": [0.01, 0.01], "mjd": [mjd, mjd + 0.1]})
    >>> epochs = Epochs({"2023": {"ut": night(60100.0), "at": night(60101.0)},
    ...                  "2024": night(60500.0)})
    >>> epochs.dataset_names, epochs.epoch_of
    (('ut', 'at', '2024'), ('2023', '2023', '2024'))
    >>> [round(float(t), 2) for t in epochs.times]
    [60100.05, 60101.05, 60500.05]
    >>> epochs.noise({"2023": {"vis_scale": 2.0}, "2024": {"vis_scale": 3.0}})
    [{'vis_scale': 2.0}, {'vis_scale': 2.0}, {'vis_scale': 3.0}]
    """

    def __init__(self, epochs, at="dataset", times=None):
        if at not in _AT:
            raise ValueError(f"at must be one of {_AT}, got {at!r}.")
        if not isinstance(epochs, dict) or not epochs:
            raise ValueError(
                "epochs must be a non-empty dict {epoch name: data}."
            )
        names, epoch_of, data = [], [], []
        for epoch, group in epochs.items():
            if isinstance(group, dict):
                items = list(group.items())
            elif isinstance(group, (list, tuple)):
                items = [(f"{epoch}[{k}]", d) for k, d in enumerate(group)]
            else:
                items = [(epoch, group)]
            if not items:
                raise ValueError(f"Epoch {epoch!r} has no data.")
            for name, d in items:
                names.append(name)
                epoch_of.append(epoch)
                data.append(d)
        duplicated = sorted(
            {n for n in names if names.count(n) > 1}
            | {n for n, e in zip(names, epoch_of) if n in epochs and n != e}
        )
        if duplicated:
            raise ValueError(
                f"Names must be unique; {duplicated} name both an epoch "
                "and a dataset, or two datasets."
            )
        self.names = tuple(epochs)
        self.dataset_names = tuple(names)
        self.epoch_of = tuple(epoch_of)
        self.data = tuple(data)
        self.at = at
        self.times, self.spread_days = self._snapshot_times(times or {})

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (
            f"Epochs({len(self.names)} epochs, {len(self.data)} datasets, "
            f"at={self.at!r})"
        )

    def _snapshot_times(self, given):
        """Snapshot time and spread of each dataset."""
        unknown = set(given) - set(self.names) - set(self.dataset_names)
        if unknown:
            raise ValueError(f"times for unknown names: {sorted(unknown)}.")
        own = []
        for name, d in zip(self.dataset_names, self.data):
            mjd = getattr(d, "mjd", None)
            own.append(None if mjd is None else onp.asarray(mjd, onp.float64))
        times = onp.empty(len(self.data))
        for k, (name, epoch) in enumerate(
            zip(self.dataset_names, self.epoch_of)
        ):
            if name in given:
                times[k] = given[name]
            elif epoch in given:
                times[k] = given[epoch]
            elif own[k] is None:
                raise ValueError(
                    f"Dataset {name!r} has no times: read it from OIFITS, "
                    "give mjd per sample, or pass times={name: mjd}."
                )
            elif self.at == "dataset":
                times[k] = own[k].mean()
            else:
                together = [
                    own[j]
                    for j, e in enumerate(self.epoch_of)
                    if e == epoch and own[j] is not None
                ]
                times[k] = onp.concatenate(together).mean()
        spread = onp.array(
            [
                0.0 if t is None else float(onp.max(onp.abs(t - times[k])))
                for k, t in enumerate(own)
            ]
        )
        return times, spread

    @property
    def resolution_mas(self):
        """The finest resolution λ/B_max over the datasets (mas).

        Only unflagged samples count (see
        [`OIData.unflagged`][virgil.oidata.OIData.unflagged]).

        Fringe aliases of a companion's position are spaced by about this
        much, so starting orbits closer together than a fraction of it are
        one mode.
        """
        return float(fringe_scales(list(self.data))[0])

    def index(self, name):
        """The position of dataset ``name`` in ``data`` (and in ``noise``).

        Fitted per-dataset noise terms are the sites
        ``"noise[<index>].<term>"`` of ``fit`` and ``numpyro_model``.
        """
        try:
            return self.dataset_names.index(name)
        except ValueError:
            raise KeyError(
                f"No dataset {name!r}; the datasets are "
                f"{list(self.dataset_names)}."
            ) from None

    def snapshots(self, scene):
        """The scene at each dataset's snapshot time, one model per dataset.

        Parameters
        ----------
        scene : SourceModel or sequence of SourceModel
            One scene for all datasets, or one per dataset (e.g. with
            per-dataset fluxes). A static scene is its own snapshot.

        Returns
        -------
        list of SourceModel
            Static models, in the order of ``data``.
        """
        scenes = (
            list(scene)
            if isinstance(scene, (list, tuple))
            else [scene] * len(self.data)
        )
        if len(scenes) != len(self.data):
            raise ValueError(
                f"{len(scenes)} scenes for {len(self.data)} datasets."
            )
        return [s.at(float(t)) for s, t in zip(scenes, self.times)]

    def model_fn(self, scene_fn):
        """A model function for ``fit``/``numpyro_model``: the snapshots of
        ``scene_fn(**values)``, one per dataset."""

        def snapshots(**values):
            return self.snapshots(scene_fn(**values))

        return snapshots

    def noise(self, terms):
        """One ``noise`` dict per dataset, from terms keyed by name.

        Parameters
        ----------
        terms : dict
            ``{epoch or dataset name: {term: prior, value or tied
            callable}}``. An epoch's terms apply to each of its datasets,
            and each dataset gets its own copy (its own fitted site); a
            dataset's own terms are added to, and override, its epoch's.
            To share one fitted value between datasets, tie them to one
            parameter with callables (see
            [`noise_sites`][virgil.likelihood.noise_sites]).

        Returns
        -------
        list of dict
            In the order of ``data``, as ``fit(..., noise=...)`` takes.
        """
        unknown = set(terms) - set(self.names) - set(self.dataset_names)
        if unknown:
            raise KeyError(
                f"noise for unknown names {sorted(unknown)}; the epochs are "
                f"{list(self.names)} and the datasets "
                f"{list(self.dataset_names)}."
            )
        return [
            {**terms.get(epoch, {}), **terms.get(name, {})}
            for name, epoch in zip(self.dataset_names, self.epoch_of)
        ]

    def loglike(self, scene, noise=None):
        """The log likelihood of all the data, one snapshot per dataset.

        The sum over datasets of
        [`model_loglike`][virgil.likelihood.model_loglike] of each
        snapshot, with ``noise`` (values, keyed by name as for
        :meth:`noise`) applied to each dataset. Traceable: it can be
        jitted, differentiated or vmapped in the scene's parameters.
        """
        per_dataset = self.noise(noise or {})
        return sum(
            model_loglike(snap, d, **n)
            for snap, d, n in zip(
                self.snapshots(scene), self.data, per_dataset
            )
        )


# ---------------------------------------------------------------------------
# Starting and sampling orbit fits
# ---------------------------------------------------------------------------


def _orbit_list(orbits):
    """A list of orbits from a list, ``starting_orbits`` pairs, a
    ``RankedOrbits`` or one orbit with a leading batch axis."""
    if isinstance(orbits, (list, tuple, RankedOrbits)):
        listed = [o[0] if isinstance(o, tuple) else o for o in orbits]
    elif onp.ndim(jax.tree_util.tree_leaves(orbits)[0]) == 0:
        listed = [orbits]
    else:
        n = len(jax.tree_util.tree_leaves(orbits)[0])
        listed = [
            jax.tree_util.tree_map(lambda x, k=k: x[k], orbits)
            for k in range(n)
        ]
    if not listed:
        raise ValueError("There are no orbits to rank.")
    return listed


def _stack(trees, what="orbits"):
    """Stack pytrees of one structure along a new leading axis."""
    try:
        return jax.tree_util.tree_map(
            lambda *xs: np.stack([np.asarray(x) for x in xs]), *trees
        )
    except ValueError as err:
        raise ValueError(
            f"The {what} must share one structure (the same class and, for "
            f"orbits, the same t_ref): {err}"
        ) from None


def _blocks(data):
    """The observable blocks of ``data``'s whitened residuals, each with
    its own error scale: ``(name, start, stop, ν, von_mises)``.

    The blocks are the visibilities (``"vis"``, scaled by ``vis_scale``),
    the phases (``"phi"``, ``phi_scale``; for correlated closure phases,
    the independent combinations followed by the periodic penalty rows,
    of which only the former count in ``ν``) and each extra observable
    block (``"extra[k]"``). ``ν`` is the block's number of independent
    observables (they add up to ``data.n_independent``), and
    ``von_mises`` marks uncorrelated wrapping phases, whose exact
    likelihood is a von Mises density. Empty blocks are left out.
    """
    n_vis = int(onp.asarray(data.vis).size)
    n_phi = int(onp.asarray(data.phi).size)
    blocks = []
    if n_vis:
        blocks.append(("vis", 0, n_vis, n_vis, False))
    if n_phi:
        correlated = data._phases_wrap and data.cp_noise is not None
        nu = int(data.cp_noise.size) if correlated else n_phi
        rows = nu + n_phi if correlated else nu
        von_mises = bool(data._phases_wrap and data.cp_noise is None)
        blocks.append(("phi", n_vis, n_vis + rows, nu, von_mises))
    stop = blocks[-1][2] if blocks else 0
    for k, block in enumerate(data.extras):
        nu = int(block.n_independent)
        blocks.append((f"extra[{k}]", stop, stop + nu, nu, False))
        stop += nu
    if stop != data.n_residuals or not blocks:
        raise ValueError(
            "Cannot split the whitened residuals into observable blocks "
            f"({stop} rows for n_residuals = {data.n_residuals})."
        )
    return tuple(blocks)


def _dof_for(dof, data, index):
    """The effective-dof fraction of dataset ``index``: one number, or
    a dict keyed by dataset or epoch name (default 1)."""
    if not isinstance(dof, dict):
        return float(dof)
    name = data.dataset_names[index]
    return float(dof.get(name, dof.get(data.epoch_of[index], 1.0)))


# The bounded (s_max) marginal integrates over ln s adaptively: scans of
# _N_SCAN nodes narrow the range to where the integrand is within
# exp(-depth) of its peak, then _N_QUADRATURE Gauss–Legendre nodes
# integrate it there.
_N_SCAN = 64
_N_NARROW = 4
_N_QUADRATURE = 64
_LEGENDRE = onp.polynomial.legendre.leggauss(_N_QUADRATURE)


def _log_integral(log_f, lo, hi, guess):
    """``log ∫ exp(log_f(t)) dt`` over ``[lo, hi]``, for ``log_f`` mapping
    an array of ``t`` to the log integrand, which may be steep.

    A fixed grid in t misses steep edges (a likelihood cut off sharply by
    a bound, falling by many e-folds per grid step) and narrow peaks. Here
    each scan evaluates ``log_f`` on a uniform grid over the current
    window (plus ``guess``, the expected peak) and keeps the span of nodes
    within ``depth`` e-folds of the maximum, widened by one node on each
    side, so that the window brackets every part of the integrand that
    matters. Gauss–Legendre then integrates over the final window, all in
    log space. The window does not carry gradients: its edges are where
    the integrand is ~exp(-depth) of its peak, or the fixed bounds.
    """
    guess = jax.lax.stop_gradient(guess)
    dtype = guess.dtype
    depth = -onp.log(float(np.finfo(dtype).eps)) + 10.0
    lo = np.asarray(lo, dtype)
    hi = np.asarray(hi, dtype)
    a, b = lo, hi
    uniform = np.linspace(0.0, 1.0, _N_SCAN, dtype=dtype)
    for k in range(_N_NARROW):
        nodes = a + (b - a) * uniform
        if k == 0:
            nodes = np.sort(
                np.concatenate([nodes, np.clip(guess, lo, hi)[None]])
            )
        values = jax.lax.stop_gradient(log_f(jax.lax.stop_gradient(nodes)))
        values = np.where(np.isnan(values), -np.inf, values)
        keep = values >= np.max(values) - depth
        index = np.arange(nodes.size)
        first = np.min(np.where(keep, index, nodes.size))
        last = np.max(np.where(keep, index, -1))
        a = nodes[np.maximum(first - 1, 0)]
        b = nodes[np.minimum(last + 1, nodes.size - 1)]
    x, w = (np.asarray(v, dtype) for v in _LEGENDRE)
    half = 0.5 * (b - a)
    t = a + half * (x + 1.0)
    return jax.nn.logsumexp(log_f(t) + np.log(w)) + np.log(half)


def _check_dof(dof):
    dof = float(dof)
    if not 0.0 < dof <= 1.0:
        raise ValueError(
            f"The effective-dof fraction must be in (0, 1], got {dof}."
        )
    return dof


def _check_marginal_data(data):
    """Raise if ``data`` has a model-dependent likelihood normalizer, which
    the scale-marginalized surface does not include."""
    found = [
        what
        for what, present in (
            ("gains (OIData.with_gains)", data.gains is not None),
            (
                "closure-phase offsets (OIData.with_closure_offsets)",
                data.phase_offsets is not None,
            ),
            (
                "extra observables with a model-dependent covariance",
                bool(data.has_model_covariance),
            ),
        )
        if present
    ]
    if found:
        raise NotImplementedError(
            "The scale-marginalized surface (scales='marginal', "
            "epoch_positions, marginal_loglike) needs a likelihood whose "
            "normalization depends only on the error scales; these data "
            f"have {' and '.join(found)}, whose covariance a scale does not "
            "multiply. Use the data without them, or scales='quoted'."
        )


class _Surface:
    """The scale-marginalized score of one dataset, block by block.

    Only the layout of the data enters (the blocks, their ν and
    ``s_max``), so that one compiled kernel serves every dataset of one
    shape: the hash and equality, used as a static jit argument, never
    depend on data values. ``score(chi2, dof, data)`` maps the blocks' χ²
    on the quoted errors to ``m = -Σ_b (ν_b/2) ln χ²_b`` (Gaussian
    normalization in every block) or, with ``s_max``, to the log of each
    block's likelihood integrated over ln s in [-ln s_max, ln s_max]
    (uniform in ln s: the Jeffreys prior, bounded), with the exact von
    Mises normalization for uncorrelated closure phases (whose quoted
    errors are read from the traced ``data``). ``dof`` (a fraction in
    (0, 1], traced) tempers every block's log likelihood: ν_eff = dof · ν.
    """

    def __init__(self, data, s_max=None):
        _check_marginal_data(data)
        if s_max is not None and not float(s_max) > 1.0:
            raise ValueError(f"s_max must exceed 1, got {s_max}.")
        self.blocks = _blocks(data)
        self.names = tuple(b[0] for b in self.blocks)
        self.nu = onp.array([b[3] for b in self.blocks], dtype=float)
        self.s_max = None if s_max is None else float(s_max)

    def _key(self):
        return (self.blocks, self.s_max)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, _Surface) and self._key() == other._key()

    def chi2(self, model, data, noise=None):
        """χ² of each block of ``data`` for ``model``, on the quoted errors."""
        r = whitened_residuals(model, data, **(noise or {}))
        return np.stack([np.sum(r[a:b] ** 2) for _, a, b, _, _ in self.blocks])

    def score(self, chi2, dof, data):
        dof = np.asarray(dof, chi2.dtype)
        if self.s_max is None:
            tiny = np.finfo(chi2.dtype).tiny
            nu = np.asarray(self.nu, chi2.dtype)
            return -0.5 * dof * np.sum(nu * np.log(np.maximum(chi2, tiny)))
        bound = onp.log(self.s_max)
        total = 0.0
        for k, (_, _, _, nu, von_mises) in enumerate(self.blocks):

            def log_f(log_s, k=k, nu=nu, von_mises=von_mises):
                loglike = -0.5 * chi2[k] * np.exp(-2.0 * log_s)
                if not von_mises:
                    return dof * (loglike - nu * log_s)
                # von Mises: -log(2π I0(κ)) + κ per phase, κ = 1/(s σ)²,
                # relative to the Gaussian's -log(√(2π) σ), as in the other
                # blocks: -log(√(2π) i0e(κ)) + log σ, which is -log s for
                # κ ≫ 1.
                sigma = np.asarray(data.d_phi, chi2.dtype).reshape(-1)
                kappa = 1.0 / (np.exp(2.0 * log_s)[:, None] * sigma**2)
                loglike = (
                    loglike
                    - np.sum(np.log(np.sqrt(2.0 * np.pi) * i0e(kappa)), axis=1)
                    + np.sum(np.log(sigma))
                )
                return dof * loglike

            # The Gaussian block's peak, ln ŝ = ln √(χ²/ν).
            tiny = np.finfo(chi2.dtype).tiny
            guess = 0.5 * np.log(np.maximum(chi2[k], tiny) / nu)
            total = total + _log_integral(log_f, -bound, bound, guess)
        return total

    def weights(self, chi2, dof, data):
        """The IRLS weight of each block, ``w_b = 2 ∂(-score)/∂χ²_b``.

        Near a point, the score falls as ``Σ_b w_b χ²_b / 2``, so a
        Gauss–Newton step on the blocks' residuals multiplied by √w_b has
        the curvature of the score. ``w_b ≥ 0`` because the score falls
        as any χ²_b rises. Without ``s_max`` it is ``dof ν_b/χ²_b``, which
        is ``dof/ŝ_b²`` with ŝ_b the block's profile scale. ``dof`` is a
        scalar here.
        """
        return 2.0 * jax.grad(lambda c: -self.score(c, dof, data))(chi2)

    def weighted_rows(self, r, dof, data):
        """The whitened residuals ``r`` with each block multiplied by
        √(stop_gradient ``w_b``): curvature rows of the score (see
        [`weights`][virgil.epochs._Surface.weights]). The weights are
        frozen, so ``½ |rows|²`` has the score's Gauss–Newton curvature."""
        chi2 = np.stack([np.sum(r[a:b] ** 2) for _, a, b, _, _ in self.blocks])
        w = jax.lax.stop_gradient(self.weights(chi2, dof, data))
        return np.concatenate(
            [
                np.sqrt(w[k]) * r[a:b]
                for k, (_, a, b, _, _) in enumerate(self.blocks)
            ]
        )

    def scales(self, chi2):
        """ŝ_b = √(χ²_b/ν_b), the profile maximum of each block's scale."""
        return onp.sqrt(onp.asarray(chi2, float) / self.nu)


def marginal_loglike(model, data, *, dof=1.0, s_max=None, **noise):
    """The log likelihood with each block's error scale marginalized.

    Each observable block *b* of ``data`` (visibilities, phases, and each
    extra observable) has an unknown factor ``s_b`` on its quoted errors.
    Integrating it out under its Jeffreys prior ``1/s_b`` (Jeffreys 1946,
    https://doi.org/10.1098/rspa.1946.0056) gives, up to a constant,

        m = -Σ_b (ν_b/2) ln χ²_b,

    where χ²_b is the block's χ² on the quoted errors (from
    [`whitened_residuals`][virgil.likelihood.whitened_residuals]) and
    ν_b its number of independent observables
    ([`n_independent`][virgil.oidata.OIData.n_independent], split by
    block). Profiling ``s_b`` gives the same function, at
    ŝ_b² = χ²_b/ν_b (the re-estimation of error bars of MacKay 1992,
    Neural Comput. 4, 415, https://doi.org/10.1162/neco.1992.4.3.415). ``m`` does not change when any block's errors are
    multiplied by a constant, so its differences between models (e.g.
    the gap between two peaks) do not depend on how well the errors were
    quoted. Near a peak, m ≈ -χ²/(2ŝ²): differences are those of the
    quoted-error log likelihood divided by ŝ².

    Every block uses the Gaussian (small-σ) normalization, including
    uncorrelated closure phases, whose likelihood in
    [`model_loglike`][virgil.likelihood.model_loglike] is a von Mises
    density; for those the unbounded marginal would be improper as
    s → ∞. ``m`` is a search surface, not a replacement for
    ``model_loglike``. Where sσ is not small (weak closure phases with a
    large scale), pass ``s_max``.

    Data whose likelihood has a model-dependent normalization are refused
    with a ``NotImplementedError``: gains
    ([`OIData.with_gains`][virgil.oidata.OIData.with_gains]), closure-phase
    offsets
    ([`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets])
    and extra observables with a model-dependent covariance. Their
    marginalized nuisance covariance is not multiplied by an error scale,
    so a block's χ² is not ∝ 1/s² and the scale would not be integrated
    out. The same holds for ``epoch_positions`` and
    ``rank_orbits(scales="marginal")``.

    Parameters
    ----------
    model : SourceModel
        Model to evaluate.
    data : OIData
        Data to compare with.
    dof : float, optional
        The effective number of degrees of freedom as a fraction of ν_b
        (default 1). Errors correlated beyond the model of the quoted
        errors carry fewer than ν_b degrees of freedom, and ``m`` with
        ν_b is then over-confident by about ν_b/ν_eff; a ``dof`` below 1
        (e.g. calibrated by a residual bootstrap) scales every difference
        of ``m`` down by that factor.
    s_max : float, optional
        Bound each scale to [1/s_max, s_max] and integrate it out
        numerically, with the exact von Mises normalization for
        uncorrelated closure phases: ``m`` is then Σ_b ln ∫ L_b(s)^dof
        d ln s over the bounds. The integral is adaptive in ln s (scans
        narrow it to where the integrand matters, then Gauss–Legendre),
        so it stays accurate where the likelihood falls steeply from a
        bound. This is finite and proper where the unbounded marginal is
        not.
    **noise
        Other noise terms (e.g. ``wavel_scale``), as for
        ``whitened_residuals``. ``vis_scale`` and ``phi_scale`` should be
        left out: the scales are integrated out.

    Returns
    -------
    float
    """
    surface = _Surface(data, s_max)
    return surface.score(
        surface.chi2(model, data, noise), _check_dof(dof), data
    )


_SCALES = ("quoted", "marginal")


def _check_scales(scales, name):
    if scales is None:
        warnings.warn(
            f"{name} ranks with the quoted errors when scales is not given; "
            "the default will become scales='marginal' (each dataset's error "
            "scales integrated out, see marginal_loglike). Pass "
            "scales='quoted' or scales='marginal' to choose.",
            FutureWarning,
            stacklevel=external_stacklevel(),
        )
        return "quoted"
    if scales not in _SCALES:
        raise ValueError(f"scales must be one of {_SCALES}, not {scales!r}.")
    return scales


def _orbit_batch_size(batch_size, datasets, n, per_candidate=1):
    """Candidates per vectorized batch of a ``lax.map`` over ``n`` of them.

    ``lax.map`` without a ``batch_size`` is a serial loop, several times
    slower than batches (3-5x for thousands of candidates on a laptop CPU).
    By default a batch holds about ``_grid.batch_visibilities()`` model
    visibilities in all: each candidate evaluates every dataset, so it
    costs the datasets' samples summed, times ``per_candidate`` for work
    that multiplies that (a flux grid). One candidate per batch at the
    least, never more than ``n``. The results do not depend on it.
    """
    if batch_size is not None:
        return min(batch_size_or_default(batch_size, None), max(int(n), 1))
    samples = sum(max(onp.size(d.u), onp.size(d.wavel), 1) for d in datasets)
    fit = batch_visibilities() // max(samples * per_candidate, 1)
    return min(max(fit, 1), max(int(n), 1))


def _loglikes(
    scene_of,
    inputs,
    data,
    noise,
    batch_size,
    dtype,
    scales="quoted",
    dof=1.0,
    s_max=None,
):
    """``data``'s log likelihood of ``scene_of(x)`` for each ``x`` in the
    stacked ``inputs``, as a NumPy array (``-inf`` where not finite).

    With ``scales="marginal"`` it is the scale-marginalized score
    ``m = -Σ (ν/2) ln χ²`` over every block of every dataset (see
    [`marginal_loglike`][virgil.epochs.marginal_loglike])."""
    if not isinstance(data, Epochs):
        raise TypeError(
            f"data must be an Epochs, not {type(data).__name__}: rankings "
            "take one snapshot per dataset."
        )
    per_dataset = data.noise(noise or {})
    batch_size = _orbit_batch_size(
        batch_size, data.data, jax.tree_util.tree_leaves(inputs)[0].shape[0]
    )
    if scales == "marginal":
        surfaces = tuple(_Surface(d, s_max) for d in data.data)
        dofs = np.asarray(
            [_check_dof(_dof_for(dof, data, k)) for k in range(len(data))]
        )
        with run_in(dtype):
            inputs = cast_tree(inputs, dtype)
            datasets = cast_tree(tuple(data.data), dtype)

            @jax.jit
            def every_marginal(inputs, datasets, dofs):
                def one(x):
                    snapshots = data.snapshots(scene_of(x))
                    total = sum(
                        f.score(f.chi2(s, d, n), dofs[k], d)
                        for k, (s, d, n, f) in enumerate(
                            zip(snapshots, datasets, per_dataset, surfaces)
                        )
                    )
                    return np.where(np.isfinite(total), total, -np.inf)

                return jax.lax.map(one, inputs, batch_size=batch_size)

            dofs = np.asarray(dofs, dtype)
            return onp.asarray(
                every_marginal(inputs, datasets, dofs), dtype=float
            )
    with run_in(dtype):
        inputs = cast_tree(inputs, dtype)
        datasets = cast_tree(tuple(data.data), dtype)

        @jax.jit
        def every(inputs, datasets):
            def one(x):
                snapshots = data.snapshots(scene_of(x))
                total = sum(
                    model_loglike(s, d, **n)
                    for s, d, n in zip(snapshots, datasets, per_dataset)
                )
                return np.where(np.isfinite(total), total, -np.inf)

            return jax.lax.map(one, inputs, batch_size=batch_size)

        return onp.asarray(every(inputs, datasets), dtype=float)


@dataclasses.dataclass(frozen=True)
class RankedOrbits:
    """Orbits ranked by the log likelihood of multi-epoch data, best first.

    Returned by [`rank_orbits`][virgil.epochs.rank_orbits] and
    [`chain_starts`][virgil.epochs.chain_starts]. Iterating gives
    ``(orbit, loglike)`` pairs, like the ``(orbit, χ²)`` pairs of
    [`starting_orbits`][virgil.orbits.starting_orbits], and
    ``ranked[k]`` is the ``k``-th pair.

    Attributes
    ----------
    orbits : tuple
        The orbits, best first.
    loglike : numpy.ndarray
        The log likelihood of all the data for each orbit (with the
        ``noise`` values of the ranking), or with ``scales="marginal"``
        the scale-marginalized score, in the same order.
    order : numpy.ndarray
        The position of each orbit in the list that was ranked.
    times : numpy.ndarray
        The snapshot times of the datasets (MJD), at which
        :meth:`positions` and ``chain_starts`` compare orbits.
    resolution_mas : float
        The finest resolution λ/B_max of the data (mas).
    """

    orbits: tuple
    loglike: onp.ndarray
    order: onp.ndarray
    times: onp.ndarray
    resolution_mas: float

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, k):
        return self.orbits[k], float(self.loglike[k])

    def __iter__(self):
        return iter(zip(self.orbits, (float(x) for x in self.loglike)))

    @property
    def best(self):
        """The orbit with the highest log likelihood."""
        return self.orbits[0]

    def positions(self):
        """The companion's position ``(dra, ddec)`` (mas) of each orbit at
        each snapshot time, shape ``(n_orbits, n_times, 2)``."""
        with run_in("float64"):
            stacked = cast_tree(_stack(self.orbits), "float64")
            dra, ddec, _ = jax.vmap(lambda o: o.relative(self.times))(stacked)
            return onp.stack([onp.asarray(dra), onp.asarray(ddec)], -1)

    def _take(self, index):
        index = onp.asarray(index, dtype=int)
        return dataclasses.replace(
            self,
            orbits=tuple(self.orbits[k] for k in index),
            loglike=self.loglike[index],
            order=self.order[index],
        )


def _ranked(orbits, loglike, data):
    # The same deterministic order as score_orbits(...).order(). Imported
    # here because orbit_search imports this module.
    from .orbit_search import rank_scores

    order = rank_scores(loglike)
    return RankedOrbits(
        orbits=tuple(orbits[k] for k in order),
        loglike=loglike[order],
        order=order,
        times=onp.asarray(data.times, dtype=onp.float64),
        resolution_mas=data.resolution_mas,
    )


def rank_orbits(
    model,
    data,
    orbits,
    *,
    scales=None,
    noise=None,
    dof=1.0,
    s_max=None,
    batch_size=None,
    dtype="float64",
):
    """Rank trial orbits by the log likelihood of multi-epoch data.

    Each orbit's scene is evaluated at every dataset's snapshot time and
    compared with all the data at once, by one compiled kernel mapped
    over the orbits. This judges trial orbits, e.g. from a Thiele–Innes
    grid on rough positions
    ([`starting_orbits`][virgil.orbits.starting_orbits]) or from prior
    draws, by the interferometric data themselves rather than by the
    positions that seeded them.

    Parameters
    ----------
    model : callable
        Maps an orbit to a time-dependent scene, e.g.
        ``lambda orbit: OrbitalBinary(orbit, 0.1)``. It is traced, so it
        must not convert the orbit's elements to Python floats; other
        parameters (fluxes, the shape of an extended component) are fixed
        at trial values.
    data : Epochs
        The data, one snapshot per dataset.
    orbits : sequence or orbit
        A list of orbits, the ``(orbit, χ²)`` pairs of
        ``starting_orbits``, a ``RankedOrbits``, or one orbit whose
        elements have a leading axis (e.g. prior draws). All must share
        one class and ``t_ref``.
    scales : {"quoted", "marginal"}, optional
        How the errors are scaled. ``"quoted"`` ranks by
        ``Epochs.loglike`` on the quoted errors (with ``noise``), so the
        datasets with the most underestimated errors dominate the
        ranking. ``"marginal"`` integrates each dataset's error scale out,
        block by block (visibilities, phases), under its Jeffreys prior,
        and ranks by the sum of
        [`marginal_loglike`][virgil.epochs.marginal_loglike]: the
        ranking then does not change when one dataset's errors are
        rescaled. Data with gains, closure-phase offsets or a
        model-dependent covariance are refused (see ``marginal_loglike``). The default is ``"quoted"`` for now, with a
        ``FutureWarning`` when ``scales`` is not given; it will become
        ``"marginal"``.
    noise : dict, optional
        Noise values keyed by epoch or dataset name, as for
        [`Epochs.loglike`][virgil.epochs.Epochs.loglike]; by default
        the quoted errors. With ``scales="marginal"``, error scales are
        integrated out and only the other terms (e.g. ``wavel_scale``)
        matter.
    dof, s_max : optional
        For ``scales="marginal"``: the effective-dof fraction (one
        number, or a dict keyed by dataset or epoch name) and the bound
        on the scales, as for ``marginal_loglike``.
    batch_size : int, optional
        Orbits evaluated together by ``jax.lax.map`` (default: as many as
        keep about a million model visibilities, summed over all the
        datasets, in memory at once, at least one and at most all of
        them). Lower it if the data are large.
    dtype : {"float64", "float32"}, optional
        Precision of the evaluation (default float64, in a local
        ``jax.enable_x64`` context).

    Returns
    -------
    RankedOrbits
        The orbits sorted by log likelihood, best first. An orbit whose
        likelihood is not finite ranks last, with ``-inf``.

    Examples
    --------
    >>> candidates = starting_orbits(positions, periods)  # doctest: +SKIP
    >>> ranked = rank_orbits(lambda o: OrbitalBinary(o, 0.1), epochs,
    ...                      candidates, scales="marginal")  # doctest: +SKIP
    >>> ranked.best, ranked.loglike[:3]  # doctest: +SKIP
    """
    scales = _check_scales(scales, "rank_orbits")
    listed = _orbit_list(orbits)
    loglike = _loglikes(
        model,
        _stack(listed),
        data,
        noise,
        batch_size,
        dtype,
        scales=scales,
        dof=dof,
        s_max=s_max,
    )
    return _ranked(listed, loglike, data)


def _distinct(ranked, n, min_distance_mas):
    """Indices of up to ``n`` orbits of ``ranked``, best first, whose
    positions at the snapshot times differ from every better choice by
    more than ``min_distance_mas`` at some time."""
    finite = int(onp.sum(onp.isfinite(ranked.loglike)))
    if finite == 0:
        return []
    positions = ranked.positions()
    chosen = []
    for k in range(finite):
        offsets = [positions[k] - positions[j] for j in chosen]
        if all(
            onp.max(onp.hypot(o[:, 0], o[:, 1])) > min_distance_mas
            for o in offsets
        ):
            chosen.append(k)
            if len(chosen) == n:
                break
    return chosen


def chain_starts(ranked, n_chains, *, min_distance_mas=None):
    """The best distinct orbits of a ranking, one per chain.

    Starting every chain at one best fit hides other modes; starting
    chains in different good modes (the best few distinct orbits) lets
    the chains show whether the posterior is multimodal. Two orbits are
    the same mode when the companion's positions at every snapshot time
    agree to within ``min_distance_mas``: the mirror orbit (Ω + 180°,
    ω + 180°), which has the same sky motion, is the same mode, while a
    fringe alias or a period alias is not.

    Parameters
    ----------
    ranked : RankedOrbits
        From [`rank_orbits`][virgil.epochs.rank_orbits].
    n_chains : int
        Number of starts.
    min_distance_mas : float, optional
        The largest difference in position (mas) at any snapshot time for
        two orbits to count as one mode; by default half the finest
        resolution λ/B_max of the data.

    Returns
    -------
    RankedOrbits
        ``n_chains`` orbits, best first. If there are fewer distinct
        orbits than chains, the distinct orbits are repeated in turn.
        Map each orbit to your model's parameters (e.g. ``period``,
        ``dt_peri``) to start a chain there.
    """
    if n_chains < 1:
        raise ValueError(f"n_chains must be at least 1, got {n_chains}.")
    if min_distance_mas is None:
        min_distance_mas = 0.5 * ranked.resolution_mas
    chosen = _distinct(ranked, n_chains, min_distance_mas)
    if not chosen:
        raise ValueError("No orbit has a finite log likelihood.")
    return ranked._take([chosen[k % len(chosen)] for k in range(n_chains)])


@dataclasses.dataclass(frozen=True)
class EpochPeaks:
    """The catalogue of one dataset's peaks, from
    [`epoch_positions`][virgil.epochs.epoch_positions]: the highest
    distinct local maxima of its grid, each refined with ``refine``,
    best first.

    Attributes
    ----------
    dra, ddec, flux : numpy.ndarray
        The peaks' positions (mas, East and North) and companion/primary
        fluxes, after refinement.
    loglike : numpy.ndarray
        The log likelihood of each peak on the quoted errors.
    marginal : numpy.ndarray
        The scale-marginalized score m of each peak
        ([`marginal_loglike`][virgil.epochs.marginal_loglike]); the peaks
        are sorted by it, highest first.
    weight : numpy.ndarray
        The posterior mass of each peak in this dataset alone, summing
        to 1: the Laplace approximation w ∝ exp(m) det(H)^(-1/2), with H
        the Hessian of -m at the peak over (dra, ddec, flux), so that a
        broad peak outweighs a narrow one of the same height. Where
        ``laplace`` is false the peak's height exp(m) stands in.
    edge : numpy.ndarray
        Whether each peak started from a grid point on the edge of the
        position grid. A best peak at the edge may stand for a companion
        outside the grid: treat that dataset as ambiguous.
    laplace : numpy.ndarray
        Whether each peak's weight is its Laplace mass: false without
        ``refine``, or where the Hessian is not positive definite (a
        peak on a bound, or a flat direction).
    """

    dra: onp.ndarray
    ddec: onp.ndarray
    flux: onp.ndarray
    loglike: onp.ndarray
    marginal: onp.ndarray
    weight: onp.ndarray
    edge: onp.ndarray
    laplace: onp.ndarray

    def __len__(self):
        return len(self.marginal)


@dataclasses.dataclass(frozen=True)
class NightFlag:
    """Why one night's position should not seed an orbit.

    Attributes
    ----------
    code : str
        A short code: ``"alias-peaks"``, ``"low-gap"``, ``"chi2-high"``,
        ``"sep-below-resolution"``, ``"flux-at-bound"``.
    reason : str
        A human-readable reason with the numbers behind it.
    """

    code: str
    reason: str


@dataclasses.dataclass(frozen=True)
class EpochPositions:
    """The companion's position in each dataset, from
    [`epoch_positions`][virgil.epochs.epoch_positions].

    Positions, covariances and ``gap_marginal`` come from the
    scale-marginalized surface
    ([`marginal_loglike`][virgil.epochs.marginal_loglike]), on which each
    observable block's error scale is integrated out: they do not change
    when a dataset's quoted errors, or only its closure-phase errors, are
    multiplied by a constant.

    Attributes
    ----------
    names : tuple of str
        The datasets.
    mjd : numpy.ndarray
        Each dataset's snapshot time (MJD).
    dra, ddec : numpy.ndarray
        The fitted positions (mas, East and North).
    cov : numpy.ndarray
        The covariance of each position, shape ``(n, 2, 2)`` (mas²), from
        the curvature of the scale-marginalized surface.
    flux : numpy.ndarray
        The fitted companion/primary flux of each dataset (at most 1).
    gap : numpy.ndarray
        The gap on the **quoted** errors, as before ``gap_marginal`` was
        added: the log likelihood of the best grid position minus that of
        the best grid position more than ``gap_mas`` from it. It grows as
        1/s² when the errors are underestimated by s, so an ambiguous
        dataset can look decisive. It is a grid quantity, not refined,
        and its best grid position need not be the fitted one. Kept
        unchanged for one release, after
        which ``gap`` will hold the marginal value.
    gap_marginal : numpy.ndarray
        How decisive each dataset is, on the scale-marginalized surface:
        the same difference between the best grid position and its best
        rival more than ``gap_mas`` away, with each block's error scale
        integrated out. A small gap means another peak (a fringe alias,
        or a mirror image when closure phases are weak) fits almost as
        well. ``decisive`` and ``min_gap`` compare this. With ``refine``
        it compares refined peaks: the best of the catalogue (``peaks``)
        and the best of its refined rivals and of the grid points more
        than ``gap_mas`` from it.
    chi2_raw : tuple of dict
        For each dataset, χ²/N on the quoted errors at the fitted
        position, N = ``n_independent``: per block (``"vis"``, ``"phi"``,
        ``"extra[k]"``) and for the whole dataset (``"all"``). A value
        well above 1 means the quoted errors are too small; it is not
        made 1 by the scale marginalization.
    scale : tuple of dict
        For each dataset, the fitted error scale ŝ = √(χ²/ν) of each
        block (``"vis_scale"``, ``"phi_scale"``, ``"extra[k]_scale"``).
    peaks : tuple of EpochPeaks
        For each dataset, the catalogue of its peaks, best first: the
        first is the fitted position.
    flags : tuple of tuple of NightFlag
        For each dataset, why its position should not seed an orbit (empty
        for a trustworthy night), from the relative thresholds of
        ``epoch_positions``; see ``flagged``.
    primary_diam : float or None
        The uniform-disc primary the positions were fitted with (mas), or
        None for a point primary.
    """

    names: tuple
    mjd: onp.ndarray
    dra: onp.ndarray
    ddec: onp.ndarray
    cov: onp.ndarray
    flux: onp.ndarray
    gap: onp.ndarray
    gap_marginal: onp.ndarray
    chi2_raw: tuple
    scale: tuple
    peaks: tuple = ()
    flags: tuple = ()
    primary_diam: float | None = None

    @property
    def flagged(self):
        """Whether each dataset carries any flag (``flags``): a flagged
        night seeds nothing in
        [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases], but its
        data still count in the likelihood."""
        if not self.flags:
            return onp.zeros(len(self.names), dtype=bool)
        return onp.array([len(f) > 0 for f in self.flags], dtype=bool)

    def flag_codes(self):
        """The flag codes of each dataset, ``{name: [code, ...]}``, for
        the flagged datasets only."""
        return {
            n: [f.code for f in fl]
            for n, fl in zip(self.names, self.flags)
            if fl
        }

    def decisive(self, min_gap):
        """Whether each dataset's ``gap_marginal`` exceeds ``min_gap``."""
        return self.gap_marginal > min_gap

    @property
    def edge(self):
        """Whether each dataset's best peak started at the edge of the
        position grid (see ``EpochPeaks.edge``)."""
        return onp.array([bool(p.edge[0]) for p in self.peaks], dtype=bool)

    def positions(self, *, t_ref, min_gap=None, skip_flagged=True):
        """The positions as [`PositionData`][virgil.orbits.PositionData],
        for [`starting_orbits`][virgil.orbits.starting_orbits].

        Parameters
        ----------
        t_ref : float
            The reference time (MJD) of the orbits fitted to them: give
            the ``t_ref`` of your model.
        min_gap : float, optional
            Keep only the datasets whose ``gap_marginal`` exceeds this.
            Seeding orbits with an indecisive dataset's position can start
            every orbit at the wrong peak; the visibility fit judges those
            datasets instead.
        skip_flagged : bool, optional
            Drop the flagged datasets (``flagged``; default True, so that
            the flags cannot be ignored silently): the same idea as
            ``min_gap`` with several reasons. ``min_gap`` and the
            ``flag_min_gap`` of ``epoch_positions`` are the same
            threshold on ``gap_marginal``. Pass False to keep every
            dataset.
        """
        keep = (
            onp.ones(len(self.names), bool)
            if min_gap is None
            else self.decisive(min_gap)
        )
        if skip_flagged:
            keep = keep & ~self.flagged
        if not keep.any():
            raise ValueError(
                "No dataset is decisive and unflagged: the marginal gaps are "
                f"{self.gap_marginal.round(1)}, min_gap is {min_gap} and "
                f"the flagged datasets are {self.flag_codes()}."
            )
        return PositionData(
            self.mjd[keep],
            self.dra[keep],
            self.ddec[keep],
            self.cov[keep],
            t_ref=t_ref,
        )


def _binary(x, diam=None):
    """The static binary ``x = (dra, ddec, flux)``: two point stars, or,
    when ``diam`` (mas, traced) is given, a point companion beside a
    uniform-disc primary. ``diam=None`` keeps the fast point-star path
    (``None`` is an empty pytree, so it never triggers a recompilation)."""
    if diam is None:
        return BinaryModelCartesian(x[0], x[1], x[2])
    # Resolved primaries: a point primary absorbs the V^2 deficit of a
    # resolved star into the companion's flux and position (iota Peg,
    # 1.05 mas), so the per-night seeds are biased and their chi2 poor.
    return System(
        primary=UniformDisk(diam),
        companion=PointSource(x[2], x[0], x[1]),
    )


@functools.partial(jax.jit, static_argnames=("surface", "batch_size"))
def _grid_scores(points, data, surface, batch_size, diam=None):
    """The quoted-error log likelihood and the blocks' χ² at each point."""

    def one(x):
        whitened, errors = _whitened_and_errors(_binary(x, diam), data, {})
        chi2 = np.stack(
            [np.sum(whitened[a:b] ** 2) for _, a, b, _, _ in surface.blocks]
        )
        return _gaussian_loglike(whitened, errors), chi2

    return jax.lax.map(one, points, batch_size=batch_size)


@functools.partial(jax.jit, static_argnames="surface")
def _grid_marginal(chi2, dof, data, surface):
    """The scale-marginalized score of each grid point's block χ²."""
    return jax.vmap(lambda c: surface.score(c, dof, data))(chi2)


def _negative_marginal(x, dof, data, surface, diam=None):
    return -surface.score(surface.chi2(_binary(x, diam), data), dof, data)


_negative_marginal_hessian = jax.jit(
    jax.hessian(_negative_marginal), static_argnames="surface"
)
_block_chi2 = jax.jit(
    lambda x, data, surface, diam=None: surface.chi2(_binary(x, diam), data),
    static_argnames="surface",
)


def _gap(score, xx, yy, rival):
    """The best position of a (dra, ddec, flux) grid of scores, and its
    score minus that of the best point more than ``rival`` from it."""
    by_position = score.max(axis=2)
    i, j = onp.unravel_index(onp.argmax(by_position), by_position.shape)
    far = onp.hypot(xx[..., 0] - xx[i, j, 0], yy[..., 0] - yy[i, j, 0]) > rival
    gap = by_position[i, j] - by_position[far].max() if far.any() else onp.inf
    return (i, j, int(onp.argmax(score[i, j]))), float(gap)


def _check_n_peaks(n_peaks):
    if (
        isinstance(n_peaks, bool)
        or not isinstance(n_peaks, (int, onp.integer))
        or n_peaks < 1
    ):
        raise ValueError(
            f"n_peaks must be an integer of at least 1, not {n_peaks!r}."
        )


def _grid_peaks(score, xx, yy, rival, n_peaks):
    """The ``(i, j, k)`` cells of up to ``n_peaks`` local maxima of a
    (dra, ddec, flux) grid of scores, highest first, each more than
    ``rival`` from every higher one (k is the best flux there)."""
    from scipy.ndimage import maximum_filter

    by_position = score.max(axis=2)
    finite = onp.isfinite(by_position)
    local = finite & (
        by_position == maximum_filter(by_position, size=3, mode="nearest")
    )
    cells = onp.argwhere(local)
    cells = cells[onp.argsort(-by_position[local], kind="stable")]
    if not len(cells):
        cells = [onp.unravel_index(onp.argmax(by_position), by_position.shape)]
    chosen = []
    for i, j in cells:
        if all(
            onp.hypot(xx[i, j, 0] - xx[a, b, 0], yy[i, j, 0] - yy[a, b, 0])
            > rival
            for a, b, _ in chosen
        ):
            chosen.append((int(i), int(j), int(onp.argmax(score[i, j]))))
            if len(chosen) == n_peaks:
                break
    return chosen


def _rank_peaks(grid_points, grid_marginal, refine, score, rival):
    """Refine each grid peak and rank the peaks by their marginal score.

    ``refine(x0)`` returns a refined point and its marginal score, kept
    only when it is no worse than the grid point's; ``score(points)``
    returns the quoted log likelihood and the marginal score of each
    point. Two grid peaks can refine to one maximum, so a peak within
    ``rival`` of a higher one is dropped. Returns the points, quoted and
    marginal scores, best first, and the order (indices into
    ``grid_points``).
    """
    points = onp.array(grid_points, dtype=float)
    if refine is not None:
        for n, (x0, m0) in enumerate(zip(grid_points, grid_marginal)):
            refined, value = refine(onp.asarray(x0, dtype=float))
            if value >= m0:
                points[n] = refined
    loglike, marginal = score(points)
    loglike = onp.asarray(loglike, dtype=float)
    marginal = onp.where(
        onp.isfinite(marginal), onp.asarray(marginal, dtype=float), -onp.inf
    )
    order = []
    for n in onp.argsort(-marginal, kind="stable"):
        if all(
            onp.hypot(*(points[n, :2] - points[k, :2])) > rival for k in order
        ):
            order.append(int(n))
    order = onp.array(order, dtype=int)
    return points[order], loglike[order], marginal[order], order


def _peak_weights(marginal, hessians):
    """Normalized Laplace masses of the peaks, and whether each is one.

    log w_k = m_k - ½ log det H_k, with H_k the Hessian of -m at peak k
    over (dra, ddec, flux). A peak whose Hessian is missing (``None``)
    or not positive definite falls back to its height m_k.
    """
    log_w, laplace = [], []
    for m, hess in zip(marginal, hessians):
        hess = None if hess is None else onp.asarray(hess, dtype=float)
        positive = (
            hess is not None
            and bool(onp.all(onp.isfinite(hess)))
            and bool(onp.all(onp.linalg.eigvalsh(hess) > 0))
        )
        log_w.append(m - 0.5 * onp.linalg.slogdet(hess)[1] if positive else m)
        laplace.append(positive)
    log_w = onp.asarray(log_w, dtype=float)
    top = onp.max(log_w)
    weight = (
        onp.exp(log_w - top) if onp.isfinite(top) else onp.ones(len(log_w))
    )
    return weight / weight.sum(), onp.array(laplace, dtype=bool)


def _refined_gap(points, marginal, grid_score, xx, yy, rival):
    """The best peak's marginal score minus that of its best rival more
    than ``rival`` from it: a refined peak or, as a floor, a grid
    point."""
    x, y = points[0, :2]
    far = onp.hypot(points[1:, 0] - x, points[1:, 1] - y) > rival
    rivals = list(marginal[1:][far])
    by_position = grid_score.max(axis=2)
    far_grid = onp.hypot(xx[..., 0] - x, yy[..., 0] - y) > rival
    if far_grid.any():
        rivals.append(by_position[far_grid].max())
    return float(marginal[0] - max(rivals)) if rivals else onp.inf


class _RefineProblem(eqx.Module):
    """The negative scale-marginalized surface of one dataset's static
    binary ``(dra, ddec, flux)``, as a kernel problem
    ([`_gauss_newton`][virgil._gauss_newton]).

    The curvature rows are the blocks' whitened residuals weighted by the
    IRLS weights of the surface ([`_Surface.weights`][virgil.epochs._Surface.weights]);
    the priors are flat in these coordinates.
    """

    data: object  # an OIData
    dof: jax.Array
    surface: _Surface = eqx.field(static=True)
    diam: object = None  # the primary's diameter (mas), or None: a point

    def loss(self, x):
        return _negative_marginal(
            x, self.dof, self.data, self.surface, self.diam
        )

    def rows(self, x):
        r = whitened_residuals(_binary(x, self.diam), self.data)
        return self.surface.weighted_rows(r, self.dof, self.data)

    def diag(self, x):
        return np.zeros_like(x)


def _refine_marginal(x0, dof, d64, surface, bounds, diam=None):
    """Maximize the scale-marginalized surface from ``x0`` (damped Newton on
    the IRLS curvature, in float64, deterministic) within ``bounds``.
    Returns the point and the surface's negative value there; a run that
    does not converge (or starts non-finite) returns ``x0`` and ``inf``,
    which the caller reads as no improvement."""
    state = _gauss_newton.solve(
        _RefineProblem(d64, dof, surface, diam),
        onp.asarray(x0, float),
        lower=[b[0] for b in bounds],
        upper=[b[1] for b in bounds],
        dtol=1e-12,
        max_steps=200,
    )
    if int(state.stop) != _gauss_newton.CONVERGED:
        return onp.asarray(x0, dtype=float), float("inf")
    return onp.asarray(state.z, dtype=float), float(state.value)


def _flag_one_night(
    catalogue,
    gap_marginal,
    flux_bounds,
    resolution,
    *,
    alias_nats,
    min_gap,
    sep_fraction,
    flux_tol,
):
    """The flags of one night that need only that night: ``alias-peaks``,
    ``low-gap``, ``sep-below-resolution`` and ``flux-at-bound``. A
    threshold of None disables its flag. The cross-night ``chi2-high`` is
    added by [`_flag_chi2`][virgil.epochs._flag_chi2]."""
    flags = []
    best = float(catalogue.marginal[0])
    # Single-night alias peaks: the true position is often not the best
    # peak (Gl 229 2024-02-27 and 2024-03-28), so a night whose catalogue
    # holds another peak within a few nats of the best must not seed orbits.
    rivals = catalogue.marginal[1:]
    close = (
        0 if alias_nats is None else int(onp.sum(best - rivals <= alias_nats))
    )
    if close:
        flags.append(
            NightFlag(
                "alias-peaks",
                f"{close} other peak(s) within {alias_nats:g} nats of the "
                f"best (marginal gap {gap_marginal:.1f}"
                + (
                    ", so also below the low-gap limit"
                    if min_gap is not None and gap_marginal < min_gap
                    else ""
                )
                + "): a single-night alias, the true position may be a rival",
            )
        )
    elif min_gap is not None and gap_marginal < min_gap:
        flags.append(
            NightFlag(
                "low-gap",
                f"marginal gap {gap_marginal:.1f} nats is below {min_gap:g}: "
                "the position is not decisive",
            )
        )
    sep = float(onp.hypot(catalogue.dra[0], catalogue.ddec[0]))
    if sep_fraction is not None and sep < sep_fraction * resolution:
        flags.append(
            NightFlag(
                "sep-below-resolution",
                f"separation {sep:.2f} mas is below {sep_fraction:g} "
                f"lambda/B_max = {sep_fraction * resolution:.2f} mas: the "
                "companion is not resolved",
            )
        )
    if flux_tol is not None:
        lo, hi = flux_bounds
        flux = float(catalogue.flux[0])
        if flux >= hi * (1.0 - flux_tol):
            flags.append(
                NightFlag(
                    "flux-at-bound",
                    f"flux {flux:.3g} is at the upper bound {hi:g}: the "
                    "labels of the stars are undetermined (genuine for a "
                    "near-equal pair)",
                )
            )
        elif flux <= lo * (1.0 + flux_tol):
            flags.append(
                NightFlag(
                    "flux-at-bound",
                    f"flux {flux:.3g} is at the lower bound {lo:.3g}: no "
                    "companion is detected",
                )
            )
    return flags


def _flag_chi2(chi2_all, factor):
    """The ``chi2-high`` flag of each night: raw χ²/N more than ``factor``
    times the median night's (None disables it; fewer than three nights
    have no meaningful median, so they are never flagged)."""
    chi2_all = onp.asarray(chi2_all, dtype=float)
    out = [[] for _ in chi2_all]
    if factor is None or len(chi2_all) < 3:
        return out
    median = float(onp.median(chi2_all))
    # A night whose chi2 stands far above the others (a bad night, a
    # triple, extra light) fits no binary well, so its position is not a
    # trustworthy seed whatever its peaks look like. The floor of 1 stops
    # nights with chi2/N well below 1 (overestimated errors) from making a
    # fair night an outlier. Outlier detection only: uniform failures and
    # a majority of bad nights pass.
    limit = factor * max(median, 1.0)
    for k, c in enumerate(chi2_all):
        if c > limit:
            out[k].append(
                NightFlag(
                    "chi2-high",
                    f"raw chi2/N {c:.1f} exceeds {limit:.1f} "
                    f"({factor:g} x max(median night {median:.1f}, 1))",
                )
            )
    return out


def epoch_positions(
    data,
    grid,
    *,
    gap_mas=None,
    refine=True,
    n_peaks=5,
    dof=1.0,
    s_max=None,
    batch_size=4096,
    primary_diam=None,
    flag_nights=True,
    flag_alias_nats=2.0,
    flag_min_gap=5.0,
    flag_chi2_factor=3.0,
    flag_sep_fraction=0.5,
    flag_flux_tol=0.02,
):
    """The companion's position in each dataset of a binary.

    For each dataset: a static binary
    ([`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]) on a
    grid of positions and fluxes, scored on the scale-marginalized
    surface m = -Σ_b (ν_b/2) ln χ²_b
    ([`marginal_loglike`][virgil.epochs.marginal_loglike]), in which the
    error scale of each observable block (visibilities, closure phases)
    is integrated out under its Jeffreys prior. The ``n_peaks`` highest
    distinct local maxima of the grid (more than ``gap_mas`` apart) are
    each refined (with ``refine``) by maximizing m, which is the fit with
    a free ``vis_scale`` and ``phi_scale`` per dataset, profiled, and
    ranked by their refined m: the best grid point is not always the best
    peak once refined, since fringe peaks are often narrower than the
    grid step. The best refined peak is the position, and the curvature
    of m there gives its covariance. Every peak is kept in the catalogue
    ``EpochPositions.peaks``. The
    positions, covariances and ``gap_marginal`` therefore do not depend
    on how well each block's errors were quoted, and nights whose errors
    are underestimated do not look more decisive than they are. These
    positions are only a starting point for a fit of the orbit to the
    visibilities: when the scene is more than two point stars they can be
    biased.

    **Night flags.** Each night also gets flags with reasons
    (``EpochPositions.flags``, see ``flag_nights``): a night with a rival
    peak within ``flag_alias_nats`` of the best (the single-night alias of
    Gl 229 on 2024-02-27 and 2024-03-28, where the true position is often
    not the best peak), a low ``gap_marginal``, a raw χ²/N far above the
    median night's, a separation below λ/2B_max, or a flux at its bound.
    No threshold is a per-system number. Two are in nats of the
    scale-marginalized score (``flag_alias_nats``, ``flag_min_gap``): they
    are not dimensionless, and the gaps grow with the amount of data (the
    number of frames and baselines), so retune them for very different
    data volumes. Three are relative: a multiple of the median night
    (``flag_chi2_factor``), a fraction of the night's own λ/B_max
    (``flag_sep_fraction``) and a fraction of a flux bound
    (``flag_flux_tol``). ``chi2-high`` is outlier detection only: it
    cannot catch a uniform failure (every night underestimated alike) or
    a majority of bad nights, and the raw χ²/N column of
    ``EpochPositions.chi2_raw`` should be read in any case. A flagged
    night should seed nothing: [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases]
    skips it when seeding, but its data still count in the likelihood.

    **Resolved primaries.** With ``primary_diam`` the companion is fitted
    beside a uniform-disc primary instead of a point star, since a point
    primary absorbs a resolved star's visibility deficit into the
    companion's flux and position (ι Peg's primary is 1.05 mas).

    The raw χ²/N on the quoted errors and the fitted scale of each block
    are recorded, and a ``UserWarning`` names the datasets whose raw
    χ²/N exceeds 4 (errors underestimated by more than 2): their orbit
    fits need fitted error scales, and a rescaled χ²/N ≈ 1 does not make
    them good fits.

    Parameters
    ----------
    data : Epochs
        The data. Datasets with gains, closure-phase offsets or a
        model-dependent covariance are refused (see ``marginal_loglike``).
    grid : dict
        ``{"dra": axis, "ddec": axis, "flux": values}``: the positions
        (mas) and companion/primary fluxes to try. The position step
        should be finer than the resolution λ/B of the longest baseline.
        Fluxes must be in (0, 1]: a binary with flux ratio f > 1 at r is
        the one with 1/f at -r, so f > 1 would only add a twin of every
        peak.
    gap_mas : float, optional
        The distance (mas) beyond which another peak counts as a rival
        for ``gap`` and ``gap_marginal``; by default each dataset's
        resolution λ/B_max.
    refine : bool, optional
        Refine each grid peak by maximizing m (default). Otherwise the
        position is the best grid point, with a covariance of one grid
        step squared, and the peaks are the grid points.
    n_peaks : int, optional
        The peaks catalogued and refined per dataset (default 5). Each
        refinement is a small fit, so the cost of ``refine`` grows
        linearly with ``n_peaks``; 1 refines only the best grid point,
        which can commit to the wrong peak.
    dof : float or dict, optional
        The effective-dof fraction ν_eff/ν of each dataset's blocks (one
        number, or a dict keyed by dataset or epoch name; default 1),
        for errors correlated beyond the quoted ones (see
        ``marginal_loglike``). It multiplies ``gap_marginal`` by ``dof``
        and ``cov`` by 1/dof.
    s_max : float, optional
        Bound the scales and integrate them out numerically (see
        ``marginal_loglike``), for weak closure phases with a large
        scale.
    batch_size : int, optional
        Grid points evaluated together.
    primary_diam : float, optional
        The primary's uniform-disc diameter (mas), fixed. ``None``
        (default) or 0 is a point primary (the fast path). Give a value
        estimated from the data or from a calibrator, never from the
        orbit being tested. Fitting it (``"fit"``) is not supported yet
        and raises.
    flag_nights : bool, optional
        Compute the night flags (default True). With False every night is
        unflagged.
    flag_alias_nats : float or None, optional
        ``alias-peaks`` (nats): flag a night whose catalogue has another
        peak within this many nats (scale-marginalized score m, Δχ² ≈
        2 × nats) of the best (default 2). It needs ``n_peaks`` of at
        least 2 to see a rival. None disables it.
    flag_min_gap : float or None, optional
        ``low-gap`` (nats): flag a night whose ``gap_marginal`` is below
        this (default 5), the same quantity as the ``min_gap`` of
        ``decisive`` and ``EpochPositions.positions``. An ``alias-peaks``
        night has a gap below ``flag_alias_nats``, so it is also below
        this while ``flag_alias_nats <= flag_min_gap`` (the defaults):
        ``alias-peaks`` then implies ``low-gap``, and only the more
        specific code is listed. None disables it.
    flag_chi2_factor : float or None, optional
        ``chi2-high`` (relative): flag a night whose raw χ²/N exceeds this
        multiple of ``max(median, 1)`` over the nights (default 3; only
        with three or more nights). The floor of 1 keeps a set of nights
        with small χ²/N from flagging a night that fits well. None
        disables it.
    flag_sep_fraction : float or None, optional
        ``sep-below-resolution`` (relative): flag a night whose separation
        is below this fraction of its own λ/B_max, from that night's
        baselines and wavelengths (default 0.5, i.e. λ/2B_max). None
        disables it.
    flag_flux_tol : float or None, optional
        ``flux-at-bound`` (relative): flag a night whose flux is within
        this relative distance of the fitted range's bounds (default
        0.02). The upper bound is fixed at 1 (a brighter companion is the
        fainter one on the other side); the lower bound is a tenth of the
        grid's lowest flux. A flux at 1 may be genuine for a near-equal
        pair: the flag says the labels of the stars are not determined,
        not that the night is wrong.

    Returns
    -------
    EpochPositions
    """
    if not isinstance(data, Epochs):
        raise TypeError(f"data must be an Epochs, not {type(data).__name__}.")
    _check_n_peaks(n_peaks)
    if isinstance(primary_diam, str):
        raise ValueError(
            f"primary_diam={primary_diam!r}: epoch_positions supports only "
            "a fixed diameter (a float, in mas) or None for now; fitting "
            "it ('fit') may be added later."
        )
    diam = None if primary_diam is None else float(primary_diam)
    if diam is not None and not diam >= 0:
        raise ValueError(f"primary_diam must be None or >= 0, not {diam}.")
    diam = diam or None  # 0 is a point primary
    missing = {"dra", "ddec", "flux"} - set(grid)
    if missing:
        raise ValueError(f"grid needs the axes {sorted(missing)}.")
    axes = [onp.asarray(grid[k], float) for k in ("dra", "ddec", "flux")]
    if (axes[2] <= 0).any() or (axes[2] > 1).any():
        raise ValueError(
            "The grid's fluxes must be in (0, 1]: a companion brighter than "
            "the primary is the fainter one on the other side (1/f at -r)."
        )
    xx, yy, ff = onp.meshgrid(*axes, indexing="ij")
    points = onp.stack([xx.ravel(), yy.ravel(), ff.ravel()], -1)
    steps = [
        float(onp.min(onp.diff(onp.sort(a)))) if a.size > 1 else 1.0
        for a in axes[:2]
    ]
    bounds = [(a.min() - s, a.max() + s) for a, s in zip(axes[:2], steps)]
    bounds.append((0.1 * float(axes[2].min()), 1.0))
    rows, inflated, edge_grid, edge_peak = [], [], [], []
    night_flags = []
    for index, (name, d) in enumerate(zip(data.dataset_names, data.data)):
        surface = _Surface(d, s_max)
        dof_k = _check_dof(_dof_for(dof, data, index))
        with run_in("float64"):
            d64 = cast_tree(d, "float64")
            diam64 = None if diam is None else np.asarray(diam)
            loglike, chi2 = _grid_scores(
                np.asarray(points),
                d64,
                surface,
                min(batch_size, len(points)),
                diam64,
            )
            marginal = _grid_marginal(chi2, np.asarray(dof_k), d64, surface)
            loglike = onp.asarray(loglike).reshape(xx.shape)
            marginal = onp.asarray(marginal).reshape(xx.shape)
        loglike = onp.where(onp.isfinite(loglike), loglike, -onp.inf)
        marginal = onp.where(onp.isfinite(marginal), marginal, -onp.inf)
        rival = (
            Epochs({name: d}).resolution_mas if gap_mas is None else gap_mas
        )
        _, gap = _gap(loglike, xx, yy, rival)
        _, gap_marginal = _gap(marginal, xx, yy, rival)
        cells = _grid_peaks(marginal, xx, yy, rival, n_peaks)
        grid_points = [
            [axes[0][a], axes[1][b], axes[2][c]] for a, b, c in cells
        ]
        with run_in("float64"):
            dof64 = np.asarray(dof_k)

        def score(x, d64=d64, surface=surface, dof64=dof64, diam64=diam64):
            with run_in("float64"):
                quoted, chi2 = _grid_scores(
                    np.asarray(x), d64, surface, len(x), diam64
                )
                m = _grid_marginal(chi2, dof64, d64, surface)
            return onp.asarray(quoted), onp.asarray(m)

        def refine_one(
            x0, d64=d64, surface=surface, dof64=dof64, diam64=diam64
        ):
            with run_in("float64"):
                x, value = _refine_marginal(
                    x0, dof64, d64, surface, bounds, diam64
                )
            return x, -value

        peaks, peak_loglike, peak_marginal, order = _rank_peaks(
            grid_points,
            [marginal[c] for c in cells],
            refine_one if refine else None,
            score,
            rival,
        )
        if refine:
            gap_marginal = _refined_gap(
                peaks, peak_marginal, marginal, xx, yy, rival
            )
        at_edge = [
            a in (0, xx.shape[0] - 1) or b in (0, xx.shape[1] - 1)
            for a, b, _ in cells
        ]
        on_edge = onp.array([at_edge[n] for n in order], dtype=bool)
        if at_edge[0]:
            edge_grid.append(name)
        elif on_edge[0]:
            edge_peak.append(name)
        hessians = [None] * len(peaks)
        if refine:
            with run_in("float64"):
                hessians = [
                    onp.asarray(
                        _negative_marginal_hessian(
                            np.asarray(x), dof64, d64, surface, diam64
                        ),
                        dtype=float,
                    )
                    for x in peaks
                ]
        weight, laplace = _peak_weights(peak_marginal, hessians)
        catalogue = EpochPeaks(
            dra=peaks[:, 0],
            ddec=peaks[:, 1],
            flux=peaks[:, 2],
            loglike=peak_loglike,
            marginal=peak_marginal,
            weight=weight,
            edge=on_edge,
            laplace=laplace,
        )
        best = peaks[0]
        cov = max(steps) ** 2 * onp.eye(2)
        with run_in("float64"):
            if refine:
                full = onp.linalg.pinv(hessians[0])
                if onp.all(onp.isfinite(full)) and onp.all(
                    onp.linalg.eigvalsh(full[:2, :2]) > 0
                ):
                    cov = full[:2, :2]
            chi2_best = onp.asarray(
                _block_chi2(np.asarray(best), d64, surface, diam64),
                dtype=float,
            )
        per_block = dict(zip(surface.names, chi2_best / surface.nu))
        per_block["all"] = float(chi2_best.sum() / surface.nu.sum())
        if per_block["all"] > 4.0:
            inflated.append(f"{name} ({per_block['all']:.1f})")
        scale = {
            f"{n}_scale": float(v)
            for n, v in zip(surface.names, surface.scales(chi2_best))
        }
        rows.append(
            (best, cov, gap, gap_marginal, per_block, scale, catalogue)
        )
        night_flags.append(
            _flag_one_night(
                catalogue,
                gap_marginal,
                (bounds[2][0], bounds[2][1]),
                Epochs({name: d}).resolution_mas,
                alias_nats=flag_alias_nats,
                min_gap=flag_min_gap,
                sep_fraction=flag_sep_fraction,
                flux_tol=flag_flux_tol,
            )
            if flag_nights
            else []
        )
    if inflated:
        warnings.warn(
            "Raw chi2/N on the quoted errors exceeds 4 (errors underestimated "
            f"by more than 2) in {', '.join(inflated)}: fit error scales for "
            "these datasets, and do not read a rescaled chi2/N of 1 as a "
            "good fit.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    if edge_grid:
        warnings.warn(
            f"The best grid position of {', '.join(edge_grid)} is at the "
            "edge of the grid: the companion may lie outside it. Widen the "
            "grid.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    if edge_peak:
        warnings.warn(
            f"The best refined peak of {', '.join(edge_peak)} started from "
            "the edge of the grid: the companion may lie outside it. Widen "
            "the grid.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    flags = [[] for _ in rows]
    if flag_nights:
        extra = _flag_chi2([r[4]["all"] for r in rows], flag_chi2_factor)
        flags = [a + b for a, b in zip(night_flags, extra)]
    return EpochPositions(
        names=data.dataset_names,
        mjd=onp.asarray(data.times, dtype=onp.float64),
        dra=onp.array([r[0][0] for r in rows]),
        ddec=onp.array([r[0][1] for r in rows]),
        cov=onp.array([r[1] for r in rows]),
        flux=onp.array([r[0][2] for r in rows]),
        gap=onp.array([float(r[2]) for r in rows]),
        gap_marginal=onp.array([float(r[3]) for r in rows]),
        chi2_raw=tuple(r[4] for r in rows),
        scale=tuple(r[5] for r in rows),
        peaks=tuple(r[6] for r in rows),
        flags=tuple(tuple(f) for f in flags),
        primary_diam=diam,
    )


# Resolved primary, problem 10 of the orbit-consolidation plan (ι Peg's
# primary is 1.05 mas; a point primary gives χ²/N in the thousands). The
# data resolve the primary when a shared uniform-disc diameter improves the
# summed scale-marginalized score of the nights by this many nats over a
# point primary: one extra parameter, so 5 nats is Δχ² ≈ 10, about 3σ.
RESOLVE_NATS = 5.0
# The diameters tried, as fractions of the finest fringe spacing λ/B_max: a
# disc much smaller than 0.03 λ/B_max changes V² by less than a part in
# 10⁴, and one of 1.5 λ/B_max has crossed the first visibility null.
_DIAM_FRACTIONS = (0.03, 0.06, 0.12, 0.25, 0.4, 0.6, 0.9, 1.3)
# A position is searched within this many resolution elements of its start.
_REFINE_WINDOW = 3.0


@dataclasses.dataclass(frozen=True)
class PrimaryDiameter:
    """The primary's uniform-disc diameter estimated from per-night
    positions, from
    [`estimate_primary_diameter`][virgil.epochs.estimate_primary_diameter].

    Attributes
    ----------
    diam : float
        The best diameter (mas); 0 when a point primary is as good (or
        better) on the grid.
    sigma : float
        Its 1σ uncertainty (mas) from the curvature of the profile around
        the maximum, or NaN when the maximum sits on the edge of the grid
        and the curvature is undetermined.
    delta_nats : float
        The gain in summed scale-marginalized score of the nights, at
        ``diam`` over a point primary (nats; 0 for ``diam == 0``).
    resolved : bool
        ``diam > 0`` and ``delta_nats`` at least the threshold
        ``resolve_nats``: the data resolve the primary.
    grid, profile : numpy.ndarray
        The diameters tried (``grid[0]`` is 0, a point primary) and the
        summed score of the nights at each (positions re-refined at each
        diameter).
    positions : EpochPositions
        The input positions, re-seeded for the chosen model: refined at
        ``diam`` with ``primary_diam=diam`` when ``resolved``, else
        unchanged for a point primary. See Notes of
        ``estimate_primary_diameter``.
    """

    diam: float
    sigma: float
    delta_nats: float
    resolved: bool
    grid: onp.ndarray
    profile: onp.ndarray
    positions: EpochPositions


def estimate_primary_diameter(
    data,
    positions,
    *,
    resolve_nats=RESOLVE_NATS,
    diam_grid=None,
    s_max=None,
):
    """Estimate a shared uniform-disc primary diameter from per-night
    positions, and decide whether the data resolve the primary.

    This is the cheap first pass of ``primary_diam="auto"`` in
    [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases] (a resolved
    primary, e.g. ι Peg's at 1.05 mas, makes a point-star model fit badly
    and biases the seeds). Each night's static binary is re-refined from
    its ``positions`` entry at each diameter of a grid, on the
    scale-marginalized score ``m`` of
    [`marginal_loglike`][virgil.epochs.marginal_loglike], and the nights'
    scores are summed: the diameter is shared by every night. Flagged
    nights are left out of the sum (their positions may be wrong peaks)
    unless every night is flagged.

    **The rule.** The diameters tried are 0 (a point primary) and the
    fractions 0.03 to 1.3 of the finest fringe spacing λ/B_max. The best
    diameter is the maximum of the summed score, refined by a parabola in
    the diameter through its two neighbours, whose curvature gives
    ``sigma``. The data **resolve** the primary when the best diameter is
    positive and its summed score exceeds the point primary's by at least
    ``resolve_nats`` (default 5 nats: one extra parameter, Δχ² ≈ 10, about
    3σ). Because ``m`` integrates out each block's error scale, a night
    whose errors are underestimated does not look more decisive than it
    is. The rule uses no published value.

    Parameters
    ----------
    data : Epochs or sequence of OIData
        The datasets, in the order of ``positions``.
    positions : EpochPositions
        The per-night positions (from
        [`epoch_positions`][virgil.epochs.epoch_positions]); each night's
        best position and flux start its refinements.
    resolve_nats : float, optional
        The gain (nats) at which the primary counts as resolved.
    diam_grid : array-like, optional
        Positive diameters (mas) to try besides 0; by default the
        fractions of λ/B_max above.
    s_max : float, optional
        As in ``marginal_loglike``.

    Returns
    -------
    PrimaryDiameter

    Notes
    -----
    The static snapshot ignores motion within a night, as the seeds do.
    A V² contrast loss that is the same on every baseline (a calibration
    error) is partly degenerate with a small disc here; the orbit fit's
    per-night amplitude (``vis_amp``) absorbs it, and the diameter is
    refitted there. The returned ``positions`` keep the original night
    flags, ``gap`` and peak catalogues (which belong to the point-primary
    surface); only the best position, flux and covariance are refitted.
    """
    sets = list(data.data) if isinstance(data, Epochs) else list(data)
    if len(sets) != len(positions.names):
        raise ValueError(
            f"{len(sets)} datasets but positions for "
            f"{len(positions.names)}: they must be the same nights."
        )
    use = ~positions.flagged
    if not use.any():
        use = onp.ones(len(sets), bool)
    with run_in("float64"):
        d64 = [cast_tree(d, "float64") for d in sets]
        surfaces = [_Surface(d, s_max) for d in d64]
        resolution = [float(fringe_scales(d)[0]) for d in sets]
        finest = min(resolution)
        grid = onp.concatenate(
            [
                [0.0],
                finest * onp.asarray(_DIAM_FRACTIONS)
                if diam_grid is None
                else onp.sort(onp.asarray(diam_grid, float)),
            ]
        )
        if (grid[1:] <= 0).any():
            raise ValueError("diam_grid must hold positive diameters.")
        start = [
            onp.array([positions.dra[i], positions.ddec[i], positions.flux[i]])
            for i in range(len(sets))
        ]
        dof = np.asarray(1.0)

        def refit(i, x0, diam):
            """Refine night ``i`` from ``x0`` at ``diam``: (x, score)."""
            window = _REFINE_WINDOW * resolution[i]
            x0 = onp.clip(
                x0,
                [x0[0] - window, x0[1] - window, 1e-4],
                [x0[0] + window, x0[1] + window, 1.0],
            )
            bounds = [
                (x0[0] - window, x0[0] + window),
                (x0[1] - window, x0[1] + window),
                (1e-4, 1.0),
            ]
            dm = None if diam <= 0 else np.asarray(float(diam))
            x, value = _refine_marginal(
                x0, dof, d64[i], surfaces[i], bounds, dm
            )
            if not onp.isfinite(value):  # no improvement: score the start
                value = float(
                    _negative_marginal(
                        np.asarray(x0), dof, d64[i], surfaces[i], dm
                    )
                )
            return x, -value

        def profile_at(diam, begin):
            """Summed score over the used nights at ``diam``, and the
            fitted positions (None for the nights left out)."""
            total, xs = 0.0, list(begin)
            for i in onp.flatnonzero(use):
                xs[i], score = refit(i, begin[i], diam)
                total += score
            return total, xs

        profile, xs, fits = [], start, []
        for diam in grid:  # ascending: each starts from the last fit
            total, xs = profile_at(diam, xs)
            profile.append(total)
            fits.append(xs)
        profile = onp.asarray(profile)
        best = int(onp.argmax(profile))
        diam, sigma = float(grid[best]), float("nan")
        gain = float(profile[best] - profile[0])
        if 0 < best < len(grid) - 1:
            # A parabola through the best grid point and its neighbours
            # locates the maximum between grid points.
            a, b, _ = onp.polyfit(
                grid[best - 1 : best + 2], profile[best - 1 : best + 2], 2
            )
            vertex = -b / (2 * a) if a < 0 else float("nan")
            if grid[best - 1] <= vertex <= grid[best + 1]:
                total, _ = profile_at(vertex, fits[best])
                if total >= profile[best]:
                    diam, gain = float(vertex), float(total - profile[0])
            # The uncertainty is the curvature of the profile at the
            # maximum, from steps of 15% of the diameter.
            h = 0.15 * diam
            up, _ = profile_at(diam + h, fits[best])
            down, _ = profile_at(diam - h, fits[best])
            peak = gain + profile[0]
            curve = 2 * peak - up - down
            if curve > 0:
                sigma = float(h / onp.sqrt(curve))
        resolved = bool(diam > 0 and gain >= resolve_nats)
        chosen = diam if resolved else 0.0
        reseed = resolved or positions.primary_diam is not None
        if reseed:
            dm = None if chosen <= 0 else np.asarray(float(chosen))
            xs_all, covs = [], []
            for i in range(len(sets)):
                x, _ = refit(i, start[i], chosen)
                xs_all.append(x)
                full = onp.asarray(
                    _negative_marginal_hessian(
                        np.asarray(x), dof, d64[i], surfaces[i], dm
                    ),
                    dtype=float,
                )
                cov = positions.cov[i]
                if onp.all(onp.isfinite(full)):
                    inv = onp.linalg.pinv(full)[:2, :2]
                    if onp.all(onp.linalg.eigvalsh(inv) > 0):
                        cov = inv
                covs.append(cov)
            xs_all = onp.asarray(xs_all)
            seeded = dataclasses.replace(
                positions,
                dra=xs_all[:, 0],
                ddec=xs_all[:, 1],
                flux=xs_all[:, 2],
                cov=onp.asarray(covs),
                primary_diam=chosen or None,
            )
        else:
            seeded = positions
    return PrimaryDiameter(
        diam=diam,
        sigma=sigma,
        delta_nats=float(gain),
        resolved=resolved,
        grid=grid,
        profile=profile,
        positions=seeded,
    )


@dataclasses.dataclass(frozen=True)
class OrbitStart:
    """A start for a multi-epoch orbit fit, from
    [`start_from_positions`][virgil.epochs.start_from_positions].

    Attributes
    ----------
    positions : EpochPositions
        The per-dataset positions that seeded the orbits.
    candidates : RankedOrbits
        The starting orbits, ranked by the visibilities.
    fits : tuple of FitResult
        The fits to the visibilities from the distinct best candidates,
        lowest loss first, and those with a non-finite loss last. Each
        ``info`` holds the ``candidate`` it started from (its rank in
        ``candidates``); a fit that did not converge, or stopped at its
        ``time_limit``, says so in ``info["converged"]`` and
        ``info["stop"]``.
    data : Epochs
        The data.
    failed : tuple of (int, str)
        The candidates whose refinement fit raised an error, with the
        error.
    seeded : tuple of str
        The datasets whose positions seeded the starting orbits (a
        ``gap_marginal`` above ``min_gap``).
    ambiguous : tuple of str
        The other datasets: their best peak has a rival within
        ``min_gap``, so their positions did not seed the orbits. They
        still enter the ranking and the fits through their visibilities;
        their peaks are in ``positions.peaks``.
    """

    positions: EpochPositions
    candidates: RankedOrbits
    fits: tuple
    data: Epochs
    failed: tuple = ()
    seeded: tuple = ()
    ambiguous: tuple = ()

    @property
    def best(self):
        """The fit with the lowest loss."""
        return self.fits[0]

    def modes(self, *, max_delta_loss=10.0, same_mode_chi2=1.0):
        """The distinct fits, lowest loss first.

        Two fits are one mode when their predictions differ by less than
        ``same_mode_chi2`` in χ² (the sum of squared differences of their
        whitened residuals, with the quoted errors). Fits whose loss is
        more than ``max_delta_loss`` above the best's are dropped: a
        difference Δ in loss is a factor of about e^Δ in posterior
        density. Fits with a non-finite loss are left out.
        """
        finite = [r for r in self.fits if math.isfinite(r.info["loss"])]
        if not finite:
            raise ValueError("No refinement fit has a finite loss.")
        floor = finite[0].info["loss"]
        kept, residuals = [], []
        for result in finite:
            if result.info["loss"] - floor > max_delta_loss:
                break
            models = result.model
            if not isinstance(models, (list, tuple)):
                models = [models] * len(self.data)
            r = onp.concatenate(
                [
                    onp.asarray(whitened_residuals(m, d))
                    for m, d in zip(models, self.data.data)
                ]
            )
            if all(onp.sum((r - q) ** 2) > same_mode_chi2 for q in residuals):
                kept.append(result)
                residuals.append(r)
        return kept

    def chain_values(self, n_chains, **options):
        """One start (fitted values) per chain, cycling through the
        distinct modes (see :meth:`modes`, which takes ``options``).

        Pass them to
        [`chain_init_params`][virgil.likelihood.chain_init_params] to
        start each chain of numpyro's ``MCMC`` in its own mode.
        """
        modes = self.modes(**options)
        return [dict(modes[k % len(modes)].values) for k in range(n_chains)]


def start_from_positions(
    model,
    priors,
    data,
    start_values,
    *,
    grid,
    periods,
    t_ref,
    scales=None,
    noise=None,
    dof=1.0,
    s_max=None,
    eccs=None,
    n_phase=36,
    n_candidates=200,
    n_refine=4,
    min_gap=5.0,
    refine_positions=True,
    n_peaks=5,
    batch_size=None,
    primary_diam=None,
    skip_flagged=True,
    flag_options=None,
    **fit_options,
):
    """Start an orbit fit to multi-epoch visibilities from a positions fit.

    The likelihood of visibilities is multimodal on the scale of the
    resolution λ/B (fringe aliases), and a sampler started from default
    values can stick in an alias. This builds a start in four steps:

    1. **Positions**: the companion's position in each dataset, the
       best of its refined grid peaks
       ([`epoch_positions`][virgil.epochs.epoch_positions]).
    2. **Starting orbits**: a Thiele–Innes grid over period, eccentricity
       and time of periastron on the decisive datasets' positions
       ([`starting_orbits`][virgil.orbits.starting_orbits]). An
       ambiguous dataset does not seed them; its other peaks are kept in
       ``positions.peaks``, but only its visibilities judge between
       them here.
    3. **Ranking** of those orbits by the likelihood of all the
       visibilities, with each orbit mapped to the model's parameters by
       ``start_values`` and the flux at the median of the positions fits.
    4. **Refinement** with [`fit`][virgil.fitting.fit] of the full model
       (noise terms included) from the ``n_refine`` best *distinct*
       orbits (as for [`chain_starts`][virgil.epochs.chain_starts]).

    Only the start comes from positions: nothing here enters the
    posterior. Sample from the best fit with numpyro's ``init_to_value``,
    or start one chain in each mode with
    [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values] and
    [`chain_init_params`][virgil.likelihood.chain_init_params].

    Parameters
    ----------
    model : callable
        The scene function, called with the parameters in ``priors`` as
        keyword arguments and returning a time-dependent scene, as for
        [`Epochs.model_fn`][virgil.epochs.Epochs.model_fn].
    priors : dict
        The priors of the fit, as for ``fit``.
    data : Epochs
        The data.
    start_values : callable
        ``start_values(orbit, flux)``: the starting values (a dict with
        a value for every key of ``priors``) for a
        [`KeplerOrbit`][virgil.orbits.KeplerOrbit] (with ``t_ref``) and
        a companion/primary flux. It may use Python floats.
    grid : dict
        The per-dataset grid of positions and fluxes (see
        ``epoch_positions``).
    periods : array-like
        Trial periods (days) for ``starting_orbits``. Build them with
        [`period_grid`][virgil.orbits.period_grid] over the times of the
        datasets: a ``UserWarning`` says when neighbouring periods would
        drift apart by more than 1/9 of a cycle over the baseline of the
        datasets that seed the orbits, so that the grid can miss the
        true period.
    t_ref : float
        The reference time (MJD) of the starting orbits: the ``t_ref`` of
        your model.
    scales : {"quoted", "marginal"}, optional
        How step 3 ranks the candidates (see ``rank_orbits``):
        ``"quoted"`` on the quoted errors, where the datasets with the
        most underestimated errors dominate, or ``"marginal"`` with each
        dataset's per-block error scale integrated out, so that each
        dataset counts by its own fitted scale ŝ. The default is
        ``"quoted"`` for now, with a ``FutureWarning`` when ``scales`` is
        not given; it will become ``"marginal"``.
    noise : dict or list of dict, optional
        Noise terms of the refinement fits, as for ``fit`` (e.g. from
        [`Epochs.noise`][virgil.epochs.Epochs.noise]). The ranking does
        not use them.
    dof, s_max : optional
        Passed to ``epoch_positions`` and, with ``scales="marginal"``, to
        the ranking.
    eccs, n_phase : optional
        Passed to ``starting_orbits``.
    n_candidates : int, optional
        Starting orbits to rank (default 200).
    n_refine : int, optional
        Distinct orbits to refine (default 4).
    min_gap : float, optional
        Seed orbits only with datasets whose positions are decisive: a
        ``gap_marginal`` above this (see ``EpochPositions``). This
        compares the scale-marginalized gap, so a night whose errors are
        underestimated no longer passes it with a gap inflated by s².
        The datasets left out are named in ``OrbitStart.ambiguous``, with
        a ``UserWarning``.
    refine_positions : bool, optional
        Refine each grid peak with a fit (see ``epoch_positions``).
    n_peaks : int, optional
        Peaks refined per dataset (see ``epoch_positions``); the cost of
        step 1 grows linearly with it.
    batch_size : int, optional
        Orbits ranked together (see ``rank_orbits``).
    primary_diam : float, optional
        The primary's uniform-disc diameter (mas) for the per-dataset
        positions (see ``epoch_positions``); ``None`` is a point primary.
    skip_flagged : bool, optional
        Seed orbits only with datasets that ``epoch_positions`` did not
        flag (default True); flagged datasets join ``OrbitStart.ambiguous``
        and still enter the ranking and the fits.
    flag_options : dict, optional
        The ``flag_*`` keywords of ``epoch_positions``. ``flag_min_gap``
        defaults to ``min_gap``, so that decisive and unflagged agree.
    **fit_options
        Passed to the refinement fits (e.g. ``method``, ``max_steps``,
        ``time_limit`` in seconds per fit, ``progress``). A fit that raises
        an error is recorded in ``OrbitStart.failed``, with a
        ``RuntimeWarning``, and the others go on; one that stops at its
        ``time_limit`` is kept, unconverged. A ``TypeError`` or
        ``ValueError`` (such as a misspelt option) does not depend on the
        start, and is raised at once.

    Returns
    -------
    OrbitStart

    Raises
    ------
    RuntimeError
        If every refinement fit raised an error (chained to the last).
    """
    scales = _check_scales(scales, "start_from_positions")
    _check_n_peaks(n_peaks)
    positions = epoch_positions(
        data,
        grid,
        refine=refine_positions,
        n_peaks=n_peaks,
        dof=dof,
        s_max=s_max,
        primary_diam=primary_diam,
        **{"flag_min_gap": min_gap, **(flag_options or {})},
    )
    decisive = positions.decisive(min_gap)
    if skip_flagged:
        decisive = decisive & ~positions.flagged
    seeded = tuple(n for n, d in zip(positions.names, decisive) if d)
    ambiguous = tuple(n for n, d in zip(positions.names, decisive) if not d)
    if ambiguous and seeded:
        warnings.warn(
            f"The positions of {', '.join(ambiguous)} are ambiguous (a "
            f"marginal gap of at most min_gap={min_gap}) or flagged "
            f"({positions.flag_codes()}) and do not seed the orbits; their "
            "visibilities still enter the ranking and the fits. See "
            "OrbitStart.ambiguous and positions.peaks.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    with run_in("float64"):
        seeds = positions.positions(
            t_ref=t_ref, min_gap=min_gap, skip_flagged=skip_flagged
        )
    if len(seeds.dt) < 2:
        raise ValueError(
            "At least two decisive, unflagged datasets are needed to seed "
            f"orbits; the marginal gaps are {positions.gap_marginal.round(1)} "
            f"and the flagged datasets are {positions.flag_codes()}. Lower "
            "min_gap, refine the grid, or (if flags are the cause) pass "
            "skip_flagged=False or loosen flag_options."
        )
    _warn_if_coarse(periods, seeds.dt)
    with run_in("float64"):
        candidates = starting_orbits(
            seeds, periods, eccs=eccs, n_phase=n_phase, n_best=n_candidates
        )
    flux = float(onp.median(positions.flux[decisive]))
    orbits = [orbit for orbit, _ in candidates]
    values = [dict(start_values(orbit, flux)) for orbit in orbits]
    missing = set(priors) - set(values[0])
    if missing:
        raise KeyError(
            f"start_values returned no value for {sorted(missing)}; it "
            "must give one for every key of priors."
        )
    stacked = _stack(
        [{k: v[k] for k in priors} for v in values], "starting values"
    )
    loglike = _loglikes(
        lambda v: model(**v),
        stacked,
        data,
        None,
        batch_size,
        "float64",
        scales=scales,
        dof=dof,
        s_max=s_max,
    )
    ranked = _ranked(orbits, loglike, data)
    chosen = _distinct(ranked, n_refine, 0.5 * ranked.resolution_mas)
    if not chosen:
        raise ValueError("No starting orbit has a finite log likelihood.")
    fits, failed, error = [], [], None
    # One function for every refinement, so ``fit`` compiles once: a fresh
    # closure per start would recompile for each.
    model_fn = data.model_fn(model)
    for k in chosen:
        # One bad start (a NaN, a solver error) must not lose the others.
        # TypeError and ValueError are the caller's (a misspelt option,
        # Adam with a time limit): they do not depend on the start, so
        # they propagate.
        try:
            result = fit(
                model_fn,
                priors,
                data.data,
                noise=noise,
                init=values[ranked.order[k]],
                **fit_options,
            )
        except (TypeError, ValueError):
            raise
        except Exception as caught:
            error = caught
            reason = f"{type(error).__name__}: {error}"
            failed.append((int(k), reason))
            warnings.warn(
                f"The refinement fit from candidate {k} failed ({reason}); "
                "the others go on.",
                RuntimeWarning,
                stacklevel=external_stacklevel(),
            )
            continue
        result.info["candidate"] = int(k)
        fits.append(result)
    if not fits:
        raise RuntimeError(
            f"Every refinement fit failed: {failed}."
        ) from error
    fits.sort(key=_loss_order)
    return OrbitStart(
        positions=positions,
        candidates=ranked,
        fits=tuple(fits),
        data=data,
        failed=tuple(failed),
        seeded=seeded,
        ambiguous=ambiguous,
    )


def _loss_order(result):
    """Sort key: lowest loss first, non-finite losses last."""
    loss = result.info["loss"]
    return (0, loss) if math.isfinite(loss) else (1, 0.0)


def _warn_if_coarse(periods, times):
    """Warn when trial periods drift apart by more than 1/k of a cycle
    over the baseline of ``times`` (see ``period_grid``)."""
    times = onp.asarray(times, float)
    baseline = float(onp.ptp(times)) if times.size else 0.0
    drift = _phase_drift(periods, baseline)
    if drift > 1.0 / _PHASE_COHERENCE:
        warnings.warn(
            f"Neighbouring trial periods drift apart by up to {drift:.2g} "
            f"cycles over the {baseline:.0f}-day baseline of the seeding "
            f"datasets (more than 1/{_PHASE_COHERENCE:g}), so the starting "
            "orbits can miss the true period; use "
            "virgil.orbits.period_grid(times, p_min, p_max).",
            UserWarning,
            stacklevel=external_stacklevel(),
        )

"""Radio-interferometric (u, v, complex visibility) data and calibration.

[`ComplexVisData`][virgil.radio.ComplexVisData] holds calibrated complex
visibilities at arbitrary (u, v) points, read from a CASA measurement set or
built from arrays. Its [`term`][virgil.radio.ComplexVisData.term] is a
likelihood term for [`fit`][virgil.fitting.fit] (and `numpyro_model`) whose
model is any [`SourceModel`][virgil.models.SourceModel], multiplied by
per-execution-block nuisances ([`CalibrationNuisance`][virgil.radio.CalibrationNuisance]):
a flux scale, relative gains, phase-centre offsets, error scales per
spectral window and a spectral index.

Conventions (pinned by tests, see the Radio interferometry guide): the
visibility is ``V(u, v) = Σ I(x, y) exp(-2πi (u x + v y))`` with ``x`` towards
East and ``y`` towards North and ``u``, ``v`` in wavelengths, as everywhere in
virgil; a weight is ``1 / σ²`` for each of the real and imaginary parts.
"""

from __future__ import annotations

import dataclasses
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from ._geometry import offset_phase, pixel_offsets
from ._utils import mas2rad, rad2mas
from .models import SourceModel

__all__ = [
    "CalibrationNuisance",
    "ComplexVisData",
    "dartboard_folds",
    "primary_beam",
]

C_LIGHT = 299_792_458.0  # m/s

# How the CASA ``DATA`` column maps to virgil's convention, as decided by
# ``tests/test_radio.py::test_read_ms_convention`` (a point source predicted
# by CASA's simulator): the data are V* when ``MS_CONJUGATE`` is true, and
# ``u = MS_U_SIGN * UVW[0] * ν / c``.
MS_CONJUGATE = True
MS_U_SIGN = 1.0

_CROSS = {(9, 12), (5, 8)}  # XX, YY and RR, LL


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


def _relabel(ids):
    """Sorted unique original ids and the 0..n-1 label of every entry."""
    originals, labels = onp.unique(onp.asarray(ids), return_inverse=True)
    return tuple(int(i) for i in originals), labels.astype(onp.int32)


class ComplexVisData(eqx.Module):
    """Complex visibilities at (u, v) points, with calibration labels.

    Build one with [`from_arrays`][virgil.radio.ComplexVisData.from_arrays]
    or [`read_ms`][virgil.radio.ComplexVisData.read_ms].

    Attributes
    ----------
    u, v : (N,) float64
        Spatial frequencies in wavelengths (``UVW * ν / c`` per channel).
    vis : (N,) complex
        Calibrated visibilities (Jy).
    weight : (N,) float
        ``1 / σ²`` of each of the real and imaginary parts (Jy⁻²).
    freq : (N,) float64
        Channel frequency (Hz).
    ddid, eb : (N,) int32
        Data-description and execution-block labels, relabelled 0..n-1 in
        sorted order of the originals ``ddid_ids`` and ``eb_ids``.
    ddid_eb : tuple of int
        The (relabelled) execution block of each ddid.
    time, ant1, ant2, chan : optional arrays
        MJD seconds, antenna indices and channel index, when known.
    """

    u: jax.Array
    v: jax.Array
    vis: jax.Array
    weight: jax.Array
    freq: jax.Array
    ddid: jax.Array
    eb: jax.Array
    ddid_ids: tuple = eqx.field(static=True)
    eb_ids: tuple = eqx.field(static=True)
    ddid_eb: tuple = eqx.field(static=True)
    time: jax.Array | None = None
    ant1: jax.Array | None = None
    ant2: jax.Array | None = None
    chan: jax.Array | None = None

    # -- construction -------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        u,
        v,
        vis,
        weight,
        freq,
        ddid=0,
        eb=0,
        *,
        time=None,
        ant1=None,
        ant2=None,
        chan=None,
    ):
        """Build from arrays; scalars broadcast over the points.

        ``ddid`` and ``eb`` are the original labels, relabelled 0..n-1 in
        sorted order. Each ddid must belong to one execution block.
        """
        u = onp.atleast_1d(onp.asarray(u, dtype=onp.float64))
        n = u.size

        def full(x, dtype):
            x = onp.asarray(x, dtype=dtype)
            if x.ndim == 0:
                x = onp.full(n, x)
            return x

        v = full(v, onp.float64)
        vis = full(vis, onp.complex128)
        weight = full(weight, onp.float64)
        freq = full(freq, onp.float64)
        ddid_orig = full(ddid, onp.int64)
        eb_orig = full(eb, onp.int64)
        optional = {
            "time": time,
            "ant1": ant1,
            "ant2": ant2,
            "chan": chan,
        }
        for name, x in list(optional.items()):
            if x is not None:
                optional[name] = full(x, None)
        for name, x in dict(
            u=u,
            v=v,
            vis=vis,
            weight=weight,
            freq=freq,
            ddid=ddid_orig,
            eb=eb_orig,
        ).items():
            if x.shape != (n,):
                raise ValueError(f"{name} has shape {x.shape}, not ({n},).")
        for name, x in optional.items():
            if x is not None and x.shape != (n,):
                raise ValueError(f"{name} has shape {x.shape}, not ({n},).")
        if not (onp.all(onp.isfinite(u)) and onp.all(onp.isfinite(v))):
            raise ValueError("u and v must be finite.")
        if not onp.all(onp.isfinite(vis)):
            raise ValueError("vis must be finite.")
        if not (onp.all(onp.isfinite(weight)) and onp.all(weight > 0)):
            raise ValueError("weight must be finite and positive.")
        if not (onp.all(onp.isfinite(freq)) and onp.all(freq > 0)):
            raise ValueError("freq must be finite and positive.")
        ddid_ids, ddid_l = _relabel(ddid_orig)
        eb_ids, eb_l = _relabel(eb_orig)
        ddid_eb = []
        for d in range(len(ddid_ids)):
            ebs = onp.unique(eb_l[ddid_l == d])
            if ebs.size != 1:
                raise ValueError(
                    f"ddid {ddid_ids[d]} spans several execution blocks."
                )
            ddid_eb.append(int(ebs[0]))
        return cls(
            u=np.asarray(u),
            v=np.asarray(v),
            vis=np.asarray(vis),
            weight=np.asarray(weight),
            freq=np.asarray(freq),
            ddid=np.asarray(ddid_l),
            eb=np.asarray(eb_l),
            ddid_ids=ddid_ids,
            eb_ids=eb_ids,
            ddid_eb=tuple(ddid_eb),
            **{
                k: None if x is None else np.asarray(x)
                for k, x in optional.items()
            },
        )

    @classmethod
    def read_ms(
        cls,
        path,
        ddids=None,
        stride=1,
        datacolumn="DATA",
        conjugate=None,
        u_sign=None,
    ):
        """Read a CASA measurement set (needs the ``radio`` extra).

        Cross-correlations only; one sample per (row, channel), kept if the
        row and both parallel-hand polarizations (XX, YY or RR, LL) are
        unflagged and both weights positive. The two polarizations are
        averaged with their weights (Stokes I for equal weights) and the
        weight is their sum. ``WEIGHT_SPECTRUM`` is used when present and
        non-empty, else ``WEIGHT``. ``u = UVW[0] ν / c``, ``v = UVW[1] ν / c``
        per channel. One ddid is read at a time, so memory is that of the
        largest ddid at the given ``stride``.

        Parameters
        ----------
        path : str
            The measurement set.
        ddids : sequence of int, optional
            ``DATA_DESC_ID`` values to read (default all).
        stride : int, optional
            Keep every ``stride``-th row of each ddid.
        datacolumn : str, optional
            ``"DATA"`` or ``"CORRECTED_DATA"``.
        conjugate, u_sign : optional
            Override the module's pinned convention (``MS_CONJUGATE``,
            ``MS_U_SIGN``) for comparisons.
        """
        try:
            from casatools import table as casa_table
        except ImportError as err:  # pragma: no cover - needs no casatools
            raise ImportError(
                "read_ms needs casatools: pip install 'virgil-astro[radio]'."
            ) from err
        conjugate = MS_CONJUGATE if conjugate is None else conjugate
        u_sign = MS_U_SIGN if u_sign is None else u_sign
        stride = int(stride)
        if stride < 1:
            raise ValueError("stride must be a positive integer.")
        path = str(path)

        tb = casa_table()
        try:
            tb.open(f"{path}/DATA_DESCRIPTION")
            spw_of = tb.getcol("SPECTRAL_WINDOW_ID")
            pol_of = tb.getcol("POLARIZATION_ID")
            tb.close()
            tb.open(f"{path}/SPECTRAL_WINDOW")
            chan_freqs = [
                onp.asarray(tb.getcell("CHAN_FREQ", int(s)), dtype=onp.float64)
                for s in range(tb.nrows())
            ]
            tb.close()
            tb.open(f"{path}/POLARIZATION")
            corr_types = [
                tuple(int(c) for c in tb.getcell("CORR_TYPE", int(p)))
                for p in range(tb.nrows())
            ]
            tb.close()
            wanted = range(len(spw_of)) if ddids is None else list(ddids)
            parts = []
            tb.open(path)
            for d in wanted:
                d = int(d)
                if not 0 <= d < len(spw_of):
                    raise ValueError(
                        f"ddid {d} is not in the measurement set."
                    )
                corr = corr_types[int(pol_of[d])]
                pair = _parallel_hands(corr)
                sub = tb.query(f"DATA_DESC_ID=={d} && ANTENNA1!=ANTENNA2")
                try:
                    if sub.nrows() == 0:
                        continue
                    part = _read_ddid(
                        sub,
                        pair,
                        chan_freqs[int(spw_of[d])],
                        stride,
                        datacolumn,
                        conjugate,
                        u_sign,
                    )
                finally:
                    sub.close()
                if part is None:
                    warnings.warn(f"ddid {d} has no unflagged samples.")
                    continue
                part["ddid"] = onp.full(part["u"].size, d, dtype=onp.int64)
                parts.append(part)
            tb.close()
        finally:
            tb.done()
        if not parts:
            raise ValueError("no unflagged cross-correlations were read.")
        cat = {k: onp.concatenate([p[k] for p in parts]) for k in parts[0]}
        return cls.from_arrays(**cat)

    # -- selection ----------------------------------------------------

    def __len__(self):
        return int(self.u.shape[0])

    def subset(self, mask):
        """The points where boolean ``mask`` is true, labels unchanged.

        Labels are kept as they are (not relabelled), so a
        [`CalibrationNuisance`][virgil.radio.CalibrationNuisance] built for
        the full data still fits.
        """
        mask = onp.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError(
                f"mask has shape {mask.shape}, not ({len(self)},)."
            )
        idx = onp.flatnonzero(mask)
        leaves = {}
        for f in dataclasses.fields(self):
            x = getattr(self, f.name)
            leaves[f.name] = (
                x
                if f.name in ("ddid_ids", "eb_ids", "ddid_eb") or x is None
                else x[idx]
            )
        return type(self)(**leaves)

    def select(self, ddids=None, ebs=None):
        """Points of the given ddids and/or execution blocks.

        ``ddids`` and ``ebs`` are the original ids (``ddid_ids``,
        ``eb_ids``). Labels are kept; see
        [`subset`][virgil.radio.ComplexVisData.subset].
        """
        keep = onp.ones(len(self), dtype=bool)
        if ddids is not None:
            labels = [
                self.ddid_ids.index(int(d)) for d in onp.atleast_1d(ddids)
            ]
            keep &= onp.isin(onp.asarray(self.ddid), labels)
        if ebs is not None:
            labels = [self.eb_ids.index(int(e)) for e in onp.atleast_1d(ebs)]
            keep &= onp.isin(onp.asarray(self.eb), labels)
        return self.subset(keep)

    # -- model --------------------------------------------------------

    def model(self, scene, values, nuisance):
        """The complex model ``M_k`` at every point (all at once).

        ``M = F g_e (ν/ν₀)^α exp(-2πi(δα_e u + δδ_e v)) T(u, v)
        V_scene(u, v)``; see
        [`CalibrationNuisance`][virgil.radio.CalibrationNuisance]. ``scene``
        is a [`SourceModel`][virgil.models.SourceModel] (normalized to unit
        flux; ``cal.flux`` sets the scale), evaluated at wavelength
        ``c / ν`` of each point so that chromatic scenes work.
        """
        return _model_points(
            scene,
            nuisance._unpack(values),
            nuisance,
            self.u,
            self.v,
            self.freq,
            self.ddid,
            self.eb,
        )

    def term(self, scene, nuisance, mask=None, chunk_size=10_000_000):
        """A likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``.

        Parameters
        ----------
        scene : callable
            Maps the fitted values (a dict by site name) to a
            [`SourceModel`][virgil.models.SourceModel], as ``PositionData.term``
            takes ``orbit``.
        nuisance : CalibrationNuisance
            The calibration model; its sites are keys of the fit's priors.
        mask : (N,) bool, optional
            Points to use (default all); masked points change neither χ²
            nor the normalization.
        chunk_size : int, optional
            Points are padded with zero weight to a multiple of this and
            evaluated with ``jax.lax.map`` over chunks, bounding memory
            (the transform is never ``vmap``-ed).
        """
        return _RadioTerm(self, scene, nuisance, mask, chunk_size)

    def chi2_by_ddid(self, scene, values, nuisance, mask=None):
        """χ² of the data with the σ scales applied, per ddid.

        Returns ``(chi2, n)``: float64 arrays of length ``n_ddid``, ``n`` the
        number of unmasked points of each ddid (χ² sums two terms per point).
        """
        term = self.term(scene, nuisance, mask=mask)
        r2 = term._r2(values)
        mask_ = (
            onp.ones(len(self), dtype=bool)
            if mask is None
            else onp.asarray(mask, dtype=bool)
        )
        n_ddid = len(self.ddid_ids)
        chi2 = jax.ops.segment_sum(
            np.asarray(r2, dtype=np.float64), self.ddid, n_ddid
        )
        n = onp.bincount(onp.asarray(self.ddid)[mask_], minlength=n_ddid)
        return onp.asarray(chi2), n


def _parallel_hands(corr):
    """Indices of the parallel-hand pair in ``CORR_TYPE`` (XX, YY or RR, LL)."""
    for a, b in sorted(_CROSS):
        if a in corr and b in corr:
            return corr.index(a), corr.index(b)
    raise ValueError(
        f"no XX/YY or RR/LL polarization pair in CORR_TYPE {corr}."
    )


def _read_ddid(sub, pair, chan_freq, stride, column, conjugate, u_sign):
    """The samples of one ddid's cross-correlation query, or None."""
    data = sub.getcol(column, 0, -1, stride)  # (ncorr, nchan, nrow)
    flag = sub.getcol("FLAG", 0, -1, stride)
    flag_row = sub.getcol("FLAG_ROW", 0, -1, stride)
    uvw = sub.getcol("UVW", 0, -1, stride)
    time = sub.getcol("TIME", 0, -1, stride)
    ant1 = sub.getcol("ANTENNA1", 0, -1, stride)
    ant2 = sub.getcol("ANTENNA2", 0, -1, stride)
    obs = sub.getcol("OBSERVATION_ID", 0, -1, stride)
    weight = None
    if "WEIGHT_SPECTRUM" in sub.colnames() and sub.iscelldefined(
        "WEIGHT_SPECTRUM", 0
    ):
        ws = sub.getcol("WEIGHT_SPECTRUM", 0, -1, stride)
        if ws.size:
            weight = ws
    if weight is None:
        w = sub.getcol("WEIGHT", 0, -1, stride)  # (ncorr, nrow)
        weight = onp.broadcast_to(w[:, None, :], data.shape)
    i, j = pair
    w1, w2 = weight[i].astype(onp.float64), weight[j].astype(onp.float64)
    ok = (
        ~flag[i]
        & ~flag[j]
        & ~flag_row[None, :].astype(bool)
        & (w1 > 0)
        & (w2 > 0)
    )  # (nchan, nrow)
    if not ok.any():
        return None
    wsum = w1 + w2
    vis = (w1 * data[i] + w2 * data[j]) / onp.where(wsum > 0, wsum, 1.0)
    if conjugate:
        vis = onp.conj(vis)
    ch, rw = onp.nonzero(ok)
    freq = chan_freq[ch]
    lam_inv = freq / C_LIGHT
    return dict(
        u=u_sign * uvw[0, rw] * lam_inv,
        v=uvw[1, rw] * lam_inv,
        vis=vis[ch, rw],
        weight=wsum[ch, rw],
        freq=freq,
        eb=obs[rw].astype(onp.int64),
        time=time[rw],
        ant1=ant1[rw],
        ant2=ant2[rw],
        chan=ch.astype(onp.int16),
    )


# ----------------------------------------------------------------------
# Calibration nuisances
# ----------------------------------------------------------------------


class CalibrationNuisance:
    """Per-execution-block calibration nuisances of a radio dataset.

    Static configuration: which nuisances are fitted and their site names.
    The sites (``prefix="cal"``) are keys of the fit's ``priors``; a function
    model must accept them:

    ===============  ==============  ==========================  ============
    Site             Shape           Default prior               Not fitted
    ===============  ==============  ==========================  ============
    ``cal.flux``     ``()``          ``LogUniform(1e-4, 1e2)``   always fitted
    ``cal.gain``     ``(n_eb-1,)``   ``LogUniform(0.5, 2)``      1
    ``cal.offset``   ``(n_eb-1, 2)`` ``Uniform(-100, 100)`` mas  0
    ``cal.sigma``    ``(n_ddid,)``   ``LogUniform(0.1, 10)``     1
    ``cal.alpha``    ``()``          ``Uniform(0, 5)``           0
    ===============  ==============  ==========================  ============

    The reference execution block (``ref_eb``, an index) has gain 1 and zero
    offset. The model at a point is
    ``F g_e (ν/ν₀)^α exp(-2πi(δα_e u + δδ_e v)) T(u, v) V_scene(u, v)``,
    the offset ramp with the sign of a component's ``dra``, ``ddec``; ``T`` is
    the optional Gaussian taper ``exp(-(π θ q)² / (4 ln 2))`` (θ the FWHM in
    radians, ``q = √(u² + v²)``), a Gaussian convolution of the image. The
    error scale ``s_d`` of each ddid multiplies the errors.

    Parameters
    ----------
    data : ComplexVisData
    ref_eb : int
    gains, offsets, sigmas, alpha : bool
        Which nuisances are fitted (the flux always is). Gains and offsets
        need more than one execution block.
    nu0_hz : float
        Reference frequency of the spectral index.
    taper_fwhm_mas : float, optional
        FWHM of a Gaussian taper (a convolution), in mas.
    prefix : str
        Site-name prefix.
    """

    def __init__(
        self,
        data,
        ref_eb=0,
        gains=True,
        offsets=True,
        sigmas=True,
        alpha=True,
        nu0_hz=239e9,
        taper_fwhm_mas=None,
        prefix="cal",
    ):
        self.n_eb = len(data.eb_ids)
        self.n_ddid = len(data.ddid_ids)
        if not 0 <= int(ref_eb) < self.n_eb:
            raise ValueError(f"ref_eb must be in 0..{self.n_eb - 1}.")
        self.ref_eb = int(ref_eb)
        self.gains = bool(gains) and self.n_eb > 1
        self.offsets = bool(offsets) and self.n_eb > 1
        self.sigmas = bool(sigmas)
        self.alpha = bool(alpha)
        self.nu0_hz = float(nu0_hz)
        self.taper_fwhm_mas = (
            None if taper_fwhm_mas is None else float(taper_fwhm_mas)
        )
        self.prefix = prefix

    # site names
    def _site(self, name):
        return f"{self.prefix}.{name}"

    @property
    def sites(self):
        """Names of the fitted sites."""
        return tuple(self.priors())

    def priors(self):
        """Default priors of the fitted sites, a dict for ``fit``."""
        import numpyro.distributions as dist

        k = self.n_eb - 1
        out = {self._site("flux"): dist.LogUniform(1e-4, 1e2)}
        if self.gains:
            out[self._site("gain")] = dist.LogUniform(
                np.full(k, 0.5), np.full(k, 2.0)
            )
        if self.offsets:
            out[self._site("offset")] = dist.Uniform(
                np.full((k, 2), -100.0), np.full((k, 2), 100.0)
            )
        if self.sigmas:
            out[self._site("sigma")] = dist.LogUniform(
                np.full(self.n_ddid, 0.1), np.full(self.n_ddid, 10.0)
            )
        if self.alpha:
            out[self._site("alpha")] = dist.Uniform(0.0, 5.0)
        return out

    def defaults(self):
        """Starting values of the fitted sites."""
        k = self.n_eb - 1
        out = {self._site("flux"): 0.1}
        if self.gains:
            out[self._site("gain")] = onp.ones(k)
        if self.offsets:
            out[self._site("offset")] = onp.zeros((k, 2))
        if self.sigmas:
            out[self._site("sigma")] = onp.ones(self.n_ddid)
        if self.alpha:
            out[self._site("alpha")] = 2.0
        return out

    def _expand(self, free, fill):
        """Insert the reference execution block (value ``fill``) at ``ref_eb``."""
        return np.insert(free, self.ref_eb, fill, axis=0)

    def _unpack(self, values):
        """(flux, gain[n_eb], offset[n_eb, 2], sigma[n_ddid], alpha)."""
        site = self._site
        gain = (
            self._expand(np.asarray(values[site("gain")]), 1.0)
            if self.gains
            else np.ones(self.n_eb)
        )
        offset = (
            self._expand(
                np.reshape(np.asarray(values[site("offset")]), (-1, 2)), 0.0
            )
            if self.offsets
            else np.zeros((self.n_eb, 2))
        )
        sigma = (
            np.asarray(values[site("sigma")])
            if self.sigmas
            else np.ones(self.n_ddid)
        )
        alpha = values[site("alpha")] if self.alpha else 0.0
        return values[site("flux")], gain, offset, sigma, alpha

    def _taper(self, u, v):
        if self.taper_fwhm_mas is None:
            return 1.0
        theta = self.taper_fwhm_mas * mas2rad
        q2 = u * u + v * v
        return np.exp(-((np.pi * theta) ** 2) * q2 / (4.0 * onp.log(2.0)))


def _model_points(scene, unpacked, nuisance, u, v, freq, ddid, eb):
    flux, gain, offset, _, alpha = unpacked
    lam = C_LIGHT / freq
    vis = scene.model(u * lam, v * lam, lam)
    ramp = offset_phase(u, v, offset[eb, 0], offset[eb, 1])
    spectral = (freq / nuisance.nu0_hz) ** alpha
    return flux * gain[eb] * spectral * ramp * nuisance._taper(u, v) * vis


# ----------------------------------------------------------------------
# The likelihood term
# ----------------------------------------------------------------------


class _RadioTerm(eqx.Module):
    """Whitened residuals of a `ComplexVisData` with calibration nuisances."""

    arrays: dict
    scene: object = eqx.field(static=True)
    nuisance: object = eqx.field(static=True)
    n: int = eqx.field(static=True)

    def __init__(self, data, scene, nuisance, mask, chunk_size):
        n = len(data)
        if mask is None:
            mask = onp.ones(n, dtype=bool)
        mask = onp.asarray(mask, dtype=bool)
        if mask.shape != (n,):
            raise ValueError(f"mask has shape {mask.shape}, not ({n},).")
        chunk = max(1, min(int(chunk_size), n))
        pad = (-n) % chunk

        def padded(x, fill, dtype):
            x = onp.asarray(x, dtype=dtype)
            x = onp.concatenate([x, onp.full(pad, fill, dtype=dtype)])
            return np.asarray(x.reshape(-1, chunk))

        self.arrays = dict(
            u=padded(data.u, 0.0, onp.float64),
            v=padded(data.v, 0.0, onp.float64),
            vis=padded(data.vis, 0.0, onp.complex128),
            freq=padded(data.freq, float(data.freq[0]), onp.float64),
            ddid=padded(data.ddid, 0, onp.int32),
            eb=padded(data.eb, 0, onp.int32),
            # zero weight outside the mask and on the padding
            sqrt_w=padded(
                onp.sqrt(onp.asarray(data.weight)) * mask, 0.0, onp.float64
            ),
            m=padded(mask, False, onp.float64),
            w=padded(onp.asarray(data.weight), 1.0, onp.float64),
        )
        self.scene = scene
        self.nuisance = nuisance
        self.n = n

    @property
    def has_log_norm(self):
        return self.nuisance.sigmas

    def _residuals(self, values):
        """(re, im) whitened residual chunks of shape (nchunk, chunk)."""
        nuis = self.nuisance
        unpacked = nuis._unpack(values)
        sigma = unpacked[3]
        scene = self.scene
        if not isinstance(scene, SourceModel):
            scene = scene(values)

        def chunk(a):
            model = _model_points(
                scene,
                unpacked,
                nuis,
                a["u"],
                a["v"],
                a["freq"],
                a["ddid"],
                a["eb"],
            )
            r = (a["vis"] - model) * a["sqrt_w"] / sigma[a["ddid"]]
            return np.real(r), np.imag(r)

        return jax.lax.map(chunk, self.arrays)

    def _flat(self, x):
        return np.reshape(x, -1)[: self.n]

    def __call__(self, values):
        re, im = self._residuals(values)
        return np.concatenate([self._flat(re), self._flat(im)])

    def _r2(self, values):
        """Squared whitened residual (Re² + Im²) of each point."""
        re, im = self._residuals(values)
        return self._flat(re) ** 2 + self._flat(im) ** 2

    def _log_sigma_sum(self, values):
        sigma = self.nuisance._unpack(values)[3]
        a = self.arrays
        return np.sum(a["m"] * np.log(sigma[a["ddid"]]))

    def log_norm(self, values):
        """``Σ_k m_k 2 ln s_d(k)``, which ``fit`` adds to ½χ²."""
        return 2.0 * self._log_sigma_sum(values)

    def loglike(self, values):
        """Normalized Gaussian log density of the unmasked points."""
        a = self.arrays
        chi2 = np.sum(self._r2(values))
        const = np.sum(a["m"] * (np.log(2.0 * np.pi) - np.log(a["w"])))
        return -0.5 * chi2 - self.log_norm(values) - const


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def primary_beam(npix, pixel_scale_mas, freq_hz, dish_m=12.0, factor=1.13):
    """Log primary-beam attenuation ``-4 ln 2 r² / FWHM²`` on the pixel grid.

    ``FWHM = factor c / (ν D)`` and ``r`` is the distance from the image
    centre (pixel grid of
    [`pixel_offsets`][virgil._geometry.pixel_offsets]). Add it to an
    [`Image`][virgil.models.Image]'s log-brightness so that the image is the
    sky times the beam.
    """
    fwhm_mas = factor * C_LIGHT / (freq_hz * dish_m) * rad2mas
    x = pixel_offsets(npix, pixel_scale_mas)
    xx, yy = onp.meshgrid(x, x, indexing="xy")
    return -4.0 * onp.log(2.0) * (xx**2 + yy**2) / fwhm_mas**2


def dartboard_folds(data, k=5, n_q=12, n_phi=16, seed=0):
    """Cross-validation folds from uv cells (a "dartboard").

    The uv plane is cut into ``n_q`` log-spaced annuli between the smallest
    and largest ``q = √(u² + v²)`` and ``n_phi`` azimuth sectors of
    ``φ = arctan2(v, u) mod π``, so a point and its Hermitian partner share a
    cell. The occupied cells are shuffled into ``k`` folds with
    ``numpy.random.default_rng(seed)``.

    Returns
    -------
    fold, annulus : (N,) int arrays
        The fold (0..k-1) and the annulus index of every point.
    """
    u, v = (
        onp.asarray(data.u, dtype=onp.float64),
        onp.asarray(data.v, dtype=onp.float64),
    )
    # Fold into the upper half plane so (u, v) and (-u, -v) coincide exactly.
    flip = (v < 0) | ((v == 0) & (u < 0))
    u, v = onp.where(flip, -u, u), onp.where(flip, -v, v)
    q = onp.hypot(u, v)
    positive = q[q > 0]
    if positive.size == 0:
        raise ValueError("all points are at the origin of the uv plane.")
    edges = onp.geomspace(positive.min(), positive.max(), n_q + 1)
    annulus = onp.clip(
        onp.searchsorted(edges, q, side="right") - 1, 0, n_q - 1
    )
    phi = onp.arctan2(v, u)  # [0, π]
    sector = onp.clip((phi / onp.pi * n_phi).astype(int), 0, n_phi - 1)
    cell = annulus * n_phi + sector
    cells = onp.unique(cell)
    if cells.size < k:
        raise ValueError(f"only {cells.size} occupied uv cells for {k} folds.")
    order = onp.random.default_rng(seed).permutation(cells.size)
    fold_of = onp.empty(cells.size, dtype=int)
    fold_of[order] = onp.arange(cells.size) % k
    fold = fold_of[onp.searchsorted(cells, cell)]
    return fold, annulus

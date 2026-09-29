import jax
import jax.numpy as np

import numpy as onp

import equinox as eqx
import zodiax as zx

from .amigo import is_mixed_disco_record, mixed_disco_fields
from .oifits import read_oifits


__all__ = ["OIData", "closure_phases", "cp_indices"]


class OIData(zx.Base):  # type: ignore[reportGeneralTypeIssues]
    """
    Store and transform optical-interferometry observables.

    Parameters
    ----------
    data : dict, str, os.PathLike or astropy.io.fits.HDUList
        An OIFITS file (a path, or a file opened with ``astropy.io.fits`` or
        ``pyoifits``), or a dictionary of arrays; see ``__init__``.
    target : str or int, optional
        For OIFITS input, the target to keep (by name or ``TARGET_ID``).
        Required when the file contains more than one target.

    Notes
    -----
    Every (baseline, wavelength) sample is one element of the flat ``u``,
    ``v`` (metres) and ``wavel`` (metres) arrays; ``wavel`` has a single
    element when all samples share one wavelength. Models are evaluated on
    these samples. The observables are:

    * ``vis``/``d_vis``: squared visibilities (``v2_flag=True``) or
      amplitudes, or their projection through ``vis_mat``.
    * ``phi``/``d_phi``: closure phases (``cp_flag=True``) built from the
      samples ``i_cps1 + i_cps2 - i_cps3``, or absolute phases; always in
      radians, optionally projected through ``phi_mat``.

    Flagged samples are left out of the observables. ``vis_index`` (and
    ``phi_index`` for absolute phases) then lists the samples that are
    observed; they are ``None`` when every sample is used.
    """

    u: jax.Array
    v: jax.Array
    wavel: jax.Array
    vis: jax.Array
    d_vis: jax.Array
    phi: jax.Array
    d_phi: jax.Array
    i_cps1: jax.Array | onp.ndarray | None
    i_cps2: jax.Array | onp.ndarray | None
    i_cps3: jax.Array | onp.ndarray | None
    vis_mat: jax.Array | None
    phi_mat: jax.Array | None
    vis_index: jax.Array | None
    phi_index: jax.Array | None
    observable_kind: str = eqx.field(static=True)
    vis_mode: str = eqx.field(static=True)
    v2_flag: bool = eqx.field(static=True)
    cp_flag: bool = eqx.field(static=True)

    def __init__(self, data, target=None):
        """
        Initialize from an OIFITS file or explicit arrays.

        Parameters
        ----------
        data : dict, str, os.PathLike or astropy.io.fits.HDUList
            An OIFITS file, read with [`drpangloss.oifits.read_oifits`][drpangloss.oifits.read_oifits]
            (several wavelength channels, tables and epochs, and ``FLAG``
            columns, are supported). Or a dictionary with keys:

            * ``u``, ``v`` (metres) and ``wavel`` (metres): per sample, or
              ``u``/``v`` per baseline with ``vis`` of shape
              ``(n_baseline, n_wavel)`` for several channels.
            * ``vis``, ``d_vis``: squared visibilities or amplitudes.
            * ``phi``, ``d_phi``: closure or absolute phases, with
              ``phi_unit`` (``"rad"``, the default, or ``"deg"``;
              ``phase_unit`` is an alias). For several channels, shape
              ``(n_triangle, n_wavel)`` or ``(n_baseline, n_wavel)``.
            * ``i_cps1``, ``i_cps2``, ``i_cps3`` (optional): for each
              closure phase, the baselines ``(a, b)``, ``(b, c)`` and
              ``(a, c)`` of its triangle.
            * ``v2_flag`` (default True): ``vis`` holds squared
              visibilities; otherwise amplitudes.
            * ``cp_flag`` (default: True when closure indices are given):
              ``phi`` holds closure phases; otherwise absolute phases.
            * ``vis_flag``, ``phi_flag`` (optional): boolean masks, True
              for bad samples, shaped like ``vis`` and ``phi``. Samples with
              non-finite values or errors are flagged automatically.
            * ``vis_mode`` (``"auto"``, ``"v2"``, ``"amp"`` or ``"logamp"``;
              ``observable_kind`` is an alias): the visibility channel that
              data and model are compared in. ``"auto"`` keeps the channel
              of ``vis``.
            * ``vis_mat``, ``phi_mat`` (optional): linear operators of shape
              ``(n_out, n_in)`` projecting the channels into, e.g., kernel
              or DISCO observables. The ``disco_vis_mat``/``disco_phi_mat``
              spellings also check that the projected covariance is
              diagonal. Only diagonal uncertainties are propagated.

            A record with ``disco_coefficients`` is read as an AMIGO
            mixed-DISCO product (see [`load_oi_data`][drpangloss.amigo.load_oi_data]); its ``u`` and
            ``v`` are negated to match the drpangloss sign convention.
        target : str or int, optional
            For OIFITS input, the target to keep.
        """
        if not isinstance(data, dict):
            data = read_oifits(data, target=target)
        elif target is not None:
            raise ValueError("target only applies to OIFITS input.")

        if is_mixed_disco_record(data):
            for name, value in mixed_disco_fields(data).items():
                setattr(self, name, value)
            return

        u = onp.asarray(data["u"], dtype=float)
        v = onp.asarray(data["v"], dtype=float)
        wavel = onp.atleast_1d(onp.asarray(data["wavel"], dtype=float))
        vis = onp.asarray(data["vis"], dtype=float)
        d_vis = onp.asarray(data["d_vis"], dtype=float)
        phi_unit = data.get("phi_unit", data.get("phase_unit", "rad"))
        phi, d_phi = self._phase_to_radians(
            onp.asarray(data["phi"], dtype=float),
            onp.asarray(data["d_phi"], dtype=float),
            phi_unit,
            default_unit="rad",
        )
        phi, d_phi = onp.asarray(phi), onp.asarray(d_phi)

        indices = [data.get(key) for key in ("i_cps1", "i_cps2", "i_cps3")]
        if any(index is None for index in indices):
            indices = None
        else:
            indices = [onp.asarray(index, dtype=int) for index in indices]

        v2_flag = self._coerce_bool_flag(data.get("v2_flag", True), "v2_flag")
        cp_flag = self._coerce_bool_flag(
            data.get("cp_flag", indices is not None), "cp_flag"
        )
        if cp_flag and indices is None:
            raise ValueError(
                "cp_flag=True needs the closure-phase indices i_cps1, "
                "i_cps2 and i_cps3."
            )
        vis_flag = data.get("vis_flag")
        phi_flag = data.get("phi_flag")

        if vis.ndim == 2:
            u, v, wavel, indices = _expand_channels(u, v, wavel, vis, indices)
            vis, d_vis = vis.reshape(-1), d_vis.reshape(-1)
            phi, d_phi = phi.reshape(-1), d_phi.reshape(-1)
            vis_flag = None if vis_flag is None else onp.ravel(vis_flag)
            phi_flag = None if phi_flag is None else onp.ravel(phi_flag)
        elif wavel.size not in (1, u.size):
            raise ValueError(
                f"wavel has {wavel.size} values for {u.size} samples. For "
                "several wavelength channels give vis with shape "
                "(n_baseline, n_wavel), or give one wavelength per sample."
            )

        has_disco_vis = "disco_vis_mat" in data
        has_disco_phi = "disco_phi_mat" in data
        vis_mat_in = data.get("disco_vis_mat", data.get("vis_mat", None))
        phi_mat_in = data.get("disco_phi_mat", data.get("phi_mat", None))
        vis_mat = (
            None if vis_mat_in is None else onp.asarray(vis_mat_in, float)
        )
        phi_mat = (
            None if phi_mat_in is None else onp.asarray(phi_mat_in, float)
        )

        # Drop flagged samples from the observables. Their baselines stay in
        # u and v, because closure phases may still need them.
        vis_index = None
        keep = _good_samples(vis, d_vis, vis_flag, u.size, "vis")
        if keep is not None:
            if vis_mat is not None:
                raise ValueError(
                    "Flagged visibilities cannot be combined with vis_mat; "
                    "remove the flagged samples and the matching operator "
                    "columns first."
                )
            vis_index = onp.flatnonzero(keep)
            vis, d_vis = vis[keep], d_vis[keep]

        phi_index = None
        n_phi = len(indices[0]) if cp_flag else u.size
        keep = _good_samples(phi, d_phi, phi_flag, n_phi, "phi")
        if keep is not None:
            if phi_mat is not None:
                raise ValueError(
                    "Flagged phases cannot be combined with phi_mat; "
                    "remove the flagged samples and the matching operator "
                    "columns first."
                )
            phi, d_phi = phi[keep], d_phi[keep]
            if cp_flag:
                indices = [index[keep] for index in indices]
            else:
                phi_index = onp.flatnonzero(keep)

        self.u = np.asarray(u)
        self.v = np.asarray(v)
        self.wavel = np.asarray(wavel)
        self.vis = np.asarray(vis)
        self.d_vis = np.asarray(d_vis)
        self.phi = np.asarray(phi)
        self.d_phi = np.asarray(d_phi)
        self.i_cps1, self.i_cps2, self.i_cps3 = (
            (None, None, None)
            if indices is None
            else tuple(np.asarray(index) for index in indices)
        )
        self.v2_flag = v2_flag
        self.cp_flag = cp_flag
        self.vis_mat = None if vis_mat is None else np.asarray(vis_mat)
        self.phi_mat = None if phi_mat is None else np.asarray(phi_mat)
        self.vis_index = None if vis_index is None else np.asarray(vis_index)
        self.phi_index = None if phi_index is None else np.asarray(phi_index)
        vis_mode_in = data.get(
            "vis_mode", data.get("observable_vis_mode", "auto")
        )
        self.vis_mode = self._resolve_vis_mode(vis_mode_in)
        self.observable_kind = "split"
        self._transform_observed_channels(
            validate_vis_covariance=has_disco_vis,
            validate_phi_covariance=has_disco_phi,
        )

    def _resolve_vis_mode(self, vis_mode):
        """Resolve the visibility channel convention used before linear projection."""
        mode = str(vis_mode).strip().lower()
        if mode == "auto":
            return "v2" if self.v2_flag else "amp"
        valid = {"v2", "amp", "logamp"}
        if mode not in valid:
            raise ValueError(
                f"Unsupported vis_mode '{vis_mode}'. Expected one of {sorted(valid)} or 'auto'."
            )
        return mode

    @staticmethod
    def _coerce_bool_flag(value, name):
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "t", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "f", "0", "no", "n", "off"}:
                return False
            raise ValueError(
                f"Unsupported {name!r} value {value!r}; expected a boolean."
            )
        return bool(value)

    @staticmethod
    def _phase_unit_scale(unit, default_unit):
        """Return multiplicative factor converting the provided phase unit to rad."""
        raw_unit = default_unit if unit is None else unit
        unit_name = str(raw_unit).strip().lower()
        if unit_name in {"rad", "radian", "radians"}:
            return 1.0
        if unit_name in {"deg", "degree", "degrees"}:
            return np.pi / 180.0
        raise ValueError(
            f"Unsupported phase unit '{raw_unit}'. Expected radians or degrees."
        )

    @classmethod
    def _phase_to_radians(cls, phi, d_phi, unit, default_unit):
        """Convert phase observables and uncertainties to radians."""
        scale = cls._phase_unit_scale(unit, default_unit)
        return np.asarray(phi, dtype=float) * scale, np.asarray(
            d_phi, dtype=float
        ) * np.abs(scale)

    @staticmethod
    def _validate_operator_shape(operator, input_size, label):
        """Check an operator has shape ``(n_out, input_size)``."""
        if operator is None:
            return
        if operator.ndim != 2:
            raise ValueError(
                f"{label} must be a 2D matrix; got shape {operator.shape}."
            )
        if operator.shape[1] != input_size:
            hint = (
                " It looks transposed; pass its transpose."
                if operator.shape[0] == input_size
                else ""
            )
            raise ValueError(
                f"{label} has shape {operator.shape}, but operators must have "
                f"shape (n_out, n_in) with n_in = {input_size} samples.{hint}"
            )

    @staticmethod
    def _apply_linear_operator(values, operator):
        """Apply an ``(n_out, n_in)`` operator to a length-``n_in`` vector."""
        if operator is None:
            return values
        return operator @ np.asarray(values, dtype=float).reshape(-1)

    @classmethod
    def _propagate_uncertainty(cls, channel_sigma, operator):
        """Propagate diagonal uncertainties through a linear operator."""
        sigma = np.asarray(channel_sigma, dtype=float).reshape(-1)
        if operator is None:
            return sigma
        return np.sqrt(np.sum((operator * sigma[None, :]) ** 2, axis=1))

    @classmethod
    def _validate_diagonal_covariance(cls, channel_sigma, operator, label):
        """Assert that an explicitly labelled DISCO operator whitens covariance."""
        if operator is None:
            return
        sigma = np.asarray(channel_sigma, dtype=float).reshape(-1)
        weighted = operator * sigma[None, :]
        covariance = weighted @ weighted.T
        diagonal = np.diag(np.diag(covariance))
        scale = float(np.max(np.abs(np.diag(covariance))))
        atol = max(1e-12, 1e-7 * scale)
        if not bool(np.allclose(covariance, diagonal, rtol=1e-5, atol=atol)):
            raise ValueError(
                f"{label} does not produce diagonal propagated covariance. "
                "DISCO observables must be statistically independent."
            )

    def _visibility_channel_from_model(self, cvis):
        """Convert complex visibilities to the configured scalar visibility channel."""
        amp = np.abs(cvis)
        if self.vis_mode == "v2":
            return amp**2
        if self.vis_mode == "logamp":
            return np.log(np.maximum(amp, 1e-30))
        return amp

    def _visibility_channel_from_data(self, vis):
        """Convert stored visibility observables to the configured scalar channel."""
        vis = np.asarray(vis, dtype=float)
        if self.vis_mode == "logamp":
            if self.v2_flag:
                return 0.5 * np.log(np.maximum(vis, 1e-30))
            return np.log(np.maximum(vis, 1e-30))
        if self.vis_mode == "amp" and self.v2_flag:
            return np.sqrt(np.maximum(vis, 0.0))
        if self.vis_mode == "v2" and (not self.v2_flag):
            return vis**2
        return vis

    def _visibility_uncertainty_channel(self, vis, d_vis):
        """Convert visibility uncertainties into the configured scalar channel."""
        vis = np.asarray(vis, dtype=float)
        d_vis = np.asarray(d_vis, dtype=float)
        if self.vis_mode == "logamp":
            if self.v2_flag:
                return 0.5 * d_vis / np.maximum(vis, 1e-30)
            return d_vis / np.maximum(vis, 1e-30)
        if self.vis_mode == "amp" and self.v2_flag:
            return 0.5 * d_vis / np.sqrt(np.maximum(vis, 1e-30))
        if self.vis_mode == "v2" and (not self.v2_flag):
            # |V| is floored at its own uncertainty, so noisy amplitudes near
            # or below zero do not get a vanishing V² error.
            return 2.0 * np.hypot(vis, d_vis) * d_vis
        return d_vis

    def _transform_observed_channels(
        self, validate_vis_covariance=False, validate_phi_covariance=False
    ):
        """Convert observed channels to ``vis_mode`` and apply operators.

        Data already in the projected basis (their size differs from the
        number of observed samples) are left unchanged.
        """
        n_vis = self._n_vis_samples()
        n_phi = self._n_phi_samples()
        self._validate_operator_shape(self.vis_mat, n_vis, "vis_mat")
        self._validate_operator_shape(self.phi_mat, n_phi, "phi_mat")

        if np.asarray(self.vis).size == n_vis:
            vis_channel = self._visibility_channel_from_data(self.vis)
            vis_sigma = self._visibility_uncertainty_channel(
                self.vis, self.d_vis
            )
            if self.vis_mat is None:
                self.vis, self.d_vis = vis_channel, vis_sigma
            else:
                if validate_vis_covariance:
                    self._validate_diagonal_covariance(
                        vis_sigma, self.vis_mat, "disco_vis_mat"
                    )
                self.vis = self._apply_linear_operator(
                    vis_channel, self.vis_mat
                )
                self.d_vis = self._propagate_uncertainty(
                    vis_sigma, self.vis_mat
                )

        if self.phi_mat is not None and np.asarray(self.phi).size == n_phi:
            phi_sigma = np.asarray(self.d_phi, dtype=float)
            if validate_phi_covariance:
                self._validate_diagonal_covariance(
                    phi_sigma, self.phi_mat, "disco_phi_mat"
                )
            self.phi = self._apply_linear_operator(self.phi, self.phi_mat)
            self.d_phi = self._propagate_uncertainty(phi_sigma, self.phi_mat)

    def _n_vis_samples(self):
        """Number of visibility samples before any projection."""
        if self.vis_index is not None:
            return int(np.asarray(self.vis_index).size)
        return int(np.asarray(self.u).size)

    def _n_phi_samples(self):
        """Number of phase samples (or closure phases) before projection."""
        if self.cp_flag:
            return len(self.i_cps1)
        if self.phi_index is not None:
            return int(np.asarray(self.phi_index).size)
        return int(np.asarray(self.u).size)

    def flatten_data(self):
        """
        Return the data vector and its uncertainties.

        Returns
        -------
        tuple[array-like, array-like]
            The visibility observables followed by the phases (radians),
            in the order of
            [`model`][drpangloss.oidata.OIData.model], and matching one-sigma uncertainties.
        """
        if self.observable_kind == "mixed_log_complex":
            return self.vis, self.d_vis
        return (
            np.concatenate([self.vis, self.phi]),
            np.concatenate([self.d_vis, self.d_phi]),
        )

    @property
    def _phases_wrap(self):
        """Whether the phase block is raw angles that wrap at ±π."""
        return self.observable_kind == "split" and self.phi_mat is None

    def residuals(self, prediction, reference=None):
        """Return ``prediction - reference`` with phase residuals wrapped.

        Parameters
        ----------
        prediction : array-like
            Model vector, e.g. from [`model`][drpangloss.oidata.OIData.model].
        reference : array-like, optional
            Vector to compare against; by default the data
            (the first vector of :meth:`flatten_data`).

        Returns
        -------
        array-like
            Residual vector. Unprojected phase residuals are wrapped into
            ``[-π, π)``, so that a closure phase of ``π - ε`` against a model
            of ``-π + ε`` counts as a small residual rather than ``2π``.
        """
        if reference is None:
            reference = self.flatten_data()[0]
        resid = np.asarray(prediction) - np.asarray(reference)
        if not self._phases_wrap:
            return resid
        n_vis = np.asarray(self.vis).size
        phase = np.mod(resid[n_vis:] + np.pi, 2.0 * np.pi) - np.pi
        return np.concatenate([resid[:n_vis], phase])

    def standardize_model(self, cvis):
        """Map model complex visibilities (one per sample) to the data vector.

        The result lines up with the first vector of :meth:`flatten_data`.
        """
        if self.observable_kind == "mixed_log_complex":
            if self.vis_mat is None or self.phi_mat is None:
                raise ValueError(
                    "Mixed log-complex observables require model operators."
                )
            log_cvis = np.log(cvis)
            return self.vis_mat @ log_cvis.real + self.phi_mat @ log_cvis.imag
        return np.concatenate([self.to_vis(cvis), self.to_phases(cvis)])

    def to_vis(self, cvis):
        """
        Convert model complex visibilities to the visibility observables.

        The channel follows ``vis_mode`` (V², amplitude or log-amplitude);
        flagged samples are dropped, and ``vis_mat`` is applied if set.
        """
        vis = self._visibility_channel_from_model(cvis)
        if self.vis_index is not None:
            vis = vis[self.vis_index]
        return self._apply_linear_operator(vis, self.vis_mat)

    def to_phases(self, cvis):
        """
        Convert complex visibilities to closure or absolute phases in radians.
        """
        if self.cp_flag:
            phases = closure_phases(
                cvis, self.i_cps1, self.i_cps2, self.i_cps3
            )
        else:
            phases = np.angle(cvis)
            if self.phi_index is not None:
                phases = phases[self.phi_index]
        return self._apply_linear_operator(phases, self.phi_mat)

    def model(self, model_object):
        """
        Compute the model visibilities and phases for the given model object.
        """
        cvis = model_object.model(self.u, self.v, self.wavel)
        return self.standardize_model(cvis)

    def with_model(self, model_object, key=None, noise_scale=1.0):
        """Return a copy populated from a model with optional Gaussian noise.

        Sampling, uncertainties, conventions, closure indices, and linear
        observable operators are preserved from this object.
        """
        noise_scale = float(noise_scale)
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")

        prediction = self.model(model_object)
        n_vis = self.vis.size
        vis = prediction[:n_vis]
        phi = prediction[n_vis:]
        if key is not None:
            vis_key, phi_key = jax.random.split(key)
            vis = vis + noise_scale * self.d_vis * jax.random.normal(
                vis_key, vis.shape
            )
            phi = phi + noise_scale * self.d_phi * jax.random.normal(
                phi_key, phi.shape
            )
        return self.set(["vis", "phi"], [vis, phi])


def closure_phases(cvis, index_cps1, index_cps2, index_cps3):
    """
    Calculate closure phases from complex visibilities.

    Parameters
    ----------
    cvis : array-like
        Complex visibilities, one per sample.
    index_cps1 : array-like
        For each closure phase, the sample of baseline ``(a, b)``.
    index_cps2 : array-like
        For each closure phase, the sample of baseline ``(b, c)``.
    index_cps3 : array-like
        For each closure phase, the sample of baseline ``(a, c)``.

    Returns
    -------
    array-like
        Closure phases ``φ[i1] + φ[i2] − φ[i3]`` in radians, wrapped into
        ``[-π, π)``.

    Notes
    -----
    This helper returns radians for internal modeling consistency. Convert to
    degrees before writing OIFITS phase columns (e.g., ``T3PHI``).
    """
    phases = np.angle(np.asarray(cvis))
    cp = (
        phases[np.asarray(index_cps1)]
        + phases[np.asarray(index_cps2)]
        - phases[np.asarray(index_cps3)]
    )
    return np.mod(cp + np.pi, 2.0 * np.pi) - np.pi


def cp_indices(vis_sta_index, cp_sta_index):
    """Map closure-triangle station indices to baseline indices.

    Parameters
    ----------
    vis_sta_index : array-like
        Station index pairs ``(a, b)``, one per baseline.
    cp_sta_index : array-like
        Station index triplets ``(a, b, c)``, one per closure triangle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays ``(i_cps1, i_cps2, i_cps3)`` giving, for each triangle, the
        baselines ``(a, b)``, ``(b, c)`` and ``(a, c)``, so that the closure
        phase is ``φ[i_cps1] + φ[i_cps2] − φ[i_cps3]``.

    Raises
    ------
    ValueError
        If a triangle needs a baseline that is missing, or stored only in the
        reversed orientation.

    Notes
    -----
    Baselines are matched on station indices alone. For data with several
    epochs or wavelength channels use [`drpangloss.oifits.read_oifits`][drpangloss.oifits.read_oifits],
    which also matches on instrument and MJD.
    """
    vis_sta_index = onp.asarray(vis_sta_index, dtype=int).reshape(-1, 2)
    cp_sta_index = onp.asarray(cp_sta_index, dtype=int).reshape(-1, 3)
    lookup = {}
    for k, (a, b) in enumerate(vis_sta_index):
        lookup.setdefault((int(a), int(b)), k)

    def baseline(pair):
        pair = (int(pair[0]), int(pair[1]))
        if pair in lookup:
            return lookup[pair]
        # TODO: support reversed baselines by returning a sign per leg.
        detail = (
            f"it is only stored reversed as {pair[::-1]}"
            if pair[::-1] in lookup
            else "it is missing"
        )
        raise ValueError(
            f"A closure triangle needs baseline {pair}, but {detail}."
        )

    legs = [[], [], []]
    for a, b, c in cp_sta_index:
        for leg, pair in zip(legs, ((a, b), (b, c), (a, c))):
            leg.append(baseline(pair))
    return tuple(onp.asarray(leg, dtype=int) for leg in legs)


def _expand_channels(u, v, wavel, vis, indices):
    """Expand per-baseline arrays to one sample per (baseline, channel).

    Samples are ordered baseline-major; closure indices (per baseline) are
    mapped to the samples at the same channel.
    """
    n_baseline, n_wavel = vis.shape
    if u.shape != (n_baseline,) or v.shape != (n_baseline,):
        raise ValueError(
            f"vis has shape {vis.shape}, so u and v need {n_baseline} "
            "entries (one per baseline)."
        )
    if wavel.size != n_wavel:
        raise ValueError(
            f"vis has {n_wavel} wavelength channels but wavel has "
            f"{wavel.size} values."
        )
    channels = onp.arange(n_wavel)
    if indices is not None:
        indices = [
            (index[:, None] * n_wavel + channels[None, :]).reshape(-1)
            for index in indices
        ]
    return (
        onp.repeat(u, n_wavel),
        onp.repeat(v, n_wavel),
        onp.tile(wavel, n_baseline),
        indices,
    )


def _good_samples(values, errors, flag, n_samples, name):
    """Mask of unflagged, finite samples, or ``None`` if all are good.

    Data whose size is not ``n_samples`` are already projected and are not
    checked.
    """
    if values.size != n_samples:
        if flag is not None:
            raise ValueError(
                f"{name}_flag was given, but {name} has {values.size} values "
                f"for {n_samples} samples (it looks already projected)."
            )
        return None
    bad = ~(onp.isfinite(values) & onp.isfinite(errors))
    if flag is not None:
        flag = onp.asarray(flag, dtype=bool).reshape(-1)
        if flag.size != n_samples:
            raise ValueError(
                f"{name}_flag has {flag.size} entries for {n_samples} samples."
            )
        bad = bad | flag
    if not bad.any():
        return None
    return ~bad

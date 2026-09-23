import jax
import jax.numpy as np

import numpy as onp

import equinox as eqx
import zodiax as zx


__all__ = ["OIData", "closure_phases", "cp_indices"]


class OIData(zx.Base):  # type: ignore[reportGeneralTypeIssues]
    """
    Store and transform optical-interferometry observables.

    Parameters
    ----------
    data : dict or object
        Either a dictionary with explicit interferometric arrays, or an OIFITS
        object opened with ``pyoifits``.

    Notes
    -----
    The object stores baseline coordinates, observables, uncertainties, and
    optional closure-phase index triplets. It provides convenience methods for
    flattening data/model vectors and converting complex visibilities to the
    configured visibility/phase conventions. Phase observables are stored
    internally in radians.
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
    vis_mode: str = eqx.field(static=True)
    v2_flag: bool = eqx.field(static=True)
    cp_flag: bool = eqx.field(static=True)

    def __init__(self, data):
        """
        Initialize from an OIFITS object or explicit arrays.

        Parameters
        ----------
        data : dict or object
            OIFITS data opened with ``pyoifits``, or a dictionary containing
            ``u``, ``v``, ``wavel``, ``vis``, ``d_vis``, ``phi``, ``d_phi``,
            optional closure-phase indices, convention flags, and optional
            ``phi_unit`` (``"rad"`` or ``"deg"``). OIFITS phase columns are
            interpreted using their column unit metadata when present, and
            default to degrees when metadata is missing.
        """

        if not isinstance(data, dict):
            # assume data is an oifits file opened with pyoifits
            data_names = [d.name for d in data.get_dataHDUs()]
            assert "OI_VIS" in data_names or "OI_VIS2" in data_names, (
                "No visibility data found in OIFITS file"
            )
            assert "OI_T3" in data_names or "OI_PHI" in data_names, (
                "No phase data found in OIFITS file"
            )

            # get the data from the oifits file
            self.wavel = np.array(
                data[1].data["EFF_WAVE"], dtype=float
            )  # note that for AMI this is scalar but for CHARA it is an array

            # if square visibilities are available, get them, otherwise get unsquared visibilities
            if "OI_VIS2" in data_names:
                visdata = data["OI_VIS2"]
                self.vis = np.array(visdata.data["VIS2DATA"], dtype=float)
                self.d_vis = np.array(visdata.data["VIS2ERR"], dtype=float)
                vis_sta_index = visdata.data["STA_INDEX"]

                self.u, self.v = (
                    np.array(visdata.data["UCOORD"], dtype=float),
                    np.array(visdata.data["VCOORD"], dtype=float),
                )

                self.v2_flag = True

            elif "OI_VIS" in data_names:
                visdata = data["OI_VIS"]
                vis_key = (
                    "VISAMP" if "VISAMP" in visdata.data.names else "VISPHI"
                )
                d_vis_key = (
                    "VISAMPERR"
                    if "VISAMPERR" in visdata.data.names
                    else "VISERR"
                )
                self.vis = np.array(visdata.data[vis_key], dtype=float)
                self.d_vis = np.array(visdata.data[d_vis_key], dtype=float)
                self.u, self.v = (
                    np.array(visdata.data["UCOORD"], dtype=float),
                    np.array(visdata.data["VCOORD"], dtype=float),
                )
                vis_sta_index = np.array(visdata.data["STA_INDEX"], dtype=int)

                self.v2_flag = False

            # if absolute phases are available, get them, otherwise get closure phases
            if "OI_PHI" in data_names:
                phidata = data["OI_PHI"]
                phi = np.array(phidata.data["VISPHI"], dtype=float)
                d_phi = np.array(phidata.data["VISERR"], dtype=float)
                phase_unit = self._extract_oifits_phase_unit(phidata, "VISPHI")
                self.phi, self.d_phi = self._phase_to_radians(
                    phi, d_phi, phase_unit, default_unit="deg"
                )
                self.i_cps1, self.i_cps2, self.i_cps3 = None, None, None

                self.cp_flag = False

            elif "OI_T3" in data_names:
                phidata = data["OI_T3"]
                phi = np.array(phidata.data["T3PHI"], dtype=float)
                d_phi = np.array(phidata.data["T3PHIERR"], dtype=float)
                phase_unit = self._extract_oifits_phase_unit(phidata, "T3PHI")
                self.phi, self.d_phi = self._phase_to_radians(
                    phi, d_phi, phase_unit, default_unit="deg"
                )

                cp_sta_index = np.array(phidata.data["STA_INDEX"], dtype=int)
                self.i_cps1, self.i_cps2, self.i_cps3 = cp_indices(
                    vis_sta_index, cp_sta_index
                )

                self.cp_flag = True

        else:
            # assume data is a dict of the form {'u':u,'v':v,'wavel':wavel,'vis':vis,'d_vis':d_vis,
            #'phi':phi,'d_phi':d_phi,'i_cps1':i_cps1,'i_cps2':i_cps2,'i_cps3':i_cps3,'v2_flag':v2_flag,'cp_flag':cp_flag}

            self.u = np.array(data["u"], dtype=float)
            self.v = np.array(data["v"], dtype=float)
            self.wavel = np.array(data["wavel"], dtype=float)

            self.vis = np.array(data["vis"], dtype=float)
            self.d_vis = np.array(data["d_vis"], dtype=float)

            self.phi = np.array(data["phi"], dtype=float)
            self.d_phi = np.array(data["d_phi"], dtype=float)
            phi_unit = data.get("phi_unit", data.get("phase_unit", "rad"))
            self.phi, self.d_phi = self._phase_to_radians(
                self.phi, self.d_phi, phi_unit, default_unit="rad"
            )

            try:
                idx1 = data["i_cps1"]
                idx2 = data["i_cps2"]
                idx3 = data["i_cps3"]
                if idx1 is None or idx2 is None or idx3 is None:
                    raise KeyError
                self.i_cps1 = np.array(idx1, dtype=int)
                self.i_cps2 = np.array(idx2, dtype=int)
                self.i_cps3 = np.array(idx3, dtype=int)
            except KeyError:
                self.i_cps1 = None
                self.i_cps2 = None
                self.i_cps3 = None

            v2_flag = data.get("v2_flag", True)
            self.v2_flag = self._coerce_bool_flag(v2_flag, "v2_flag")
            cp_flag = data.get("cp_flag", self.i_cps1 is not None)
            self.cp_flag = self._coerce_bool_flag(cp_flag, "cp_flag")

            has_disco_vis = "disco_vis_mat" in data
            has_disco_phi = "disco_phi_mat" in data
            vis_mat_in = data.get("disco_vis_mat", data.get("vis_mat", None))
            phi_mat_in = data.get("disco_phi_mat", data.get("phi_mat", None))
            self.vis_mat = (
                None
                if vis_mat_in is None
                else np.asarray(vis_mat_in, dtype=float)
            )
            self.phi_mat = (
                None
                if phi_mat_in is None
                else np.asarray(phi_mat_in, dtype=float)
            )
            vis_mode_in = data.get(
                "vis_mode", data.get("observable_vis_mode", "auto")
            )
            self.vis_mode = self._resolve_vis_mode(vis_mode_in)
            self._transform_observed_channels(
                validate_vis_covariance=has_disco_vis,
                validate_phi_covariance=has_disco_phi,
            )
            return

        self.vis_mat = None
        self.phi_mat = None
        self.vis_mode = self._resolve_vis_mode("auto")

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
    def _extract_oifits_phase_unit(phidata, column_name):
        """Extract phase-column unit from an OIFITS table, if available."""
        columns = getattr(phidata, "columns", None)
        if columns is None or column_name not in columns.names:
            return None
        return getattr(columns[column_name], "unit", None)

    @staticmethod
    def _validate_operator_shape(operator, input_size, label):
        """Validate a linear operator can act on vectors of length ``input_size``."""
        if operator is None:
            return
        if operator.ndim != 2:
            raise ValueError(
                f"{label} must be a 2D matrix; got shape {operator.shape}."
            )
        if operator.shape[0] != input_size and operator.shape[1] != input_size:
            raise ValueError(
                f"{label} shape {operator.shape} is incompatible with vector length {input_size}."
            )

    @staticmethod
    def _apply_linear_operator(values, operator):
        """Apply a 2D linear operator to a 1D vector, supporting left or right multiplication."""
        if operator is None:
            return values
        vec = np.asarray(values, dtype=float).reshape(-1)
        if operator.shape[1] == vec.size:
            return operator @ vec
        if operator.shape[0] == vec.size:
            return vec @ operator
        raise ValueError(
            f"Operator shape {operator.shape} is incompatible with vector length {vec.size}."
        )

    @staticmethod
    def _operator_weights(channel_sigma, operator):
        """Orient a linear operator to act on the supplied channel vector."""
        sigma = np.asarray(channel_sigma, dtype=float).reshape(-1)
        if operator is None:
            return None
        op = operator
        if op.shape[1] == sigma.size:
            weights = op
        elif op.shape[0] == sigma.size:
            weights = op.T
        else:
            raise ValueError(
                f"Operator shape {op.shape} is incompatible with uncertainty length {sigma.size}."
            )
        return weights

    @classmethod
    def _propagate_uncertainty(cls, channel_sigma, operator):
        """Propagate diagonal uncertainties through a linear operator."""
        sigma = np.asarray(channel_sigma, dtype=float).reshape(-1)
        if operator is None:
            return sigma
        weights = cls._operator_weights(sigma, operator)
        assert weights is not None
        return np.sqrt(np.sum((weights * sigma[None, :]) ** 2, axis=1))

    @classmethod
    def _validate_diagonal_covariance(cls, channel_sigma, operator, label):
        """Assert that an explicitly labelled DISCO operator whitens covariance."""
        if operator is None:
            return
        sigma = np.asarray(channel_sigma, dtype=float).reshape(-1)
        weights = cls._operator_weights(sigma, operator)
        assert weights is not None
        weighted = weights * sigma[None, :]
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
            return 2.0 * np.maximum(vis, 1e-30) * d_vis
        return d_vis

    def _transform_observed_channels(
        self, validate_vis_covariance=False, validate_phi_covariance=False
    ):
        """Optionally project observed channels into linear self-calibrated observables."""
        n_vis = np.asarray(self.u).size
        n_phi = (
            np.asarray(self.u).size
            if not self.cp_flag
            else np.asarray(self.phi).size
        )
        self._validate_operator_shape(self.vis_mat, n_vis, "vis_mat")
        self._validate_operator_shape(self.phi_mat, n_phi, "phi_mat")

        if self.vis_mat is not None and np.asarray(self.vis).size == n_vis:
            vis_channel = self._visibility_channel_from_data(self.vis)
            vis_sigma = self._visibility_uncertainty_channel(
                self.vis, self.d_vis
            )
            if validate_vis_covariance:
                self._validate_diagonal_covariance(
                    vis_sigma, self.vis_mat, "disco_vis_mat"
                )
            self.vis = self._apply_linear_operator(vis_channel, self.vis_mat)
            self.d_vis = self._propagate_uncertainty(vis_sigma, self.vis_mat)

        if self.phi_mat is not None and np.asarray(self.phi).size == n_phi:
            phi_sigma = np.asarray(self.d_phi, dtype=float)
            if validate_phi_covariance:
                self._validate_diagonal_covariance(
                    phi_sigma, self.phi_mat, "disco_phi_mat"
                )
            self.phi = self._apply_linear_operator(self.phi, self.phi_mat)
            self.d_phi = self._propagate_uncertainty(phi_sigma, self.phi_mat)

    def flatten_data(self):
        """
        Flatten closure phases and uncertainties.
        """
        return np.concatenate([self.vis, self.phi]), np.concatenate(
            [self.d_vis, self.d_phi]
        )

    def unpack_all(self):
        """
        Unpack all data to be used in some legacy model functions.
        """
        return (
            self.u / self.wavel,
            self.v / self.wavel,
            self.phi,
            self.d_phi,
            self.vis,
            self.d_vis,
            self.i_cps1,
            self.i_cps2,
            self.i_cps3,
        )

    def flatten_model(self, cvis):
        """
        Flatten model visibilities and phases.

        Parameters
        ----------
        cvis : array-like
            Complex visibilities from a model evaluation.

        Returns
        -------
        array-like
            Concatenated visibility and phase model vector in the same
            convention/order as ``flatten_data``.
        """

        return np.concatenate([self.to_vis(cvis), self.to_phases(cvis)])

    def to_vis(self, cvis):
        """
        Convert complex visibilities to visibilities or squared visibilities.
        """
        vis = self._visibility_channel_from_model(cvis)
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
        return self._apply_linear_operator(phases, self.phi_mat)

    def model(self, model_object):
        """
        Compute the model visibilities and phases for the given model object.
        """
        cvis = model_object.model(self.u, self.v, self.wavel)
        return self.flatten_model(cvis)

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
        Complex visibilities.
    index_cps1 : array-like
        First baseline indices for each closure triangle.
    index_cps2 : array-like
        Second baseline indices for each closure triangle.
    index_cps3 : array-like
        Third baseline indices for each closure triangle.

    Returns
    -------
    array-like
        Closure phases in radians.

    Notes
    -----
    This helper returns radians for internal modeling consistency. Convert to
    degrees before writing OIFITS phase columns (e.g., ``T3PHI``).

    """
    visphiall = np.angle(cvis)
    visphiall = np.mod(visphiall + np.pi, 2.0 * np.pi) - np.pi
    visphi = np.reshape(visphiall, (len(cvis), 1))
    cp = (
        visphi[np.array(index_cps1)]
        + visphi[np.array(index_cps2)]
        - visphi[np.array(index_cps3)]
    )
    out = np.reshape(np.mod(cp + np.pi, 2.0 * np.pi) - np.pi, len(index_cps1))
    return out


def cp_indices(vis_sta_index, cp_sta_index):
    """Map closure-triangle station indices to baseline indices.

    Parameters
    ----------
    vis_sta_index : array-like
        Baseline station index pairs from visibility data.
    cp_sta_index : array-like
        Triangle station index triplets from closure-phase data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays ``(i_cps1, i_cps2, i_cps3)`` identifying the three baselines
        composing each closure phase.
    """
    vis_sta_index, cp_sta_index = (
        onp.array(vis_sta_index, dtype=int),
        onp.array(cp_sta_index, dtype=int),
    )
    i_cps1 = onp.zeros(len(onp.array(cp_sta_index)), dtype=int)
    i_cps2 = onp.zeros(len(onp.array(cp_sta_index)), dtype=int)
    i_cps3 = onp.zeros(len(onp.array(cp_sta_index)), dtype=int)

    for i in range(len(cp_sta_index)):
        i_cps1[i] = onp.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][1] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps2[i] = onp.argwhere(
            (cp_sta_index[i][1] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps3[i] = onp.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
    return (
        onp.array(i_cps1, dtype=int),
        onp.array(i_cps2, dtype=int),
        onp.array(i_cps3, dtype=int),
    )

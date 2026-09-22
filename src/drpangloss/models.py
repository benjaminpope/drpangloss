import jax.numpy as np
import jax

import numpy as onp
from scipy.stats import chi2 as scipy_chi2

import equinox as eqx
import zodiax as zx

from .inference import (
    fisher_matrix as _fisher_matrix,
    laplace_covariance as _laplace_covariance,
)

rad2mas = 180.0 / np.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = np.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad

dtor = np.pi / 180.0
i2pi = 1j * 2.0 * np.pi


class OIData(zx.Base):
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
    i_cps1: jax.Array
    i_cps2: jax.Array
    i_cps3: jax.Array
    vis_mat: jax.Array
    phi_mat: jax.Array
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

            self.v2_flag = bool(data.get("v2_flag", True))
            self.cp_flag = bool(data.get("cp_flag", self.i_cps1 is not None))

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


def _image_coordinates(npix, fov_mas):
    """Return Cartesian image-plane coordinates in milliarcseconds."""
    npix = int(npix)
    pixel_scale_mas = float(fov_mas) / npix
    pixel_indices = np.arange(npix)
    center = 0.5 * (npix - 1)
    x = -(pixel_indices - center) * pixel_scale_mas
    y = (center - pixel_indices) * pixel_scale_mas
    return np.meshgrid(x, y, indexing="xy")


def _normalize_image(image):
    """Return a finite unit-sum image for render outputs."""
    image = np.nan_to_num(np.asarray(image), nan=0.0, posinf=0.0, neginf=0.0)
    total = np.sum(image)
    if bool(np.isfinite(total)) and bool(total > 0.0):
        return image / total
    raise ValueError("Rendered image must contain positive finite flux.")


class SourceModel(zx.Base):
    """Base class for sky-brightness source models."""

    def model(self, u, v, wavel):
        """Evaluate complex visibilities on interferometric baselines."""
        raise NotImplementedError

    def render(self, npix=256, fov_mas=200.0):
        """Render an image-plane model in milliarcseconds."""
        raise NotImplementedError


class BinaryModelAngular(SourceModel):
    """
    Represent a binary companion using angular separation and position angle.

    Parameters
    ----------
    sep : float or array-like
        On-sky separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees, measured East of North.
    contrast : float or array-like
        Brightness contrast ratio ``star/companion``.

    Notes
    -----
    This parameterization is often convenient for reporting astrophysical
    constraints directly in polar-like coordinates. The model evaluates complex
    visibilities on the provided interferometric baseline geometry.
    """

    sep: jax.Array
    pa: jax.Array
    contrast: jax.Array

    def __init__(self, sep, pa, contrast):
        """
        Initialize a binary model in angular coordinates.

        Parameters
        ----------
        sep : float or array-like
            Separation in milliarcseconds.
        pa : float or array-like
            Position angle in degrees.
        contrast : float or array-like
            Contrast ratio between primary and companion (``star/companion``).

        """

        self.sep = np.asarray(sep, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.contrast = np.asarray(contrast, dtype=float)

    def unpack_all(self):
        """
        Return all model parameters in angular form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(sep, pa, contrast)``.
        """
        return self.sep, self.pa, self.contrast

    def to_cartesian(self):
        """
        Convert this angular parameterization into Cartesian sky offsets.

        Returns
        -------
        BinaryModelCartesian
            Equivalent binary model expressed as ``(dra, ddec, flux)``.
        """
        th = self.pa * dtor
        dra = -self.sep * np.sin(th)
        ddec = self.sep * np.cos(th)
        flux = 1.0 / self.contrast
        return BinaryModelCartesian(dra, ddec, flux)

    def model(self, u, v, wavel):
        """
        Evaluate complex visibilities for this angular binary model.

        Parameters
        ----------
        u : array-like
            Baseline ``u`` coordinates in meters.
        v : array-like
            Baseline ``v`` coordinates in meters.
        wavel : array-like
            Effective wavelength(s) in meters.

        Returns
        -------
        array-like
            Complex visibility samples on the provided baselines.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary_angular(uu, vv, self.sep, self.pa, self.contrast)

    def render(self, npix=256, fov_mas=200.0):
        """
        Render a two-point-source approximation on a Cartesian image grid.
        """
        xx, yy = _image_coordinates(npix, fov_mas)
        th = self.pa * dtor
        ddec = self.sep * np.cos(th)
        dra = -1.0 * self.sep * np.sin(th)

        l2 = 1.0 / (self.contrast + 1.0)
        l1 = 1.0 - l2
        sigma = max(float(fov_mas) / float(npix), 1e-6)
        star = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
        comp = np.exp(
            -0.5 * (((xx - dra) / sigma) ** 2 + ((yy - ddec) / sigma) ** 2)
        )
        image = l1 * star + l2 * comp
        return image / np.sum(image)


class BinaryModelCartesian(SourceModel):
    """
    Represent a binary companion using Cartesian sky offsets.

    Parameters
    ----------
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    ddec : float or array-like
        Declination offset in milliarcseconds.
    flux : float or array-like
        Companion-to-primary flux ratio.

    Notes
    -----
    This parameterization is useful for optimization and inference workflows
    that operate directly in Cartesian offsets.
    """

    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array

    def __init__(self, dra, ddec, flux):
        """
        Initialize a binary model in Cartesian offsets.

        Parameters
        ----------
        dra : float or array-like
            Right-ascension offset in milliarcseconds.
        ddec : float or array-like
            Declination offset in milliarcseconds.
        flux : float or array-like
            Flux ratio for the companion component.

        """

        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

    def unpack_all(self):
        """
        Return all model parameters in Cartesian form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(dra, ddec, flux)``.
        """
        return self.dra, self.ddec, self.flux

    def to_angular(self):
        """
        Convert this Cartesian parameterization into angular coordinates.

        Returns
        -------
        BinaryModelAngular
            Equivalent binary model expressed as ``(sep, pa, contrast)``.
        """
        sep = np.sqrt(self.dra**2 + self.ddec**2)
        pa = np.mod(np.rad2deg(np.arctan2(-self.dra, self.ddec)), 360.0)
        contrast = 1.0 / self.flux
        return BinaryModelAngular(sep, pa, contrast)

    def model(self, u, v, wavel):
        """
        Evaluate complex visibilities for this Cartesian binary model.

        Parameters
        ----------
        u : array-like
            Baseline ``u`` coordinates in meters.
        v : array-like
            Baseline ``v`` coordinates in meters.
        wavel : array-like
            Effective wavelength(s) in meters.

        Returns
        -------
        array-like
            Complex visibility samples on the provided baselines.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary(uu, vv, self.ddec, self.dra, self.flux)

    def render(self, npix=256, fov_mas=200.0):
        """
        Render a two-point-source approximation on a Cartesian image grid.
        """
        xx, yy = _image_coordinates(npix, fov_mas)
        l2 = self.flux / (1.0 + self.flux)
        l1 = 1.0 - l2
        sigma = max(float(fov_mas) / float(npix), 1e-6)
        star = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
        comp = np.exp(
            -0.5
            * (
                ((xx - self.dra) / sigma) ** 2
                + ((yy - self.ddec) / sigma) ** 2
            )
        )
        image = l1 * star + l2 * comp
        return image / np.sum(image)


class GaussianDiskModel(SourceModel):
    """Centered or offset circular Gaussian disk model."""

    sigma: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def __init__(self, sigma, dra=0.0, ddec=0.0):
        self.sigma = np.asarray(sigma, dtype=float)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def __repr__(self):
        return (
            "GaussianDiskModel("
            f"sigma={self.sigma}, dra={self.dra}, ddec={self.ddec})"
        )

    def unpack_all(self):
        return self.sigma, self.dra, self.ddec

    def model(self, u, v, wavel):
        uu, vv = u / wavel, v / wavel
        return cvis_gaussian_disk(uu, vv, self.sigma, self.dra, self.ddec)

    def render(self, npix=256, fov_mas=200.0):
        xx, yy = _image_coordinates(npix, fov_mas)
        sigma_mas = np.maximum(self.sigma, 1e-9)
        log_image = -0.5 * (
            ((xx - self.dra) / sigma_mas) ** 2
            + ((yy - self.ddec) / sigma_mas) ** 2
        )
        image = np.exp(log_image - np.max(log_image))
        return _normalize_image(image)


class HarmonixModel(SourceModel):
    """
    Wrapper for external source models with harmonix-like visibility methods.
    """

    source: object = eqx.field(static=True)
    visibility_method: str = eqx.field(static=True)
    render_method: str = eqx.field(static=True)
    expects_wavelength_units: bool = eqx.field(static=True)
    observation_time: object = eqx.field(static=True)

    def __init__(
        self,
        source,
        visibility_method="model",
        render_method="render",
        expects_wavelength_units=True,
        observation_time=None,
    ):
        self.source = source
        self.visibility_method = str(visibility_method)
        self.render_method = str(render_method)
        self.expects_wavelength_units = bool(expects_wavelength_units)
        self.observation_time = observation_time

    def model(self, u, v, wavel):
        method = getattr(self.source, self.visibility_method)
        args = (
            [u / wavel, v / wavel] if self.expects_wavelength_units else [u, v]
        )
        if self.observation_time is not None:
            args.append(self.observation_time)
        return np.asarray(method(*args))

    def render(self, npix=256, fov_mas=200.0):
        if not hasattr(self.source, self.render_method):
            if hasattr(self.source, "surface"):
                theta = 0.0
                if hasattr(self.source, "rotational_phase"):
                    theta = self.source.rotational_phase(
                        0.0
                        if self.observation_time is None
                        else self.observation_time
                    )
                image = np.asarray(
                    self.source.surface.render(res=npix, theta=theta)
                )
                return _normalize_image(image)
            raise NotImplementedError(
                "Wrapped source does not expose a compatible render method."
            )
        return _normalize_image(
            getattr(self.source, self.render_method)(npix, fov_mas)
        )


HarmonixAdapter = HarmonixModel


def cvis_binary_angular(u, v, sep, pa, contrast):
    # adapted from pymask
    """Compute complex visibilities for an angular-parameterized binary model.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    sep : float or array-like
        Separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees.
    contrast : float or array-like
        Contrast ratio ``star/companion``.

    Returns
    -------
    array-like
        Complex visibility samples.
    """

    # normalize visibilities so total power is 1

    th = pa * dtor

    ddec = mas2rad * (sep * np.cos(th))
    dra = -1 * mas2rad * (sep * np.sin(th))

    # decompose into two "luminosity"
    l2 = 1.0 / (contrast + 1)
    l1 = 1 - l2

    # phase-factor
    phi = np.exp(-i2pi * (u * dra + v * ddec))
    cvis = l1 + l2 * phi

    return cvis


def cvis_binary(u, v, ddec, dra, planet):
    # adapted from pymask
    """Compute complex visibilities for a Cartesian-parameterized binary model.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    ddec : float or array-like
        Declination offset in milliarcseconds.
    dra : float or array-like
        Right-ascension offset in milliarcseconds.
    planet : float or array-like
        Flux ratio of the companion.

    Returns
    -------
    array-like
        Complex visibility samples.
    """

    star = 1

    # normalize visibilities so total power is 1
    p3 = star / (star + planet)
    p2 = planet / (star + planet)

    # relative locations
    ddec = ddec * np.pi / (180.0 * 3600.0 * 1000.0)
    dra = dra * np.pi / (180.0 * 3600.0 * 1000.0)
    phi_r = np.cos(-2 * np.pi * (u * dra + v * ddec))
    phi_i = np.sin(-2 * np.pi * (u * dra + v * ddec))

    cvis = p3 + p2 * phi_r + p2 * phi_i * 1.0j

    return cvis


def cvis_gaussian_disk(u, v, sigma, dra=0.0, ddec=0.0):
    """Compute complex visibilities for a circular Gaussian disk."""
    sigma_rad = mas2rad * sigma
    rho2 = u**2 + v**2
    envelope = np.exp(-2.0 * (np.pi**2) * (sigma_rad**2) * rho2)

    dra_rad = mas2rad * dra
    ddec_rad = mas2rad * ddec
    phase = np.exp(-i2pi * (u * dra_rad + v * ddec_rad))
    return envelope * phase


def model_loglike(model_object, data_obj):
    """Evaluate a Gaussian log likelihood for an instantiated model object."""
    model_data = data_obj.model(model_object)
    data, errors = data_obj.flatten_data()
    return jax.scipy.stats.norm.logpdf(
        model_data, loc=data, scale=errors
    ).sum()


def joint_prediction(params, observations, model_fn):
    """Concatenate predictions for a parameter pytree and multiple observations.

    ``model_fn(params, index)`` defines which parameters are shared and which
    are specific to each observation.
    """
    return np.concatenate(
        [
            observation.model(model_fn(params, index))
            for index, observation in enumerate(observations)
        ]
    )


def joint_loglike(params, observations, model_fn):
    """Sum independent Gaussian log likelihoods over multiple observations."""
    return sum(
        model_loglike(model_fn(params, index), observation)
        for index, observation in enumerate(observations)
    )


def loglike(values, params, data_obj, model_class):
    """
    Abstract log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    """

    param_dict = dict(zip(params, values))

    return model_loglike(model_class(**param_dict), data_obj)


def loglike_nosignal(values, params, data_obj, model_class):
    """
    Abstract null log-likelihood function for a given model class and data object, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    """

    param_dict = dict(zip(params, values))

    model_data = data_obj.model(model_class(**param_dict))
    _, errors = data_obj.flatten_data()
    data = np.concatenate(
        [np.ones_like(data_obj.vis), np.zeros_like(data_obj.phi)]
    )

    return jax.scipy.stats.norm.logpdf(
        model_data, loc=data, scale=errors
    ).sum()


def laplace_cov(values, params, data_obj, model_class):
    """
    Compute the full Laplace covariance matrix for all model parameters jointly.

    Computes the inverse of the Hessian of the negative log-likelihood with
    respect to all parameters in ``params`` simultaneously, returning an
    ``N x N`` covariance matrix (where ``N = len(params)``).

    .. note::
        This function returns the *full* covariance matrix over all ``N``
        parameters.  To obtain only the marginal flux uncertainty at a fixed
        position, use :func:`laplace_contrast_uncertainty` instead.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    array-like
        ``N x N`` covariance matrix, where ``N = len(params)``.
    """

    objective = lambda vals: -loglike(vals, params, data_obj, model_class)
    return _laplace_covariance(objective, np.asarray(values, dtype=float))


def laplace_contrast_uncertainty(
    flux, dra, ddec, data_obj, model_class, params=None
):
    """
    Compute the Laplace uncertainty in flux at a fixed sky position.

    Unlike :func:`laplace_cov`, which inverts the *full* N-parameter Hessian,
    this function **fixes** ``dra`` and ``ddec`` and computes only the scalar
    curvature of the negative log-likelihood along the **flux axis alone**:

    .. math::

        \\sigma_f = \\left(\\frac{\\partial^2 (-\\log L)}{\\partial f^2}\\right)^{-1/2}

    This is a 1-D (scalar) second derivative, not a matrix inversion.  It is
    appropriate when the position is held fixed (e.g. on a detection grid) and
    only the contrast uncertainty at that grid point is needed.  For the joint
    uncertainty over all parameters, use :func:`laplace_cov` instead.

    Parameters
    ----------
    flux : float
        Flux ratio value at which the local Laplace uncertainty is evaluated.
    dra : float
        Right ascension offset in mas (held fixed).
    ddec : float
        Declination offset in mas (held fixed).
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    params : list[str] or tuple[str, str, str], optional
        Parameter names corresponding to ``(dra, ddec, flux)``. Defaults to
        ``["dra", "ddec", "flux"]``.

    Returns
    -------
    float
        Scalar uncertainty in the contrast (standard deviation along flux axis).
    """

    if params is None:
        params = ["dra", "ddec", "flux"]

    values = np.asarray([dra, ddec, flux], dtype=float)
    return laplace_parameter_uncertainty(
        values,
        params,
        data_obj,
        model_class,
        target_param=params[-1],
    )


def laplace_parameter_uncertainty(
    values, params, data_obj, model_class, target_param
):
    """Compute scalar Laplace uncertainty for one parameter with all others fixed."""
    params = list(params)
    if target_param not in params:
        raise ValueError(
            f"target_param '{target_param}' is not present in params={params}."
        )
    idx = params.index(target_param)
    values = np.asarray(values, dtype=float)

    objective = lambda x: -loglike(
        values.at[idx].set(x), params, data_obj, model_class
    )
    d2_axis = jax.grad(jax.grad(objective))(values[idx])
    return np.sqrt(1.0 / np.asarray(d2_axis, dtype=float))


def fisher(values, params, data_obj, model_class, ridge=0.0):
    """Approximate the local Fisher matrix at a parameter point.

    Parameters
    ----------
    values : array-like
        Parameter vector at which to evaluate the local curvature.
    params : list[str]
        Parameter names corresponding to ``values``.
    data_obj : OIData
        Observational data object.
    model_class : class
        Model class used to evaluate the likelihood.
    ridge : float, optional
        Diagonal regularization term.

    Returns
    -------
    array-like
        Fisher information matrix.
    """
    objective = lambda vals: -loglike(vals, params, data_obj, model_class)
    return _fisher_matrix(
        objective, np.asarray(values, dtype=float), ridge=ridge
    )


def chi2ppf(p, df):
    """
    Percentile function for chi-square.

    For ``df=1`` (the path used in ``nsigma``), use the closed-form identity
    based on the standard normal quantile, i.e. square ``norm.ppf((p+1)/2)``.
    This remains JAX-native,
    differentiable, and fast.

    For ``df != 1``, use JAX's native inverse lower incomplete gamma when it
    is available. On older JAX versions where that function does not exist,
    fall back to SciPy's ``chi2.ppf`` for concrete (non-traced) inputs.
    Traced/JIT execution with ``df != 1`` on those versions is unsupported.

    Parameters
    ----------
    p : array-like
        Percentile value
    df : array-like
        Degrees of freedom

    Returns
    -------
    array-like
        Corresponding chi2 value to the percentile
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    df = np.asarray(df, dtype=float)
    z = jax.scipy.stats.norm.ppf((p + 1.0) / 2.0)
    q_df1 = z**2

    if hasattr(jax.scipy.special, "gammaincinv"):
        q_general = jax.scipy.special.gammaincinv(df / 2.0, p) * 2.0
        return np.where(df == 1.0, q_df1, q_general)

    try:
        p_host = onp.asarray(p)
        df_host = onp.asarray(df)
    except Exception as exc:
        raise NotImplementedError(
            "chi2ppf(df != 1) requires either jax.scipy.special.gammaincinv "
            "or concrete (non-traced) inputs for the SciPy fallback."
        ) from exc

    p_host, df_host = onp.broadcast_arrays(p_host, df_host)
    q_host = onp.asarray(np.broadcast_to(q_df1, p_host.shape))
    mask = df_host != 1.0
    if onp.any(mask):
        q_host = q_host.copy()
        q_host[mask] = scipy_chi2.ppf(p_host[mask], df_host[mask])
    return np.asarray(q_host)


def nsigma(chi2r_test, chi2r_true, ndof):
    """
    Parameters
    ----------
    chi2r_test: float
        Reduced chi-squared of test model.
    chi2r_true: float
        Reduced chi-squared of true model.
    ndof: int
        Number of degrees of freedom.

    Returns
    -------
    nsigma: float
        Detection significance.
    """

    q = jax.scipy.stats.chi2.cdf(ndof * chi2r_test / chi2r_true, ndof)
    p = np.clip(1.0 - q, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    nsigma = jax.scipy.stats.norm.ppf((p + 1.0) / 2.0)

    return nsigma


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

import jax.numpy as np
import jax

import numpy as onp

import equinox as eqx
import zodiax as zx

from .inference import (
    fisher_matrix as _fisher_matrix,
    hessian_matrix as _hessian_matrix,
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
    configured visibility/phase conventions.
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
            optional closure-phase indices, and convention flags.
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
                self.phi = np.array(phidata.data["VISPHI"], dtype=float)
                self.d_phi = np.array(phidata.data["VISERR"], dtype=float)
                self.i_cps1, self.i_cps2, self.i_cps3 = None, None, None

                self.cp_flag = False

            elif "OI_T3" in data_names:
                phidata = data["OI_T3"]
                self.phi = np.array(phidata.data["T3PHI"], dtype=float)
                self.d_phi = np.array(phidata.data["T3PHIERR"], dtype=float)

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

    def __repr__(self):
        """Return a compact string summary of the loaded interferometric data."""
        phname = "CP" if self.cp_flag else "Phi"
        visname = "V2" if self.v2_flag else "Vis"
        return (
            f"OIData(u={self.u}, v={self.v}, {phname}={self.phi}, d_{phname}={self.d_phi}, "
            f"{visname}={self.vis}, d_{visname}={self.d_vis}, "
            f"i_cps1={self.i_cps1}, i_cps2={self.i_cps2}, i_cps3={self.i_cps3})"
        )

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
        if self.v2_flag:
            return np.abs(cvis) ** 2
        else:
            return np.abs(cvis)

    def to_phases(self, cvis):
        """
        Convert complex visibilities to closure phases or absolute phases.
        """
        if self.cp_flag:
            return closure_phases(cvis, self.i_cps1, self.i_cps2, self.i_cps3)
        else:
            return np.rad2deg(np.angle(cvis))

    def model(self, model_object):
        """
        Compute the model visibilities and phases for the given model object.
        """
        cvis = model_object.model(self.u, self.v, self.wavel)
        return self.flatten_model(cvis)


class BinaryModelAngular(zx.Base):
    """
    Class for a binary star model.
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

    def __repr__(self):
        """Return a readable representation of binary angular parameters."""
        return f"BinaryModel(sep={self.sep}, pa={self.pa}, contrast={self.contrast})"

    def unpack_all(self):
        """
        Convenience function to unpack all data to be used in model functions.
        """
        return self.sep, self.pa, self.contrast

    def model(self, u, v, wavel):
        """
        Model for binary star system.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary_angular(uu, vv, self.sep, self.pa, self.contrast)


class BinaryModelCartesian(zx.Base):
    """
    Class for a binary star model.
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

    def __repr__(self):
        """Return a readable representation of binary Cartesian parameters."""
        return f"BinaryModelAngular(dra={self.dra}, pa={self.ddec}, flux={self.flux})"

    def unpack_all(self):
        """
        Convenience function to unpack all data to be used in model functions.
        """
        return self.dra, self.ddec, self.flux

    def model(self, u, v, wavel):
        """
        Model for binary star system.
        """
        uu, vv = u / wavel, v / wavel
        return cvis_binary(uu, vv, self.ddec, self.dra, self.flux)


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

    model_data = data_obj.model(model_class(**param_dict))
    data, errors = data_obj.flatten_data()

    return -0.5 * np.sum((data - model_data) ** 2 / errors**2)


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

    return -0.5 * np.sum((data - model_data) ** 2 / errors**2)


def laplace_cov(values, params, data_obj, model_class):
    """
    Calculate the uncertainty with the Laplace method from an optimized fit between a model and data object.

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
        Covariance matrix.
    """

    objective = lambda vals: -loglike(vals, params, data_obj, model_class)
    return _laplace_covariance(objective, np.asarray(values, dtype=float))


def laplace_contrast_uncertainty(flux, dra, ddec, data_obj, model_class):
    """
    Calculate the uncertainty with the Laplace method from an optimized fit between a model and data object.

    Parameters
    ----------
    flux : float
        Flux ratio value at which the local Laplace uncertainty is evaluated.
    dra : float
        Right ascension offset in mas.
    ddec : float
        Declination offset in mas.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    array-like
        Uncertainty in the contrast.
    """

    params = ["dra", "ddec", "flux"]  # TODO: make this more general

    objective = lambda f: -loglike(
        [dra, ddec, f], params, data_obj, model_class
    )
    hess = _hessian_matrix(objective, np.asarray(flux, dtype=float))
    cov = 1 / np.asarray(hess)
    return np.sqrt(cov)


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

    For ``df != 1``, this falls back to numpyro's gammaincinv backend when
    available.

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

    try:
        if float(onp.asarray(df)) == 1.0:
            z = jax.scipy.stats.norm.ppf((p + 1.0) / 2.0)
            return z**2
    except Exception:
        pass

    from numpyro.distributions.util import gammaincinv

    return gammaincinv(df / 2.0, p) * 2.0


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
    p = 1.0 - q
    nsigma = np.sqrt(chi2ppf(p, 1.0))

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
        Closure phases in degrees.

    """
    visphiall = np.rad2deg(np.angle(cvis))
    visphiall = np.mod(visphiall + 180.0, 360.0) - 180.0
    visphi = np.reshape(visphiall, (len(cvis), 1))
    cp = (
        visphi[np.array(index_cps1)]
        + visphi[np.array(index_cps2)]
        - visphi[np.array(index_cps3)]
    )
    out = np.reshape(np.mod(cp + 180.0, 360.0) - 180.0, len(index_cps1))
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

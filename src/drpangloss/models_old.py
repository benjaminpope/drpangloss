import warnings

from drpangloss import models as _models


_LEGACY_MODULE_MSG = (
    "drpangloss.models_old is deprecated and will be removed in a future "
    "release. Migrate to drpangloss.models and drpangloss.grid_fit APIs."
)


def _warn_legacy(symbol_name, replacement=None):
    msg = f"{symbol_name} from drpangloss.models_old is deprecated."
    if replacement is not None:
        msg += f" Use {replacement} instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)


warnings.warn(_LEGACY_MODULE_MSG, DeprecationWarning, stacklevel=2)


class OIData(_models.OIData):
    """Deprecated compatibility wrapper around :class:`drpangloss.models.OIData`."""

    def __init__(self, data):
        _warn_legacy("OIData", replacement="drpangloss.models.OIData")
        super().__init__(data)


class BinaryModelAngular(_models.BinaryModelAngular):
    """Deprecated compatibility wrapper around the modern binary model."""

    def __init__(self, sep, pa, contrast):
        _warn_legacy(
            "BinaryModelAngular",
            replacement="drpangloss.models.BinaryModelAngular",
        )
        super().__init__(sep, pa, contrast)


class BinaryModelCartesian(_models.BinaryModelCartesian):
    """Deprecated compatibility wrapper around the modern Cartesian model."""

    def __init__(self, dra, ddec, flux):
        _warn_legacy(
            "BinaryModelCartesian",
            replacement="drpangloss.models.BinaryModelCartesian",
        )
        super().__init__(dra, ddec, flux)


class GaussianDiskModel(_models.GaussianDiskModel):
    """Deprecated compatibility wrapper around :class:`drpangloss.models.GaussianDiskModel`."""

    def __init__(self, sigma, dra=0.0, ddec=0.0):
        _warn_legacy(
            "GaussianDiskModel",
            replacement="drpangloss.models.GaussianDiskModel",
        )
        super().__init__(sigma, dra=dra, ddec=ddec)


class HarmonixModel(_models.HarmonixModel):
    """Deprecated compatibility wrapper around :class:`drpangloss.models.HarmonixModel`."""

    def __init__(
        self,
        source,
        visibility_method="model",
        render_method="render",
        expects_wavelength_units=True,
        observation_time=None,
    ):
        _warn_legacy(
            "HarmonixModel",
            replacement="drpangloss.models.HarmonixModel",
        )
        super().__init__(
            source,
            visibility_method=visibility_method,
            render_method=render_method,
            expects_wavelength_units=expects_wavelength_units,
            observation_time=observation_time,
        )


HarmonixAdapter = _models.HarmonixAdapter


rad2mas = _models.rad2mas
mas2rad = _models.mas2rad
dtor = _models.dtor
i2pi = _models.i2pi


def cvis_binary_angular(u, v, sep, pa, contrast):
    _warn_legacy(
        "cvis_binary_angular",
        replacement="drpangloss.models.cvis_binary_angular",
    )
    return _models.cvis_binary_angular(u, v, sep, pa, contrast)


def cvis_binary(u, v, ddec, dra, planet):
    _warn_legacy("cvis_binary", replacement="drpangloss.models.cvis_binary")
    return _models.cvis_binary(u, v, ddec, dra, planet)


def cvis_gaussian_disk(u, v, sigma, dra=0.0, ddec=0.0):
    _warn_legacy(
        "cvis_gaussian_disk",
        replacement="drpangloss.models.cvis_gaussian_disk",
    )
    return _models.cvis_gaussian_disk(u, v, sigma, dra=dra, ddec=ddec)


def model_loglike(model_object, data_obj):
    _warn_legacy("model_loglike", replacement="drpangloss.models.model_loglike")
    return _models.model_loglike(model_object, data_obj)


def joint_prediction(params, observations, model_fn):
    _warn_legacy(
        "joint_prediction",
        replacement="drpangloss.models.joint_prediction",
    )
    return _models.joint_prediction(params, observations, model_fn)


def joint_loglike(params, observations, model_fn):
    _warn_legacy(
        "joint_loglike",
        replacement="drpangloss.models.joint_loglike",
    )
    return _models.joint_loglike(params, observations, model_fn)


def loglike(values, params, data_obj, model_class):
    _warn_legacy("loglike", replacement="drpangloss.models.loglike")
    return _models.loglike(values, params, data_obj, model_class)


def loglike_nosignal(values, params, data_obj, model_class):
    _warn_legacy(
        "loglike_nosignal",
        replacement="drpangloss.models.loglike_nosignal",
    )
    return _models.loglike_nosignal(values, params, data_obj, model_class)


def laplace_cov(values, params, data_obj, model_class):
    _warn_legacy("laplace_cov", replacement="drpangloss.models.laplace_cov")
    return _models.laplace_cov(values, params, data_obj, model_class)


def laplace_contrast_uncertainty(
    flux, dra, ddec, data_obj, model_class, params=None
):
    _warn_legacy(
        "laplace_contrast_uncertainty",
        replacement="drpangloss.models.laplace_contrast_uncertainty",
    )
    return _models.laplace_contrast_uncertainty(
        flux, dra, ddec, data_obj, model_class, params=params
    )


def laplace_parameter_uncertainty(
    values, params, data_obj, model_class, target_param
):
    _warn_legacy(
        "laplace_parameter_uncertainty",
        replacement="drpangloss.models.laplace_parameter_uncertainty",
    )
    return _models.laplace_parameter_uncertainty(
        values, params, data_obj, model_class, target_param
    )


def fisher(values, params, data_obj, model_class, ridge=0.0):
    _warn_legacy("fisher", replacement="drpangloss.models.fisher")
    return _models.fisher(values, params, data_obj, model_class, ridge=ridge)


def chi2ppf(p, df):
    _warn_legacy("chi2ppf", replacement="drpangloss.models.chi2ppf")
    return _models.chi2ppf(p, df)


def nsigma(chi2r_test, chi2r_true, ndof):
    _warn_legacy("nsigma", replacement="drpangloss.models.nsigma")
    return _models.nsigma(chi2r_test, chi2r_true, ndof)


def closure_phases(cvis, index_cps1, index_cps2, index_cps3):
    _warn_legacy(
        "closure_phases",
        replacement="drpangloss.models.closure_phases",
    )
    return _models.closure_phases(cvis, index_cps1, index_cps2, index_cps3)


def cp_indices(vis_sta_index, cp_sta_index):
    _warn_legacy("cp_indices", replacement="drpangloss.models.cp_indices")
    return _models.cp_indices(vis_sta_index, cp_sta_index)


__all__ = [
    "OIData",
    "SourceModel",
    "BinaryModelAngular",
    "BinaryModelCartesian",
    "GaussianDiskModel",
    "HarmonixModel",
    "HarmonixAdapter",
    "rad2mas",
    "mas2rad",
    "dtor",
    "i2pi",
    "cvis_binary_angular",
    "cvis_binary",
    "cvis_gaussian_disk",
    "model_loglike",
    "joint_prediction",
    "joint_loglike",
    "loglike",
    "loglike_nosignal",
    "laplace_cov",
    "laplace_contrast_uncertainty",
    "laplace_parameter_uncertainty",
    "fisher",
    "chi2ppf",
    "nsigma",
    "closure_phases",
    "cp_indices",
]
    dra: float
        Right ascension offset of companion (mas).
    planet_contrast: float
        Relative flux of companion.
    xs: array
        x values of PPF.
    ppf_arr: array
        PPF values.
    ndof: int
        Number of degrees of freedom.
    sigma: int
        Confidence level for which the detection limits shall be computed.

    Returns
    -------
    res: float
        Maximum relative flux of companion.
    """

    _warn_legacy(
        "optimize_nsigma", replacement="drpangloss.grid_fit.absil_limits"
    )

    sol = optx.compat.minimize(
        nsigma_wrap,
        method="BFGS",
        x0=jnp.array([planet_contrast]),
        args=(
            u,
            v,
            cp,
            d_cp,
            vis2,
            d_vis2,
            i_cps1,
            i_cps2,
            i_cps3,
            ddec,
            dra,
            xs,
            ppf_arr,
            ndof,
            sigma,
        ),
        options={"maxiter": 100},
    )

    res = sol.x

    return res


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

    _warn_legacy("nsigma", replacement="drpangloss.models.nsigma")

    q = stats.chi2.cdf(ndof * chi2r_test / chi2r_true, ndof)
    p = 1.0 - q
    nsigma = np.sqrt(stats.chi2.ppf(1.0 - p, 1.0))
    if p < 1e-15:
        nsigma = np.sqrt(stats.chi2.ppf(1.0 - 1e-15, 1.0))

    return nsigma


@jit
def chi2all(cp_modelr, v2_modelr, oidata, const=0.0):
    """Compute total chi-square from modeled closure phases and squared visibilities."""

    cp_obsr, vis2_obsr, cp_errr, vis2_errr = (
        oidata.phi,
        oidata.vis,
        oidata.d_phi,
        oidata.d_vis,
    )
    # chi2

    chi2_closurer = jnp.sum((cp_obsr - cp_modelr.flatten()) ** 2 / cp_errr**2)

    chi2_v2r = jnp.sum((vis2_obsr - v2_modelr.flatten()) ** 2 / (vis2_errr**2))

    return (chi2_closurer + chi2_v2r) + const


@jit
def chi2_suball(oidata, cont, vis_in, imsum, ddec, dra):
    """Compute chi-square for a binary-plus-input-visibility composite model."""
    u21, v21 = oidata.u / oidata.wavel, oidata.v / oidata.wavel
    i_cps121, i_cps221, i_cps321 = oidata.i_cps1, oidata.i_cps2, oidata.i_cps3
    cont = 10**cont
    cvis_t211 = vis_binary2(
        u21,
        v21,
        ddec=ddec,
        dra=dra,
        p2=cont / (1.0 + cont + imsum),
        p3=1.0 / (1.0 + cont + imsum),
    )
    cvis_t211 += vis_in / (1 + cont + imsum)
    cp_model_t211 = closure_phases(cvis_t211, i_cps121, i_cps221, i_cps321)
    return chi2all(cp_model_t211, jnp.abs(cvis_t211) ** 2, oidata)


def lim_absil(f0, oidata, ddec, dra, chi2_true, ndof, sigma=3):
    """
    Parameters
    ----------
    f0: float
        Relative flux of companion.
    func: method
        Method to compute chi-squared.
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion.
        p0[2]: float
            Declination offset of companion.
        p0[3]: float
            Uniform disk diameter (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list
        contains one data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    chi2r_true: float
        Reduced chi-squared of true model.
    ndof: int
        Number of degrees of freedom.
    sigma: int
        Confidence level for which the detection limits shall be computed.

    Returns
    -------
    chi2: float
        Chi-squared of Absil method.
    """

    _warn_legacy("lim_absil", replacement="drpangloss.grid_fit.absil_limits")

    chi2_test = chi2_suball(
        oidata, f0, vis_in=0.0, imsum=0.0, ddec=ddec, dra=dra
    )
    nsigmavar = nsigma(
        chi2r_test=chi2_test / ndof, chi2r_true=chi2_true / ndof, ndof=ndof
    )

    return np.abs(nsigmavar - sigma) ** 2

import jax.numpy as jnp
from jax import jit, vmap
import jax
import warnings

import numpy as np

import optimistix as optx
import equinox as eqx
import zodiax as zx
from functools import partial


rad2mas = 180.0 / jnp.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = jnp.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad

dtor = np.pi / 180.0
i2pi = 1j * 2.0 * np.pi


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
    This is a legacy implementation retained for compatibility with older
    workflows.
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
            visibility/phase arrays and associated metadata.
        """

        _warn_legacy("OIData", replacement="drpangloss.models.OIData")

        try:
            # assume data is an oifits file opened with pyoifits
            data_names = [d.name for d in data.get_dataHDUs()]
            assert "OI_VIS" in data_names or "OI_VIS2" in data_names, (
                "No visibility data found in OIFITS file"
            )
            assert "OI_T3" in data_names or "OI_PHI" in data_names, (
                "No phase data found in OIFITS file"
            )

            # get the data from the oifits file
            self.wavel = jnp.array(
                data[1].data["EFF_WAVE"], dtype=float
            )  # note that for AMI this is scalar but for CHARA it is an array

            # if square visibilities are available, get them, otherwise get unsquared visibilities
            if "OI_VIS2" in data_names:
                visdata = data["OI_VIS2"]
                self.vis = jnp.array(visdata.data["VIS2DATA"], dtype=float)
                self.d_vis = jnp.array(visdata.data["VIS2ERR"], dtype=float)
                vis_sta_index = visdata.data["STA_INDEX"]

                self.u, self.v = (
                    jnp.array(visdata.data["UCOORD"], dtype=float),
                    jnp.array(visdata.data["VCOORD"], dtype=float),
                )

                self.v2_flag = True

            elif "OI_VIS" in data_names:
                visdata = data["OI_VIS"]
                self.vis = jnp.array(visdata.data["VISPHI"], dtype=float)
                self.d_vis = jnp.array(visdata.data["VISERR"], dtype=float)
                self.u, self.v = (
                    jnp.array(visdata.data["UCOORD"], dtype=float),
                    jnp.array(visdata.data["VCOORD"], dtype=float),
                )
                vis_sta_index = jnp.array(visdata.data["STA_INDEX"], dtype=int)

                self.v2_flag = False

            # if absolute phases are available, get them, otherwise get closure phases
            if "OI_PHI" in data_names:
                phidata = data["OI_PHI"]
                self.phi = jnp.array(phidata.data["VISPHI"], dtype=float)
                self.d_phi = jnp.array(phidata.data["VISERR"], dtype=float)
                self.i_cps1, self.i_cps2, self.i_cps3 = None, None, None

                self.cp_flag = True

            elif "OI_T3" in data_names:
                phidata = data["OI_T3"]
                self.phi = jnp.array(phidata.data["T3PHI"], dtype=float)
                self.d_phi = jnp.array(phidata.data["T3PHIERR"], dtype=float)

                cp_sta_index = jnp.array(phidata.data["STA_INDEX"], dtype=int)
                self.i_cps1, self.i_cps2, self.i_cps3 = cp_indices(
                    vis_sta_index, cp_sta_index
                )

                self.cp_flag = True

        except:
            # assume data is a dict of the form {'u':u,'v':v,'wavel':wavel,'vis':vis,'d_vis':d_vis,
            #'phi':phi,'d_phi':d_phi,'i_cps1':i_cps1,'i_cps2':i_cps2,'i_cps3':i_cps3,'v2_flag':v2_flag,'cp_flag':cp_flag}

            self.u = jnp.array(data["u"], dtype=float)
            self.v = jnp.array(data["v"], dtype=float)
            self.wavel = jnp.array(data["wavel"], dtype=float)

            self.vis = jnp.array(data["vis"], dtype=float)
            self.d_vis = jnp.array(data["d_vis"], dtype=float)

            self.phi = jnp.array(data["phi"], dtype=float)
            self.d_phi = jnp.array(data["d_phi"], dtype=float)

            self.i_cps1 = jnp.array(data["i_cps1"], dtype=int)
            self.i_cps2 = jnp.array(data["i_cps2"], dtype=int)
            self.i_cps3 = jnp.array(data["i_cps3"], dtype=int)

            self.v2_flag = data["v2_flag"]
            self.cp_flag = data["cp_flag"]

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
        return jnp.concatenate([self.vis, self.phi]), jnp.concatenate(
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
            Concatenated visibility and phase model vector.
        """

        return jnp.concatenate([self.to_vis(cvis), self.to_phases(cvis)])

    def to_vis(self, cvis):
        """
        Convert complex visibilities to visibilities or squared visibilities.
        """
        if self.v2_flag:
            return jnp.abs(cvis) ** 2
        else:
            return jnp.abs(cvis)

    def to_phases(self, cvis):
        """
        Convert complex visibilities to closure phases or absolute phases.
        """
        if self.cp_flag:
            return closure_phases(cvis, self.i_cps1, self.i_cps2, self.i_cps3)
        else:
            jnp.angle(cvis)

    def model(self, model_object):
        """
        Compute the model visibilities and phases for the given model object.
        """
        cvis = model_object.model(self.u, self.v, self.wavel)
        return self.flatten_model(cvis)


class BinaryModelAngular(zx.Base):
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
            Contrast ratio ``star/companion``.

        """

        _warn_legacy(
            "BinaryModelAngular",
            replacement="drpangloss.models.BinaryModelAngular",
        )

        self.sep = jnp.asarray(sep, dtype=float)
        self.pa = jnp.asarray(pa, dtype=float)
        self.contrast = jnp.asarray(contrast, dtype=float)

    def __repr__(self):
        """Return a readable representation of binary angular parameters."""
        return f"BinaryModel(sep={self.sep}, pa={self.pa}, contrast={self.contrast})"

    def unpack_all(self):
        """
        Return all model parameters in angular form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(sep, pa, contrast)``.
        """
        return self.sep, self.pa, self.contrast

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


class BinaryModelCartesian(zx.Base):
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
            Flux ratio of the companion.

        """

        _warn_legacy(
            "BinaryModelCartesian",
            replacement="drpangloss.models.BinaryModelCartesian",
        )

        self.dra = jnp.asarray(dra, dtype=float)
        self.ddec = jnp.asarray(ddec, dtype=float)
        self.flux = jnp.asarray(flux, dtype=float)

    def __repr__(self):
        """Return a readable representation of binary Cartesian parameters."""
        return f"BinaryModelAngular(dra={self.dra}, pa={self.ddec}, flux={self.flux})"

    def unpack_all(self):
        """
        Return all model parameters in Cartesian form.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            Tuple ``(dra, ddec, flux)``.
        """
        return self.dra, self.ddec, self.flux

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

    ddec = mas2rad * (sep * jnp.cos(th))
    dra = -1 * mas2rad * (sep * jnp.sin(th))

    # decompose into two "luminosity"
    l2 = 1.0 / (contrast + 1)
    l1 = 1 - l2

    # phase-factor
    phi = jnp.exp(-i2pi * (u * dra + v * ddec))
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
    phi_r = jnp.cos(-2 * np.pi * (u * dra + v * ddec))
    phi_i = jnp.sin(-2 * np.pi * (u * dra + v * ddec))

    cvis = p3 + p2 * phi_r + p2 * phi_i * 1.0j

    return cvis


@jit
def vis_binary2(u, v, ddec, dra, p2, p3):
    # adapted from pymask
    """Compute complex visibilities for explicit star/planet flux fractions.

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
    p2 : float or array-like
        Companion flux fraction.
    p3 : float or array-like
        Primary flux fraction.

    Returns
    -------
    array-like
        Complex visibility samples.
    """

    _warn_legacy("vis_binary2", replacement="No direct replacement")

    # relative locations
    ddec = (ddec) * np.pi / (180.0 * 3600.0 * 1000.0)
    dra = (dra) * np.pi / (180.0 * 3600.0 * 1000.0)
    phi_r = jnp.cos(-2 * np.pi * (u * dra + v * ddec))
    phi_i = jnp.sin(-2 * np.pi * (u * dra + v * ddec))

    cvis = p3 + p2 * phi_r + p2 * phi_i * 1.0j

    return cvis


@jit
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
    real = jnp.real(cvis)
    imag = jnp.imag(cvis)
    visphiall = jnp.arctan2(imag, real)
    visphiall = jnp.mod(visphiall + 10980.0, 360.0) - 180.0
    visphi = jnp.reshape(visphiall, (len(cvis), 1))
    cp = (
        visphi[jnp.array(index_cps1)]
        + visphi[jnp.array(index_cps2)]
        - visphi[jnp.array(index_cps3)]
    )
    out = jnp.reshape(cp * 180 / np.pi, len(index_cps1))
    return out


def log_like_binary(
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
    planet_contrast,
):
    # adapted from pymask
    """Compute the Gaussian log-likelihood for a binary model.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units.
    cp : array-like
        Observed closure phases in degrees.
    d_cp : array-like
        Closure-phase uncertainties in degrees.
    vis2 : array-like
        Observed squared visibilities.
    d_vis2 : array-like
        Squared-visibility uncertainties.
    i_cps1, i_cps2, i_cps3 : array-like
        Closure-phase baseline indices.
    ddec : float
        Declination offset in milliarcseconds.
    dra : float
        Right-ascension offset in milliarcseconds.
    planet_contrast : float
        Companion flux ratio.

    Returns
    -------
    float
        Log-likelihood value.
    """

    _warn_legacy("log_like_binary", replacement="drpangloss.models.loglike")

    cvis_model = cvis_binary(u, v, ddec, dra, planet_contrast)

    # calculate model observables
    cp_obs = closure_phases(cvis_model, i_cps1, i_cps2, i_cps3)
    vis2_obs = jnp.abs(cvis_model) ** 2

    ll_cp = jnp.sum((cp_obs - cp) ** 2 / d_cp**2)
    ll_vis2 = jnp.sum((vis2_obs - vis2) ** 2 / d_vis2**2)

    return -0.5 * (ll_cp + ll_vis2)


def chi2_binary(
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
    planet_contrast,
):
    # adapted from pymask
    """Compute a scaled chi-square proxy from ``log_like_binary``.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units.
    cp : array-like
        Observed closure phases in degrees.
    d_cp : array-like
        Closure-phase uncertainties in degrees.
    vis2 : array-like
        Observed squared visibilities.
    d_vis2 : array-like
        Squared-visibility uncertainties.
    i_cps1, i_cps2, i_cps3 : array-like
        Closure-phase baseline indices.
    ddec : float
        Declination offset in milliarcseconds.
    dra : float
        Right-ascension offset in milliarcseconds.
    planet_contrast : float
        Companion flux ratio.

    Returns
    -------
    float
        Half-scaled negative log-likelihood value.
    """

    _warn_legacy("chi2_binary", replacement="No direct replacement")

    return -0.5 * (
        log_like_binary(
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
            planet_contrast,
        )
    )


def log_like_star(cp, d_cp, vis2, d_vis2):
    """Compute the Gaussian log-likelihood for a single unresolved star.

    Parameters
    ----------
    cp : array-like
        Observed closure phases in degrees.
    d_cp : array-like
        Closure-phase uncertainties in degrees.
    vis2 : array-like
        Observed squared visibilities.
    d_vis2 : array-like
        Squared-visibility uncertainties.

    Returns
    -------
    float
        Log-likelihood value under a null-companion model.
    """

    _warn_legacy(
        "log_like_star", replacement="drpangloss.models.loglike_nosignal"
    )

    # calculate model observables
    cp_obs = 0.0
    vis2_obs = 1.0

    ll_cp = jnp.sum((cp_obs - cp) ** 2 / d_cp**2)
    ll_vis2 = jnp.sum((vis2_obs - vis2) ** 2 / d_vis2**2)

    return -0.5 * (ll_cp + ll_vis2)


def log_like_wrap(
    planet_contrast,
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
):
    """Wrapper returning the negative binary log-likelihood for scalar optimization."""
    _warn_legacy("log_like_wrap", replacement="No direct replacement")
    return -log_like_binary(
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
        planet_contrast,
    )


def optimize_log_like(
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
    planet_contrast,
):
    """Optimize binary contrast by minimizing ``log_like_wrap`` with BFGS."""
    _warn_legacy(
        "optimize_log_like",
        replacement="drpangloss.grid_fit.optimized_contrast_grid",
    )
    sol = optx.compat.minimize(
        log_like_wrap,
        method="BFGS",
        x0=jnp.array([planet_contrast]),
        args=(u, v, cp, d_cp, vis2, d_vis2, i_cps1, i_cps2, i_cps3, ddec, dra),
        options={"maxiter": 100},
    )
    res = sol.x
    return res


# define a function to find the contrast that maximizes the log likelihood
vmap_fun = partial(
    vmap(
        optimize_log_like,
        in_axes=(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            0,
        ),
    )
)
optimize_log_like_map = jit(vmap_fun)


# calc sigma with laplace approximation
def sigma(
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
    planet_contrast,
):
    """Estimate contrast uncertainty from the Hessian (Laplace approximation)."""
    _warn_legacy(
        "sigma", replacement="drpangloss.models.laplace_contrast_uncertainty"
    )
    hess = jax.hessian(log_like_binary, argnums=[11])(
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
        planet_contrast,
    )
    cov = -jnp.linalg.inv(jnp.array(hess))

    return jnp.sqrt(cov)


def cp_indices(vis_sta_index, cp_sta_index):
    """Map closure-triangle station indices to baseline indices."""
    _warn_legacy("cp_indices", replacement="drpangloss.models.cp_indices")
    vis_sta_index, cp_sta_index = (
        np.array(vis_sta_index, dtype=int),
        np.array(cp_sta_index, dtype=int),
    )
    i_cps1 = np.zeros(len(np.array(cp_sta_index)), dtype=int)
    i_cps2 = np.zeros(len(np.array(cp_sta_index)), dtype=int)
    i_cps3 = np.zeros(len(np.array(cp_sta_index)), dtype=int)

    for i in range(len(cp_sta_index)):
        i_cps1[i] = np.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][1] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps2[i] = np.argwhere(
            (cp_sta_index[i][1] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
        i_cps3[i] = np.argwhere(
            (cp_sta_index[i][0] == vis_sta_index[:, 0])
            & (cp_sta_index[i][2] == vis_sta_index[:, 1])
        )[0, 0]
    return (
        np.array(i_cps1, dtype=int),
        np.array(i_cps2, dtype=int),
        np.array(i_cps3, dtype=int),
    )


def nsigma_wrap(
    planet_contrast,
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
):
    """Objective for solving the contrast that matches a target sigma threshold."""

    _warn_legacy("nsigma_wrap", replacement="drpangloss.grid_fit.absil_limits")

    # constraints
    planet_contrast = jnp.where(planet_contrast < 1e-6, 1e-6, planet_contrast)
    planet_contrast = jnp.where(planet_contrast > 1.0, 1.0, planet_contrast)

    chi2_s = (
        chi2_binary(
            u, v, cp, d_cp, vis2, d_vis2, i_cps1, i_cps2, i_cps3, 0.0, 0.0, 0.0
        )
        / ndof
    )
    chi2_b = (
        chi2_binary(
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
            planet_contrast,
        )
        / ndof
    )

    q = jsp.stats.chi2.cdf(ndof * chi2_b / chi2_s, ndof)
    p = 1.0 - q

    nsigma = jnp.sqrt(jnp.interp(p, xs, ppf_arr))

    nsigma_overflow = jnp.sqrt(jnp.interp(1e-15, xs, ppf_arr))

    nsigmavar = jnp.where(p < 1e-15, nsigma_overflow, nsigma)

    return (sigma - nsigmavar) ** 2


def optimize_nsigma(
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
    planet_contrast,
    xs,
    ppf_arr,
    ndof,
    sigma,
):
    """


    Parameters
    ----------
    oidata: object
        Observational data, including:
        - u: array
            Baselines coordinates.
        - v: array
            Baselines coordinates.
        - cp: array
            Closure phases.
        - d_cp: array
            Closure phase uncertainties.
        - vis2: array
            Squared visibilities.
        - d_vis2: array
            Squared visibility uncertainties.
        - i_cps1: array
            Indices of closure phases for triangle 1.
        - i_cps2: array
            Indices of closure phases for triangle 2.
        - i_cps3: array
            Indices of closure phases for triangle 3.
    ddec: float
        Declination offset of companion (mas).
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

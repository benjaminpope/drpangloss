"""Low-level numerical helpers shared across ``drpangloss.models``.

The Bessel functions of the first kind are based on the
[CEPHES](https://www.netlib.org/cephes/) implementation, and the JAX
translations are adopted (almost) verbatim from
[Harmonix](https://github.com/shashankdholakia/harmonix).
"""

from functools import partial

import jax.numpy as np
from jax import jit
from jax.lax import scan

# === CONSTANTS ===

rad2mas = 180.0 / np.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = np.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad
dtor = np.pi / 180.0  # convert deg to rad

i2pi = 1j * 2.0 * np.pi

# ===


# === BESSEL FUNCTIONS OF THE FIRST KIND, BASED ON THE CEPHES IMPLEMENTATION ===

RP1 = np.array(
    [
        -8.99971225705559398224e8,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    ]
)
RQ1 = np.array(
    [
        1.0,
        6.20836478118054335476e2,
        2.56987256757748830383e5,
        8.35146791431949253037e7,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    ]
)

PP1 = np.array(
    [
        7.62125616208173112003e-4,
        7.31397056940917570436e-2,
        1.12719608129684925192e0,
        5.11207951146807644818e0,
        8.42404590141772420927e0,
        5.21451598682361504063e0,
        1.00000000000000000254e0,
    ]
)
PQ1 = np.array(
    [
        5.71323128072548699714e-4,
        6.88455908754495404082e-2,
        1.10514232634061696926e0,
        5.07386386128601488557e0,
        8.39985554327604159757e0,
        5.20982848682361821619e0,
        9.99999999999999997461e-1,
    ]
)

QP1 = np.array(
    [
        5.10862594750176621635e-2,
        4.98213872951233449420e0,
        7.58238284132545283818e1,
        3.66779609360150777800e2,
        7.10856304998926107277e2,
        5.97489612400613639965e2,
        2.11688757100572135698e2,
        2.52070205858023719784e1,
    ]
)
QQ1 = np.array(
    [
        1.0,
        7.42373277035675149943e1,
        1.05644886038262816351e3,
        4.98641058337653607651e3,
        9.56231892404756170795e3,
        7.99704160447350683650e3,
        2.82619278517639096600e3,
        3.36093607810698293419e2,
    ]
)

YP1 = np.array(
    [
        1.26320474790178026440e9,
        -6.47355876379160291031e11,
        1.14509511541823727583e14,
        -8.12770255501325109621e15,
        2.02439475713594898196e17,
        -7.78877196265950026825e17,
    ]
)
YQ1 = np.array(
    [
        5.94301592346128195359e2,
        2.35564092943068577943e5,
        7.34811944459721705660e7,
        1.87601316108706159478e10,
        3.88231277496238566008e12,
        6.20557727146953693363e14,
        6.87141087355300489866e16,
        3.97270608116560655612e18,
    ]
)

Z1 = 1.46819706421238932572e1
Z2 = 4.92184563216946036703e1
PIO4 = 0.78539816339744830962  # pi/4
THPIO4 = 2.35619449019234492885  # 3*pi/4
SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)


def j1_small(x):
    z = x * x
    w = np.polyval(RP1, z) / np.polyval(RQ1, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w


def j1_large_c(x):
    w = 5.0 / x
    z = w * w
    p = np.polyval(PP1, z) / np.polyval(PQ1, z)
    q = np.polyval(QP1, z) / np.polyval(QQ1, z)
    xn = x - THPIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    return p * SQ2OPI / np.sqrt(x)


def j1(x):
    """Bessel function of order one, translated from the CEPHES implementation."""
    return np.sign(x) * np.where(
        np.abs(x) < 5.0, j1_small(np.abs(x)), j1_large_c(np.abs(x))
    )


PP0 = np.array(
    [
        7.96936729297347051624e-4,
        8.28352392107440799803e-2,
        1.23953371646414299388e0,
        5.44725003058768775090e0,
        8.74716500199817011941e0,
        5.30324038235394892183e0,
        9.99999999999999997821e-1,
    ]
)
PQ0 = np.array(
    [
        9.24408810558863637013e-4,
        8.56288474354474431428e-2,
        1.25352743901058953537e0,
        5.47097740330417105182e0,
        8.76190883237069594232e0,
        5.30605288235394617618e0,
        1.00000000000000000218e0,
    ]
)

QP0 = np.array(
    [
        -1.13663838898469149931e-2,
        -1.28252718670509318512e0,
        -1.95539544257735972385e1,
        -9.32060152123768231369e1,
        -1.77681167980488050595e2,
        -1.47077505154951170175e2,
        -5.14105326766599330220e1,
        -6.05014350600728481186e0,
    ]
)
QQ0 = np.array(
    [
        1.0,
        6.43178256118178023184e1,
        8.56430025976980587198e2,
        3.88240183605401609683e3,
        7.24046774195652478189e3,
        5.93072701187316984827e3,
        2.06209331660327847417e3,
        2.42005740240291393179e2,
    ]
)

YP0 = np.array(
    [
        1.55924367855235737965e4,
        -1.46639295903971606143e7,
        5.43526477051876500413e9,
        -9.82136065717911466409e11,
        8.75906394395366999549e13,
        -3.46628303384729719441e15,
        4.42733268572569800351e16,
        -1.84950800436986690637e16,
    ]
)
YQ0 = np.array(
    [
        1.04128353664259848412e3,
        6.26107330137134956842e5,
        2.68919633393814121987e8,
        8.64002487103935000337e10,
        2.02979612750105546709e13,
        3.17157752842975028269e15,
        2.50596256172653059228e17,
    ]
)

DR10 = 5.78318596294678452118e0
DR20 = 3.04712623436620863991e1

RP0 = np.array(
    [
        -4.79443220978201773821e9,
        1.95617491946556577543e12,
        -2.49248344360967716204e14,
        9.70862251047306323952e15,
    ]
)
RQ0 = np.array(
    [
        1.0,
        4.99563147152651017219e2,
        1.73785401676374683123e5,
        4.84409658339962045305e7,
        1.11855537045356834862e10,
        2.11277520115489217587e12,
        3.10518229857422583814e14,
        3.18121955943204943306e16,
        1.71086294081043136091e18,
    ]
)


def j0_small(x):
    """Implementation of J0 for x < 5."""
    z = x * x
    p = (z - DR10) * (z - DR20)
    p = p * np.polyval(RP0, z) / np.polyval(RQ0, z)
    return np.where(x < 1e-5, 1 - z / 4.0, p)


def j0_large(x):
    """Implementation of J0 for x >= 5."""
    w = 5.0 / x
    q = 25.0 / (x * x)
    p = np.polyval(PP0, q) / np.polyval(PQ0, q)
    q = np.polyval(QP0, q) / np.polyval(QQ0, q)
    xn = x - PIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    return p * SQ2OPI / np.sqrt(x)


def j0(x):
    """Implementation of J0 for all x in Jax."""
    return np.where(np.abs(x) < 5.0, j0_small(np.abs(x)), j0_large(np.abs(x)))


# Modified from Harmonix implementation to return only J0 if called with n=0
# (previously returned stacked J0 and J1 results for this case).
@partial(jit, static_argnums=0)
def bessel_jn(n, x):
    """Compute the Bessel function $J_n(x)$, for $n >= 0$. Returns the function output
    for all orders up to the requested order $n$ evaluated for the kernel $x$, where the
    the different orders are stacked along the first axis. The shape of the final result
    is thus (n + 1, shape(x))."""
    if n == 0:
        return np.array([j0(x)])
    else:
        # use recurrence relations
        def body(carry, i):
            jnm1, jn = carry
            jnplus = (2 * i) / x * jn - jnm1
            return (jn, jnplus), jnplus

        j0_val, j1_val = j0(x), j1(x)
        _, jn = scan(body, (j0_val, j1_val), np.arange(1, n))

        return np.concatenate((np.array([j0_val]), np.array([j1_val]), jn))


# ===


# === ELLIPTICAL TRANSFORM HELPERS ===
#
# Position angles follow the on-sky convention used throughout drpangloss:
# North is +y (up), East is +x (left, i.e. x decreases left-to-right across
# an image array's columns), and position angle is measured North to East,
# i.e. counter-clockwise from the top in a plot with East to the left. This
# matches the coordinate grid produced by ``models._image_coordinates``.


def undo_elliptical_transf_spat_freq(u, v, pa, stretch):
    """When considering an ellipticaly rotated and stretched (along the apparent minor
    axis) object, this function takes spatial frequency coordinates, and transforms
    them to spatial frequencies in the frame of reference where an elliptical object
    appears circular and is aligned with the major axis pointing North.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East (i.e. counter-clockwise in conventional astronomical image orientation)
        in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $u$ and $v$ spatial frequencies, but in a frame of reference where the
        rotation and stretch of the original elliptical transformation is undone.
    """
    pa_rad = pa * dtor

    ut = (u * np.cos(pa_rad) - v * np.sin(pa_rad)) / stretch
    vt = u * np.sin(pa_rad) + v * np.cos(pa_rad)

    return ut, vt


def apply_elliptical_transf_spat_freq(u, v, pa, stretch):
    """Inverse of :func:`undo_elliptical_transf_spat_freq`. Takes spatial frequency
    coordinates in the frame of reference where an elliptical object appears circular
    and aligned with its major axis pointing North, and transforms them into the
    original (rotated and stretched) frame of reference.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units, in the de-rotated/unstretched
        frame of reference.
    v : array-like
        Baseline ``v`` coordinates in wavelength units, in the de-rotated/unstretched
        frame of reference.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East, in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $u$ and $v$ spatial frequencies in the original (rotated and stretched)
        frame of reference.
    """
    pa_rad = pa * dtor

    u_scaled = u * stretch
    v_scaled = v

    ut = u_scaled * np.cos(pa_rad) + v_scaled * np.sin(pa_rad)
    vt = -u_scaled * np.sin(pa_rad) + v_scaled * np.cos(pa_rad)

    return ut, vt


def undo_elliptical_transf_coord(x, y, pa, stretch):
    """When considering an ellipticaly rotated and stretched (along the apparent minor
    axis) object, this function takes spatial coordinates, and transforms
    them to coordinates in the frame of reference where the elliptical object
    appears circular and is aligned with the major axis pointing North.

    Parameters
    ----------
    x : array-like
        Spatial coordinates along the x-axis (East-positive) in milliarcseconds.
    y : array-like
        Spatial coordinates along the y-axis (North-positive) in milliarcseconds.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East (i.e. counter-clockwise in conventional astronomical image orientation)
        in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $x$ and $y$ spatial coordinates, but in a frame of reference where the
        rotation and stretch of the original elliptical transformation is undone.
    """
    pa_rad = pa * dtor

    xt = (x * np.cos(pa_rad) - y * np.sin(pa_rad)) / stretch
    yt = x * np.sin(pa_rad) + y * np.cos(pa_rad)

    return xt, yt


def apply_elliptical_transf_coord(x, y, pa, stretch):
    """Inverse of :func:`undo_elliptical_transf_coord`. Takes spatial coordinates in
    the frame of reference where an elliptical object appears circular and aligned
    with its major axis pointing North, and transforms them into the original
    (rotated and stretched) frame of reference.

    Parameters
    ----------
    x : array-like
        Spatial coordinates along the x-axis, in the de-rotated/unstretched frame
        of reference, in milliarcseconds.
    y : array-like
        Spatial coordinates along the y-axis, in the de-rotated/unstretched frame
        of reference, in milliarcseconds.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East, in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $x$ and $y$ spatial coordinates in the original (rotated and stretched)
        frame of reference, in milliarcseconds.
    """
    pa_rad = pa * dtor

    x_scaled = x * stretch
    y_scaled = y

    xt = x_scaled * np.cos(pa_rad) + y_scaled * np.sin(pa_rad)
    yt = -x_scaled * np.sin(pa_rad) + y_scaled * np.cos(pa_rad)

    return xt, yt


# ===


# === MISCELLANEOUS ===


def check_az_prof_nonnegative(az_amps, az_pas, tol=1e-6):
    r"""Returns `False` if $1 + f(theta)$ drops below 0 at any point, where $f$ is a
    harmonic series of form $I(r, \theta) = \sum_{m=0}^{n} A_m \cos{(m(\theta - \pa_m)$.
    This is done using a Laurent polynomial + companion matrix approach, and should
    thus be pretty quick.

    Parameters
    ----------
    az_amps : array-like
        1D array containing the azimuthal modulation order amplitudes, starting from
        order 1.
    az_pas : array-like
        1D array containing the azimuthal modulation order position angles, starting from
        order 1.

    Returns
    -------
    bool
        Whether the intensity profile $1 + f(\theta)$, where $f(\theta)$ is described
        by the azimuthal modulations, remains positive.
    """
    k = len(az_amps)
    deg = 2 * k

    # Orders start from 1 up to k
    idx = np.arange(1, k + 1)
    phases_terms_rad = az_pas * dtor * idx

    # Initialize polynomial coefficients (must be complex).
    coeffs = np.zeros(deg + 1) + 0j

    # Correctly aligned Fourier derivative polterms for z^k * f'(z) = 0.
    lower_vals = -0.5j * az_amps * idx * np.exp(1j * phases_terms_rad)
    upper_vals = 0.5j * az_amps * idx * np.exp(-1j * phases_terms_rad)

    # Set complex polynomial coefficients.
    coeffs = coeffs.at[k - np.arange(1, k + 1)].set(lower_vals)
    coeffs = coeffs.at[k + np.arange(1, k + 1)].set(upper_vals)

    # Prevent division-by-zero errors during matrix normalization.
    leading_coef = np.where(np.abs(coeffs[-1]) > 1e-6, coeffs[-1], 1.0 + 0.0j)
    coeffs_norm = coeffs / leading_coef

    # Build Companion Matrix (must be complex).
    companion_matrix = np.zeros((deg, deg)) + 0j
    if deg > 1:
        companion_matrix = companion_matrix.at[1:, :-1].set(np.eye(deg - 1))
    companion_matrix = companion_matrix.at[:, -1].set(-coeffs_norm[:-1])

    # Extract all complex points where derivative is 0.
    roots = np.linalg.eigvals(companion_matrix)
    angles = np.angle(roots)

    # Evaluate at these local minima.
    harmonics = az_amps * np.cos(idx * angles[:, None] - phases_terms_rad)
    f_at_peaks = np.sum(harmonics, axis=1)

    # Isolate valid peaks near unit circle.
    global_min = np.min(f_at_peaks)

    # Find global minimum of azimuthal profile.
    f_global_min = 1.0 + global_min

    return f_global_min > 0.0 - tol


# ===

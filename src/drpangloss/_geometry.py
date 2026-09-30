"""Sky-plane and uv-plane geometry shared by the source models.

Position angles follow the on-sky convention used throughout drpangloss:
North is +y (up), East is +x (left, i.e. x decreases left-to-right across
an image array's columns), and position angle is measured North to East,
i.e. counter-clockwise from the top in a plot with East to the left. This
matches the coordinate grid produced by :func:`image_coordinates`.
"""

import jax
import jax.numpy as np

from ._utils import dtor, mas2rad


def image_coordinates(npix, fov_mas):
    """Return Cartesian image-plane coordinates in milliarcseconds."""
    npix = int(npix)
    pixel_scale_mas = float(fov_mas) / npix
    pixel_indices = np.arange(npix)
    center = 0.5 * (npix - 1)
    x = -(pixel_indices - center) * pixel_scale_mas
    y = (center - pixel_indices) * pixel_scale_mas
    return np.meshgrid(x, y, indexing="xy")


def pixel_offsets(npix, pixel_scale_mas):
    """Sky offsets (mas) of the pixel centres along one image axis.

    Along columns this is ``dra``, decreasing to the right (East left); along
    rows it is ``ddec``, decreasing downwards (North up). Both are
    ``(centre - index) * pixel_scale_mas`` with the centre at index
    ``(npix - 1) / 2``, as in :func:`image_coordinates`.
    """
    return (0.5 * (npix - 1) - np.arange(npix)) * pixel_scale_mas


def image_visibilities(brightness, uu, vv, pixel_scale_mas):
    """Exact Fourier transform of a pixel image at arbitrary frequencies.

    Each pixel is treated as a point at its centre, so
    ``V(u, v) = sum_{row, col} I[row, col] exp(-2πi (u x_col + v y_row))``
    with the sign convention of :func:`offset_phase`. The sum is separable
    into two matrix products, done at ``Precision.HIGHEST`` (on A100/H100
    GPUs the default is TF32, with ~1e-3 relative error).

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes in the orientation of :func:`pixel_offsets` (East left,
        North up). Not normalised here.
    uu, vv : array-like
        Spatial frequencies, baseline / wavelength (per radian), of
        broadcastable shapes.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.

    Returns
    -------
    array-like
        Complex visibilities with the broadcast shape of ``uu`` and ``vv``.
    """
    brightness = np.asarray(brightness)
    uu, vv = np.broadcast_arrays(uu, vv)
    nrow, ncol = brightness.shape
    x = pixel_offsets(ncol, pixel_scale_mas)
    y = pixel_offsets(nrow, pixel_scale_mas)
    fu = mas2rad * np.ravel(uu)
    fv = mas2rad * np.ravel(vv)
    cols = np.exp(-2j * np.pi * np.outer(fu, x))
    rows = np.exp(-2j * np.pi * np.outer(fv, y))
    highest = jax.lax.Precision.HIGHEST
    partial = np.matmul(rows, brightness.astype(rows.dtype), precision=highest)
    return np.sum(partial * cols, axis=-1).reshape(np.shape(uu))


def offset_phase(uu, vv, dra, ddec):
    """Fourier shift factor for an offset of ``(dra, ddec)`` milliarcseconds."""
    arg = 2.0 * np.pi * mas2rad * (uu * dra + vv * ddec)
    return jax.lax.complex(np.cos(arg), -np.sin(arg))


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

    # Fourier similarity theorem: compressing the image by `stretch` dilates uv by 1/stretch.
    ut = (u * np.cos(pa_rad) - v * np.sin(pa_rad)) * stretch
    vt = u * np.sin(pa_rad) + v * np.cos(pa_rad)

    return ut, vt


def apply_elliptical_transf_spat_freq(u, v, pa, stretch):
    """Inverse of [`undo_elliptical_transf_spat_freq`][drpangloss._geometry.undo_elliptical_transf_spat_freq]. Takes spatial frequency
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

    u_scaled = u / stretch
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
    """Inverse of [`undo_elliptical_transf_coord`][drpangloss._geometry.undo_elliptical_transf_coord]. Takes spatial coordinates in
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


def check_az_prof_nonnegative(az_amps, az_pas, tol=1e-6):
    r"""Check that an azimuthal brightness modulation stays non-negative.

    The modulation is $1 + f(\theta)$ with
    $f(\theta) = \sum_{m=1}^{n} A_m \cos(m(\theta - \mathrm{pa}_m))$. Its
    minimum is found from the roots of $f'$, using a Laurent polynomial and
    its companion matrix, so the check is fast and exact.

    Parameters
    ----------
    az_amps : array-like
        1D array containing the azimuthal modulation order amplitudes, starting from
        order 1.
    az_pas : array-like
        1D array containing the azimuthal modulation order position angles, starting from
        order 1, in degrees.
    tol : float, optional
        Tolerance: minima down to ``-tol`` still count as non-negative.

    Returns
    -------
    bool or jax.Array
        Whether $1 + f(\theta) \geq -\mathrm{tol}$ everywhere. For array
        inputs this is a 0-d JAX boolean array; use ``bool()`` on it.
    """
    k = len(az_amps)
    if k == 0:
        # No modulation: the profile is the constant 1.
        return True
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

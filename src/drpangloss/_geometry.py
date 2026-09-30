"""Sky-plane and uv-plane geometry shared by the source models.

Position angles follow the on-sky convention used throughout drpangloss:
North is +y (up), East is +x (left, i.e. x decreases left-to-right across
an image array's columns), and position angle is measured North to East,
i.e. counter-clockwise from the top in a plot with East to the left. This
matches the coordinate grid produced by :func:`image_coordinates`.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

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


def image_visibilities(brightness, uu, vv, pixel_scale_mas, backend="dft"):
    """Fourier transform of a pixel image at arbitrary frequencies.

    Each pixel is treated as a point at its centre, so
    ``V(u, v) = sum_{row, col} I[row, col] exp(-2πi (u x_col + v y_row))``
    with the sign convention of :func:`offset_phase`.

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
    backend : {"dft", "nufft"}
        ``"dft"`` (default) is the exact sum, done as two matrix products
        at ``Precision.HIGHEST`` (on A100/H100 GPUs the default is TF32,
        with ~1e-3 relative error). ``"nufft"`` uses a non-uniform FFT from
        the optional ``jax-finufft`` package (``pip install
        'drpangloss[nufft]'``). It is approximate, with a requested
        relative tolerance ``eps`` of 1e-7 in float64 and 1e-5 in float32
        (FINUFFT's accuracy is relative to the 2-norm of the output, so
        single visibilities can do worse; ``tests/test_nufft.py`` checks
        ``3 eps`` per point for unit-sum images). It is faster for large
        images and many irregularly placed frequencies.

    Returns
    -------
    array-like
        Complex visibilities with the broadcast shape of ``uu`` and ``vv``.
    """
    brightness = np.asarray(brightness)
    uu, vv = np.broadcast_arrays(uu, vv)
    if backend == "dft":
        vis = _dft(brightness, np.ravel(uu), np.ravel(vv), pixel_scale_mas)
    elif backend == "nufft":
        fu, fv = mas2rad * np.ravel(uu), mas2rad * np.ravel(vv)
        vis = _nufft(brightness, fu, fv, pixel_scale_mas)
    else:
        raise ValueError(f"backend must be 'dft' or 'nufft', not {backend!r}.")
    return vis.reshape(np.shape(uu))


def _fourier_matrix(freq_per_rad, npix, pixel_scale_mas):
    """``exp(-2πi f x)`` for frequencies ``f`` and one axis's pixel offsets."""
    offsets = pixel_offsets(npix, pixel_scale_mas)
    return np.exp(-2j * np.pi * np.outer(mas2rad * freq_per_rad, offsets))


def _dft(brightness, uu, vv, pixel_scale_mas):
    nrow, ncol = brightness.shape
    cols = _fourier_matrix(uu, ncol, pixel_scale_mas)
    rows = _fourier_matrix(vv, nrow, pixel_scale_mas)
    highest = jax.lax.Precision.HIGHEST
    partial = np.matmul(rows, brightness.astype(rows.dtype), precision=highest)
    return np.sum(partial * cols, axis=-1)


def grid_visibilities(brightness, uu_axis, vv_axis, pixel_scale_mas):
    """Exact Fourier transform of a pixel image onto a regular uv grid.

    The two-sided matrix Fourier transform (Soummer et al. 2007,
    arXiv:0711.0368): ``V = A_v @ I @ A_u.T``, the same sum as
    :func:`image_visibilities` but costing ``nv·N² + nv·nu·N`` rather than
    ``nv·nu·N²`` for an N x N image. Done at ``Precision.HIGHEST``.

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes, oriented as in :func:`pixel_offsets`.
    uu_axis, vv_axis : array-like, shapes (nu,) and (nv,)
        The grid's spatial frequencies along each image axis, baseline /
        wavelength (per radian).
    pixel_scale_mas : float
        Pixel size in milliarcseconds.

    Returns
    -------
    array-like, shape (nv, nu)
        ``V[j, i]`` is the visibility at ``(uu_axis[i], vv_axis[j])``.
    """
    brightness = np.asarray(brightness)
    nrow, ncol = brightness.shape
    cols = _fourier_matrix(np.ravel(uu_axis), ncol, pixel_scale_mas)
    rows = _fourier_matrix(np.ravel(vv_axis), nrow, pixel_scale_mas)
    highest = jax.lax.Precision.HIGHEST
    partial = np.matmul(rows, brightness.astype(rows.dtype), precision=highest)
    return np.matmul(partial, cols.T, precision=highest)


def rotate(x, y, rotation_deg):
    """Rotate a frame by a position angle, North towards East.

    Returns the sky coordinates of the point ``(x, y)`` of a frame whose
    "up" axis points at position angle ``rotation_deg``: the frame's
    ``(0, 1)`` lands at ``(sin θ, cos θ)``. The same matrix maps
    frequencies, and ``rotate(..., -rotation_deg)`` inverts it.
    """
    c, s = np.cos(rotation_deg * dtor), np.sin(rotation_deg * dtor)
    return c * x + s * y, -s * x + c * y


class UVGrid(eqx.Module):
    """uv samples that lie on a regular lattice, possibly rotated on the sky.

    AMI data, for example, are sampled on the detector's Fourier grid,
    which is rotated on the sky by the parallactic angle. The sample ``k``
    is at grid-frame coordinates
    ``(u_axis[index[k] % nu], v_axis[index[k] // nu])`` (metres), and on
    the sky at ``rotate(u_g, v_g, rotation_deg)``. Found from the samples by
    :func:`find_uv_grid`.
    """

    u_axis: jax.Array
    v_axis: jax.Array
    index: jax.Array
    rotation_deg: float = eqx.field(static=True)


def find_uv_grid(u, v, tolerance=1e-6):
    """Return the lattice that ``(u, v)`` samples lie on, or ``None``.

    The lattice axes are found from the shortest separation between
    samples, and every sample must lie within ``tolerance`` of a cell of a
    lattice point (so that a Fourier transform onto the lattice is exact).

    Parameters
    ----------
    u, v : array-like
        Sample coordinates (concrete arrays, e.g. in metres).
    tolerance : float, optional
        Largest allowed offset from a lattice point, in cells.

    Returns
    -------
    UVGrid or None
        ``rotation_deg`` is in ``(-45, 45]``.
    """
    u, v = onp.asarray(u, dtype=float), onp.asarray(v, dtype=float)
    if u.ndim != 1 or u.size < 2:
        return None
    gap = onp.hypot(u - u[0], v - v[0])
    gap[0] = onp.inf
    nearest = int(onp.argmin(gap))
    angle = onp.degrees(onp.arctan2(v[nearest] - v[0], u[nearest] - u[0]))
    rotation = float(-(((angle + 45.0) % 90.0) - 45.0))
    c, s = onp.cos(onp.radians(rotation)), onp.sin(onp.radians(rotation))
    grid_u, grid_v = c * u - s * v, s * u + c * v  # rotate(u, v, -rotation)
    axes, cells = [], []
    for coord in (grid_u, grid_v):
        start = coord.min()
        steps = onp.diff(onp.sort(coord))
        steps = steps[steps > tolerance * gap[nearest]]
        pitch = steps.min() if steps.size else 1.0
        cell = (coord - start) / pitch
        if onp.max(onp.abs(cell - onp.round(cell))) > tolerance:
            return None
        cell = onp.round(cell).astype(int)
        axes.append(start + pitch * onp.arange(cell.max() + 1))
        cells.append(cell)
    index = cells[1] * axes[0].size + cells[0]
    if onp.unique(index).size != index.size:
        return None
    return UVGrid(
        np.asarray(axes[0]), np.asarray(axes[1]), np.asarray(index), rotation
    )


def _nufft(brightness, fu, fv, pixel_scale_mas):
    try:
        from jax_finufft import nufft2
    except ImportError as err:
        raise ImportError(
            "backend='nufft' needs jax-finufft: pip install "
            "'drpangloss[nufft]'."
        ) from err
    # finufft sums f[k] exp(+i k·t) over modes k = index - n // 2. Our pixel
    # offsets are (n - 1)/2 - index = -(k + n//2 - (n - 1)/2) pixels, so
    # t = 2π s f (in radians per pixel), and even n picks up a half-pixel
    # phase exp(+i t / 2) from the centre at (n - 1)/2 rather than n/2.
    source = brightness.astype(np.result_type(brightness.dtype, np.complex64))
    t_col = 2.0 * np.pi * pixel_scale_mas * fu
    t_row = 2.0 * np.pi * pixel_scale_mas * fv
    eps = 1e-7 if source.dtype == np.complex128 else 1e-5
    vis = nufft2(source, t_row, t_col, iflag=1, eps=eps)
    nrow, ncol = brightness.shape
    shift = 0.5 * ((1 - nrow % 2) * t_row + (1 - ncol % 2) * t_col)
    return vis * np.exp(1j * shift)


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

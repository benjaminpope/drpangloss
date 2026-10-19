"""Sky-plane and uv-plane geometry shared by the source models.

Position angles follow the on-sky convention used throughout virgil:
North is +y (up), East is +x (left, i.e. x decreases left-to-right across
an image array's columns), and position angle is measured North to East,
i.e. counter-clockwise from the top in a plot with East to the left. This
matches the coordinate grid produced by :func:`image_coordinates`.
"""

import warnings

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

    ``(centre - index) * pixel_scale_mas``, centre at ``(npix - 1) / 2``: ``dra``
    along columns (East left), ``ddec`` along rows (North up).
    """
    return (0.5 * (npix - 1) - np.arange(npix)) * pixel_scale_mas


def image_visibilities(brightness, uu, vv, pixel_scale_mas, ft="auto"):
    """Fourier transform of a pixel image at arbitrary frequencies.

    Each pixel is a point at its centre: ``V = Σ I[row, col] exp(-2πi (u x_col
    + v y_row))``, the sign convention of :func:`offset_phase`, done at
    ``Precision.HIGHEST`` (the A100/H100 default, TF32, is ~1e-3 accurate).
    On a regular uv grid :func:`grid_visibilities` is faster. Large problems
    use the optional ``nufftax`` transform (see ``ft``), accurate to a
    requested tolerance rather than exactly.

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes in the orientation of :func:`pixel_offsets`. Not
        normalized here.
    uu, vv : array-like
        Spatial frequencies, baseline / wavelength (per radian), of
        broadcastable shapes.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    ft : {"auto", "dft", "nufft"}
        ``"dft"`` is the exact sum; ``"nufft"`` needs ``pip install
        "virgil-astro[nufft]"``; ``"auto"`` picks the faster by problem size
        (:func:`_use_nufft`) and uses the DFT without nufftax.

    Returns
    -------
    array-like
        Complex visibilities with the broadcast shape of ``uu`` and ``vv``.
    """
    brightness = np.asarray(brightness)
    uu, vv = np.broadcast_arrays(uu, vv)
    shape = np.shape(uu)
    uu, vv = np.ravel(uu), np.ravel(vv)
    if ft not in ("auto", "dft", "nufft"):
        raise ValueError(f"ft must be 'auto', 'dft' or 'nufft', not {ft!r}.")
    if ft == "auto":
        ft = "nufft" if _use_nufft(brightness.shape, uu.size) else "dft"
    if ft == "nufft":
        vis = _nufft(brightness, uu, vv, pixel_scale_mas)
    else:
        vis = _dft(brightness, uu, vv, pixel_scale_mas)
    return vis.reshape(shape)


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


# virgil#75: nufftax (pure JAX) against the DFT, value+grad, f32. A100
# (DFT job 17776437, nufftax job 18762629; ozstar_scripts#35): no faster below
# M·N1·N2 ~ 1e8 or at 64², 4-50x above. Apple M1 Max CPU (2026-10-19,
# ozstar_scripts@4f4ad9e): nufftax's FFT floor grows with N² and the DFT loses
# from M ~ 150-400 points at every N from 64 to 1024, so the CPU rule is on M.
_NUFFT_GPU_WORK = 1e8
_NUFFT_GPU_MIN_SIDE = 128
_NUFFT_CPU_POINTS = 500
_NUFFT_CPU_MIN_SIDE = 64
# Requested relative tolerance (of the zero-baseline visibility). 1e-5 in f32
# is the f32 floor: as accurate as the f32 DFT (7e-6 to 9e-5), and 1.2-1.5x
# faster than 1e-6.
_NUFFT_EPS_F32 = 1e-5
_NUFFT_EPS_F64 = 1e-12


def _nufftax():
    """The nufftax package, with an install hint if it is missing."""
    try:
        import nufftax
    except ImportError as err:
        raise ImportError(
            'ft="nufft" needs nufftax: pip install "virgil-astro[nufft]".'
        ) from err
    return nufftax


def _nufftax_available():
    try:
        _nufftax()
    except ImportError:
        return False
    return True


def _use_nufft(shape, n_points):
    """Whether ``ft="auto"`` uses the NUFFT: large enough, and installed.

    Without nufftax a problem above the threshold warns once and uses the DFT.
    """
    nrow, ncol = shape
    backend = jax.default_backend()
    if backend == "gpu":
        big = (
            min(nrow, ncol) >= _NUFFT_GPU_MIN_SIDE
            and n_points * nrow * ncol >= _NUFFT_GPU_WORK
        )
    else:
        big = (
            min(nrow, ncol) >= _NUFFT_CPU_MIN_SIDE
            and n_points >= _NUFFT_CPU_POINTS
        )
    if not big:
        return False
    if _nufftax_available():
        return True
    warnings.warn(
        "nufftax would be faster for this image and number of points: "
        'pip install "virgil-astro[nufft]".',
        stacklevel=4,
    )
    return False


def _nufft(brightness, uu, vv, pixel_scale_mas, eps=None):
    """:func:`_dft` by nufftax's type-2 transform, to relative tolerance eps."""
    nufft2d2 = _nufftax().nufft2d2
    dtype = np.result_type(brightness.dtype, np.float32)
    if eps is None:
        eps = _NUFFT_EPS_F64 if dtype == np.float64 else _NUFFT_EPS_F32
    nrow, ncol = brightness.shape
    # x pairs with the last axis. V = Σ f[k] exp(+i k t) exp(+i t (1 - n%2)/2) over modes k = index - n//2,
    # t = 2π s f (rad per pixel): the DFT's centre is (n - 1)/2, not n//2.
    t_row = 2.0 * np.pi * mas2rad * pixel_scale_mas * vv.astype(dtype)
    t_col = 2.0 * np.pi * mas2rad * pixel_scale_mas * uu.astype(dtype)
    # nufftax wraps t modulo 2π, but the half-pixel phase on an even axis is
    # 4π-periodic: transform the wrapped t and apply that phase to the original.
    wrap = lambda t: t - 2.0 * np.pi * np.round(t / (2.0 * np.pi))  # noqa: E731
    source = brightness.astype(np.result_type(dtype, np.complex64))
    vis = nufft2d2(wrap(t_col), wrap(t_row), source, eps=eps, isign=1)
    shift = 0.5 * ((1 - nrow % 2) * t_row + (1 - ncol % 2) * t_col)
    return vis * np.exp(1j * shift)


def grid_visibilities(brightness, uu_axis, vv_axis, pixel_scale_mas):
    """Exact Fourier transform of a pixel image onto a regular uv grid.

    The two-sided matrix Fourier transform (Soummer et al. 2007),
    ``V = A_v @ I @ A_u.T``: the sum of :func:`image_visibilities`, but
    O(``nv·N² + nv·nu·N``) rather than O(``nv·nu·N²``).

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes, oriented as in :func:`pixel_offsets`.
    uu_axis, vv_axis : array-like, shapes (nu,) and (nv,)
        The grid's spatial frequencies along each image axis (per radian).
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
    """Sky coordinates of ``(x, y)`` in a frame whose "up" is at position angle ``rotation_deg``.

    The frame's ``(0, 1)`` lands at ``(sin θ, cos θ)``; ``-rotation_deg``
    inverts it, and the same matrix maps frequencies. NumPy-like inputs are
    rotated in NumPy at their own precision, JAX arrays and tracers in JAX.
    """
    xp = onp if _numpy_inputs(x, y, rotation_deg) else np
    if xp is onp:
        x, y, rotation_deg = (
            a
            if isinstance(a, (int, float, onp.ndarray, onp.generic))
            else onp.asarray(a)
            for a in (x, y, rotation_deg)
        )
    c, s = xp.cos(rotation_deg * dtor), xp.sin(rotation_deg * dtor)
    return c * x + s * y, -s * x + c * y


def _is_numpy_like(a):
    """Whether ``a`` is a number, NumPy, or array-like, but not JAX."""
    if isinstance(a, jax.Array):  # includes tracers
        return False
    if isinstance(a, (int, float, onp.ndarray, onp.generic)):
        return True
    if isinstance(a, (list, tuple)):
        return all(_is_numpy_like(b) for b in a)
    return hasattr(a, "__array__")


def _numpy_inputs(*values):
    """Whether every value is NumPy-like (numbers, lists, ``__array__``)."""
    return all(_is_numpy_like(a) for a in values)


def position_angle(dra, ddec):
    """Position angle in degrees, in [0, 360), of the offset ``(dra, ddec)``.

    ``dra`` is East and ``ddec`` North (any common unit); ``(0, 1)`` is 0°
    and ``(1, 0)`` is 90°. NumPy-like inputs give NumPy, otherwise JAX.
    """
    xp = onp if _numpy_inputs(dra, ddec) else np
    if xp is np:
        # arctan2 has a NaN gradient at the origin, where the angle is
        # undefined: give it a harmless point there (0° either way), so that
        # a sampler or optimizer that starts on the origin sees finite
        # derivatives. The angle at the origin (and at signed zeros, which
        # compare equal to 0) is 0°; NumPy's arctan2 still tells them apart.
        origin = (dra == 0) & (ddec == 0)
        dra, ddec = np.where(origin, 0.0, dra), np.where(origin, 1.0, ddec)
    pa = xp.mod(xp.rad2deg(xp.arctan2(dra, ddec)), 360.0)
    # mod rounds a tiny negative angle up to exactly 360. [()] turns the
    # 0-d array NumPy's where makes of a scalar back into a scalar.
    return xp.where(pa >= 360.0, pa - 360.0, pa)[()]


def separation_pa(dra, ddec):
    """``(sep, pa)`` of the offset ``(dra, ddec)``; ``pa`` as in :func:`position_angle`."""
    xp = onp if _numpy_inputs(dra, ddec) else np
    return xp.hypot(dra, ddec), position_angle(dra, ddec)


def _unflagged_samples(data):
    """Baseline, wavelength and channel width (flat arrays, metres) of the unflagged samples.

    ``data`` is an ``OIData`` or a sequence. Samples that no observable uses
    (``OIData.unflagged``) and zero baselines are dropped; the width is NaN
    where the data carry none. Raises if no sample remains.
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    baseline, wavel, width = [], [], []
    for d in observations:
        u = onp.ravel(onp.asarray(d.u, float))
        v = onp.ravel(onp.asarray(d.v, float))
        shape = u.shape
        lam = onp.broadcast_to(onp.ravel(onp.asarray(d.wavel, float)), shape)
        bandwidth = getattr(d, "bandwidth", None)
        if bandwidth is None:
            dlam = onp.full(shape, onp.nan)
        else:
            dlam = onp.broadcast_to(
                onp.ravel(onp.asarray(bandwidth, float)), shape
            )
        unflagged = getattr(d, "unflagged", None)
        keep = (
            onp.ones(shape, bool)
            if unflagged is None
            else onp.asarray(unflagged, bool)
        )
        b = onp.hypot(u, v)
        keep = keep & (b > 0)
        baseline.append(b[keep]), wavel.append(lam[keep])
        width.append(dlam[keep])
    baseline, wavel, width = (
        onp.concatenate(x) for x in (baseline, wavel, width)
    )
    if baseline.size == 0:
        raise ValueError(
            "The data have no unflagged sample with a non-zero baseline."
        )
    return baseline, wavel, width


def fringe_scales(data):
    """The finest and coarsest fringe spacings of ``data``, in mas.

    ``data`` is an ``OIData`` or a sequence. With each sample's own
    ``B/λ``, these are ``1 / max(B/λ)`` and ``1 / min(B/λ)`` over the
    unflagged samples with a non-zero baseline.

    Returns
    -------
    tuple[float, float]
        ``(finest, coarsest)``.
    """
    baseline, wavel, _ = _unflagged_samples(data)
    rho = baseline / wavel
    return 1.0 / (rho.max() * mas2rad), 1.0 / (rho.min() * mas2rad)


def smearing_scale(data):
    """The field (mas) beyond which bandwidth smearing washes out a source.

    The smallest ``λ² / (Δλ B)`` (first null of the channel's sinc) over the
    unflagged samples with a non-zero baseline: an order-of-magnitude limit
    (the visibility is down 36% at half this offset). Infinite when the
    data carry no channel widths (``OIData.with_bandwidth``).
    """
    baseline, wavel, width = _unflagged_samples(data)
    known = onp.isfinite(width) & (width > 0)
    if not known.any():
        return onp.inf
    scale = wavel[known] ** 2 / (width[known] * baseline[known])
    return float(scale.min() / mas2rad)


class UVGrid(eqx.Module):
    """uv samples on a regular lattice, possibly rotated on the sky (see :func:`find_uv_grid`).

    Sample ``k`` is at grid-frame ``(u_axis[index[k] % nu], v_axis[index[k] // nu])``
    (metres), and on the sky at ``rotate(u_g, v_g, rotation_deg)``.
    """

    u_axis: jax.Array
    v_axis: jax.Array
    index: jax.Array
    rotation_deg: float = eqx.field(static=True)


def find_uv_grid(u, v, tolerance=1e-6):
    """Return the lattice that ``(u, v)`` samples lie on, or ``None``.

    Axes come from the shortest sample separation; every sample must lie
    within ``tolerance`` cells of a lattice point.

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


def offset_phase(uu, vv, dra, ddec):
    """Fourier shift factor for an offset of ``(dra, ddec)`` milliarcseconds."""
    arg = 2.0 * np.pi * mas2rad * (uu * dra + vv * ddec)
    return jax.lax.complex(np.cos(arg), -np.sin(arg))


def undo_elliptical_transf_spat_freq(u, v, pa, stretch):
    """Map ``(u, v)`` to the frame where the elliptical object is circular, major axis North.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units.
    pa : float or array-like
        Position angle of the projected major axis (degrees, North to East).
    stretch : float or array-like
        Factor (typically < 1) applied to the minor axis.

    Returns
    -------
    tuple[array-like, array-like]
        ``(u, v)`` with the rotation and stretch undone.
    """
    pa_rad = pa * dtor

    # Fourier similarity theorem: compressing the image by `stretch` dilates uv by 1/stretch.
    ut = (u * np.cos(pa_rad) - v * np.sin(pa_rad)) * stretch
    vt = u * np.sin(pa_rad) + v * np.cos(pa_rad)

    return ut, vt


def undo_elliptical_transf_coord(x, y, pa, stretch):
    """Map ``(x, y)`` (mas, East and North) to the frame where the elliptical object is circular, major axis North.

    ``pa`` (degrees, North to East) and ``stretch`` (typically < 1, applied
    to the minor axis) are as in :func:`undo_elliptical_transf_spat_freq`.
    Returns ``(x, y)`` in mas with the rotation and stretch undone.
    """
    pa_rad = pa * dtor

    xt = (x * np.cos(pa_rad) - y * np.sin(pa_rad)) / stretch
    yt = x * np.sin(pa_rad) + y * np.cos(pa_rad)

    return xt, yt


def apply_elliptical_transf_coord(x, y, pa, stretch):
    """Inverse of :func:`undo_elliptical_transf_coord`; ``(x, y)`` in mas."""
    pa_rad = pa * dtor

    x_scaled = x * stretch
    y_scaled = y

    xt = x_scaled * np.cos(pa_rad) + y_scaled * np.sin(pa_rad)
    yt = -x_scaled * np.sin(pa_rad) + y_scaled * np.cos(pa_rad)

    return xt, yt


def check_az_prof_nonnegative(az_amps, az_pas, tol=1e-6):
    r"""Check that an azimuthal brightness modulation stays non-negative.

    The modulation is $1 + f(\theta)$, $f(\theta) = \sum_{m=1}^{n} A_m
    \cos(m(\theta - \mathrm{pa}_m))$. Its minimum is the lowest of $64 n$
    grid values polished by Newton steps, so it is never underestimated.
    ``jit``-safe.

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
    az_amps = np.asarray(az_amps)
    orders = np.arange(1, k + 1)
    phases = np.asarray(az_pas) * dtor * orders

    def derivatives(theta):
        # f and its first two derivatives at the angles ``theta``.
        arg = orders * theta[..., None] - phases
        cos, sin = np.cos(arg), np.sin(arg)
        f = np.sum(az_amps * cos, axis=-1)
        df = -np.sum(az_amps * orders * sin, axis=-1)
        d2f = -np.sum(az_amps * orders**2 * cos, axis=-1)
        return f, df, d2f

    theta = np.linspace(0.0, 2 * np.pi, 64 * k, endpoint=False)
    f_grid = derivatives(theta)[0]
    f_min = np.min(f_grid)
    # Newton steps on f' from the lowest grid point, where f'' > 0 near a
    # minimum; a step is skipped where it is not.
    t = theta[np.argmin(f_grid)]
    for _ in range(3):
        f, df, d2f = derivatives(t)
        t = t - np.where(d2f > 0, df / np.where(d2f > 0, d2f, 1.0), 0.0)
        f_min = np.minimum(f_min, derivatives(t)[0])

    return 1.0 + f_min > -tol

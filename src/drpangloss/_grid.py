"""Machinery shared by the grid searches and the contrast limits (private).

Both [`drpangloss.grid_fit`][drpangloss.grid_fit] and
[`drpangloss.limits`][drpangloss.limits] import from here, so neither
depends on the other's internals.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from ._utils import concrete, is_flux_param, resolve_flux_param


# Grid points are evaluated in batches of this many, which bounds memory on
# large grids: each point holds a full model evaluation and, in the
# optimizers, a BFGS state. 256 keeps multi-component float64 models over
# DISCO data within a laptop's memory. The public functions take
# ``batch_size=`` to change it, e.g. larger for small models.
DEFAULT_BATCH_SIZE = 256


def batch_size_or_default(batch_size):
    """Validate a ``batch_size`` argument, defaulting to 256."""
    if batch_size is None:
        return DEFAULT_BATCH_SIZE
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    return batch_size


def map_points(fn, *xs, batch_size=DEFAULT_BATCH_SIZE):
    """Apply ``fn`` to every row of ``xs``, ``batch_size`` rows at a time."""
    return jax.lax.map(lambda args: fn(*args), xs, batch_size=batch_size)


def check_flux_axes(samples_dict, flux_param=None):
    """Reject grid axes that would give a flux parameter negative values.

    Axes whose name ends in ``flux`` are checked, as is the explicitly
    selected ``flux_param`` whatever its name.
    """
    for key, values in samples_dict.items():
        if not (is_flux_param(key) or key == flux_param):
            continue
        values = concrete(values)
        if values is not None and np.any(values < 0.0):
            raise ValueError(
                f"The grid axis {key!r} contains negative values, but fluxes "
                "must be non-negative."
            )


def resolve_grid_keys(samples_dict, flux_param=None):
    """Return ``(params, coord_keys, flux_key)`` for a grid-fitting call."""
    params = tuple(samples_dict.keys())
    check_flux_axes(samples_dict, flux_param)
    flux_key = resolve_flux_param(params, flux_param)
    coord_keys = tuple(key for key in params if key != flux_key)
    if not coord_keys:
        raise ValueError(
            "samples_dict needs at least one coordinate parameter besides "
            f"the flux {flux_key!r}."
        )
    return params, coord_keys, flux_key


def meshgrid_vectors(samples_dict, params):
    """Build flattened meshgrid vectors with axis order matching ``params``."""
    samples = [jnp.asarray(samples_dict[param]) for param in params]
    grid_shape = tuple(sample.shape[0] for sample in samples)
    grids = jnp.meshgrid(*samples, indexing="ij")
    vals_vec = jnp.stack([grid.reshape(-1) for grid in grids], axis=1)
    return vals_vec, grid_shape


def ordered_values(flux, coord_vals, params, coord_keys, flux_key):
    """Build parameter values in ``params`` order without traced dict objects."""
    flux_value = jnp.asarray(flux).reshape(-1)[0]
    coord_vals = jnp.asarray(coord_vals)
    return [
        flux_value
        if param == flux_key
        else coord_vals[coord_keys.index(param)]
        for param in params
    ]


def coordinate_points(samples_dict, coord_keys):
    """Flattened ``(n_points, n_coords)`` coordinate grid and its shape.

    The grid uses ``indexing="ij"``: axis ``k`` follows ``coord_keys[k]``,
    so for ``(dra, ddec)`` axis 0 is ``dra``.
    """
    coord_grids = jnp.meshgrid(
        *[jnp.asarray(samples_dict[key]) for key in coord_keys],
        indexing="ij",
    )
    points = jnp.stack([grid.reshape(-1) for grid in coord_grids], axis=1)
    return points, coord_grids[0].shape


def warn_unconverged(success, caller):
    """Warn if an optimizer failed to converge at some grid positions."""
    failed = int(np.sum(~np.asarray(success, dtype=bool)))
    if failed:
        warnings.warn(
            f"{caller}(): the optimizer did not converge at {failed} of "
            f"{np.size(success)} grid positions; values there may be "
            "inaccurate.",
            RuntimeWarning,
            # user -> public function -> here
            stacklevel=3,
        )

"""Small helpers shared across virgil: unit constants, traced-value
checks, and the rules for naming flux parameters."""

import jax
import jax.numpy as np
import numpy as onp


# === CONSTANTS ===

rad2mas = 180.0 / np.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = np.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad
dtor = np.pi / 180.0  # convert deg to rad

i2pi = 1j * 2.0 * np.pi


# === TRACED VALUES ===


def concrete(value):
    """``value`` as a NumPy array, or ``None`` inside a traced computation.

    Checks that need actual numbers (e.g. "fluxes are non-negative") use
    this to run on concrete inputs and skip silently under ``jax.jit``,
    ``jax.vmap`` or ``jax.grad``.
    """
    try:
        return onp.asarray(value)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


# === PER-DATASET MODELS ===


def _reference(model):
    """The model regularisers act on: the first if there is one per dataset."""
    return model[0] if isinstance(model, (list, tuple)) else model


def _per_dataset(model, n_data):
    """One model per dataset: ``model`` repeated, or its list checked."""
    if not isinstance(model, (list, tuple)):
        return [model] * n_data
    if len(model) != n_data:
        raise ValueError(
            f"The model function returned {len(model)} models for "
            f"{n_data} datasets."
        )
    return list(model)


# === FLUX PARAMETERS ===
#
# A parameter is a flux when the last part of its name (or zodiax path) is
# ``flux`` (``flux``, ``comp.flux``, ``comp.disk.flux``), or when it is a flux
# spectrum's reference ratio (``comp.flux.ratio``). Fluxes are relative
# to a reference component (usually the primary star at flux 1), so for a
# companion ``flux`` is its companion/primary flux ratio.


def is_flux_param(name):
    """Whether ``name`` (a parameter name or path) is a flux.

    That is, its last part is ``flux`` (``flux``, ``comp.flux``), or it is
    the reference ratio of a flux spectrum (``comp.flux.ratio``).
    """
    parts = str(name).split(".")
    return parts[-1] == "flux" or parts[-2:] == ["flux", "ratio"]


def resolve_flux_param(keys, flux_param=None):
    """Return the key among ``keys`` holding the flux to fit or optimize.

    ``flux_param`` is returned if given (and present). Otherwise the one key
    whose last part is ``flux`` is used; if there is none, or more than one,
    ``flux_param`` must be passed.
    """
    keys = list(keys)
    if flux_param is not None:
        if flux_param not in keys:
            raise ValueError(
                f"flux_param {flux_param!r} is not one of the keys {keys}."
            )
        return flux_param
    candidates = [key for key in keys if is_flux_param(key)]
    if len(candidates) != 1:
        found = f"several ({candidates})" if candidates else "none"
        raise ValueError(
            f"Could not tell which of {keys} is the flux: {found} end in "
            "'flux'. Pass flux_param=<key>."
        )
    return candidates[0]

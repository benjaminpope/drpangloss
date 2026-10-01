"""Local numerical precision for fitting and sampling entry points.

drpangloss never enables float64 globally. Entry points such as
[`fit`][drpangloss.fitting.fit] instead run inside :func:`run_in`, a local
``jax.enable_x64`` context, after casting their inputs with
:func:`cast_tree`. Forward-model code works in either precision.
"""

import contextlib

import jax
import jax.numpy as np
import numpy as onp

_DTYPES = {
    "float32": (np.float32, np.complex64),
    "float64": (np.float64, np.complex128),
}


def _check(dtype):
    if dtype not in _DTYPES:
        raise ValueError(
            f"dtype must be 'float32' or 'float64', not {dtype!r}."
        )


@contextlib.contextmanager
def run_in(dtype):
    """Run the enclosed code with JAX's 64-bit mode on or off.

    ``dtype`` is ``"float64"`` or ``"float32"``; the setting is restored on
    exit.
    """
    _check(dtype)
    with jax.enable_x64(dtype == "float64"):
        yield


def cast_tree(tree, dtype):
    """Cast every floating-point and complex array in a pytree to ``dtype``.

    Real arrays become ``dtype`` and complex arrays the matching complex
    type; everything else (integers, booleans, Python scalars) is left
    alone. Call it inside :func:`run_in` for float64.
    """
    _check(dtype)
    real, cplx = _DTYPES[dtype]

    def cast(leaf):
        if not isinstance(leaf, (jax.Array, onp.ndarray)):
            return leaf
        if np.issubdtype(leaf.dtype, np.complexfloating):
            return np.asarray(leaf, dtype=cplx)
        if np.issubdtype(leaf.dtype, np.floating):
            return np.asarray(leaf, dtype=real)
        return leaf

    return jax.tree_util.tree_map(cast, tree)

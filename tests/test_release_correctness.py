"""Correctness fixes from the 0.4.0 release review.

* ``OIData._diagonalized`` keeps only the real directions of an operator
  applied to correlated closure phases (a singular correlation), in float32
  as well as float64.
* ``PolarDisk.is_physical`` works with a ``VonMises`` azimuthal model.
* ``SharedFlux(grid=(lo, hi, n))`` is a range only when ``n`` is an ``int``.
* Matmuls and contractions in forward models use ``Precision.HIGHEST``
  (checked on the jaxpr, since on CPU the default is already exact).
"""

import itertools

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.amigo import load_oi_data
from virgil.likelihood import model_loglike
from virgil.models import (
    GaussianRing,
    LimbDarkenedDisk,
    PolarDisk,
    PolarKernel,
    RadialGP,
    VonMises,
)
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData
from virgil.orbit_search import SharedFlux, _axis
from tests._shared import CALIBRATED_VISIBILITY

WAVEL = 2.2e-6


# --- R2: singular correlation of projected closure phases in float32 -------


def _four_telescope_correlation(n_blocks):
    """Correlation of the 4 closure phases of each of ``n_blocks`` frames.

    Four telescopes have four triangles but three independent closure
    phases, so each 4 x 4 block is singular (rank 3).
    """
    pairs = list(itertools.combinations(range(4), 2))
    incidence = onp.zeros((4, 6))
    for row, (i, j, k) in enumerate(itertools.combinations(range(4), 3)):
        incidence[row, pairs.index((i, j))] = 1.0
        incidence[row, pairs.index((j, k))] = 1.0
        incidence[row, pairs.index((i, k))] = -1.0
    return onp.kron(onp.eye(n_blocks), incidence @ incidence.T)


@pytest.mark.parametrize("seed", range(5))
def test_projected_correlated_closure_phases_keep_the_real_directions(seed):
    """A projection of correlated closure phases has rank 3 per frame; in
    float32 the cut at 1e-10 of the largest eigenvalue kept the null one."""
    n_blocks = 10
    rng = onp.random.default_rng(seed)
    sigma = 0.01 * (1.0 + rng.uniform(size=4 * n_blocks))
    operator = rng.standard_normal((4 * n_blocks, 4 * n_blocks))
    assert not jax.config.jax_enable_x64  # the default float32 path
    matrix, errors = OIData._diagonalized(
        operator, sigma, _four_telescope_correlation(n_blocks)
    )
    assert matrix.shape == (3 * n_blocks, 4 * n_blocks)
    assert errors.shape == (3 * n_blocks,)
    assert matrix.dtype == errors.dtype == np.asarray(0.0).dtype
    # The errors are those of real directions, not numerical noise.
    assert float(errors.min()) > 1e-3


@pytest.mark.parametrize("seed", range(3))
def test_rank_deficient_operator_keeps_only_real_directions(seed):
    """The same for an operator of rank 15 with 40 outputs (no correlation
    given, so the outputs are first found to be correlated)."""
    rng = onp.random.default_rng(seed)
    operator = rng.standard_normal((40, 15)) @ rng.standard_normal((15, 60))
    sigma = 0.01 * (1.0 + rng.uniform(size=60))
    matrix, errors = OIData._diagonalized(operator, sigma)
    assert matrix.shape == (15, 60)
    assert float(errors.min()) > 1e-3


# --- R3: PolarDisk.is_physical with a VonMises azimuthal model --------------


def _polar(harmonics):
    return PolarDisk(
        GaussianRing(15.0, 2.0), harmonics=harmonics, inc=40.0, pa=10.0, n_r=24
    )


def test_polar_disk_is_physical_with_von_mises():
    assert bool(_polar(VonMises(4.0, 30.0, degree=20)).is_physical())


def test_reject_unphysical_with_von_mises_harmonics():
    model = _polar(VonMises(4.0, 30.0, degree=20))
    u = onp.array([20.0, 40.0, 60.0, 80.0])
    data = OIData(
        {
            "u": u,
            "v": onp.zeros_like(u),
            "wavel": onp.full(u.size, WAVEL),
            "vis": onp.full(u.size, 0.5),
            "d_vis": onp.full(u.size, 0.05),
        }
    )
    loglike = model_loglike(model, data, reject_unphysical=True)
    assert onp.isfinite(float(loglike))


# --- R4: the (lo, hi, n) grid form is unambiguous ---------------------------


def test_axis_range_form_needs_a_python_int():
    x, w = _axis((0.01, 0.1, 5), True, "flux")
    assert onp.allclose(onp.exp(x), onp.geomspace(0.01, 0.1, 5))
    assert x.shape == w.shape
    x, _ = _axis((0.01, 0.1, 1), True, "flux")
    assert x.shape == (1,)


@pytest.mark.parametrize(
    "spec",
    [
        (0.01, 0.1, 1.0),
        [0.01, 0.1, 1.0],
        onp.array([0.01, 0.1, 1.0]),
        (0.01, 0.1, 2.0),
    ],
)
def test_axis_floats_are_values(spec):
    x, _ = _axis(spec, True, "flux")
    assert onp.allclose(onp.exp(x), onp.asarray(spec, dtype=float))


def test_shared_flux_grid_ending_at_one_is_three_fluxes():
    x, _ = _axis(SharedFlux(grid=(0.01, 0.1, 1.0)).grid, True, "flux")
    assert x.shape == (3,)


def test_axis_numpy_integer_size_is_a_range():
    x, _ = _axis((0.01, 0.1, onp.int64(5)), True, "flux")
    assert x.shape == (5,)


def test_axis_bool_size_is_not_a_range():
    x, _ = _axis((0.01, 0.1, True), True, "flux")
    assert x.shape == (3,)


def test_axis_range_size_must_be_positive():
    with pytest.raises(ValueError, match="size"):
        _axis((0.01, 0.1, 0), True, "flux")


# --- Precision.HIGHEST in contractions --------------------------------------


def _dot_precisions(jaxpr):
    """The ``precision`` of every dot_general in a (nested) jaxpr."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            found.append(eqn.params["precision"])
        for value in eqn.params.values():
            subs = value if isinstance(value, (list, tuple)) else [value]
            for sub in subs:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    found.extend(_dot_precisions(inner))
    return found


def _assert_highest(fn, *args):
    precisions = _dot_precisions(jax.make_jaxpr(fn)(*args).jaxpr)
    assert precisions, "no matmul found"
    for precision in precisions:
        assert precision is not None
        assert all(p == jax.lax.Precision.HIGHEST for p in precision)


def test_limb_darkening_coefficient_matmul_is_highest():
    disk = LimbDarkenedDisk(1.0, u=[0.4, 0.2, 0.1])
    _assert_highest(lambda d: d._profile()[0], disk)


def test_radial_gp_contractions_are_highest():
    gp = RadialGP(0.5, 10.0, sigma=1.0, ell=2.0, z=np.zeros(24), n=24)
    _assert_highest(lambda g: g._values()[1], gp)
    _assert_highest(lambda g: g.draws(jax.random.key(0), 2), gp)


def test_polar_kernel_values_are_highest():
    kernel = PolarKernel(3.0, 4.0, 0.4, r_in=1.0, r_out=9.0, n=12)
    z = np.zeros((2, 3, 12))
    _assert_highest(lambda k, zz: k.values(zz, np.ones(5)), kernel, z)


def test_disco_operator_application_is_highest():
    n = 6
    values = np.arange(1.0, n + 1)
    operator = np.eye(n)
    _assert_highest(
        lambda v: OIData._apply_linear_operator(v, operator), values
    )


def test_mixed_disco_standardize_model_is_highest():
    data = load_oi_data(CALIBRATED_VISIBILITY, "F430M")
    model = BinaryModelCartesian(100.0, -50.0, 1e-3)
    cvis = model.model(data.u, data.v, data.wavel)
    _assert_highest(data.standardize_model, cvis)

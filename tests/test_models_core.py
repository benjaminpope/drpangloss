import jax
import jax.numpy as np
import jax.scipy as jsp
import numpy as onp
import pytest

import drpangloss.models as models
from drpangloss.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    cvis_binary,
    fisher,
    joint_data,
    joint_errors,
    joint_loglike,
    joint_prediction,
    laplace_cov,
    loglike,
    loglike_nosignal,
    model_loglike,
)
from drpangloss.oidata import OIData, closure_phases

from tests._test_data import i_cps1, i_cps2, i_cps3, oidata, u, v


ddec, dra, planet = 0.1, 0.2, 10


def test_oidata_compatibility_reexports():
    assert models.OIData is OIData
    assert models.closure_phases is closure_phases


def test_cvis_binary():
    vis = cvis_binary(u, v, ddec, dra, planet)
    vis2 = np.abs(vis) ** 2
    assert vis.shape == (u.shape[0],)
    assert np.all(vis2 >= 0.0)
    assert np.all(vis2 <= 1.0)
    assert np.all(np.isfinite(vis))


def test_closure_phases():
    vis = cvis_binary(u, v, ddec, dra, planet)
    cps = closure_phases(vis, i_cps1, i_cps2, i_cps3)
    assert cps.shape == (35,)
    assert np.all(np.isfinite(cps))


def test_likelihood():
    binary = BinaryModelAngular(50, 45, 10)
    model_data = oidata.model(binary)
    data, errors = oidata.flatten_data()

    like = jsp.stats.norm.logpdf(model_data, loc=data, scale=errors).sum()
    assert np.all(np.isfinite(like))


def test_BinaryModelAngular():
    binary = BinaryModelAngular(50, 45, 0.1)
    model_data = oidata.model(binary)
    assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_data))


def test_BinaryModelCartesian():
    binary = BinaryModelCartesian(150, 150, 1e-3)
    model_data = oidata.model(binary)
    assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_data))


def test_BinaryModelAngular_to_cartesian_preserves_model():
    angular = BinaryModelAngular(50.0, 45.0, 10.0)
    cartesian = angular.to_cartesian()

    assert isinstance(cartesian, BinaryModelCartesian)
    assert np.allclose(
        angular.model(oidata.u, oidata.v, oidata.wavel),
        cartesian.model(oidata.u, oidata.v, oidata.wavel),
    )


def test_BinaryModelCartesian_to_angular_roundtrip():
    cartesian = BinaryModelCartesian(150.0, -120.0, 1e-3)
    angular = cartesian.to_angular()
    roundtrip = angular.to_cartesian()

    assert isinstance(angular, BinaryModelAngular)
    assert np.allclose(roundtrip.dra, cartesian.dra)
    assert np.allclose(roundtrip.ddec, cartesian.ddec)
    assert np.allclose(roundtrip.flux, cartesian.flux)


@pytest.mark.parametrize(
    ("pa", "expected_index"),
    [
        # pa=0deg -> North (+y, low row index).
        (0.0, (0, 2)),
        # pa=90deg -> East (+x, low column index).
        (90.0, (2, 0)),
    ],
)
def test_binary_model_angular_render_follows_north_to_east_pa_convention(
    pa, expected_index
):
    image = BinaryModelAngular(sep=4.0, pa=pa, contrast=1e-6).render(
        npix=5, fov_mas=10.0
    )
    assert (
        onp.unravel_index(onp.asarray(image).argmax(), image.shape)
        == expected_index
    )


def test_binary_model_angular_matches_cartesian_at_east_position_angle():
    """A companion placed at pa=90deg (East) must match the
    already-validated BinaryModelCartesian placed at dra=+sep, ddec=0.
    Comparing against an independently constructed BinaryModelCartesian
    (rather than angular.to_cartesian()) is what actually catches a
    sign error in the angular parameterization: a self-consistency
    round-trip through to_cartesian()/to_angular() alone would stay
    internally consistent even if both used the same wrong sign.
    """
    sep, contrast = 40.0, 5.0
    angular = BinaryModelAngular(sep=sep, pa=90.0, contrast=contrast)
    cartesian = BinaryModelCartesian(dra=sep, ddec=0.0, flux=1.0 / contrast)

    assert np.allclose(
        angular.model(oidata.u, oidata.v, oidata.wavel),
        cartesian.model(oidata.u, oidata.v, oidata.wavel),
    )


def test_laplace_and_fisher_wrappers_are_finite():
    params = ["dra", "ddec", "flux"]
    values = np.array([120.0, -80.0, 2e-3])
    param_dict = dict(zip(params, values))
    model_data = oidata.model(BinaryModelCartesian(**param_dict))
    data, errors = oidata.flatten_data()

    cov = laplace_cov(values, params, oidata, BinaryModelCartesian)
    fmat = fisher(values, params, oidata, BinaryModelCartesian, ridge=1e-10)
    like = loglike(values, params, oidata, BinaryModelCartesian)
    expected_like = jsp.stats.norm.logpdf(
        model_data, loc=data, scale=errors
    ).sum()

    assert cov.shape == (3, 3)
    assert fmat.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    assert np.all(np.isfinite(fmat))
    assert np.allclose(fmat, fmat.T)
    assert np.isfinite(like)
    assert np.allclose(like, expected_like)


def test_loglike_nosignal_matches_normalized_gaussian_logpdf():
    params = ["dra", "ddec", "flux"]
    values = np.array([120.0, -80.0, 2e-3])
    param_dict = dict(zip(params, values))
    model_data = oidata.model(BinaryModelCartesian(**param_dict))
    _, errors = oidata.flatten_data()
    null_data = np.concatenate(
        [np.ones_like(oidata.vis), np.zeros_like(oidata.phi)]
    )

    like = loglike_nosignal(values, params, oidata, BinaryModelCartesian)
    expected_like = jsp.stats.norm.logpdf(
        model_data, loc=null_data, scale=errors
    ).sum()

    assert np.isfinite(like)
    assert np.allclose(like, expected_like)


def test_model_and_joint_loglike_helpers_match_legacy_loglike():
    values = np.array([120.0, -80.0, 2e-3])
    params = ["dra", "ddec", "flux"]
    model = BinaryModelCartesian(*values)
    observations = (oidata, oidata)
    tree = {"dra": values[0], "ddec": values[1], "flux": values[2:]}
    model_fn = lambda values, index: BinaryModelCartesian(
        values["dra"], values["ddec"], values["flux"][index]
    )

    legacy = loglike(values, params, oidata, BinaryModelCartesian)
    assert np.allclose(model_loglike(model, oidata), legacy)
    assert np.allclose(joint_loglike(tree, observations, model_fn), 2 * legacy)
    assert joint_data(observations).shape[0] == 2 * (
        oidata.vis.size + oidata.phi.size
    )
    assert joint_errors(observations).shape == joint_data(observations).shape
    assert joint_prediction(tree, observations, model_fn).shape[0] == 2 * (
        oidata.vis.size + oidata.phi.size
    )


def test_oidata_with_model_preserves_structure_and_seeded_noise():
    model = BinaryModelCartesian(120.0, -80.0, 2e-3)
    noiseless = oidata.with_model(model)
    noisy_a = oidata.with_model(model, key=jax.random.key(12))
    noisy_b = oidata.with_model(model, key=jax.random.key(12))

    assert np.allclose(noiseless.flatten_data()[0], oidata.model(model))
    assert np.allclose(noisy_a.vis, noisy_b.vis)
    assert np.allclose(noisy_a.phi, noisy_b.phi)
    assert np.allclose(noiseless.u, oidata.u)
    assert np.allclose(noiseless.d_vis, oidata.d_vis)
    assert noiseless.cp_flag == oidata.cp_flag
    assert noiseless.vis_mode == oidata.vis_mode

    with pytest.raises(ValueError, match="non-negative"):
        oidata.with_model(model, noise_scale=-1.0)


def test_oidata_linear_observables_transform_and_model_alignment():
    cvis = cvis_binary(
        oidata.u / oidata.wavel, oidata.v / oidata.wavel, 0.0, 50.0, 1e-3
    )
    n = cvis.shape[0]
    m_vis = 12
    m_phi = 10
    vis_mat = np.eye(n)[:m_vis, :]
    phi_mat = np.eye(n)[:m_phi, :]

    sim_data = {
        "u": oidata.u,
        "v": oidata.v,
        "wavel": oidata.wavel,
        "vis": np.abs(cvis),
        "d_vis": np.ones_like(np.abs(cvis)) * 1e-3,
        "phi": np.angle(cvis),
        "d_phi": np.ones_like(np.angle(cvis)) * 0.1,
        "v2_flag": False,
        "cp_flag": False,
        "vis_mode": "logamp",
        "disco_vis_mat": vis_mat,
        "disco_phi_mat": phi_mat,
    }

    disco_data = OIData(sim_data)
    flattened_data, errors = disco_data.flatten_data()
    model_vector = disco_data.model(BinaryModelCartesian(50.0, 0.0, 1e-3))
    cvis_model = BinaryModelCartesian(50.0, 0.0, 1e-3).model(
        disco_data.u, disco_data.v, disco_data.wavel
    )

    assert disco_data.vis.shape == (m_vis,)
    assert disco_data.phi.shape == (m_phi,)
    assert disco_data.observable_kind == "split"
    assert np.allclose(flattened_data, disco_data.standardize_data())
    assert np.allclose(errors, disco_data.standardize_errors())
    assert np.allclose(model_vector, disco_data.standardize_model(cvis_model))
    assert flattened_data.shape == model_vector.shape
    assert errors.shape == flattened_data.shape
    assert np.all(np.isfinite(flattened_data))
    assert np.all(np.isfinite(model_vector))


def test_oidata_linear_observables_invalid_shape_raises():
    n = oidata.u.shape[0]
    bad_data = {
        "u": oidata.u,
        "v": oidata.v,
        "wavel": oidata.wavel,
        "vis": np.ones(n),
        "d_vis": np.ones(n) * 1e-3,
        "phi": np.zeros(n),
        "d_phi": np.ones(n) * 0.1,
        "v2_flag": False,
        "cp_flag": False,
        "disco_vis_mat": np.ones((5, 7)),
    }

    raised = False
    try:
        OIData(bad_data)
    except ValueError:
        raised = True
    assert raised


def test_disco_operator_requires_diagonal_propagated_covariance():
    base = {
        "u": np.zeros(2),
        "v": np.zeros(2),
        "wavel": np.array([1e-6]),
        "vis": np.ones(2),
        "d_vis": np.ones(2),
        "phi": np.zeros(2),
        "d_phi": np.ones(2),
        "v2_flag": False,
        "cp_flag": False,
    }
    orthogonal = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    independent = OIData({**base, "disco_vis_mat": orthogonal})

    assert np.allclose(independent.d_vis, np.ones(2))
    with pytest.raises(ValueError, match="diagonal propagated covariance"):
        OIData(
            {
                **base,
                "disco_vis_mat": np.array([[1.0, 0.0], [1.0, 1.0]]),
            }
        )

    correlated_generic = OIData(
        {**base, "vis_mat": np.array([[1.0, 0.0], [1.0, 1.0]])}
    )
    assert correlated_generic.vis.shape == (2,)

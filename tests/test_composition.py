import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density

from drpangloss.grid_fit import (
    absil_limits,
    likelihood_grid,
    optimized_contrast_grid,
)
from drpangloss.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    GaussianDisk,
    GaussianDiskModel,
    ModulatedGaussianRim,
    PointSource,
    System,
    UniformDisk,
    build_model,
    laplace_cov,
    loglike,
    numpyro_model,
)
from tests._test_data import oidata, samples_dict

U, V, WAVEL = oidata.u, oidata.v, oidata.wavel


def _composed_binary(dra=120.0, ddec=-80.0, flux=4e-3):
    return System(
        star=PointSource(), comp=PointSource(dra=dra, ddec=ddec, flux=flux)
    )


def test_composed_binary_matches_binary_model_cartesian():
    assert np.allclose(
        _composed_binary().model(U, V, WAVEL),
        BinaryModelCartesian(120.0, -80.0, 4e-3).model(U, V, WAVEL),
        atol=1e-12,
    )


def test_operator_sugar_matches_named_system():
    sugar = PointSource() + 4e-3 * PointSource(dra=120.0, ddec=-80.0)
    assert list(sugar.components) == ["c0", "c1"]
    assert np.allclose(
        sugar.model(U, V, WAVEL), _composed_binary().model(U, V, WAVEL)
    )


def test_adding_systems_merges_named_components():
    rim = ModulatedGaussianRim(14.0, 3.0, 20.0, 10.0, flux=0.5)
    combined = System(star=PointSource()) + System(rim=rim) + GaussianDisk(4.0)
    assert list(combined.components) == ["star", "rim", "c0"]


def test_adding_systems_with_clashing_names_raises():
    with pytest.raises(ValueError, match="both operands"):
        System(star=PointSource()) + System(star=PointSource())


def test_reserved_component_names_are_rejected():
    with pytest.raises(ValueError, match="not a valid component name"):
        System({"flux": PointSource()})


def test_weighting_a_binary_wraps_instead_of_rescaling_companion():
    weighted = 3.0 * BinaryModelCartesian(120.0, -80.0, 4e-3)
    assert isinstance(weighted, System)
    assert float(weighted.flux) == pytest.approx(3.0)
    assert float(weighted.c0.flux) == pytest.approx(4e-3)


def test_weighting_a_component_scales_its_flux():
    assert float((0.25 * PointSource(flux=2.0)).flux) == pytest.approx(0.5)


def test_paths_get_and_set_through_named_components():
    system = System(
        star=PointSource(),
        comp=System(core=PointSource(), disk=GaussianDisk(2.0, flux=0.3)),
    )
    assert float(system.get("comp.disk.sigma")) == pytest.approx(2.0)
    updated = system.set(["comp.disk.sigma", "comp.flux"], [5.0, 0.1])
    assert float(updated.comp.disk.sigma) == pytest.approx(5.0)
    assert float(updated.comp.flux) == pytest.approx(0.1)
    assert float(system.comp.disk.sigma) == pytest.approx(2.0)


def test_zero_companion_flux_is_unresolved_star():
    assert np.allclose(
        _composed_binary(flux=0.0).model(U, V, WAVEL), 1.0 + 0j, atol=1e-12
    )


def test_visibility_is_flux_weighted_mean_of_components():
    disk = GaussianDisk(5.0, flux=2.0, dra=3.0)
    ring = UniformDisk(8.0, flux=0.5, ddec=-4.0)
    expected = (
        2.0 * disk.model(U, V, WAVEL) + 0.5 * ring.model(U, V, WAVEL)
    ) / 2.5
    assert np.allclose(System(a=disk, b=ring).model(U, V, WAVEL), expected)


def test_shifted_subsystem_renders_east_on_the_left():
    image = onp.asarray(
        System(
            star=PointSource(),
            comp=System(core=PointSource(), dra=4.0, flux=3.0),
        ).render(npix=5, fov_mas=10.0)
    )
    assert onp.unravel_index(image.argmax(), image.shape) == (2, 0)


def test_gaussian_disk_model_is_star_plus_disk():
    legacy = GaussianDiskModel(sigma=6.0, flux=0.3, dra=4.0, ddec=-2.0)
    composed = System(
        star=PointSource(),
        disk=GaussianDisk(6.0, flux=0.3, dra=4.0, ddec=-2.0),
    )
    assert np.allclose(
        legacy.model(U, V, WAVEL), composed.model(U, V, WAVEL), atol=1e-12
    )


def test_build_model_accepts_classes_and_templates():
    params, values = ("dra", "ddec", "flux"), (10.0, -5.0, 0.01)
    from_class = build_model(BinaryModelCartesian, params, values)
    from_template = build_model(
        _composed_binary(), [f"comp.{p}" for p in params], values
    )
    assert np.allclose(
        from_class.model(U, V, WAVEL), from_template.model(U, V, WAVEL)
    )


def _path_samples():
    return {f"comp.{key}": value for key, value in samples_dict.items()}


def test_likelihood_grid_with_paths_matches_model_class():
    assert np.allclose(
        likelihood_grid(oidata, _composed_binary(), _path_samples()),
        likelihood_grid(oidata, BinaryModelCartesian, samples_dict),
        rtol=1e-4,
    )


def test_optimized_contrast_grid_with_paths_matches_model_class():
    composed = optimized_contrast_grid(
        oidata, _composed_binary(), _path_samples()
    )
    reference = optimized_contrast_grid(
        oidata, BinaryModelCartesian, samples_dict
    )
    composed, reference = onp.asarray(composed), onp.asarray(reference)
    scale = onp.abs(reference).max()
    # float32 BFGS may settle differently where there is no signal (contrast ~ 0).
    signal = onp.abs(reference) > 0.1 * scale
    assert onp.allclose(composed[signal], reference[signal], rtol=1e-3)
    assert onp.abs(composed - reference).max() < 0.1 * scale


def test_absil_limits_with_paths_matches_model_class():
    small = {key: value[::10] for key, value in samples_dict.items()}
    paths = {f"comp.{key}": value for key, value in small.items()}
    assert np.allclose(
        absil_limits(paths, oidata, _composed_binary(), 3.0),
        absil_limits(small, oidata, BinaryModelCartesian, 3.0),
        rtol=1e-3,
    )


def test_laplace_cov_with_paths_matches_model_class():
    values = np.array([120.0, -80.0, 4e-3])
    params = ["dra", "ddec", "flux"]
    assert np.allclose(
        laplace_cov(
            values, [f"comp.{p}" for p in params], oidata, _composed_binary()
        ),
        laplace_cov(values, params, oidata, BinaryModelCartesian),
        rtol=1e-3,
    )


def test_numpyro_model_log_density_is_prior_plus_loglike():
    priors = {
        "comp.dra": dist.Uniform(0.0, 200.0),
        "comp.ddec": dist.Uniform(-200.0, 0.0),
        "comp.flux": dist.LogUniform(1e-5, 1e-1),
    }
    point = {"comp.dra": 120.0, "comp.ddec": -80.0, "comp.flux": 4e-3}
    logp, _ = log_density(
        numpyro_model(_composed_binary(), priors, oidata), (), {}, point
    )
    expected = sum(
        priors[key].log_prob(value) for key, value in point.items()
    ) + loglike(list(point.values()), list(point), oidata, _composed_binary())
    assert np.isclose(logp, expected, rtol=1e-6)


def test_binary_angular_is_unchanged_by_composition_machinery():
    angular = BinaryModelAngular(sep=144.2, pa=123.7, contrast=250.0)
    assert np.allclose(
        angular.model(U, V, WAVEL),
        angular.to_cartesian().model(U, V, WAVEL),
        atol=1e-6,
    )
    assert jax.tree_util.tree_structure(angular).num_leaves == 3

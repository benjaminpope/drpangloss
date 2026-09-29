"""Chromatic scenes: spectra, resolved flux, several files, likelihood options."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from drpangloss import (
    GaussianDisk,
    ModulatedGaussianRim,
    OIData,
    PointSource,
    PowerLaw,
    Resolved,
    System,
    UniformDisk,
    write_oifits,
)
from drpangloss._utils import is_flux_param, resolve_flux_param
from drpangloss.likelihood import (
    joint_loglike,
    model_loglike,
    numpyro_model,
)
from tests.test_oifits import TRUTH, _tables

WAVES = onp.array([1.55e-6, 1.65e-6, 1.75e-6])


def _scene(flux_s=PowerLaw(0.05, -1.0, 1.65e-6)):
    return System(
        primary=UniformDisk(0.6, flux=PowerLaw(0.6, -4.0, 1.65e-6)),
        secondary=PointSource(flux=flux_s, dra=3.0, ddec=-1.0),
        rim=GaussianDisk(4.0, flux=PowerLaw(0.3, 2.0, 1.65e-6)),
        background=Resolved(flux=PowerLaw(0.05, 1.0, 1.65e-6)),
    )


def test_power_law_at_index_zero_is_achromatic():
    u, v = onp.array([30.0, -20.0]), onp.array([10.0, 40.0])
    chromatic = System(
        star=PointSource(flux=PowerLaw(1.0, 0.0)),
        comp=PointSource(flux=PowerLaw(0.1, 0.0), dra=5.0),
    )
    plain = System(star=PointSource(), comp=PointSource(flux=0.1, dra=5.0))
    for wavel in WAVES:
        assert np.allclose(
            chromatic.model(u, v, wavel), plain.model(u, v, wavel)
        )


def test_chromatic_system_matches_sparco_formula():
    u, v = onp.array([30.0, -20.0, 5.0]), onp.array([10.0, 40.0, -60.0])
    scene = _scene()
    for wavel in WAVES:
        ratio = wavel / 1.65e-6
        weights = {
            "primary": 0.6 * ratio**-4.0,
            "secondary": 0.05 * ratio**-1.0,
            "rim": 0.3 * ratio**2.0,
            "background": 0.05 * ratio**1.0,
        }
        numerator = sum(
            weights[name] * scene.components[name].model(u, v, wavel)
            for name in ("primary", "secondary", "rim")
        )
        expected = numerator / sum(weights.values())
        assert np.allclose(scene.model(u, v, wavel), expected, atol=1e-6)


def test_spectrum_parameters_are_paths():
    scene = _scene().set("secondary.flux.index", 3.0)
    assert float(scene.secondary.flux.index) == 3.0
    assert is_flux_param("secondary.flux.ratio")
    assert not is_flux_param("secondary.flux.index")
    assert (
        resolve_flux_param(["secondary.dra", "secondary.flux.ratio"])
        == "secondary.flux.ratio"
    )
    with pytest.raises(ValueError, match="negative"):
        PowerLaw(-0.1, 1.0)


def test_resolved_flux_only_lowers_the_visibilities():
    u, v = onp.array([0.0, 30.0]), onp.array([0.0, 10.0])
    scene = System(star=PointSource(), background=Resolved(flux=0.25))
    cvis = scene.model(u, v, 1.65e-6)
    assert np.allclose(cvis, np.array([1.0, 1.0 / 1.25]))
    image = scene.render(npix=16, fov_mas=10.0)
    assert np.isclose(float(image.sum()), 1.0)


def test_several_files_read_as_one(tmp_path):
    paths = [
        write_oifits(_tables(waves=(1.6e-6, 1.7e-6)), tmp_path / "a.fits"),
        write_oifits(_tables(waves=(1.55e-6,)), tmp_path / "b.fits"),
    ]
    joined = OIData(paths)
    singles = [OIData(path) for path in paths]
    assert joined.u.size == sum(single.u.size for single in singles)
    assert joined.phi.size == sum(single.phi.size for single in singles)
    assert joined.wavel.shape == joined.u.shape
    assert np.allclose(
        model_loglike(TRUTH, joined),
        joint_loglike(None, singles, lambda params, index: TRUTH),
        rtol=1e-6,
    )


def _toon_loglike(model, data_obj, vis_error_rel, phi_error):
    # The hand-written likelihood this option replaces (iras08_modelling).
    model_data = data_obj.model(model)
    data, errors = data_obj.flatten_data()
    n_vis = data_obj.vis.size
    errors_vis = np.hypot(errors[:n_vis], vis_error_rel * model_data[:n_vis])
    errors_phi = np.hypot(errors[n_vis:], phi_error)
    errors = np.concatenate([errors_vis, errors_phi])
    return (
        -0.5 * np.sum((data - model_data) ** 2 / errors**2)
        - np.sum(np.log(errors))
        - data.size / 2 * np.log(2 * np.pi)
    )


def test_error_inflation_matches_hand_written_likelihood(tmp_path):
    data = OIData(write_oifits(_tables(waves=WAVES), tmp_path / "x.fits"))
    model = _scene()
    expected = _toon_loglike(model, data, 0.02, 0.01)
    got = model_loglike(model, data, vis_error_rel=0.02, phi_error=0.01)
    assert np.allclose(got, expected, rtol=1e-6)
    assert not np.allclose(got, model_loglike(model, data), rtol=1e-3)


def test_unphysical_models_are_rejected_under_jit(tmp_path):
    data = OIData(write_oifits(_tables(waves=WAVES), tmp_path / "x.fits"))

    def rim_scene(rim_flux, az_amp):
        return System(
            star=PointSource(),
            rim=ModulatedGaussianRim(
                10.0,
                2.0,
                30.0,
                20.0,
                az_amps=az_amp,
                az_pas=0.0,
                flux=rim_flux,
            ),
        )

    @jax.jit
    def logl(rim_flux, az_amp):
        return model_loglike(
            rim_scene(rim_flux, az_amp), data, reject_unphysical=True
        )

    assert np.isfinite(logl(0.3, 0.5))
    assert logl(-0.1, 0.5) == -np.inf  # negative (e.g. remainder) flux
    assert logl(0.3, 1.5) == -np.inf  # rim brightness goes negative


def test_flux_ratio_priors_must_be_non_negative():
    with pytest.raises(ValueError, match="negative"):
        numpyro_model(
            _scene(), {"secondary.flux.ratio": dist.Normal(0.0, 1.0)}, None
        )

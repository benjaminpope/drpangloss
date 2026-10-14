"""Opt-in bandwidth smearing: boxcar-channel envelope on component offsets."""

import numpy as onp
import pytest

from virgil.models import GaussianDisk, PointSource, System
from virgil.oidata import OIData
from virgil.oifits import read_oifits, write_oifits

MAS = onp.pi / 180.0 / 3600.0 / 1000.0
LAMBDA0 = 2.2e-6
R = 500.0  # GRAVITY MED-like
FLUX = 0.3


def _channels(n=9):
    waves = LAMBDA0 * (1.0 + onp.linspace(-0.2, 0.2, n))
    return waves, waves / R  # full channel width, lambda / R


def _data(dra=200.0, ddec=0.0, bandwidth=False):
    """Samples on 130 m baselines at several position angles and channels."""
    waves, widths = _channels()
    pa = onp.deg2rad([0.0, 35.0, 75.0, 110.0])
    u = onp.repeat(130.0 * onp.sin(pa), waves.size)
    v = onp.repeat(130.0 * onp.cos(pa), waves.size)
    wavel = onp.tile(waves, pa.size)
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": wavel,
            "vis": onp.full(u.size, 0.5),
            "d_vis": onp.full(u.size, 0.01),
        }
    )
    return (
        data.with_bandwidth(onp.tile(widths, pa.size)) if bandwidth else data
    )


def _binary(dra=200.0, ddec=0.0):
    return System(
        star=PointSource(), comp=PointSource(dra=dra, ddec=ddec, flux=FLUX)
    )


def _brute_force(data, dra, ddec, n_sub=2001):
    """|V| of the binary averaged over each channel, by quadrature."""
    u, v = onp.asarray(data.u), onp.asarray(data.v)
    wavel, width = onp.asarray(data.wavel), onp.asarray(data.bandwidth)
    out = onp.empty(u.size)
    frac = (onp.arange(n_sub) + 0.5) / n_sub - 0.5
    for i in range(u.size):
        lam = wavel[i] + width[i] * frac
        phase = 2 * onp.pi * MAS * (u[i] * dra + v[i] * ddec) / lam
        vis = (1.0 + FLUX * onp.exp(-1j * phase)) / (1.0 + FLUX)
        out[i] = onp.abs(vis.mean())
    return out


@pytest.mark.parametrize(
    "offset", [(200.0, 0.0), (120.0, -160.0), (-250.0, 30.0)]
)
def test_smeared_binary_matches_channel_integral(offset):
    data = _data(bandwidth=True)
    model = _binary(*offset)
    smeared = onp.abs(onp.asarray(data._cvis(model)))
    exact = _brute_force(data, *offset)
    unsmeared = onp.abs(onp.asarray(data.with_bandwidth(None)._cvis(model)))
    assert onp.max(onp.abs(smeared - exact)) < 1e-3
    # The effect is real: smearing is much larger than the tolerance.
    assert onp.max(onp.abs(unsmeared - exact)) > 5e-3


def test_zero_offset_component_is_unaffected():
    data = _data(bandwidth=True)
    model = System(star=PointSource(), disk=GaussianDisk(sigma=5.0, flux=0.4))
    assert onp.array_equal(
        onp.asarray(data._cvis(model)),
        onp.asarray(data.with_bandwidth(None)._cvis(model)),
    )


def test_default_is_bit_identical():
    data = _data()
    assert data.bandwidth is None
    model = _binary()
    assert onp.array_equal(
        onp.asarray(data._cvis(model)),
        onp.asarray(model.model(data.u, data.v, data.wavel)),
    )
    # Zero width is also the unsmeared model.
    zero = data.with_bandwidth(0.0)
    assert onp.allclose(
        onp.asarray(zero._cvis(model)),
        onp.asarray(data._cvis(model)),
        atol=1e-6,
    )


def test_nested_system_offsets_add():
    data = _data(bandwidth=True)
    nested = System(
        star=PointSource(),
        group=System(
            comp=PointSource(dra=50.0, flux=1.0), dra=150.0, flux=FLUX
        ),
    )
    flat = _binary(200.0, 0.0)
    assert onp.allclose(
        onp.asarray(data._cvis(nested)),
        onp.asarray(data._cvis(flat)),
        atol=1e-5,
    )


def test_bandwidth_survives_subset_and_wavelength_scale():
    data = _data(bandwidth=True)
    kept = data.select(wavel_min=LAMBDA0)
    assert kept.bandwidth.shape == kept.u.shape
    assert onp.allclose(
        onp.asarray(kept.bandwidth), onp.asarray(kept.wavel) / R
    )
    scaled = data.with_wavelength_scale(1.01)
    assert onp.allclose(
        onp.asarray(scaled.bandwidth), onp.asarray(data.bandwidth) * 1.01
    )


def _tables(waves, band):
    pairs = onp.array([[1, 2], [1, 3], [2, 3]])
    return {
        "info": {"TARGET": "STAR", "INSTRUME": "TEST", "MJD": 60000.0},
        "OI_WAVELENGTH": {"EFF_WAVE": waves, "EFF_BAND": band},
        "OI_VIS2": {
            "VIS2DATA": onp.full((3, waves.size), 0.5),
            "VIS2ERR": onp.full((3, waves.size), 0.01),
            "UCOORD": onp.array([10.0, 20.0, 30.0]),
            "VCOORD": onp.array([5.0, -5.0, 15.0]),
            "STA_INDEX": pairs,
        },
    }


def test_eff_band_round_trip(tmp_path):
    waves, widths = _channels(5)
    path = write_oifits(_tables(waves, widths), tmp_path / "band.fits")
    kept = OIData(path, bandwidth=True)
    assert onp.allclose(
        onp.asarray(kept.bandwidth), onp.tile(widths, 3), rtol=1e-5
    )
    # Ignored unless asked for, so reading is unchanged by default.
    assert OIData(path).bandwidth is None
    # Several files merge their widths sample by sample.
    both = OIData([path, path], bandwidth=True)
    assert both.bandwidth.shape == both.u.shape
    # A file with no widths gives none.
    plain = write_oifits(_tables(waves, 0.0), tmp_path / "plain.fits")
    assert OIData(plain, bandwidth=True).bandwidth is None


def test_unsupported_model_raises():
    from virgil.models import BinaryModelCartesian

    with pytest.raises(TypeError, match="Bandwidth smearing"):
        _data(bandwidth=True)._cvis(
            BinaryModelCartesian(dra=5.0, ddec=2.0, flux=0.1)
        )


def test_read_oifits_leaves_widths_out_by_default(tmp_path):
    waves, widths = _channels(5)
    path = write_oifits(_tables(waves, widths), tmp_path / "b.fits")
    assert "bandwidth" not in read_oifits(path)
    assert OIData(read_oifits(path)).bandwidth is None
    assert "bandwidth" in read_oifits(path, bandwidth=True)


def test_conflicting_widths_at_one_wavelength_raise(tmp_path):
    waves, widths = _channels(5)
    tables = _tables(waves, widths)
    path = write_oifits(tables, tmp_path / "c.fits")
    from astropy.io import fits

    with fits.open(path) as hdul:
        wave = [h for h in hdul if h.name == "OI_WAVELENGTH"][0]
        dup = wave.copy()
        dup.header["INSNAME"] = "OTHER"
        dup.data["EFF_BAND"] *= 2.0
        hdul.append(dup)
        with pytest.raises(ValueError, match="different EFF_BAND"):
            read_oifits(hdul, bandwidth=True)


def test_image_component_is_refused():
    from virgil.models import Image

    image = Image(onp.ones((8, 8)), pixel_scale_mas=10.0)
    with pytest.raises(TypeError, match="Bandwidth smearing"):
        _data(bandwidth=True)._cvis(System(star=PointSource(), img=image))


def test_time_dependent_path_matches_smeared_snapshot():
    import jax

    pytest.importorskip("jaxoplanet")
    from virgil.models import Attached, smeared_model
    from virgil.orbits import KeplerOrbit

    orbit = KeplerOrbit(
        period=700.0,
        dt_peri=-120.0,
        ecc=0.3,
        inc=50.0,
        omega=60.0,
        Omega=120.0,
        a_mas=200.0,
        t_ref=0.0,
    )
    scene = System(
        primary=PointSource(), comp=Attached(PointSource(flux=FLUX), orbit)
    )
    mjd = 120.0
    with jax.enable_x64(True):
        base = _data()
        data = OIData(
            {
                "u": onp.asarray(base.u),
                "v": onp.asarray(base.v),
                "wavel": onp.asarray(base.wavel),
                "vis": onp.full(base.u.size, 0.5),
                "d_vis": onp.full(base.u.size, 0.01),
                "mjd": onp.full(base.u.size, mjd),
                "bandwidth": onp.asarray(base.wavel) / R,
            }
        )
        got = onp.asarray(data._cvis(scene))
        want = onp.asarray(
            smeared_model(
                scene.at(mjd), data.u, data.v, data.wavel, data.bandwidth
            )
        )
    # The stored sample times are float32, which moves the companion by
    # ~1e-6 in the phasor; the smearing envelope itself is ~1e-2 or more.
    onp.testing.assert_allclose(got, want, atol=1e-4)

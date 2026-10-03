"""OIFITS reading/writing and the OIData observables built from it."""

import jax.numpy as np
import numpy as onp
import pyoifits
import pytest
from astropy.io import fits

from virgil.likelihood import loglike, model_loglike
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData, closure_phases, cp_indices
from virgil.oifits import read_oifits, write_oifits


STATIONS = onp.array([[0.0, 0.0], [3.2, 0.2], [1.4, 2.6], [-1.1, 1.8]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
TRUTH = BinaryModelCartesian(dra=60.0, ddec=-40.0, flux=0.05)


def _baselines(pairs=PAIRS):
    delta = STATIONS[pairs[:, 1] - 1] - STATIONS[pairs[:, 0] - 1]
    return delta[:, 0], delta[:, 1]


def _tables(waves=(4.8e-6,), model=TRUTH, target="STAR"):
    """OIFITS tables sampling ``model`` noiselessly at ``waves``."""
    waves = onp.asarray(waves)
    u, v = _baselines()
    cvis = onp.asarray(
        model.model(u[:, None], v[:, None], waves[None, :])
    )  # (n_baseline, n_wave)
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    cp = onp.angle(cvis[i1] * cvis[i2] / cvis[i3])
    u1, v1 = _baselines(TRIANGLES[:, [0, 1]])
    u2, v2 = _baselines(TRIANGLES[:, [1, 2]])
    return {
        "info": {"TARGET": target, "INSTRUME": "TEST", "MJD": 60000.0},
        "OI_WAVELENGTH": {"EFF_WAVE": waves, "EFF_BAND": 0.1e-6},
        "OI_VIS2": {
            "VIS2DATA": onp.abs(cvis) ** 2,
            "VIS2ERR": onp.full(cvis.shape, 1e-3),
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": PAIRS,
        },
        "OI_T3": {
            "T3PHI": onp.rad2deg(cp),
            "T3PHIERR": onp.full(cp.shape, 0.5),
            "U1COORD": u1,
            "V1COORD": v1,
            "U2COORD": u2,
            "V2COORD": v2,
            "STA_INDEX": TRIANGLES,
        },
    }


def test_roundtrip_single_channel(tmp_path):
    tables = _tables()
    path = write_oifits(tables, tmp_path / "one.oifits")
    data = OIData(path)

    assert data.v2_flag and data.cp_flag
    assert np.allclose(data.u, tables["OI_VIS2"]["UCOORD"])
    assert np.allclose(data.vis, tables["OI_VIS2"]["VIS2DATA"].ravel())
    assert np.allclose(
        data.phi, onp.deg2rad(tables["OI_T3"]["T3PHI"].ravel()), atol=1e-6
    )
    assert data.wavel.shape == (1,)
    # Noiseless data: the model reproduces the file exactly.
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)


def test_written_file_is_standard_oifits(tmp_path):
    path = write_oifits(_tables(waves=(4.6e-6, 5.0e-6)), tmp_path / "x.fits")
    with fits.open(path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert hdul[0].header["CONTENT"] == "OIFITS2"
    # OI_WAVELENGTH is not the first extension, which the old reader assumed.
    assert names[1] != "OI_WAVELENGTH"
    opened = pyoifits.open(path)
    assert {"OI_VIS2", "OI_T3"} <= {h.name for h in opened.get_dataHDUs()}
    # pyoifits objects are astropy HDULists and read the same way.
    assert np.allclose(OIData(opened).vis, OIData(path).vis)


def test_multi_wavelength_closure_phases_use_their_own_channel(tmp_path):
    waves = onp.array([4.4e-6, 4.8e-6, 5.2e-6])
    path = write_oifits(_tables(waves=waves), tmp_path / "multi.oifits")
    data = OIData(path)

    n_bl, n_cp, n_wave = len(PAIRS), len(TRIANGLES), waves.size
    assert data.u.shape == (n_bl * n_wave,)
    assert data.wavel.shape == (n_bl * n_wave,)
    assert data.phi.shape == (n_cp * n_wave,)
    # Chromatic data: the model at each sample's own wavelength matches.
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)
    # ... while ignoring the chromaticity does not.
    grey = TRUTH.model(data.u, data.v, waves[1])
    assert not np.allclose(
        closure_phases(grey, data.i_cps1, data.i_cps2, data.i_cps3),
        data.phi,
        atol=1e-3,
    )


def test_flags_and_non_finite_values_are_dropped(tmp_path):
    waves = onp.array([4.4e-6, 4.8e-6])
    tables = _tables(waves=waves)
    vis_flag = onp.zeros((len(PAIRS), 2), dtype=bool)
    vis_flag[2, 1] = True
    tables["OI_VIS2"]["FLAG"] = vis_flag
    tables["OI_T3"]["T3PHI"][1, 0] = onp.nan
    data = OIData(write_oifits(tables, tmp_path / "flags.oifits"))

    assert data.vis.size == len(PAIRS) * 2 - 1
    assert data.phi.size == len(TRIANGLES) * 2 - 1
    # The flagged baseline is still available to the closure phases.
    assert data.u.size == len(PAIRS) * 2
    assert data.vis_index.size == data.vis.size
    assert np.isfinite(model_loglike(TRUTH, data))
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)


def test_absolute_phases_from_oi_vis_with_reversed_baseline(tmp_path):
    tables = _tables()
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u, v, 4.8e-6))
    pairs = PAIRS.copy()
    pairs[0] = pairs[0][::-1]  # stored as (2, 1) in OI_VIS
    phase = onp.rad2deg(onp.angle(cvis))
    phase[0] = -phase[0]
    tables["OI_VIS"] = {
        "VISAMP": onp.abs(cvis),
        "VISAMPERR": onp.full(cvis.shape, 1e-3),
        "VISPHI": phase,
        "VISPHIERR": onp.full(cvis.shape, 0.5),
        "UCOORD": onp.r_[-u[0], u[1:]],
        "VCOORD": onp.r_[-v[0], v[1:]],
        "STA_INDEX": pairs,
    }
    del tables["OI_T3"]
    data = OIData(write_oifits(tables, tmp_path / "vis.oifits"))

    assert not data.cp_flag
    assert np.allclose(data.phi, onp.angle(cvis), atol=1e-6)


def test_several_targets_need_a_choice(tmp_path):
    science = _tables()
    calibrator = _tables(model=BinaryModelCartesian(0.0, 0.0, 0.0))
    tables = dict(science)
    for table in ("OI_VIS2", "OI_T3"):
        n = len(science[table]["STA_INDEX"])
        tables[table] = {
            key: onp.concatenate([science[table][key], calibrator[table][key]])
            for key in science[table]
        }
        tables[table]["TARGET_ID"] = onp.repeat([1, 2], n)
    tables["OI_TARGET"] = {"TARGET_ID": [1, 2], "TARGET": ["SCI", "CAL"]}
    path = write_oifits(tables, tmp_path / "targets.oifits")

    with pytest.raises(ValueError, match="several targets"):
        OIData(path)
    sci = OIData(path, target="SCI")
    cal = OIData(path, target=2)
    assert sci.vis.size == cal.vis.size == len(PAIRS)
    assert np.allclose(sci.vis, science["OI_VIS2"]["VIS2DATA"].ravel())
    assert np.allclose(cal.vis, 1.0)


def test_missing_or_reversed_closure_leg_raises_clearly():
    with pytest.raises(ValueError, match="stored reversed"):
        cp_indices([[1, 2], [3, 2], [1, 3]], [[1, 2, 3]])
    with pytest.raises(ValueError, match="missing"):
        cp_indices([[1, 2], [1, 3]], [[1, 2, 3]])


def test_read_oifits_record_matches_oidata(tmp_path):
    path = write_oifits(_tables(), tmp_path / "rec.oifits")
    record = read_oifits(path)
    assert record["phi_unit"] == "rad"
    assert np.allclose(OIData(record).phi, OIData(path).phi)


# === OIData observables built from dictionaries ===


def _dict_data(**extra):
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u, v, 4.8e-6))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    return {
        "u": u,
        "v": v,
        "wavel": 4.8e-6,
        "vis": onp.abs(cvis) ** 2,
        "d_vis": onp.full(len(u), 1e-3),
        "phi": onp.angle(cvis[i1] * cvis[i2] / cvis[i3]),
        "d_phi": onp.full(len(i1), 1e-2),
        "i_cps1": i1,
        "i_cps2": i2,
        "i_cps3": i3,
        **extra,
    }


def test_vis_mode_converts_data_without_an_operator():
    v2 = OIData(_dict_data())
    amp = OIData(_dict_data(vis_mode="amp"))
    assert np.allclose(amp.vis, np.sqrt(v2.vis))
    # Data and model are compared in the same (amplitude) basis.
    assert np.allclose(amp.model(TRUTH), amp.flatten_data()[0], atol=1e-5)


def test_pre_projected_closure_phases_are_not_projected_again():
    # One triangle: any two of four share a baseline, so their outputs
    # correlate and are rotated (see test_closure).
    raw = _dict_data()
    phi_mat = onp.eye(len(TRIANGLES))[:1]
    projected = OIData({**raw, "phi_mat": phi_mat})
    again = OIData({**raw, "phi": phi_mat @ raw["phi"], "phi_mat": phi_mat})
    assert projected.phi.shape == again.phi.shape == (1,)
    assert np.allclose(projected.phi, again.phi)
    assert again.model(TRUTH).shape == again.flatten_data()[0].shape


def test_dict_with_several_channels_is_expanded():
    waves = onp.array([4.4e-6, 5.2e-6])
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u[:, None], v[:, None], waves[None, :]))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": waves,
            "vis": onp.abs(cvis) ** 2,
            "d_vis": onp.full(cvis.shape, 1e-3),
            "phi": onp.angle(cvis[i1] * cvis[i2] / cvis[i3]),
            "d_phi": onp.full((len(i1), 2), 1e-2),
            "i_cps1": i1,
            "i_cps2": i2,
            "i_cps3": i3,
        }
    )
    assert data.u.size == 2 * len(u)
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)


def test_ambiguous_wavelengths_are_rejected():
    with pytest.raises(ValueError, match="wavelength"):
        OIData(_dict_data(wavel=onp.array([4.4e-6, 4.8e-6, 5.2e-6])))


def test_phase_residuals_wrap_around_pi():
    data = OIData(_dict_data())
    prediction = data.flatten_data()[0]
    n_vis = data.vis.size
    shifted = prediction.at[n_vis:].add(2.0 * onp.pi)
    assert np.allclose(data.residuals(shifted), 0.0, atol=1e-5)

    params = ["dra", "ddec", "flux"]
    values = [60.0, -40.0, 0.05]
    near_pi = OIData(_dict_data(phi=onp.full(len(TRIANGLES), onp.pi - 1e-3)))
    flipped = OIData(_dict_data(phi=onp.full(len(TRIANGLES), -onp.pi + 1e-3)))
    # Data at +π and -π are the same angle, so the likelihood is the same.
    assert np.allclose(
        loglike(values, params, near_pi, BinaryModelCartesian),
        loglike(values, params, flipped, BinaryModelCartesian),
        rtol=1e-4,
    )


def test_v2_error_from_negative_amplitude_is_not_zero():
    data = OIData(
        _dict_data(
            v2_flag=False,
            vis_mode="v2",
            vis=onp.full(len(PAIRS), -0.01),
            d_vis=onp.full(len(PAIRS), 0.05),
        )
    )
    assert np.all(data.d_vis > 1e-3)


def test_closure_phase_only_file_round_trips(tmp_path):
    waves = onp.array([4.4e-6, 5.2e-6])
    tables = _tables(waves=waves)
    del tables["OI_VIS2"]
    data = OIData(write_oifits(tables, tmp_path / "t3.oifits"))

    assert data.cp_flag
    assert data.vis.size == 0
    assert data.phi.size == len(TRIANGLES) * waves.size
    # Shared baselines are merged: six unique pairs per channel.
    assert data.u.size == len(PAIRS) * waves.size
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)
    assert np.isfinite(model_loglike(TRUTH, data))


def _two_instrument_file(names):
    """One HDUList holding the same tables under two INSNAMEs."""
    from virgil.oifits import build_hdulist

    hduls = []
    for name, waves in zip(names, ((2.2e-6,), (2.0e-6, 2.2e-6, 2.4e-6))):
        tables = _tables(waves=waves)
        tables["info"]["INSNAME"] = name
        hduls.append(build_hdulist(tables))
    combined = fits.HDUList([hdu.copy() for hdu in hduls[0]])
    for hdu in hduls[1]:
        if hdu.header.get("EXTNAME", "").strip() in {
            "OI_WAVELENGTH",
            "OI_VIS2",
            "OI_T3",
        }:
            combined.append(hdu.copy())
    return combined


def test_gravity_fringe_tracker_and_science_tables_are_not_merged():
    hdul = _two_instrument_file(["GRAVITY_FT", "GRAVITY_SC"])
    with pytest.raises(ValueError, match="insname="):
        read_oifits(hdul)
    record = read_oifits(hdul, insname="GRAVITY_SC")
    assert onp.unique(record["wavel"]).size == 3  # the SC table's channels
    with pytest.raises(ValueError, match="No tables have INSNAME"):
        read_oifits(hdul, insname="GRAVITY_SC_P1")


def test_insname_selects_tables_from_other_instruments():
    hdul = _two_instrument_file(["PIONIER_A", "PIONIER_B"])
    both = read_oifits(hdul)  # not GRAVITY FT + SC: merged as before
    one = read_oifits(hdul, insname=["PIONIER_A"])
    assert both["u"].size == 4 * one["u"].size  # 1 + 3 channels against 1


def test_differential_visphi_is_not_read_as_absolute(tmp_path):
    from virgil.oifits import build_hdulist

    tables = _tables()
    u, v = _baselines()
    tables["OI_VIS"] = {
        "VISAMP": onp.ones(u.size),
        "VISAMPERR": onp.full(u.size, 1e-3),
        "VISPHI": onp.zeros(u.size),
        "VISPHIERR": onp.full(u.size, 0.5),
        "UCOORD": u,
        "VCOORD": v,
        "STA_INDEX": PAIRS,
    }
    del tables["OI_T3"]
    hdul = build_hdulist(tables)
    for hdu in hdul:
        if hdu.header.get("EXTNAME", "").strip() == "OI_VIS":
            hdu.header["PHITYP"] = "differential"
    with pytest.raises(ValueError, match="PHITYP"):
        read_oifits(hdul)


def test_insname_selection_drops_emptied_table_types():
    from virgil.oifits import _collect_tables, _select_insname

    hdul = _two_instrument_file(["PIONIER_A", "PIONIER_B"])
    for hdu in hdul:
        if hdu.header.get("EXTNAME", "").strip() == "OI_T3" and (
            hdu.header.get("INSNAME", "").strip() == "PIONIER_B"
        ):
            hdu.header["EXTNAME"] = "OI_IGNORED"  # B has no closure phases
    tables = _select_insname(_collect_tables(hdul), ["PIONIER_B"])
    assert "OI_T3" not in tables
    assert len(tables["OI_VIS2"]) == 1


def test_phityp_is_checked_only_for_the_chosen_target():
    from virgil.oifits import build_hdulist

    tables = _tables()
    u, v = _baselines()
    tables["OI_VIS"] = {
        "VISAMP": onp.ones(u.size),
        "VISAMPERR": onp.full(u.size, 1e-3),
        "VISPHI": onp.zeros(u.size),
        "VISPHIERR": onp.full(u.size, 0.5),
        "UCOORD": u,
        "VCOORD": v,
        "STA_INDEX": PAIRS,
    }
    del tables["OI_T3"]
    hdul = build_hdulist(tables)
    vis = next(
        h for h in hdul if h.header.get("EXTNAME", "").strip() == "OI_VIS"
    )
    other = vis.copy()
    other.data["TARGET_ID"] = 99  # rows of another target
    other.header["PHITYP"] = "differential"
    hdul.append(other)
    record = read_oifits(hdul, target="STAR")  # its table is absolute
    assert onp.isfinite(record["phi"]).all()

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from astropy.io import fits
from matplotlib.ticker import FuncFormatter

from drpangloss.models import OIData, chi2ppf, closure_phases
from drpangloss.plotting import (
    plot_contrast_limit_map,
    plot_data_model_correlation,
)


def _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None):
    return {
        "u": np.array([0.0, 0.0, 0.0]),
        "v": np.array([0.0, 0.0, 0.0]),
        "wavel": np.array([4.8e-6]),
        "vis": np.array([1.0, 1.0, 1.0]),
        "d_vis": np.array([0.1, 0.1, 0.1]),
        "phi": np.array([0.0, 0.0, 0.0]),
        "d_phi": np.array([1.0, 1.0, 1.0]),
        "v2_flag": True,
        "cp_flag": cp_flag,
        "i_cps1": i_cps1,
        "i_cps2": i_cps2,
        "i_cps3": i_cps3,
    }


class _FakeOIFITS:
    def __init__(self, hdus):
        self._hdus = hdus
        self._by_name = {
            hdu.name: hdu for hdu in hdus if getattr(hdu, "name", "")
        }

    def get_dataHDUs(self):
        return [
            hdu
            for hdu in self._hdus[1:]
            if hdu.name in {"OI_VIS", "OI_VIS2", "OI_T3", "OI_PHI"}
        ]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._hdus[key]
        return self._by_name[key]


def _make_fake_oifits_phase_input(phase_ext, phi, d_phi, unit):
    wavelength_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="EFF_WAVE",
                format="1E",
                array=onp.array([4.8e-6], dtype=onp.float32),
            )
        ]
    )
    wavelength_hdu.name = "OI_WAVELENGTH"

    vis2_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="VIS2DATA",
                format="1E",
                array=onp.array([1.0, 0.9, 0.8], dtype=onp.float32),
            ),
            fits.Column(
                name="VIS2ERR",
                format="1E",
                array=onp.array([0.01, 0.01, 0.01], dtype=onp.float32),
            ),
            fits.Column(
                name="UCOORD",
                format="1D",
                array=onp.array([0.0, 1.0, -1.0], dtype=float),
            ),
            fits.Column(
                name="VCOORD",
                format="1D",
                array=onp.array([0.0, 1.0, 1.0], dtype=float),
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=onp.array([[1, 2], [2, 3], [1, 3]], dtype=onp.int16),
            ),
        ]
    )
    vis2_hdu.name = "OI_VIS2"

    if phase_ext == "OI_PHI":
        phase_columns = [
            fits.Column(
                name="VISPHI",
                format="1D",
                unit=unit,
                array=onp.asarray(phi, dtype=float),
            ),
            fits.Column(
                name="VISERR",
                format="1D",
                unit=unit,
                array=onp.asarray(d_phi, dtype=float),
            ),
        ]
    else:
        phase_columns = [
            fits.Column(
                name="T3PHI",
                format="1D",
                unit=unit,
                array=onp.asarray(phi, dtype=float),
            ),
            fits.Column(
                name="T3PHIERR",
                format="1D",
                unit=unit,
                array=onp.asarray(d_phi, dtype=float),
            ),
            fits.Column(
                name="STA_INDEX",
                format="3I",
                array=onp.array([[1, 2, 3]], dtype=onp.int16),
            ),
        ]

    phase_hdu = fits.BinTableHDU.from_columns(phase_columns)
    phase_hdu.name = phase_ext

    return _FakeOIFITS(
        [fits.PrimaryHDU(), wavelength_hdu, vis2_hdu, phase_hdu]
    )


def test_to_phases_absolute_returns_radians():
    data = OIData(
        _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    )
    cvis = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    phases = data.to_phases(cvis)
    assert np.allclose(phases, np.array([0.0, np.pi / 2.0]))


def test_closure_phases_radian_convention():
    cvis = np.exp(1j * np.deg2rad(np.array([10.0, 30.0, 25.0])))
    cps = closure_phases(
        cvis,
        np.array([0]),
        np.array([1]),
        np.array([2]),
    )
    assert np.allclose(cps, np.deg2rad(np.array([15.0])))


def test_dict_phase_unit_degrees_converted_to_radians():
    data = _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    data["phi"] = np.array([0.0, 90.0, -180.0])
    data["d_phi"] = np.array([1.0, 2.0, 3.0])
    data["phi_unit"] = "deg"

    oidata = OIData(data)

    assert np.allclose(oidata.phi, np.deg2rad(np.array([0.0, 90.0, -180.0])))
    assert np.allclose(oidata.d_phi, np.deg2rad(np.array([1.0, 2.0, 3.0])))


def test_oifits_phase_units_convert_values_and_uncertainties_to_radians():
    cases = [
        ("OI_PHI", "DEGREES", onp.array([10.0, -20.0]), onp.array([1.0, 2.0])),
        ("OI_PHI", "RADIANS", onp.array([0.1, -0.2]), onp.array([0.01, 0.02])),
        ("OI_PHI", None, onp.array([15.0, -30.0]), onp.array([0.5, 0.75])),
        ("OI_T3", "DEGREES", onp.array([12.0]), onp.array([1.5])),
        ("OI_T3", "RADIANS", onp.array([0.4]), onp.array([0.05])),
        ("OI_T3", None, onp.array([-25.0]), onp.array([2.5])),
    ]

    for phase_ext, unit, phi, d_phi in cases:
        oidata = OIData(
            _make_fake_oifits_phase_input(phase_ext, phi, d_phi, unit)
        )
        if unit == "RADIANS":
            expected_phi = phi
            expected_d_phi = d_phi
        else:
            expected_phi = onp.deg2rad(phi)
            expected_d_phi = onp.deg2rad(d_phi)

        assert np.allclose(oidata.phi, expected_phi)
        assert np.allclose(oidata.d_phi, expected_d_phi)


def test_cp_flag_inferred_from_indices_when_missing():
    data = _base_dict(
        cp_flag=False,
        i_cps1=np.array([0]),
        i_cps2=np.array([1]),
        i_cps3=np.array([2]),
    )
    del data["cp_flag"]
    oidata = OIData(data)
    assert oidata.cp_flag is True


def test_cp_flag_string_false_is_parsed_as_false():
    data = _base_dict(
        cp_flag="False",
        i_cps1=np.array([0]),
        i_cps2=np.array([1]),
        i_cps3=np.array([2]),
    )
    oidata = OIData(data)
    assert oidata.cp_flag is False


def test_cp_flag_string_true_is_parsed_as_true():
    data = _base_dict(cp_flag="true", i_cps1=None, i_cps2=None, i_cps3=None)
    oidata = OIData(data)
    assert oidata.cp_flag is True


def test_v2_flag_string_false_is_parsed_as_false():
    data = _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    data["v2_flag"] = "False"
    oidata = OIData(data)
    assert oidata.v2_flag is False


def test_chi2ppf_df1_returns_finite_values():
    p = np.array([1e-6, 0.5, 0.95, 1.0 - 1e-6])
    q = chi2ppf(p, 1.0)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0.0)


def test_visibility_correlation_ticks_use_adaptive_float_formatter():
    data = OIData(_base_dict())
    pred = {
        "vis_mean": np.array([0.98, 1.0, 1.02]),
        "vis_std": np.array([0.01, 0.01, 0.01]),
        "phi_mean": np.array([0.0, 0.0, 0.0]),
        "phi_std": np.array([1.0, 1.0, 1.0]),
    }

    fig, (ax1, _) = plot_data_model_correlation(
        data,
        predictions_by_label={"demo": pred},
    )
    xfmt = ax1.xaxis.get_major_formatter()
    yfmt = ax1.yaxis.get_major_formatter()
    assert isinstance(xfmt, FuncFormatter)
    assert xfmt is yfmt
    assert xfmt(0.991, 0) != xfmt(0.992, 0)
    assert xfmt(0.985, 0) == "98.5"
    assert "%" not in xfmt(0.991, 0)
    assert "(V2, %)" in ax1.get_xlabel()
    assert "(V2, %)" in ax1.get_ylabel()
    plt.close(fig)


def test_delta_mag_map_uses_reversed_colormap_by_default():
    limit_map = np.array([[1e-3, 2e-3], [5e-4, 1e-3]])
    dra = np.array([-1.0, 1.0])
    ddec = np.array([-1.0, 1.0])

    fig, ax = plot_contrast_limit_map(
        limit_map,
        dra,
        ddec,
        unit_mode="delta_mag",
        cmap="inferno",
    )
    assert ax.images[0].get_cmap().name == "inferno_r"
    plt.close(fig)


def test_delta_mag_map_keeps_explicit_reversed_colormap():
    limit_map = np.array([[1e-3, 2e-3], [5e-4, 1e-3]])
    dra = np.array([-1.0, 1.0])
    ddec = np.array([-1.0, 1.0])

    fig, ax = plot_contrast_limit_map(
        limit_map,
        dra,
        ddec,
        unit_mode="delta_mag",
        cmap="inferno_r",
    )
    assert ax.images[0].get_cmap().name == "inferno_r"
    plt.close(fig)

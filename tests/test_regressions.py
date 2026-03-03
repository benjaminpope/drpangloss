import jax.numpy as np
import matplotlib.pyplot as plt
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


def test_to_phases_absolute_returns_degrees():
    data = OIData(
        _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    )
    cvis = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    phases = data.to_phases(cvis)
    assert np.allclose(phases, np.array([0.0, 90.0]))


def test_closure_phases_degree_convention():
    cvis = np.exp(1j * np.deg2rad(np.array([10.0, 30.0, 25.0])))
    cps = closure_phases(
        cvis,
        np.array([0]),
        np.array([1]),
        np.array([2]),
    )
    assert np.allclose(cps, np.array([15.0]))


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

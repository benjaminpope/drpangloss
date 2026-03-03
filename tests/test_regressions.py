import jax.numpy as np

from drpangloss.models import OIData, chi2ppf, closure_phases


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
    data = OIData(_base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None))
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

"""DiskPipeline end to end on tiny synthetic data: API contract, outputs,
round trip, resume, quicklook and the CLI.

The data are one Gaussian ring with a lopsided harmonic and a companion,
observed with five channels (so the spectral index is fitted). The runs
are deliberately small, a coarse geometry grid, two ring counts, two
length scales and a few NUTS steps: these tests check the plumbing and the
contract, not the sampler.
"""

import inspect
import json

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

import virgil  # noqa: E402
from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.models import (  # noqa: E402
    GaussianRing,
    Harmonics,
    PointSource,
    MultipoleDisk,
    System,
)
from virgil.pipeline import (  # noqa: E402
    SCHEMA,
    Check,
    ConfigMismatchError,
    DiskPipeline,
    load,
)
from virgil.pipeline import _io, cli, disk  # noqa: E402
from virgil.pipeline._common import _quantiles  # noqa: E402

pytestmark = pytest.mark.xdist_group("pipeline_disk")

TRUTH = System(
    star=PointSource(flux=1.0),
    ring=MultipoleDisk(
        GaussianRing(5.0, 1.0),
        inc=50.0,
        pa=30.0,
        flux=0.4,
        n_r=48,
        harmonics=Harmonics(re=(0.4,), im=(0.2,)),
    ),
    planet=PointSource(dra=-6.0, ddec=-3.0, flux=0.05),
)
SMALL = {
    "n_inc": 5,
    "n_pa": 6,
    "max_rings": 2,
    "r_out_mas": 12.0,
    "ell_halffringes": [1.0, 3.0],
    "companion": True,
    "quicklook_execute": False,
}
TINY_NUTS = {
    "posterior": True,
    "num_warmup": 20,
    "num_samples": 20,
    "num_chains": 2,
}


@pytest.fixture(scope="module")
def data():
    template = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 5),
        hour_angles_h=(-2.0, 0.0, 2.0),
        sigma_v2=0.01,
        sigma_cp_deg=3.0,
    )
    return template.with_model(TRUTH, key=jax.random.PRNGKey(3))


@pytest.fixture(scope="module")
def gp_run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("disk_gp") / "run"
    pipeline = DiskPipeline(
        data, output=out, **{**SMALL, "quicklook_execute": True}
    )
    return pipeline, pipeline.run()


@pytest.fixture(scope="module")
def rings_run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("disk_rings") / "run"
    pipeline = DiskPipeline(data, "rings", output=out, **SMALL, **TINY_NUTS)
    return pipeline, pipeline.run()


# --- the contract --------------------------------------------------------------

API = {
    "DiskPipeline": "(data, model=None, *, output='run', **settings)",
    "DiskPipeline.run": "(self, through=None, *, resume=True)",
    "DiskPipeline.from_oifits": "(paths, model=None, *, output='run', **settings)",
    "DiskPipeline.from_config": "(path, model=None, **overrides)",
    "DiskPipeline.defaults": "()",
}
STAGES = (
    "load",
    "overview",
    "geometry",
    "profile",
    "compare",
    "posterior",
    "quicklook",
)
DEFAULTS = {
    "r_in_mas": None,
    "r_out_mas": None,
    "n_r": None,
    "n_inc": 10,
    "n_pa": 12,
    "inc_max_deg": 85.0,
    "max_rings": 4,
    "ell_halffringes": [1.0, 2.0, 4.0],
    "harmonic_degree": 1,
    "spectral_index": None,
    "spectral_index_min_channels": 3,
    "ring_colour": "power_law",
    "wavel0_m": None,
    "star_temperature_k": None,
    "companion": False,
    "companion_min_delta_chi2": 25.0,
    "companion_max_sep_mas": None,
    "companion_step_mas": None,
    "length_scale_nats": 3.0,
    "ring_count_nats": 3.0,
    "inner_edge_fraction": 0.1,
    "inner_edge_correlation": 0.9,
    "outer_edge_fraction": 0.01,
    "inclination_min_deg": 15.0,
    "wavel_range": None,
    "error_floor": None,
    "error_scale": "quoted",
    "batch_size": None,
    "posterior": False,
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "chain_method": "vectorized",
    "seed": 0,
    "quicklook_execute": True,
}
COMMON_KEYS = {
    "schema": None,
    "pipeline": None,
    "stability": None,
    "status": None,
    "worst_check": None,
    "warnings": None,
    "checks": None,
    "data": {
        "n_vis",
        "n_phi",
        "n_independent",
        "closure_phases",
        "wavel_min_m",
        "wavel_max_m",
        "baseline_min_m",
        "baseline_max_m",
        "resolution_mas",
        "fov_mas",
    },
    "model": {
        "name",
        "label",
        "resolution_mas",
        "r_in_mas",
        "r_out_mas",
        "n_r_rings",
        "n_r_gp",
        "n_channels",
        "colour",
        "wavel0_m",
        "ell_mas",
    },
    "geometry": {
        "inc_deg",
        "pa_deg",
        "r_mas",
        "sigma_mas",
        "flux",
        "chi2_reduced",
        "converged",
        "gp_sigma",
        "laplace_sd",
        "grid",
    },
    "profile": {"candidates"},
    "chi2": {"n_independent", "error_scale", "reduced", "error_scales"},
    "compare": {"best", "rows", "residual_search", "raised_search"},
}
GP_KEYS = {
    **COMMON_KEYS,
    "disk": {
        "model",
        "best",
        "ell_mas",
        "flux",
        "inc_deg",
        "pa_deg",
        "companion",
    },
}
RINGS_KEYS = {
    **COMMON_KEYS,
    "disk": {
        "model",
        "best",
        "n_rings",
        "rings",
        "inc_deg",
        "pa_deg",
        "companion",
    },
    "posterior": {
        "best",
        "params",
        "noise",
        "num_chains",
        "num_samples",
        "num_warmup",
        "r_hat_max",
        "ess_bulk_min",
        "divergence_fraction",
    },
}
GP_CHECKS = [
    "chi2_reduced",
    "length_scale",
    "inner_edge",
    "outer_edge",
    "inclination",
    "absorbed_companion",
]
RINGS_CHECKS = [
    "chi2_reduced",
    "ring_count",
    "outer_edge",
    "inclination",
    "absorbed_companion",
    "prior_bound",
    "r_hat",
    "ess",
    "divergences",
]


def _signature(name):
    obj = {"DiskPipeline": DiskPipeline}[name.split(".")[0]]
    for part in name.split(".")[1:]:
        obj = getattr(obj, part)
    params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in inspect.signature(obj).parameters.values()
    ]
    return str(
        inspect.signature(obj).replace(
            parameters=params, return_annotation=inspect.Signature.empty
        )
    )


def _keys(summary):
    return {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in summary.items()
    }


def test_api_snapshot():
    assert {k: _signature(k) for k in API} == API
    assert DiskPipeline.STAGES == STAGES
    assert DiskPipeline.defaults() == DEFAULTS
    # model=None is "gp", the option that assumes least about the disk.
    assert DiskPipeline._default_model(None) == "gp"
    assert DiskPipeline.STABILITY == "provisional"
    assert DiskPipeline.NAME == "disk"
    assert SCHEMA == "virgil-pipeline-run-v1"
    assert "DiskPipeline" in virgil.pipeline.__all__
    assert sorted(disk.models()) == ["gp", "rings"]
    assert disk.CHECKS == (
        "chi2_reduced",
        "length_scale",
        "ring_count",
        "inner_edge",
        "outer_edge",
        "inclination",
        "absorbed_companion",
    )
    for entry in disk.models().values():
        assert set(entry.checks) <= set(disk.CHECKS)
        assert [c for c in disk.CHECKS if c in entry.checks] == list(
            entry.checks
        )


def test_provisional_tier_and_default_model(data):
    from virgil import _stability

    assert DiskPipeline.STABILITY == "provisional"
    assert (
        _stability.ADMONITION.splitlines()[0].strip() in DiskPipeline.__doc__
    )
    pipeline = DiskPipeline(data)
    assert pipeline.name == "gp" and pipeline.model == "gp"
    assert repr(pipeline) == "DiskPipeline(output='run', model='gp')"
    assert DiskPipeline(data, "rings").name == "rings"


def test_settings_and_model_are_validated(data):
    with pytest.raises(ValueError, match="model must be one of"):
        DiskPipeline(data, "cone")
    with pytest.raises(ValueError, match="model must be one of"):
        DiskPipeline(data, PointSource())
    with pytest.raises(ValueError, match="r_out_mas must exceed r_in_mas"):
        DiskPipeline(data, r_in_mas=5.0, r_out_mas=3.0)
    with pytest.raises(ValueError, match="n_r"):
        DiskPipeline(data, n_r=4)
    with pytest.raises(ValueError, match="inc_max_deg"):
        DiskPipeline(data, inc_max_deg=95.0)
    with pytest.raises(ValueError, match="max_rings"):
        DiskPipeline(data, max_rings=0)
    with pytest.raises(ValueError, match="ell_halffringes"):
        DiskPipeline(data, ell_halffringes=[1.0, -2.0])
    with pytest.raises(ValueError, match="spectral_index"):
        DiskPipeline(data, spectral_index="yes")
    with pytest.raises(ValueError, match="error_scale"):
        DiskPipeline(data, error_scale="rescaled")
    with pytest.raises(TypeError, match="unknown settings"):
        DiskPipeline(data, num_sample=10)
    # The Gaussian process's evidence needs the quoted errors.
    with pytest.raises(ValueError, match="not available for model="):
        DiskPipeline(data, "gp", error_scale="fit")
    DiskPipeline(data, "rings", error_scale="fit")


# --- the run folder -----------------------------------------------------------


def test_gp_run_contract(gp_run):
    pipeline, res = gp_run
    assert res.status == "complete"
    assert _keys(res.summary) == GP_KEYS
    # stage_warnings appears only when a stage warned.
    names = [c.name for c in res.checks if c.name != "stage_warnings"]
    assert names == GP_CHECKS
    assert all(isinstance(c, Check) for c in res.checks)
    assert res.summary["stability"] == "provisional"
    assert res.summary["model"]["name"] == "gp"
    assert res.checks[0].name == "chi2_reduced"

    out = pipeline.output
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "complete" and list(run["stages"]) == list(STAGES)
    assert set(run["config"]) == set(DEFAULTS)
    assert run["model"]["models"] == ["gp"]
    for name in (
        "summary.json",
        "grids.h5",
        "quicklook.ipynb",
        "data/processed.oifits",
        "models/best/manifest.json",
        "models/geometry/values.npz",
        "models/gp_ell0/manifest.json",
        "models/gp_ell1/manifest.json",
        "plots/geometry_grid.png",
        "plots/disk_summary.png",
        "plots/overview_uv.png",
        "stages/compare/report.json",
    ):
        assert (out / name).exists(), name
    # posterior=False leaves no samples behind.
    assert not (out / "samples.h5").exists()
    assert not list(out.rglob("*.pkl")) and not list(out.rglob("*.pickle"))


def test_gp_run_recovers_the_disk_and_the_companion(gp_run):
    pipeline, res = gp_run
    s = res.summary
    # The ring is lopsided with a companion, so the single-ring geometry is
    # only approximate; the point is that the pipeline finds the geometry's
    # neighbourhood, the companion and a good fit on the quoted errors.
    assert abs(s["geometry"]["inc_deg"] - 50.0) < 6.0
    assert abs(s["geometry"]["pa_deg"] - 30.0) < 15.0
    # The Gaussian process releases the geometry: the disk's inclination and
    # position angle are the fitted ones, closer to the truth than a ring's.
    best_values = _io.load_model_values(pipeline.output / "models" / "best")
    assert "disk.inc" in best_values and "disk.pa" in best_values
    assert s["disk"]["inc_deg"] == pytest.approx(
        float(best_values["disk.inc"])
    )
    assert s["disk"]["pa_deg"] == pytest.approx(float(best_values["disk.pa"]))
    assert abs(s["disk"]["inc_deg"] - 50.0) < 6.0
    assert abs(s["disk"]["pa_deg"] - 30.0) < 8.0
    assert s["chi2"]["reduced"][s["disk"]["best"]] < 1.6
    planet = s["disk"]["companion"]
    assert planet is not None
    assert abs(planet["dra_mas"] + 6.0) < 0.5
    assert abs(planet["ddec_mas"] + 3.0) < 0.5
    assert planet["flux"] == pytest.approx(0.05, rel=0.3)
    assert planet["detection_delta_chi2"] > 100.0
    assert s["compare"]["residual_search"]["delta_chi2"] < 25.0
    assert s["model"]["colour"] is True
    # The profile itself: the light is where the ring is (within a
    # half-fringe) and its total is the truth's, not traded against the star.
    hf = s["model"]["resolution_mas"]
    disk_model = res.model().disk
    r, w = (onp.asarray(a) for a in disk_model.profile.quadrature())
    assert abs(float(onp.sum(w * r) / onp.sum(w)) - 5.0) < hf
    assert float(disk_model.flux) == pytest.approx(0.4, rel=0.3)
    # ℓ is a Matérn length: the grid is √3 times the squared-exponential
    # equivalents of one and three half-fringes.
    assert s["model"]["ell_mas"] == pytest.approx(
        [3**0.5 * hf, 3**0.5 * 3 * hf]
    )


def test_geometry_search_leaves_a_wrong_mode(gp_run):
    pipeline, _ = gp_run
    # A single ring can settle in a mode far from a lopsided multi-ring disk's
    # geometry; the Gaussian process must not start there.
    wrong = {**disk._gp_geometry(pipeline), "inc": 25.0, "pa": 117.0}
    geo, fit, rows = disk._geometry_search(
        pipeline, wrong, pipeline._ctx().ell_mid
    )
    assert rows[0]["inc_deg"] == 25.0 and rows[0]["pa_deg"] == 117.0
    assert abs(geo["inc"] - 50.0) <= 20.0
    assert min(abs(geo["pa"] - 30.0), 180.0 - abs(geo["pa"] - 30.0)) <= 25.0
    assert (
        fit.chi2 / int(pipeline.processed.n_independent)
        < rows[0]["chi2_reduced"]
    )
    # The search is recorded with the profile stage.
    assert pipeline._report("profile")["geometry_search"]


def test_gp_round_trip(gp_run, data):
    pipeline, _ = gp_run
    res = load(pipeline.output)
    assert res.status == "complete"
    onp.testing.assert_allclose(res.data().vis, data.vis, rtol=1e-6)

    model = res.model()
    assert isinstance(model, System)
    assert set(model.components) == {"star", "disk", "planet"}
    assert type(model.disk) is MultipoleDisk
    # The rebuilt model is a live virgil object: it predicts the data.
    pred = res.data().model(model)
    assert onp.all(onp.isfinite(onp.asarray(pred)))
    best = res.summary["disk"]["best"]
    onp.testing.assert_allclose(
        float(res.model_values()["disk.flux"]),
        res.summary["disk"]["flux"],
        rtol=1e-6,
    )
    assert type(res.model(best)) is System
    assert res.model("geometry").components.keys() == {"star", "ring1"}
    assert res.fit_result(best).info["method"]

    grid = res.grid()
    assert list(grid["axes"]) == ["cos_inc", "pa"]
    assert grid["delta_chi2"].shape == (5, 6)
    assert grid["delta_chi2"].min() == 0.0
    assert grid["axes"]["pa"].max() < 180.0


def test_quicklook_notebook(gp_run):
    pipeline, _ = gp_run
    nb = json.loads((pipeline.output / "quicklook.ipynb").read_text())
    cells = nb["cells"]
    assert (
        cells[0]["cell_type"] == "markdown"
        and cells[-1]["cell_type"] == "markdown"
    )
    code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    for i in code:
        assert cells[i - 1]["cell_type"] == "markdown"
        assert len(cells[i]["outputs"]) == 1, cells[i]["source"]
        assert cells[i]["outputs"][0]["output_type"] != "error"
    images = sum(
        "image/png" in o.get("data", {})
        for c in cells
        if c["cell_type"] == "code"
        for o in c["outputs"]
    )
    assert images == 3  # overview, geometry grid, disk summary
    text = "".join("".join(cells[i]["source"]) for i in code)
    assert "plot_summary" in text
    assert "| `absorbed_companion` |" in "".join(cells[-1]["source"])


def test_rings_run_contract_and_recovery(rings_run):
    pipeline, res = rings_run
    s = res.summary
    assert res.status == "complete"
    assert _keys(s) == RINGS_KEYS
    names = [c.name for c in res.checks if c.name != "stage_warnings"]
    assert names == RINGS_CHECKS
    assert (pipeline.output / "samples.h5").exists()
    assert s["disk"]["n_rings"] == 1
    ring = s["disk"]["rings"][0]
    # The geometry is the truth's, with pa folded into [0, 180).
    assert abs(s["disk"]["inc_deg"] - 50.0) < 8.0
    assert abs(s["disk"]["pa_deg"] - 30.0) < 10.0
    assert abs(ring["r_mas"] - 5.0) < 0.3
    assert ring["flux"] == pytest.approx(0.4, rel=0.2)
    # Raw chi^2/N on the quoted errors, with the harmonics and companion
    # fitted, is close to one.
    assert s["chi2"]["reduced"][s["disk"]["best"]] < 1.4
    planet = s["disk"]["companion"]
    assert (
        abs(planet["dra_mas"] + 6.0) < 0.5
        and abs(planet["ddec_mas"] + 3.0) < 0.5
    )
    # The profile stage reports one row per ring count tried, BIC and the
    # Laplace evidence included.
    rows = s["compare"]["rows"]
    assert [r["n"] for r in rows] == [1, 2]
    assert all("bic" in r and "log_evidence" in r for r in rows)


def test_rings_posterior_is_folded(rings_run):
    _, res = rings_run
    samples = res.samples()
    assert samples["pa"].shape == (2, 20)
    # pa is folded to within 90 degrees of the MAP, harmonics rotated to match.
    map_pa = res.summary["disk"]["pa_deg"]
    assert onp.all(abs(samples["pa"] - map_pa) <= 90.0)
    assert {"inc", "pa", "r1", "s1", "f1", "dra", "ddec", "fp"} <= set(samples)
    assert res.sample_stats()["diverging"].shape == (2, 20)
    params = res.summary["posterior"]["params"]
    assert {"inc", "pa", "r1"} <= set(params)
    assert "pa_vec" not in params


def test_ring_colour_is_validated(data, tmp_path):
    with pytest.raises(ValueError, match="ring_colour"):
        DiskPipeline(data, "rings", ring_colour="grey")
    with pytest.raises(ValueError, match="model=.rings."):
        DiskPipeline(data, "gp", ring_colour="blackbody")
    assert DiskPipeline.defaults()["ring_colour"] == "power_law"


def test_blackbody_ring_colour_recovers_the_temperature(tmp_path):
    from virgil.spectra import BlackBody

    w0 = 3.0e-6
    truth = System(
        star=PointSource(flux=BlackBody(1.0, 6000.0, w0)),
        ring=MultipoleDisk(
            GaussianRing(5.0, 1.0),
            inc=50.0,
            pa=30.0,
            flux=BlackBody(0.4, 600.0, w0),
            n_r=48,
        ),
    )
    broad = vlti_oidata(
        wavelengths_m=onp.linspace(2.4e-6, 3.6e-6, 5),
        hour_angles_h=(-2.0, 0.0, 2.0),
        sigma_v2=0.002,
        sigma_cp_deg=0.5,
    ).with_model(truth, key=jax.random.PRNGKey(5))
    settings = {
        **SMALL,
        "companion": False,
        "max_rings": 1,
        "harmonic_degree": 0,
        "star_temperature_k": 6000.0,
        "wavel0_m": w0,
    }
    chi2 = {}
    for law in ("power_law", "blackbody"):
        pipeline = DiskPipeline(
            broad,
            "rings",
            output=tmp_path / law,
            ring_colour=law,
            **settings,
        )
        res = pipeline.run(through="profile")
        row = res.summary["profile"]["candidates"][0]
        chi2[law] = row["chi2_reduced"]
        if law == "blackbody":
            values = _io.load_model_values(
                pipeline.output / "models" / "rings_1"
            )
            assert float(values["a1"]) == pytest.approx(600.0, rel=0.2)
    # A power law cannot follow the blackbody over this band; the blackbody
    # fits at the noise.
    assert chi2["blackbody"] < 1.5
    assert chi2["power_law"] > chi2["blackbody"]


def test_without_ring_renumbers_the_later_rings():
    values = {
        "inc": 50.0,
        "fp": 0.04,
        **{
            f"{k}{i}": 10.0 * i + j
            for i in (1, 2, 3)
            for j, k in enumerate("rsfa")
        },
        **{f"h{i}_{c}1": float(i) for i in (1, 2, 3) for c in ("re", "im")},
    }
    out = disk._without_ring(values, 2)
    assert out["inc"] == 50.0 and out["fp"] == 0.04
    assert out["r1"] == values["r1"] and out["r2"] == values["r3"]
    assert out["a2"] == values["a3"] and out["h2_im1"] == 3.0
    assert not any(k.endswith("3") or "3_" in k for k in out)
    assert len(out) == len(values) - 6


def test_backfit_moves_a_ring_off_a_wrong_minimum(tmp_path):
    truth = System(
        star=PointSource(flux=1.0),
        inner=MultipoleDisk(
            GaussianRing(3.0, 0.8), inc=50.0, pa=30.0, flux=0.3, n_r=48
        ),
        outer=MultipoleDisk(
            GaussianRing(8.5, 1.0), inc=50.0, pa=30.0, flux=0.4, n_r=48
        ),
    )
    two = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 4),
        hour_angles_h=(-2.0, 0.0, 2.0),
        sigma_v2=0.005,
        sigma_cp_deg=1.0,
    ).with_model(truth, key=jax.random.PRNGKey(11))
    settings = {
        **SMALL,
        "companion": False,
        "harmonic_degree": 0,
        "max_rings": 2,
        "r_out_mas": 14.0,
    }
    pipeline = DiskPipeline(two, "rings", output=tmp_path / "run", **settings)
    pipeline.run(through="geometry")
    shape = disk._Shape(2, False, 0, False)
    start = {
        "inc": 50.0,
        "pa": 30.0,
        "r1": 6.0,
        "s1": 1.0,
        "f1": 0.3,
        "r2": 8.5,
        "s2": 1.0,
        "f2": 0.4,
    }
    wrong = disk._fit_rings(pipeline, shape, start)
    better = disk._backfit(pipeline, shape, wrong)
    assert better.chi2 <= wrong.chi2
    radii = sorted(float(better.values[f"r{i}"]) for i in (1, 2))
    assert radii == pytest.approx([3.0, 8.5], abs=0.6)


def test_fold_pa_keeps_the_disk():
    # The fold's parity rule: pa → pa + 180° with the order-m harmonic times
    # (-1)^m is the same disk. Check it against MultipoleDisk itself.
    data = vlti_oidata(
        wavelengths_m=onp.array([2.0e-6]),
        hour_angles_h=(-1.0, 1.0),
        sigma_v2=0.01,
        sigma_cp_deg=3.0,
    )

    def disk_with(pa, re, im):
        return System(
            star=PointSource(flux=1.0),
            ring=MultipoleDisk(
                GaussianRing(5.0, 1.0),
                inc=55.0,
                pa=pa,
                flux=0.4,
                n_r=24,
                harmonics=Harmonics(re=re, im=im),
            ),
        )

    re, im = (0.3, -0.2), (0.1, 0.25)
    flipped = disk_with(30.0 + 180.0, (-re[0], re[1]), (-im[0], im[1]))
    base = disk_with(30.0, re, im)
    onp.testing.assert_allclose(
        onp.asarray(data.model(base)),
        onp.asarray(data.model(flipped)),
        atol=1e-5,
    )
    # And without the parity the model differs.
    wrong = disk_with(30.0 + 180.0, re, im)
    assert not onp.allclose(
        onp.asarray(data.model(base)),
        onp.asarray(data.model(wrong)),
        atol=1e-6,
    )

    samples = {
        "pa": onp.array([[30.0, 210.0, 395.0]]),
        "h1": onp.array([[0.3, -0.3, 0.3]]),
        "z": onp.ones((1, 3, 2, 2, 1)),
    }
    folded = disk._fold_pa(
        samples,
        "pa",
        {"h1": -1.0, "z": onp.array([-1.0, 1.0])[None, :, None]},
        30.0,
    )
    onp.testing.assert_allclose(folded["pa"], [[30.0, 30.0, 35.0]])
    onp.testing.assert_allclose(folded["h1"], [[0.3, 0.3, 0.3]])
    # The order-1 entry flips with the parity, the order-2 entry never.
    assert onp.all(folded["z"][0, 1, :, 0] == -1.0)
    assert onp.all(folded["z"][0, 1, :, 1] == 1.0)
    assert onp.all(folded["z"][0, 0] == 1.0)

    # About a MAP near the cut a fixed [0, 180) fold would split the samples:
    # here they stay together, and the harmonic keeps one sign.
    near = {
        "pa": onp.array([[2.0, 358.0, 182.0, 178.0]]),
        "h1": onp.array([[0.3, 0.3, -0.3, -0.3]]),
    }
    folded = disk._fold_pa(near, "pa", {"h1": -1.0}, 2.0)
    onp.testing.assert_allclose(folded["pa"], [[2.0, -2.0, 2.0, -2.0]])
    onp.testing.assert_allclose(folded["h1"], [[0.3, 0.3, 0.3, 0.3]])
    q = _quantiles(folded["pa"], angle=True, period=180.0)
    assert min(q["median"], 180.0 - q["median"]) < 1.5
    assert q["q84"] - q["q16"] < 10.0


def test_fold_fit_folds_the_map():
    # A MAP at pa = 210° is the disk at 30° with the harmonics turned.
    data = vlti_oidata(
        wavelengths_m=onp.array([2.0e-6]),
        hour_angles_h=(-1.0, 1.0),
        sigma_v2=0.01,
        sigma_cp_deg=3.0,
    )
    ctx = _context()
    shape = disk._Shape(1, False, 1, False)
    priors = disk._ring_priors(ctx, shape)
    values = {
        "inc": 55.0,
        "pa": 210.0,
        "pa_vec": onp.array(
            [onp.cos(onp.radians(210.0)), onp.sin(onp.radians(210.0))]
        ),
        "r1": 5.0,
        "s1": 1.0,
        "f1": 0.4,
        "h1_re1": 0.3,
        "h1_im1": 0.1,
    }
    model = disk._ring_scene(ctx, shape, values)
    folded_model, folded = disk._fold_fit(ctx, shape, priors, model, values)
    assert float(folded["pa"]) == pytest.approx(30.0)
    onp.testing.assert_allclose(
        folded["pa_vec"],
        [onp.cos(onp.radians(30.0)), onp.sin(onp.radians(30.0))],
    )
    assert float(folded["h1_re1"]) == pytest.approx(-0.3)
    assert float(folded["h1_im1"]) == pytest.approx(-0.1)
    onp.testing.assert_allclose(
        onp.asarray(data.model(folded_model)),
        onp.asarray(data.model(model)),
        atol=1e-5,
    )
    # A MAP already in [0, 180) is returned as it is.
    same, unchanged = disk._fold_fit(
        ctx, shape, priors, model, {**values, "pa": 30.0}
    )
    assert same is model and float(unchanged["pa"]) == 30.0


# --- helpers --------------------------------------------------------------------


def _context():
    return disk._Context(
        halffringe=1.0,
        r_in=1.0,
        r_out=11.0,
        n_ring=32,
        n_gp=24,
        wavel0=2e-6,
        colour=False,
        law="power_law",
        degree=1,
        star_teff=None,
        companion=False,
        max_sep=11.0,
        step=0.5,
        inc_max=85.0,
        ells=(1.7, 3.5),
    )


def test_grid_minima_wrap_the_position_angle():
    # A bowl about (inc index 2, pa index 0): the pa axis is periodic.
    ii, jj = onp.meshgrid(onp.arange(4), onp.arange(6), indexing="ij")
    delta = 20.0 * abs(ii - 2) + 10.0 * onp.minimum(jj, 6 - jj)
    delta[3, 3] = 10.0  # a genuine second minimum
    inc = onp.array([2.0, 20.0, 40.0, 60.0])
    found = disk._grid_minima(delta, inc)
    assert [(i, j) for _, i, j in found] == [(2, 0), (3, 3)]
    # Face-on cells have no position angle and are skipped.
    delta[0, 2] = -5.0
    assert (0, 2) not in [(i, j) for _, i, j in disk._grid_minima(delta, inc)]


def test_fold_gp_values_turns_the_harmonics():
    ctx = _context()
    z = onp.ones((2, 2, 3))
    values = {
        "disk.pa": onp.asarray(210.0),
        "disk.pa_vec": onp.array([-0.8, -0.5]),
        "disk.harmonics.z": z,
    }
    out = disk._fold_gp_values(ctx, values, degree=2)
    assert float(out["disk.pa"]) == pytest.approx(30.0)
    onp.testing.assert_allclose(
        out["disk.pa_vec"],
        [onp.cos(onp.radians(30.0)), onp.sin(onp.radians(30.0))],
    )
    # Order 1 flips, order 2 does not.
    assert onp.all(out["disk.harmonics.z"][:, 0] == -1.0)
    assert onp.all(out["disk.harmonics.z"][:, 1] == 1.0)
    # A position angle already in [0, 180) is left alone.
    same = {**values, "disk.pa": onp.asarray(30.0)}
    assert disk._fold_gp_values(ctx, same, degree=2) is same


def test_bic_scores_the_fitted_scales():
    quoted = disk._DiskFit(None, {}, {}, 400.0, {}, 6)
    assert disk._bic(quoted, 100) == pytest.approx(400.0 + 6 * onp.log(100))
    # Under error_scale="fit" the score is -2 log L at the fitted scales, not
    # the quoted-error chi^2 that unmodelled noise inflates.
    fitted = disk._DiskFit(None, {}, {}, 400.0, {"vis_scale": 2.0}, 7, 250.0)
    assert disk._bic(fitted, 100) == pytest.approx(250.0 + 7 * onp.log(100))


def test_dynamic_range_sigma_is_clipped():
    ctx = _context()
    # A thin ring has a large peak-to-mean contrast, a broad one almost none.
    thin = disk._dynamic_range_sigma(ctx, 5.0, 0.01)
    broad = disk._dynamic_range_sigma(ctx, 5.0, 8.0)
    assert 0.5 <= broad < thin <= 5.0
    assert thin == 5.0 and broad == 0.5


# --- resume ---------------------------------------------------------------------


def test_resume_and_config_mismatch(data, tmp_path, monkeypatch):
    out = tmp_path / "run"
    settings = {**SMALL, "companion": False, "max_rings": 1}
    res = DiskPipeline(data, "rings", output=out, **settings).run(
        through="geometry"
    )
    assert res.status == "partial" and "profile" not in res.summary
    before = json.loads((out / "run.json").read_text())["stages"]

    ran = []
    for name in ("_load_disk", "_overview", "_geometry_stage"):
        monkeypatch.setattr(disk, name, lambda p, n=name: ran.append(n))
    res = DiskPipeline(data, "rings", output=out, **settings).run(
        through="geometry"
    )
    assert ran == []  # nothing was recomputed
    after = json.loads((out / "run.json").read_text())["stages"]
    for name in ("load", "overview", "geometry"):
        assert after[name]["started"] == before[name]["started"]
    assert res.status == "partial"

    with pytest.raises(ConfigMismatchError, match="n_inc"):
        DiskPipeline(
            data, "rings", output=out, **{**settings, "n_inc": 6}
        ).run()
    with pytest.raises(ConfigMismatchError, match="model template differs"):
        DiskPipeline(data, "gp", output=out, **settings).run()


# --- CLI --------------------------------------------------------------------------


def test_cli_disk_run_from_oifits(gp_run, tmp_path, capsys):
    pipeline, _ = gp_run
    path = pipeline.output / "data" / "processed.oifits"
    out = tmp_path / "cli"
    args = ["disk", str(path), "-o", str(out), "--through", "geometry"]
    args += ["--model", "rings", "--set", "n_inc=4", "--set", "n_pa=4"]
    args += ["--set", "r_out_mas=12.0"]
    assert cli.main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "partial" and run["pipeline"] == "disk"
    assert run["model"]["models"] == ["rings"]
    assert run["config"]["n_inc"] == 4 and run["config"]["r_out_mas"] == 12.0
    assert "geometry.inc_deg" in capsys.readouterr().out
    assert cli.main(["info", str(out)]) == 0
    assert cli.main(["validate", str(out)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["disk", str(path), "--model", "cone"])

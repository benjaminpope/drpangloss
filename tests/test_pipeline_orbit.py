"""OrbitPipeline: API contract, inputs, run folder, resume, CLI, quicklook and
the golden synthetic cases of the blind alias search.

The fast tests stop at the ``epochs`` stage or fake the later reports; the
golden cases (a full blind run each) are marked slow.
"""

import copy
import inspect
import json

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")
pytest.importorskip("jaxoplanet")

import virgil  # noqa: E402
import virgil.pipeline.orbit as orbit_module  # noqa: E402
from virgil.epochs import Epochs  # noqa: E402
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.oifits import write_oifits  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402
from virgil.pipeline import (  # noqa: E402
    SCHEMA,
    Check,
    ConfigMismatchError,
    OrbitPipeline,
    _io,
    cli,
    load,
)
from virgil.pipeline import _checks, _quicklook  # noqa: E402
from virgil.simulate import simulate  # noqa: E402
from tests._shared import vlti_night  # noqa: E402

# One xdist worker builds each fixture once.
pytestmark = pytest.mark.xdist_group("pipeline_orbit")

T_REF = 60000.0
TRUTH = dict(
    period=12.38,
    dt_peri=2.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=10.0,
)
FLUX = 0.3
# Two seasons: three nights, a gap of 238 d, three more.
SEASONS = T_REF + onp.array([0.0, 2.4, 5.1, 238.0, 240.6, 247.4])
TINY = {
    "max_grid": 25,
    "n_candidates": 10,
    "n_refine": 2,
    "n_is": 200,
    "num_warmup": 60,
    "num_samples": 60,
    "num_chains": 2,
    "quicklook_execute": False,
}


def _night(mjd, rotation):
    return vlti_night(mjd, 3, rotation, d_phi=0.02, stations=True)


def _simulate(times, truth=None, flux=FLUX, seed=3):
    """One simulated night per time, for the orbit ``truth``."""
    scene = None
    if flux is not None:
        orbit = KeplerOrbit(**(truth or TRUTH), t_ref=T_REF)
        scene = OrbitalBinary(orbit, flux)
    key = jax.random.PRNGKey(seed)
    nights = {}
    with jax.enable_x64(True):
        for k, mjd in enumerate(times):
            night = _night(mjd, 7.0 * k)
            if scene is None:
                from virgil import PointSource

                data = simulate(
                    PointSource(), night, key=jax.random.fold_in(key, k)
                )
            else:
                data = simulate(scene, night, key=jax.random.fold_in(key, k))
            nights[f"n{k}"] = data
    return nights


@pytest.fixture(scope="module")
def nights():
    return _simulate(SEASONS[:3])


# --- the contract -----------------------------------------------------------

API = {
    "OrbitPipeline": "(epochs, output='run', **settings)",
    "OrbitPipeline.from_oifits": (
        "(paths, metadata=None, *, output='run', **settings)"
    ),
}
STAGES = ("load", "epochs", "search", "modes", "posterior", "quicklook")
DEFAULTS = {
    "gap_days": 0.5,
    "wavel_range": None,
    "error_floor": None,
    "exclude": {},
    "flag_nights": True,
    "n_peaks": 5,
    "max_sep_mas": None,
    "grid_step_mas": None,
    "max_grid": 201,
    "period_range": None,
    "a_range": [1.0, 30.0],
    "flux_range": [0.01, 1.0],
    "ecc_max": 0.9,
    "s_max": None,
    "primary_diam": "auto",
    "vis_amp": True,
    "sign_free": True,
    "flip_flux": True,
    "temper": True,
    "n_bands_max": 200,
    "n_candidates": 100,
    "n_refine": 6,
    "n_is": 2000,
    "alias_nats": 10.0,
    "no_orbit_nats": 10.0,
    "posterior_p_min": 0.05,
    "max_modes": 3,
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "chain_method": "vectorized",
    "seed": 0,
    "quicklook_execute": True,
}
SUMMARY_KEYS = {
    "schema": None,
    "pipeline": None,
    "stability": None,
    "status": None,
    "worst_check": None,
    "data": set(orbit_module._DATA_KEYS),
    "nights": None,
    "chi2": {"n_independent", "best_orbit_reduced", "nights"},
    "search": set(orbit_module._SEARCH_KEYS),
    "modes": set(orbit_module._MODES_KEYS),
    "orbit": {"n", "verdict", "source", "elements"},
    "posterior": set(orbit_module._POSTERIOR_KEYS),
    "warnings": None,
    "checks": None,
}
CHECKS = [
    "chi2",
    "night_flags",
    "labels_free",
    "bands_failed",
    "alias_ambiguity",
    "winner_flags",
    "tempering",
    "primary_resolved",
    "no_orbit",
    "period_bound",
    "prior_bound",
    "r_hat",
    "ess",
    "divergences",
]


def _signature(path):
    obj = virgil.pipeline
    for part in path.split("."):
        obj = getattr(obj, part)
    sig = inspect.signature(obj)
    params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in sig.parameters.values()
    ]
    return str(sig.replace(parameters=params, return_annotation=sig.empty))


def test_api_snapshot():
    # Changing any of these breaks users' scripts: update the snapshot only
    # with an additive change or a schema bump (see docs/pipeline.md).
    assert {k: _signature(k) for k in API} == API
    assert OrbitPipeline.STAGES == STAGES
    assert OrbitPipeline.defaults() == DEFAULTS
    assert OrbitPipeline.NAME == "orbit"
    assert OrbitPipeline.STABILITY == "provisional"
    assert SCHEMA == "virgil-pipeline-run-v1"
    assert "OrbitPipeline" in virgil.pipeline.__all__
    assert cli._PIPELINES["orbit"] == (
        "virgil.pipeline.orbit",
        "OrbitPipeline",
    )
    assert "orbit" in _quicklook._SECTIONS


def test_the_default_run_is_blind():
    # The default path carries no published period, window, start or prior:
    # the unconfigured search must pass the engine nothing but data-derived
    # arguments (the engine's own defaults do the rest), and those defaults
    # must be the engine's, so the two cannot drift apart.
    from virgil.aliases import fit_orbit_aliases

    d = OrbitPipeline.defaults()
    assert d["period_range"] is None
    assert d["primary_diam"] == "auto"
    assert not {"init", "p_range", "positions"} & set(d)
    engine = inspect.signature(fit_orbit_aliases).parameters
    for ours, theirs in (
        ("a_range", "a_range"),
        ("flux_range", "flux_range"),
        ("ecc_max", "ecc_max"),
        ("s_max", "s_max"),
        ("primary_diam", "primary_diam"),
        ("vis_amp", "vis_amp"),
        ("sign_free", "sign_free"),
        ("flip_flux", "flip_flux"),
        ("temper", "temper"),
        ("n_bands_max", "n_bands_max"),
        ("n_candidates", "n_candidates"),
        ("n_refine", "n_refine"),
        ("n_is", "n_is"),
        ("seed", "seed"),
    ):
        assert list(np_tuple(d[ours])) == list(
            np_tuple(engine[theirs].default)
        ), ours
    assert engine["init"].default is None
    assert engine["skip_flagged"].default is True


def np_tuple(x):
    return x if isinstance(x, (list, tuple)) else (x,)


def _fake_reports():
    row = {"dataset": "n0", "mjd": 1.0, "sep_mas": 5.0, "flux": 0.3}
    chi2 = {"chi2": 10.0, "n": 10, "chi2_per_n": 1.0, "vis_amp": 1.0}
    band = {
        "rank": 0,
        "n": 20,
        "period": 12.1,
        "p_lo": 12.0,
        "p_hi": 12.2,
        "log_z": 0.0,
        "delta_log_z": 0.0,
        "p": 1.0,
        "chi2_red": 1.0,
        "labels_free": False,
        "flags": [],
    }
    return {
        "load": {
            **{k: 1.0 for k in orbit_module._DATA_KEYS},
            "excluded": {},
            "extra": "dropped",
        },
        "epochs": {
            "flagged": {"n1": ["low-gap"]},
            "nights": [
                {
                    **row,
                    "gap_marginal": 9.0,
                    "edge": False,
                    "flags": [{"code": "low-gap", "reason": "x"}],
                }
            ],
        },
        "search": {
            "p_range": [3.0, 100.0],
            "window_capped": None,
            "n_bands": 5,
            "failed_bands": [],
            "t_ref": 1.0,
            "span_days": 10.0,
            "primary_diam": None,
            "primary_resolved": False,
            "primary_diam_estimate": 0.1,
            "primary_at_grid_edge": False,
            "tempering": {"f": 1.0, "f_raw": 1.0, "out_of_range": False},
            "labels_free": False,
            "period_bound": False,
            "period_at_lower_bound": False,
            "unseeded": [],
            "chi2": {"nights": {"n0": chi2}, "chi2": 10.0, "n": 10},
            "bands": [band],
        },
        "modes": {
            "winner": {
                "n": 20,
                "p": 1.0,
                "log_z": 0.0,
                "elements": {"period": 12.1},
                "flipped": False,
                "flags": [],
            },
            "runner_up": {"n": 21, "p": 0.0, "log_z": -30.0},
            "delta_log_z": 30.0,
            "labels_free": False,
            "no_orbit": {"delta_log_z": 50.0, "verdict": "orbit"},
            "sampled": [20],
            "table": [band],
        },
        "posterior": {
            "modes": [
                {
                    "n": 20,
                    "params": {"period": {"median": 12.1}},
                    "r_hat_max": 1.0,
                    "ess_bulk_min": 900.0,
                    "divergence_fraction": 0.0,
                    "prior_bound_fraction": {"ecc": 0.0},
                }
            ],
            "num_chains": 2,
            "num_samples": 10,
            "num_warmup": 10,
            "r_hat_max": 1.0,
            "ess_bulk_min": 900.0,
            "divergence_fraction": 0.0,
            "prior_bound_fraction": {"ecc": 0.0},
        },
    }


def test_summary_keys_and_check_order_snapshot():
    sections, checks = orbit_module._summarize(
        OrbitPipeline.defaults(), _fake_reports()
    )
    keys = {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in sections.items()
    }
    expected = {
        k: v
        for k, v in SUMMARY_KEYS.items()
        if k
        not in ("schema", "pipeline", "stability", "status", "worst_check")
        and k not in ("warnings", "checks")
    }
    assert keys == expected
    assert [c.name for c in checks] == CHECKS
    assert all(isinstance(c, Check) for c in checks)
    assert sections["orbit"]["source"] == "posterior"


def test_partial_and_no_orbit_summaries():
    reports = _fake_reports()
    for later in ("posterior", "modes", "search"):
        reports.pop(later)
        sections, checks = orbit_module._summarize(
            OrbitPipeline.defaults(), reports
        )
        assert later not in sections
        names = [c.name for c in checks]
        assert names == [c for c in CHECKS if c in names]
    reports = _fake_reports()
    reports["modes"]["no_orbit"] = {"delta_log_z": -2.0, "verdict": "no-orbit"}
    reports["posterior"] = {"skipped": "no-orbit", "modes": []}
    sections, checks = orbit_module._summarize(
        OrbitPipeline.defaults(), reports
    )
    status = {c.name: c.status for c in checks}
    assert "r_hat" not in status and status["no_orbit"] == "fail"
    assert sections["orbit"]["source"] == "map"


@pytest.mark.parametrize(
    "check, args, status",
    [
        (_checks.alias_ambiguity, (30.0, 21), "pass"),
        (_checks.alias_ambiguity, (5.0, 21), "warn"),
        (_checks.alias_ambiguity, (0.5, 21), "fail"),
        (_checks.alias_ambiguity, (None, None), "pass"),
        (_checks.no_orbit, (50.0,), "pass"),
        (_checks.no_orbit, (5.0,), "warn"),
        (_checks.no_orbit, (-1.0,), "fail"),
        (_checks.bands_failed, ([],), "pass"),
        (_checks.bands_failed, ([21, 22],), "warn"),
        (_checks.winner_flags, ([],), "pass"),
        (_checks.winner_flags, (["poor-fit"],), "pass"),
        (_checks.winner_flags, (["low-ess", "not-converged"],), "warn"),
        (_checks.labels_free, (False,), "pass"),
        (_checks.labels_free, (True,), "warn"),
        (_checks.night_flags, ({}, 5), "pass"),
        (_checks.night_flags, ({"n1": ["low-gap"]}, 5), "warn"),
        (_checks.period_bound, (False, False, [1.0, 2.0]), "pass"),
        (_checks.period_bound, (True, False, [1.0, 2.0]), "warn"),
        (_checks.period_bound, (False, True, [1.0, 2.0]), "warn"),
        (_checks.tempering, (False, 1.2, 1.2), "pass"),
        (_checks.tempering, (True, 3.0, 3.0), "warn"),
        (_checks.primary_resolved, (False, 0.1, False), "pass"),
        (_checks.primary_resolved, (True, 9.0, True), "warn"),
        (_checks.tempering, (False, None, None), "pass"),
        (_checks.primary_resolved, (True, 1.0, False), "pass"),
    ],
)
def test_orbit_checks_have_the_documented_thresholds(check, args, status):
    assert check(*args).status == status


# --- settings and inputs ----------------------------------------------------


def test_settings_are_validated(nights):
    for bad, match in (
        ({"gap_days": -1.0}, "gap_days"),
        ({"a_range": [5.0, 1.0]}, "a_range"),
        ({"flux_range": [0.01, 2.0]}, "flux_range must be at most 1"),
        ({"ecc_max": 1.0}, "ecc_max"),
        ({"period_range": [10.0, 5.0]}, "period_range"),
        ({"primary_diam": "big"}, "primary_diam"),
        ({"temper": 0.5}, "temper"),
        ({"chain_method": "x"}, "chain_method"),
        ({"n_candidates": 0}, "n_candidates"),
        ({"posterior_p_min": 2.0}, "posterior_p_min"),
        ({"vis_amp": [2.0, 1.0]}, "vis_amp"),
        ({"max_grid": 2}, "max_grid"),
        ({"n_peaks": True}, "n_peaks"),
    ):
        with pytest.raises(ValueError, match=match):
            OrbitPipeline(nights, **bad)
    with pytest.raises(TypeError, match="unknown settings"):
        OrbitPipeline(nights, num_sample=10)


def test_an_orbit_needs_two_nights(nights):
    one = {"n0": nights["n0"]}
    with pytest.raises(ValueError, match="at least two nights"):
        OrbitPipeline(one)
    with pytest.raises(TypeError, match="non-empty"):
        OrbitPipeline([])
    two = OrbitPipeline(nights, exclude={"n2": "bad night"})
    assert two.data.names == ("n0", "n1")
    assert two._excluded["n2"]["reason"] == "bad night"
    with pytest.raises(ValueError, match="no file, night or dataset"):
        OrbitPipeline(nights, exclude={"n9": "x"})
    with pytest.raises(ValueError, match="at least two nights"):
        OrbitPipeline(nights, exclude={"n1": "x", "n2": "y"})


def test_exclude_works_on_an_epochs_input_too(nights):
    # Times of the excluded datasets must not reach the rebuilt Epochs.
    given = Epochs(nights, times={"n0": 60000.0, "n1": 60010.0, "n2": 60020.0})
    pipeline = OrbitPipeline(given, exclude={"n2": "bad night"})
    assert pipeline.data.names == ("n0", "n1")
    assert [float(t) for t in pipeline.data.times] == [60000.0, 60010.0]


def test_epochs_and_dicts_are_equivalent_inputs(nights):
    by_dict = OrbitPipeline(nights)
    by_epochs = OrbitPipeline(Epochs(nights))
    nested = OrbitPipeline({k: [v] for k, v in nights.items()})
    assert by_dict.data.names == by_epochs.data.names == ("n0", "n1", "n2")
    assert by_dict._data_fingerprint() == by_epochs._data_fingerprint()
    # A list with one dataset is named after its night, as in ``Epochs``.
    assert nested.data.dataset_names == by_dict.data.dataset_names
    assert "nights=3" in repr(by_dict)


def _write_nights(folder, nights):
    paths = []
    for name, data in nights.items():
        tables, reason = _io.oidata_tables(data)
        assert tables is not None, reason
        path = folder / f"{name}.oifits"
        write_oifits(tables, path)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def files(nights, tmp_path_factory):
    return _write_nights(tmp_path_factory.mktemp("orbit_files"), nights)


def test_from_oifits_groups_files_into_nights(files):
    pipeline = OrbitPipeline.from_oifits(files, gap_days=0.5)
    assert len(pipeline.data.names) == 3
    assert [round(float(t)) for t in pipeline.data.times] == [
        round(float(m + 1 / 24)) for m in SEASONS[:3]
    ]
    # Files of one night split by 0h UT are one night, and a larger gap_days
    # merges nights further apart.
    straddle = {
        files[0]: {"mjd": 60000.95},
        files[1]: {"mjd": 60001.05},
        files[2]: {"mjd": 60200.0},
    }
    one = OrbitPipeline.from_oifits(files, straddle)
    assert len(one.data.names) == 2 and len(one.data.dataset_names) == 3
    apart = {
        files[0]: {"mjd": 60000.0},
        files[1]: {"mjd": 60010.0},
        files[2]: {"mjd": 60200.0},
    }
    assert len(OrbitPipeline.from_oifits(files, apart).data.names) == 3
    merged = OrbitPipeline.from_oifits(files, apart, gap_days=30.0)
    assert len(merged.data.names) == 2
    meta = {files[0]: {"epoch": "first"}, files[1]: {"epoch": "second"}}
    labelled = OrbitPipeline.from_oifits(files, meta)
    assert {"first", "second"} <= set(labelled.data.names)
    assert pipeline.inputs == [
        str(__import__("pathlib").Path(f).resolve()) for f in files
    ]


def test_a_file_with_no_usable_data_is_skipped_with_a_note(files, monkeypatch):
    import virgil.oidata as oidata

    real = oidata.OIData

    def fake(path, *args, **kwargs):
        if path == files[2]:
            raise ValueError("No unflagged data: every visibility is flagged.")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(oidata, "OIData", fake)
    with pytest.warns(UserWarning, match="skipped"):
        pipeline = OrbitPipeline.from_oifits(files)
    assert len(pipeline.data.names) == 2
    note = next(f for f in pipeline._input_files() if "skipped" in f)
    assert note["path"].endswith(files[2].rsplit("/", 1)[-1])
    with pytest.raises(ValueError, match="bad table"):
        monkeypatch.setattr(
            oidata,
            "OIData",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad table")),
        )
        OrbitPipeline.from_oifits(files[:1], metadata={files[0]: {}})


def test_metadata_is_checked(files):
    with pytest.raises(ValueError, match="match no input file"):
        OrbitPipeline.from_oifits(files, {"nothere.oifits": {"epoch": "x"}})
    with pytest.raises(ValueError, match="unknown keys"):
        OrbitPipeline.from_oifits(files, {files[0]: {"exclude": "bad"}})
    named = OrbitPipeline.from_oifits(
        files, {"n0.oifits": {"epoch": "by-name"}}
    )
    assert "by-name" in named.data.names
    shifted = OrbitPipeline.from_oifits(files, {files[0]: {"mjd": 60100.0}})
    assert onp.isclose(float(max(shifted.data.times)), 60100.0)


def test_exclusions_by_file_are_recorded_with_their_reason(files):
    pipeline = OrbitPipeline.from_oifits(
        files, exclude={"n2.oifits": "calibrator was resolved"}
    )
    assert len(pipeline.data.names) == 2
    record = pipeline._excluded["n2.oifits"]
    assert record["reason"] == "calibrator was resolved"
    assert len(record["datasets"]) == 1


# --- run folder, resume ------------------------------------------------------


def _started(out):
    run = json.loads((out / "run.json").read_text())
    return {k: v["started"] for k, v in run["stages"].items()}, run


@pytest.fixture(scope="module")
def epochs_run(nights, tmp_path_factory):
    out = tmp_path_factory.mktemp("orbit") / "run"
    pipeline = OrbitPipeline(nights, output=out, **TINY)
    return pipeline, pipeline.run(through="epochs")


def test_run_folder_layout_through_epochs(epochs_run):
    pipeline, res = epochs_run
    out = pipeline.output
    run = json.loads((out / "run.json").read_text())
    assert run["schema"] == SCHEMA and run["pipeline"] == "orbit"
    assert run["class"] == "OrbitPipeline"
    assert run["status"] == "partial"
    assert list(run["stages"]) == ["load", "epochs"]
    for name in ("run.json", "summary.json", "stages/load/report.json"):
        assert (out / name).exists()
    assert (out / "stages" / "epochs" / "positions.json").exists()
    assert res.summary["data"]["n_nights"] == 3
    assert len(res.summary["nights"]) == 3
    assert "search" not in res.summary and "samples.h5" not in {
        p.name for p in out.iterdir()
    }
    # Without a processed.oifits per night the loader says why.
    with pytest.raises(FileNotFoundError, match="one OIFITS per night"):
        res.data()


def test_positions_round_trip_exactly(epochs_run):
    pipeline, _ = epochs_run
    path = pipeline.output / "stages" / "epochs" / "positions.json"
    stored = json.loads(path.read_text())
    again = orbit_module._positions_json(
        orbit_module._positions_from_json(stored)
    )
    assert json.dumps(again) == json.dumps(stored)
    found = orbit_module._read_positions(pipeline)
    assert found.names == ("n0", "n1", "n2")
    # The true companion is found where the orbit puts it.
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    dra, ddec, _ = orbit.relative(onp.asarray(found.mjd))
    sep = onp.hypot(onp.asarray(dra), onp.asarray(ddec))
    assert onp.allclose(onp.hypot(found.dra, found.ddec), sep, rtol=0.15)


def test_the_position_grid_widens_until_no_peak_is_on_its_edge(epochs_run):
    _, res = epochs_run
    report = json.loads(
        (res.path / "stages" / "epochs" / "report.json").read_text()
    )
    assert report["edge_nights"] == []
    assert len(report["half_widths_mas"]) >= 2  # a = 10 mas > the first box
    assert report["half_widths_mas"][-1] > report["half_widths_mas"][0]
    assert report["grid_step_mas"] <= report["grid_step_wanted_mas"] * 1.001


def test_resume_skips_completed_stages(nights, tmp_path, monkeypatch):
    out = tmp_path / "run"
    OrbitPipeline(nights, output=out, **TINY).run(through="load")
    before, _ = _started(out)
    ran = []
    monkeypatch.setattr(orbit_module, "_load", lambda p: ran.append("load"))
    res = OrbitPipeline(nights, output=out, **TINY).run(through="epochs")
    assert ran == [] and res.status == "partial"
    after, _ = _started(out)
    assert after["load"] == before["load"] and "epochs" in after

    with pytest.raises(ConfigMismatchError, match="ecc_max"):
        OrbitPipeline(nights, output=out, **dict(TINY, ecc_max=0.5)).run()
    shifted = {
        k: v.with_error_floor(absolute={"vis": 0.5}) for k, v in nights.items()
    }
    with pytest.raises(ConfigMismatchError, match="data differ"):
        OrbitPipeline(shifted, output=out, **TINY).run()
    # Dropping a night is a different run too.
    with pytest.raises(ConfigMismatchError, match="exclude"):
        OrbitPipeline(nights, output=out, exclude={"n2": "no"}, **TINY).run()


def test_band_checkpoints_survive_a_rerun_but_not_a_fresh_run(
    nights, tmp_path
):
    out = tmp_path / "run"
    pipeline = OrbitPipeline(nights, output=out, **TINY)
    pipeline.run(through="load")
    bands = orbit_module._band_checkpoints(pipeline)
    bands.mkdir(parents=True)
    (bands / "band_20.json").write_text("{}")
    # Re-running the stage (its outputs are stale) keeps the bands.
    pipeline._discard(pipeline._stages()[2])
    assert (bands / "band_20.json").exists()
    # A fresh run starts over.
    OrbitPipeline(nights, output=out, **TINY).run(through="load", resume=False)
    assert not bands.exists()


def test_a_failing_stage_is_recorded(nights, tmp_path, monkeypatch):
    out = tmp_path / "run"

    def boom(p):
        raise RuntimeError("alias engine exploded")

    monkeypatch.setattr(orbit_module, "_epochs", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        OrbitPipeline(nights, output=out, **TINY).run(through="epochs")
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed" and run["error"]["stage"] == "epochs"
    assert run["stages"]["load"]["status"] == "complete"


# --- CLI -----------------------------------------------------------------


def test_cli_orbit_runs_from_oifits_with_metadata(files, tmp_path, capsys):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({files[0]: {"epoch": "first"}}))
    out = tmp_path / "cli"
    args = ["orbit", *files, "--output", str(out), "--through", "load"]
    args += ["--metadata", str(meta)]
    for key, value in TINY.items():
        args += ["--set", f"{key}={json.dumps(value)}"]
    assert cli.main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "partial" and run["pipeline"] == "orbit"
    assert [Path_name(f["path"]) for f in run["inputs"]["files"]] == [
        Path_name(f) for f in files
    ]
    assert run["config"]["max_grid"] == 25
    assert "data.n_nights" in capsys.readouterr().out
    assert cli.main(["validate", str(out)]) == 0


def Path_name(path):
    return __import__("pathlib").Path(path).name


def test_from_config_reads_the_metadata_key(files, tmp_path):
    config = tmp_path / "orbit.json"
    config.write_text(
        json.dumps(
            {
                "pipeline": "orbit",
                "data": [str(f) for f in files],
                "metadata": {str(files[0]): {"epoch": "first"}},
                "settings": {"max_grid": 25},
                "output": "out",
            }
        )
    )
    pipeline = OrbitPipeline.from_config(config)
    assert "first" in pipeline.data.names
    assert pipeline.settings["max_grid"] == 25


# --- quicklook ---------------------------------------------------------------


def test_quicklook_cells_are_valid_code(tmp_path):
    reports = _fake_reports()
    sections, _ = orbit_module._summarize(OrbitPipeline.defaults(), reports)
    sections["modes"]["winner"]["p"] = 1.0
    sections["modes"]["no_orbit"]["verdict"] = "orbit"
    summary = {**sections, "worst_check": "pass", "checks": []}
    run = {
        "stages": {
            k: {"status": "complete"}
            for k in ("load", "epochs", "search", "modes", "posterior")
        }
    }
    cells = _quicklook._SECTIONS["orbit"](run, summary, tmp_path)
    assert len(cells) == 4
    for markdown, code in cells:
        assert markdown.startswith("##") or markdown[0].isalpha()
        compile(code, "<cell>", "exec")
    assert "OrbitalBinary" not in "".join(m for m, _ in cells)  # target-free
    partial = _quicklook._SECTIONS["orbit"](
        {"stages": {"load": {"status": "complete"}}}, summary, tmp_path
    )
    assert partial == []


def test_default_model_and_digest_do_not_need_a_template(nights):
    pipeline = OrbitPipeline(nights)
    assert pipeline.model is None
    assert pipeline._model_digest() == {
        "class": "OrbitalBinary",
        "digest": None,
    }


def test_the_class_is_exported_lazily():
    import importlib

    module = importlib.import_module("virgil.pipeline")
    assert module.OrbitPipeline is OrbitPipeline
    assert "OrbitPipeline" in dir(module)
    assert copy.deepcopy(OrbitPipeline.defaults()) == DEFAULTS


# --- golden synthetic cases (slow) --------------------------------------------

# The search keeps the engine's default effort (that is the point: the
# default run must work); only the sampler is small.
GOLDEN = {
    "num_warmup": 200,
    "num_samples": 150,
    "num_chains": 2,
    "quicklook_execute": False,
}


def _blind(nights, out, **settings):
    pipeline = OrbitPipeline(nights, output=out, **{**GOLDEN, **settings})
    return pipeline, pipeline.run()


@pytest.mark.slow
def test_blind_search_resolves_the_two_season_gap_alias(tmp_path, monkeypatch):
    # Two seasons 238 d apart leave the period aliased to many integer
    # orbit counts; only the closure phases can say which. No period window,
    # start or prior is given: the engine gets the data alone.
    import virgil.aliases as aliases

    seen = {}
    real = aliases.fit_orbit_aliases

    def spy(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(aliases, "fit_orbit_aliases", spy)
    data = _simulate(SEASONS)
    pipeline, res = _blind(data, tmp_path / "run")
    assert len(seen["args"]) == 2 and seen["args"][1] is None  # no window
    kwargs = seen["kwargs"]
    assert "init" not in kwargs and "skip_flagged" not in kwargs
    assert kwargs["primary_diam"] == "auto" and kwargs["a_range"] == (
        1.0,
        30.0,
    )
    modes = res.summary["modes"]
    n_true = round(float(onp.ptp(SEASONS)) / TRUTH["period"])
    assert modes["winner"]["n"] == n_true
    assert abs(modes["winner"]["elements"]["period"] - TRUTH["period"]) < 0.05
    assert modes["delta_log_z"] > 10.0  # the runner-up is ruled out
    assert modes["no_orbit"]["verdict"] == "orbit"
    post = res.summary["posterior"]["modes"][0]["params"]
    assert abs(post["period"]["median"] - TRUTH["period"]) < 0.05
    assert res.status == "complete"
    assert res.aliases().best.n == n_true
    assert res.samples()["period"].shape == (2, 150)
    names = [c.name for c in res.checks if c.name in CHECKS]
    assert names == CHECKS
    assert res.summary["chi2"]["best_orbit_reduced"] < 3.0


@pytest.mark.slow
def test_a_twin_is_labels_free(tmp_path):
    data = _simulate(SEASONS, flux=1.0, seed=5)
    _, res = _blind(data, tmp_path / "run")
    assert res.summary["modes"]["labels_free"]
    status = {c.name: c.status for c in res.checks}
    assert status["labels_free"] == "warn"


@pytest.mark.slow
def test_a_resolved_primary_is_fitted(tmp_path):
    import virgil.aliases as aliases

    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = aliases._UDBinary(orbit, FLUX, 1.05)
    key = jax.random.PRNGKey(7)
    data = {}
    with jax.enable_x64(True):
        for k, mjd in enumerate(SEASONS):
            data[f"n{k}"] = simulate(
                scene, _night(mjd, 7.0 * k), key=jax.random.fold_in(key, k)
            )
    _, res = _blind(data, tmp_path / "run")
    search = res.summary["search"]
    assert search["primary_resolved"] is True
    assert search["primary_diam"] == "fit"
    assert res.summary["orbit"]["elements"]["diam"]["median"] == pytest.approx(
        1.05, rel=0.3
    )


@pytest.mark.slow
def test_a_single_star_is_no_orbit(tmp_path):
    data = _simulate(SEASONS, flux=None, seed=9)
    _, res = _blind(data, tmp_path / "run")
    modes = res.summary["modes"]
    assert modes["no_orbit"]["verdict"] == "no-orbit"
    assert res.summary["posterior"].get("skipped") == "no-orbit" or (
        "modes" in res.summary["posterior"]
        and not res.summary["posterior"]["modes"]
    )
    status = {c.name: c.status for c in res.checks}
    assert status["no_orbit"] in ("warn", "fail")
    assert res.samples() == {}


@pytest.mark.slow
def test_a_highly_eccentric_orbit_is_recovered(tmp_path):
    truth = {**TRUTH, "ecc": 0.8}
    data = _simulate(SEASONS, truth=truth, seed=11)
    _, res = _blind(data, tmp_path / "run", ecc_max=0.95)
    best = res.summary["modes"]["winner"]["elements"]
    assert abs(best["period"] - TRUTH["period"]) < 0.05
    assert best["ecc"] == pytest.approx(0.8, abs=0.1)
    assert load(tmp_path / "run").summary["chi2"]["best_orbit_reduced"] < 3.0


@pytest.mark.slow
def test_the_real_summary_has_the_snapshot_keys(tmp_path):
    data = _simulate(SEASONS, seed=3)
    _, res = _blind(data, tmp_path / "run")
    keys = {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in res.summary.items()
    }
    assert keys == SUMMARY_KEYS
    assert (res.path / "plots" / "modes_sky.png").exists()
    assert (res.path / "plots" / "search_bands.png").exists()

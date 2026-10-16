"""The CI matrix chooser (.github/scripts/ci_matrix.py) and the test lists.

Pure stdlib: nothing here compiles JAX.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ci_matrix", ROOT / ".github" / "scripts" / "ci_matrix.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def _jobs(*args, **kwargs):
    return ci.choose(*args, **kwargs)["include"]


def _ids(jobs):
    return {j["id"] for j in jobs}


GATE = {
    f"ubuntu-latest-py3.12-x64{x}-shard{k}of3" for x in "01" for k in "123"
}


def test_pull_request_runs_only_the_gate():
    jobs = _jobs("pull_request")
    assert _ids(jobs) == GATE
    assert not any(j["slow"] or j["orbit_slow"] for j in jobs)


def test_pull_request_touching_orbit_modules_adds_their_slow_tests():
    jobs = _jobs("pull_request", orbit_slow=True)
    assert _ids(jobs) == GATE
    assert all(
        not j["slow"] and "test_aliases.py" in j["orbit_slow"] for j in jobs
    )


def test_pyproject_pull_request_adds_the_floors_only():
    jobs = _jobs("pull_request", pyproject=True)
    assert _ids(jobs) == GATE | {"ubuntu-latest-py3.11-x640-lowest"}
    assert not any(j["slow"] for j in jobs)


def test_main_push_is_the_gate_with_slow_tests():
    jobs = _jobs("push")
    assert _ids(jobs) == GATE
    assert all(j["slow"] and not j["orbit_slow"] for j in jobs)


def test_full_ci_label_adds_macos_py313_and_the_floors():
    jobs = _jobs("pull_request", full_label=True)
    assert _ids(jobs) == GATE | {
        "macos-latest-py3.11-x640",
        "ubuntu-latest-py3.13-x640",
        "ubuntu-latest-py3.11-x640-lowest",
    }
    gate = [j for j in jobs if "shard" in j]
    assert all(j["slow"] for j in gate)
    # The extra jobs run slow tests nightly only.
    assert not any(j["slow"] for j in jobs if "shard" not in j)


def test_dispatch_is_full_and_nightly_adds_canary_and_slow_everywhere():
    assert _ids(_jobs("workflow_dispatch")) == _ids(
        _jobs("pull_request", full_label=True)
    )
    nightly = _jobs("schedule")
    assert _ids(nightly) == _ids(_jobs("workflow_dispatch")) | {
        "ubuntu-latest-py3.12-x640-canary"
    }
    assert all(j["slow"] for j in nightly)


@pytest.mark.parametrize(
    "event", ["pull_request", "push", "schedule", "workflow_dispatch"]
)
def test_ids_are_unique(event):
    jobs = _jobs(event, full_label=True)
    assert len(_ids(jobs)) == len(jobs)


def test_changed_flags():
    assert ci.changed_flags(["README.md"]) == (False, False)
    assert ci.changed_flags(["pyproject.toml", "docs/a.md"]) == (True, False)
    for path in (
        "src/virgil/aliases.py",
        "src/virgil/orbit_search.py",
        "src/virgil/orbits.py",
        "tests/test_epochs.py",
        "tests/test_orbit_positions.py",
    ):
        assert ci.changed_flags([path]) == (False, True), path
    # Subpackages, and tests that merely mention an orbit, do not count.
    assert ci.changed_flags(["src/virgil/pipeline/orbits.py"]) == (
        False,
        False,
    )
    assert ci.changed_flags(["tests/test_pipeline_binary.py"]) == (
        False,
        False,
    )


def test_dtype_independent_list_is_safe():
    """Files skipped on the float64 jobs exist and carry no evidence."""
    from tests.conftest import DTYPE_INDEPENDENT

    assert len(set(DTYPE_INDEPENDENT)) == len(DTYPE_INDEPENDENT)
    for name in DTYPE_INDEPENDENT:
        path = ROOT / "tests" / name
        assert path.is_file(), f"{name} is listed but does not exist"
        # The float64 run is the evidence virgil-validation reads.
        assert not re.search(r"validates\(", path.read_text()), name
    # The numerical files the float64 jobs exist for stay in.
    for name in ("test_aliases.py", "test_fitting.py", "test_likelihood.py"):
        assert name not in DTYPE_INDEPENDENT


def test_main_reads_the_environment(monkeypatch, capsys):
    monkeypatch.setenv("EVENT", "workflow_dispatch")
    monkeypatch.setenv("NIGHTLY", "true")
    ci.main()
    out = capsys.readouterr().out
    assert out.startswith("matrix=")
    assert "canary" in out
    monkeypatch.setenv("NIGHTLY", "false")
    ci.main()
    assert "canary" not in capsys.readouterr().out

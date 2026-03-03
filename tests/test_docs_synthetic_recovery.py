from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "synthetic_binary_workflow.py"
SPEC = spec_from_file_location("synthetic_binary_workflow", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

run_synthetic_binary_demo = MODULE.run_synthetic_binary_demo
within_two_sigma = MODULE.within_two_sigma
fisher_within_three_sigma = MODULE.fisher_within_three_sigma


def test_synthetic_docs_binary_recovery_within_two_sigma(tmp_path: Path):
    output = tmp_path / "synthetic_binary_docs.oifits"
    summary = run_synthetic_binary_demo(output)

    checks = within_two_sigma(summary)
    fisher_checks = fisher_within_three_sigma(summary)

    assert output.exists()
    assert all(checks.values()), f"Recovered parameters exceed 2σ bounds: {checks}; summary={summary}"
    assert all(fisher_checks.values()), f"Fisher-HMC recovered parameters exceed 3σ bounds: {fisher_checks}; summary={summary}"

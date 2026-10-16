"""The quickstart page's code runs from a clone and recovers its companion."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_recovers_the_injected_companion(monkeypatch):
    page = (ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    code = page.split("```python\n")[1].split("```")[0]
    monkeypatch.chdir(ROOT)  # the page reads data/NuHor_F480M.oifits
    scope = {}
    exec(compile(code, "docs/quickstart.md", "exec"), scope)
    values = {k: float(v) for k, v in scope["result"].values.items()}
    assert values["dra"] == pytest.approx(120.0, abs=5.0)
    assert values["ddec"] == pytest.approx(-80.0, abs=5.0)
    assert values["flux"] == pytest.approx(0.02, rel=0.2)
    assert scope["result"].info["converged"]

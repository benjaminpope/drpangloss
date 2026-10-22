"""Every test file has a timing in tests/test_durations.json.

A file without one is packed as the mean file, which unbalances the shards
(tests/conftest.py). Refresh with scripts/update_test_durations.py.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def test_every_test_file_has_a_recorded_duration():
    known = json.loads((HERE / "test_durations.json").read_text())
    files = {f"tests/{p.name}" for p in HERE.glob("test_*.py")}
    assert not files - set(known), (
        f"add to tests/test_durations.json: {sorted(files - set(known))}"
    )
    assert not set(known) - files, "durations for files that no longer exist"

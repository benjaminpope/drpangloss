"""`import virgil` leaves matplotlib for first use (astropy once oidata.py stops importing oifits at top level)."""

import subprocess
import sys

CODE = """
import sys, virgil
assert "matplotlib" not in sys.modules
from virgil import read_oifits, write_oifits
assert virgil.oifits.read_oifits is read_oifits
assert virgil.plotting.plot_roc and "matplotlib" in sys.modules
assert not hasattr(virgil, "name")
"""


def test_import_virgil_is_light():
    subprocess.run([sys.executable, "-c", CODE], check=True, timeout=300)

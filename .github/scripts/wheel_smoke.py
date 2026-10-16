"""Import the installed wheel and evaluate one likelihood (tests.yml).

Run from outside the checkout with the wheel's own interpreter, so the
installed package is what gets imported.
"""

import importlib.metadata
import math

import virgil
from virgil.coverage import nrm_oidata
from virgil.likelihood import model_loglike

assert "site-packages" in virgil.__file__, virgil.__file__
version = importlib.metadata.version("virgil-astro")
assert virgil.__version__ == version, virgil.__version__
model = virgil.BinaryModelCartesian(dra=100.0, ddec=50.0, flux=0.01)
value = float(model_loglike(model, nrm_oidata()))
assert math.isfinite(value), value
print("virgil", virgil.__version__, "log likelihood", value)

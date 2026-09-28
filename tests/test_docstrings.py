import doctest

import pytest

import drpangloss.grid_fit
import drpangloss.models


@pytest.mark.parametrize(
    "module",
    [drpangloss.models, drpangloss.grid_fit],
    ids=lambda m: m.__name__,
)
def test_docstring_examples_run(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0

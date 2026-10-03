import doctest

import pytest

import virgil.grid_fit
import virgil.models


@pytest.mark.parametrize(
    "module",
    [virgil.models, virgil.grid_fit],
    ids=lambda m: m.__name__,
)
def test_docstring_examples_run(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0

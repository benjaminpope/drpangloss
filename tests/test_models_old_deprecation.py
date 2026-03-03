import importlib
import warnings

import numpy as np


def test_models_old_module_import_warns_deprecation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module("drpangloss.models_old")
        importlib.reload(module)

    messages = [str(item.message) for item in caught]
    assert any("drpangloss.models_old is deprecated" in message for message in messages)


def test_models_old_class_constructor_warns():
    import drpangloss.models_old as models_old

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        models_old.BinaryModelCartesian(1.0, -1.0, 1e-3)

    messages = [str(item.message) for item in caught]
    assert any("BinaryModelCartesian" in message for message in messages)


def test_models_old_cp_indices_warns():
    import drpangloss.models_old as models_old

    vis_sta_index = np.array([[1, 2], [2, 3], [1, 3]], dtype=int)
    cp_sta_index = np.array([[1, 2, 3]], dtype=int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        i1, i2, i3 = models_old.cp_indices(vis_sta_index, cp_sta_index)

    assert i1.shape == (1,)
    assert i2.shape == (1,)
    assert i3.shape == (1,)

    messages = [str(item.message) for item in caught]
    assert any("cp_indices" in message for message in messages)

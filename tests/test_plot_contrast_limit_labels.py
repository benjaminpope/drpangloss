import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "drpangloss" / "plotting.py"
)
SPEC = importlib.util.spec_from_file_location(
    "drpangloss_plotting", MODULE_PATH
)
PLOTTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTTING)


def _plot_and_collect_axes(**kwargs):
    plt.close("all")
    PLOTTING.plot_contrast_limits(
        np.ones((2, 2)) * 1e-4,
        {"dra": np.array([1.0, 0.0]), "ddec": np.array([0.0, 1.0])},
        np.array([0.0, 1.0]),
        np.array([10.0, 9.0]),
        np.array([0.2, 0.3]),
        **kwargs,
    )
    fig = plt.gcf()
    map_axis = next(ax for ax in fig.axes if ax.get_title())
    legend_axis = next(ax for ax in fig.axes if ax.get_legend() is not None)
    return fig, map_axis, legend_axis


def test_plot_contrast_limits_percentile_label():
    fig, map_axis, legend_axis = _plot_and_collect_axes(
        percentile=0.9772498680518208
    )

    assert map_axis.get_title() == "97.7% Upper Limit Map ($\\Delta$mag)"
    assert legend_axis.get_legend().texts[0].get_text() == "97.7% Upper Limit"
    plt.close(fig)


def test_plot_contrast_limits_sigma_label():
    fig, map_axis, legend_axis = _plot_and_collect_axes(sigma=5.0)

    assert (
        map_axis.get_title() == "5$\\sigma$ Contrast Limit Map ($\\Delta$mag)"
    )
    assert (
        legend_axis.get_legend().texts[0].get_text()
        == "5$\\sigma$ Contrast Limit"
    )
    plt.close(fig)

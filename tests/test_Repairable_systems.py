from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from reliability.Datasets import MCF_1
from reliability.Repairable_systems import (
    ROCOF,
    MCF_nonparametric,
    MCF_parametric,
    optimal_replacement_time,
    reliability_growth,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureResult

atol = 1e-8
rtol = 1e-7


def test_reliability_growth_duane():
    times = [10400, 26900, 43400, 66400, 89400, 130400, 163400, 232000, 242000, 340700]
    rg_duane = reliability_growth(times=times, model="Duane", target_MTBF=50000)
    rg_duane.print_results()
    assert_allclose(rg_duane.A, 0.002355878294089656, rtol=rtol, atol=atol)
    assert_allclose(rg_duane.Alpha, 0.33617199465228115, rtol=rtol, atol=atol)
    assert_allclose(rg_duane.DMTBF_I, 46304.175358824315, rtol=rtol, atol=atol)
    assert_allclose(rg_duane.DMTBF_C, 30738.008367719336, rtol=rtol, atol=atol)
    assert_allclose(rg_duane.DFI_I, 2.1596324570100073e-05, rtol=rtol, atol=atol)
    assert_allclose(rg_duane.DFI_C, 3.253301215996112e-05, rtol=rtol, atol=atol)
    assert_allclose(rg_duane.time_to_target, 1448446.368611323, rtol=rtol, atol=atol)


def test_reliability_growth_crow_amsaa():
    times = [10400, 26900, 43400, 66400, 89400, 130400, 163400, 232000, 242000, 340700]
    rg_crow = reliability_growth(
        times=times,
        model="Crow-AMSAA",
        target_MTBF=50000,
    )
    assert_allclose(rg_crow.Beta, 0.741656619656656, rtol=rtol, atol=atol)
    assert_allclose(rg_crow.Lambda, 0.0007886414235385733, rtol=rtol, atol=atol)
    assert_allclose(rg_crow.growth_rate, 0.25834338034334403, rtol=rtol, atol=atol)
    assert_allclose(rg_crow.DMTBF_I, 45937.70094814556, rtol=rtol, atol=atol)
    assert_allclose(rg_crow.DMTBF_C, 34070.0, rtol=rtol, atol=atol)
    assert_allclose(rg_crow.DFI_I, 2.176861225878063e-05, rtol=rtol, atol=atol)
    assert_allclose(rg_crow.DFI_C, 2.9351335485764603e-05, rtol=rtol, atol=atol)
    assert_allclose(rg_crow.time_to_target, 1503979.9172547427, rtol=rtol, atol=atol)


def test_ROCOF():
    times: list[int] = [
        104,
        131,
        1597,
        59,
        4,
        503,
        157,
        6,
        118,
        173,
        114,
        62,
        101,
        216,
        106,
        140,
        1,
        102,
        3,
        393,
        96,
        232,
        89,
        61,
        37,
        293,
        7,
        165,
        87,
        99,
    ]
    results = ROCOF(times_between_failures=times)
    results.print_results()
    results.plot()
    plt.close("all")
    assert_allclose(results.U, 2.4094382960447107, rtol=rtol, atol=atol)
    assert_allclose(results.z_crit, (-1.959963984540054, 1.959963984540054), rtol=rtol, atol=atol)
    assert results.trend == "worsening"
    assert_allclose(results.Beta_hat, 1.5880533880966818, rtol=rtol, atol=atol)  # type: ignore
    assert_allclose(results.Lambda_hat, 3.702728848984535e-05, rtol=rtol, atol=atol)  # type: ignore
    assert (
        results.ROCOF
        == "ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t."
    )


def test_old_MCF_nonparametric():
    times = MCF_1().times
    results = MCF_nonparametric(data=times)
    results.print_results()
    results.plot()
    plt.close("all")
    assert_allclose(sum(results.MCF), 22.833333333333332, rtol=rtol, atol=atol)
    assert_allclose(sum(results.variance), 3.933518518518521, rtol=rtol, atol=atol)
    assert_allclose(sum(results.lower), 13.992740081348929, rtol=rtol, atol=atol)
    assert_allclose(sum(results.upper), 38.23687898478023, rtol=rtol, atol=atol)


def test_MCF_parametric():
    times = MCF_1().times
    results = MCF_parametric(data=times)
    results.print_results()
    results.plot()
    plt.close("all")
    assert_allclose(sum(results.MCF), 22.833333333333332, rtol=rtol, atol=atol)
    assert_allclose(sum(results.times), 214, rtol=rtol, atol=atol)
    assert_allclose(results.alpha, 11.980589826209348, rtol=rtol, atol=atol)
    assert_allclose(results.beta, 1.6736221860957468, rtol=rtol, atol=atol)
    assert_allclose(results.cov_alpha_beta, 0.034638880600157955, rtol=rtol, atol=atol)
    assert_allclose(results.alpha_lower, 11.219187030973842, rtol=rtol, atol=atol)
    assert_allclose(results.alpha_upper, 12.793666081829453, rtol=rtol, atol=atol)
    assert_allclose(results.beta_lower, 1.4980169559010625, rtol=rtol, atol=atol)
    assert_allclose(results.beta_upper, 1.8698127619704332, rtol=rtol, atol=atol)


def test_MCF_nonparametric():
    data = [[1, 3, 5], [3, 6, 8]]
    results = MCF_nonparametric(data=data)
    assert_allclose(sum(results.MCF), 5.5, rtol=rtol, atol=atol)
    assert_allclose(sum(results.variance), 1.125, rtol=rtol, atol=atol)
    assert_allclose(sum(results.lower), 3.0329684147169633, rtol=rtol, atol=atol)
    assert_allclose(sum(results.upper), 10.552109216491933, rtol=rtol, atol=atol)


def test_MCF_nonparametric_with_np_array():
    data = np.array([[1, 3, 5], [3, 6, 8]])
    results = MCF_nonparametric(data=data)
    assert_allclose(sum(results.MCF), 5.5, rtol=rtol, atol=atol)
    assert_allclose(sum(results.variance), 1.125, rtol=rtol, atol=atol)
    assert_allclose(sum(results.lower), 3.0329684147169633, rtol=rtol, atol=atol)
    assert_allclose(sum(results.upper), 10.552109216491933, rtol=rtol, atol=atol)


def test_MCF_nonparametric_with_mixed_data_types():
    data = [[1, 3, 5], np.array([3, 6, 8])]
    results = MCF_nonparametric(data=data)
    assert_allclose(sum(results.MCF), 5.5, rtol=rtol, atol=atol)
    assert_allclose(sum(results.variance), 1.125, rtol=rtol, atol=atol)
    assert_allclose(sum(results.lower), 3.0329684147169633, rtol=rtol, atol=atol)
    assert_allclose(sum(results.upper), 10.552109216491933, rtol=rtol, atol=atol)


def test_MCF_nonparametric_with_single_system():
    data = [1, 3, 5]
    results = MCF_nonparametric(data=data)
    assert_allclose(sum(results.MCF), 3.0, rtol=rtol, atol=atol)
    assert_allclose(sum(results.variance), 0.0, rtol=rtol, atol=atol)
    assert_allclose(sum(results.lower), 3.0, rtol=rtol, atol=atol)
    assert_allclose(sum(results.upper), 3.0, rtol=rtol, atol=atol)


def test_MCF_nonparametric_with_invalid_data_type():
    data = "invalid"
    with pytest.raises(TypeError):
        MCF_nonparametric(data=data)


def test_MCF_nonparametric_with_invalid_CI():
    data = [[1, 3, 5], [3, 6, 8]]
    with pytest.raises(ValueError):
        MCF_nonparametric(data=data, CI=1.5)


def test_MCF_nonparametric_with_empty_data():
    data = []
    with pytest.raises(ValueError):
        MCF_nonparametric(data=data)


def test_MCF_nonparametric_with_single_event():
    data = [[1]]
    with pytest.raises(ValueError):
        MCF_nonparametric(data=data)


def test_MCF_nonparametric_with_many_systems():
    data = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
    results = MCF_nonparametric(data=data)
    assert_allclose(sum(results.MCF), 22.833333333333332, rtol=rtol, atol=atol)
    assert_allclose(sum(results.variance), 3.933518518518521, rtol=rtol, atol=atol)
    assert_allclose(sum(results.lower), 13.992740081348929, rtol=rtol, atol=atol)
    assert_allclose(sum(results.upper), 38.23687898478023, rtol=rtol, atol=atol)


def test_print_results(capsys: pytest.CaptureFixture[str]):
    data = [1, 3, 5]
    results = MCF_nonparametric(data=data)
    captured = capsys.readouterr()
    results.print_results()

    captured = capsys.readouterr()
    captured_out = captured.out
    expected_output = "Mean Cumulative Function results (95% CI):"
    assert expected_output in captured_out


def test_plot_with_CI():
    data = [1, 3, 5]
    results = MCF_nonparametric(data=data)
    plt.figure()
    results.plot(plot_CI=True)
    plt.close()


def test_plot_without_CI():
    data = [1, 3, 5]
    results = MCF_nonparametric(data=data)
    plt.figure()
    results.plot(plot_CI=False)
    plt.close()


def test_plot_with_custom_color():
    data = [1, 3, 5]
    results = MCF_nonparametric(data=data)
    plt.figure()
    results.plot(color="red")
    plt.close()


def test_plot_with_additional_kwargs():
    data = [1, 3, 5]
    results = MCF_nonparametric(data=data)
    plt.figure()
    results.plot(linestyle="--", linewidth=2)
    plt.close()


def test_optimal_replacement_time():
    ort0 = optimal_replacement_time(cost_PM=1, cost_CM=5, weibull_alpha=1000, weibull_beta=2.5, q=0)
    ort0.show_ratio_plot()
    assert_allclose(ort0.ORT, 493.1851185118512, rtol=rtol, atol=atol)
    assert_allclose(ort0.min_cost, 0.0034620429189943167, rtol=rtol, atol=atol)
    ort1 = optimal_replacement_time(cost_PM=1, cost_CM=5, weibull_alpha=1000, weibull_beta=2.5, q=1)
    assert_allclose(ort1.ORT, 1618.644582767346, rtol=rtol, atol=atol)
    assert_allclose(ort1.min_cost, 0.0051483404213951, rtol=rtol, atol=atol)


def test_ort_print_results(capsys: pytest.CaptureFixture[str]) -> None:
    """Test function for the `print_results` method of the `optimal_replacement_time` class.

    Args:
    ----
        capsys (pytest.CaptureFixture[str]): Fixture for capturing stdout.

    Returns:
    -------
        None

    """
    ort = optimal_replacement_time(cost_PM=1, cost_CM=5, weibull_alpha=1000, weibull_beta=2.5, q=0)
    ort.print_results()
    captured: CaptureResult[str] = capsys.readouterr()
    captured_out: str = captured.out
    expected_output = (
        "Results from optimal_replacement_time:\n"
        "Cost model assuming as good as new replacement (q=0):\n"
        "The minimum cost per unit time is 0.0 \n"
        "The optimal replacement time is 493.19\n"
    )
    captured_out = captured_out.replace("Backend TkAgg is interactive backend. Turning interactive mode on.\n", "")
    captured_out = captured_out.replace("\x1b[1m\x1b[23m\x1b[4m\x1b[49m\x1b[39m", "")
    captured_out = captured_out.replace("\x1b[0m", "")
    assert captured_out == expected_output
    plt.close()


def test_show_time_plot():
    """Test case for the show_time_plot method of the RepairableSystem class.

    This test verifies the properties of the plot generated by the show_time_plot method.

    Returns
    -------
        None

    """
    rs = optimal_replacement_time(cost_PM=1, cost_CM=5, weibull_alpha=1000, weibull_beta=2.5, q=0)
    ax: plt.Axes = rs.show_time_plot(color="red")

    # Assert the plot properties
    assert ax.get_xlabel() == "Replacement time"
    assert ax.get_ylabel() == "Cost per unit time"
    assert ax.get_title() == "Optimal replacement time estimation"
    assert ax.get_ylim() == (0, rs.min_cost * 2)
    assert ax.get_xlim() == (
        0,
        rs._optimal_replacement_time__weibull_alpha * rs._optimal_replacement_time__alpha_multiple,
    )

    # Assert the plotted lines
    lines = ax.get_lines()
    assert len(lines) == 2
    assert_allclose(lines[0].get_xdata(), rs._optimal_replacement_time__t)
    assert_allclose(lines[0].get_ydata(), rs._optimal_replacement_time__CPUT)
    assert lines[0].get_color() == "red"
    plt.close()


def test_show_ratio_plot():
    """Test function for the show_ratio_plot method of the optimal_replacement_time class.
    It checks if the returned object is an instance of plt.Axes and closes the plot afterwards.
    """
    ort = optimal_replacement_time(cost_PM=100, cost_CM=500, weibull_alpha=10, weibull_beta=2)
    ax = ort.show_ratio_plot()
    assert isinstance(ax, plt.Axes)
    plt.close()

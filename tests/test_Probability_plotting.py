import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from reliability.Distributions import (
    Beta_Distribution,
    Exponential_Distribution,
    Gamma_Distribution,
    Gumbel_Distribution,
    Loglogistic_Distribution,
    Lognormal_Distribution,
    Normal_Distribution,
    Weibull_Distribution,
)
from reliability.Other_functions import make_right_censored_data
from reliability.Probability_plotting import (
    Beta_probability_plot,
    Exponential_probability_plot,
    Exponential_probability_plot_Weibull_Scale,
    Gamma_probability_plot,
    Gumbel_probability_plot,
    Loglogistic_probability_plot,
    Lognormal_probability_plot,
    Normal_probability_plot,
    PP_plot_parametric,
    PP_plot_semiparametric,
    QQ_plot_parametric,
    QQ_plot_semiparametric,
    Weibull_probability_plot,
    plot_points,
    plotting_positions,
)


def test_Weibull():
    dist = Weibull_Distribution(alpha=250, beta=3)
    for i, x in enumerate([10, 100, 1000]):
        plt.subplot(131 + i)
        dist.CDF(linestyle="--", label="True CDF")
        failures = dist.random_samples(x, seed=42)  # take 10, 100, 1000 samples
        Weibull_probability_plot(failures=failures)  # this is the probability plot
        plt.title(str(str(x) + " samples"))


def test_Normal():
    dist = Normal_Distribution(mu=50, sigma=10)
    failures = dist.random_samples(100, seed=5)
    Normal_probability_plot(failures=failures)  # generates the probability plot
    dist.CDF(linestyle="--", label="True CDF")  # this is the actual distribution provided for comparison
    plt.legend()


def test_Loglogistic():
    data = Loglogistic_Distribution(alpha=50, beta=8, gamma=10).random_samples(100, seed=1)
    Loglogistic_probability_plot(failures=data)


@pytest.mark.flaky(reruns=3)
def test_Exponential():
    data1 = Exponential_Distribution(Lambda=1 / 10).random_samples(
        50, seed=42,
    )  # should give Exponential Lambda = 0.01 OR Weibull alpha = 10
    data2 = Exponential_Distribution(Lambda=1 / 100).random_samples(
        50, seed=42,
    )  # should give Exponential Lambda = 0.001 OR Weibull alpha = 100
    Exponential_probability_plot(failures=data1)
    Exponential_probability_plot(failures=data2)


def test_PP_plot_parametric():
    Field = Normal_Distribution(mu=100, sigma=30)
    Lab = Weibull_Distribution(alpha=120, beta=3)
    PP_plot_parametric(X_dist=Field, Y_dist=Lab, x_quantile_lines=[0.3, 0.6], y_quantile_lines=[0.1, 0.6])


def test_Exponential_probability_plot_Weibull_Scale():
    data1 = Exponential_Distribution(Lambda=1 / 10).random_samples(
        50, seed=42,
    )  # should give Exponential Lambda = 0.01 OR Weibull alpha = 10
    data2 = Exponential_Distribution(Lambda=1 / 100).random_samples(
        50, seed=42,
    )  # should give Exponential Lambda = 0.001 OR Weibull alpha = 100
    Exponential_probability_plot_Weibull_Scale(failures=data1)
    Exponential_probability_plot_Weibull_Scale(failures=data2)


def test_plot_points_and_plotting_positions():
    # failure data from oil pipe corrosion
    bend = [74, 52, 32, 76, 46, 35, 65, 54, 56, 20, 71, 72, 38, 61, 29]
    valve = [78, 83, 94, 76, 86, 39, 54, 82, 96, 66, 63, 57, 82, 70, 72, 61, 84, 73, 69, 97]
    joint = [74, 52, 32, 76, 46, 35, 65, 54, 56, 25, 71, 72, 37, 61, 29]

    # combine the data into a single array
    data = np.hstack([bend, valve, joint])
    color = np.hstack([["red"] * len(bend), ["green"] * len(valve), ["blue"] * len(joint)])

    # create the probability plot and hide the scatter points
    Weibull_probability_plot(failures=data, show_scatter_points=False)

    # redraw the scatter points. kwargs are passed to plt.scatter so a list of color is accepted
    plot_points(failures=data, color=color, marker="^", s=100)

    # To show the legend correctly, we need to replot some points in separate scatter plots to create different legend entries
    x, y = plotting_positions(failures=data)
    plt.scatter(x[0], y[0], color=color[0], marker="^", s=100, label="bend")
    plt.scatter(x[len(bend)], y[len(bend)], color=color[len(bend)], marker="^", s=100, label="valve")
    plt.scatter(
        x[len(bend) + len(valve)],
        y[len(bend) + len(valve)],
        color=color[len(bend) + len(valve)],
        marker="^",
        s=100,
        label="joint",
    )

def test_plotting_positions():
    # Test case 1: Basic test case with failure data
    failures: list[int] = [10, 20, 30, 40, 50]
    expected_x: npt.NDArray[np.float64] = np.array([10., 20., 30., 40., 50.], dtype=np.float64)
    expected_y: npt.NDArray[np.float64] = np.array([0.12962962962962962, 0.31481481481481477, 0.5, 0.6851851851851851, 0.8703703703703703])
    x, y = plotting_positions(failures)
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)

    # Test case 2: Test case with right censored data
    failures = [10, 20, 30, 40, 50]
    right_censored: list[int] = [25, 35]
    expected_x: npt.NDArray[np.float64] = np.array([10., 20., 30., 40., 50.], dtype=np.float64)
    expected_y: npt.NDArray[np.float64] = np.array([0.09459459, 0.22972973, 0.39189189, 0.60810811, 0.82432432])
    x, y = plotting_positions(failures, right_censored)
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)

    # Test case 3: Test case with custom value of a
    failures = [10, 20, 30, 40, 50]
    a = 0.5
    expected_x: npt.NDArray[np.float64] = np.array([10., 20., 30., 40., 50.], dtype=np.float64)
    expected_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x, y = plotting_positions(failures, a=a)
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)

    # Test case 4: Test case with sorted output
    failures = [50, 40, 30, 20, 10]
    expected_x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    expected_y = np.array([0.12962962962962962, 0.31481481481481477, 0.5, 0.6851851851851851, 0.8703703703703703])
    x, y = plotting_positions(failures, sort=True)
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)

    # Test case 5: Test case with empty input
    with pytest.raises(IndexError):
        failures = []
        plotting_positions(failures)


    # Test case 6: Test case with invalid input type
    with pytest.raises(ValueError):
        plotting_positions("invalid input")

    # Test case 7: Test case with invalid a value
    with pytest.raises(ValueError):
        plotting_positions([10, 20, 30], a=-0.5)
    with pytest.raises(ValueError):
        plotting_positions([10, 20, 30], a=1.5)

    # Test case 8: Test case with invalid sort value
    with pytest.raises(ValueError):
        plotting_positions([10, 20, 30], sort="invalid sort") # type: ignore

    # Test case 9: Test case with large input data
    failures = list(range(1, 10001))
    a = 0.3
    expected_x = np.array(failures, dtype=np.float64)
    expected_y = np.linspace((1-a) / (len(failures) + 1 -2*a), ((len(failures)-a) / (len(failures) + 1 -2*a)), len(failures))
    x, y = plotting_positions(failures=failures, a=a)
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)


def test_Gumbel_probability_plot():
    # Test case 1: Basic test case with failure data
    failures = [10, 20, 30, 40, 50]
    fig = Gumbel_probability_plot(failures=failures)
    assert isinstance(fig, plt.Figure)

    # Test case 2: Test case with right censored data
    failures = [10, 20, 30, 40, 50]
    right_censored = [25, 35]
    fig = Gumbel_probability_plot(failures=failures, right_censored=right_censored)
    assert isinstance(fig, plt.Figure)

    # Test case 3: Test case with fitted distribution
    failures = [10, 20, 30, 40, 50]
    fitted_dist_params = Gumbel_Distribution(mu=25, sigma=5)
    fig = Gumbel_probability_plot(failures=failures, _fitted_dist_params=fitted_dist_params)
    assert isinstance(fig, plt.Figure)

    # Test case 4: Test case with custom parameters
    failures = [10, 20, 30, 40, 50]
    fig = Gumbel_probability_plot(failures=failures, show_fitted_distribution=False, show_scatter_points=False)
    assert isinstance(fig, plt.Figure)

    # Test case 5: Test case with downsampling
    failures = np.random.randint(1, 100, size=10000)
    fig = Gumbel_probability_plot(failures=failures, downsample_scatterplot=True)
    assert isinstance(fig, plt.Figure)

    # Test case 6: Test case with invalid input
    with pytest.raises(ValueError):
        Gumbel_probability_plot(failures=[], show_fitted_distribution=True)

    with pytest.raises(ValueError):
        Gumbel_probability_plot(failures="invalid input")

    with pytest.raises(ValueError):
        Gumbel_probability_plot(failures=[10], _fitted_dist_params=None)

    with pytest.raises(ValueError):
        Gumbel_probability_plot(failures=[10, 20], CI=1.5)

    with pytest.raises(ValueError):
        Gumbel_probability_plot(failures=[10, 20], CI_type="invalid type")

def test_QQ_plot_semiparametric():
    # Test case 1: Basic test case with Weibull distribution
    X_data_failures = [10, 20, 30, 40, 50]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    expected_model = [2.411095944079598, 3.0221225173625923, -22.404307687043136]
    fig, model  = QQ_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist)
    assert isinstance(fig, plt.Figure)
    assert np.allclose(model, expected_model)

    # Test case 2: Test case with right censored data
    X_data_failures = [10, 20, 30, 40, 50]
    X_data_right_censored = [25, 35]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    expected_model = [2.762983200925992, 3.3508143133647574, -21.553807456088077]
    fig, model = QQ_plot_semiparametric(X_data_failures=X_data_failures, X_data_right_censored=X_data_right_censored, Y_dist=Y_dist)
    assert isinstance(fig, plt.Figure)
    assert np.allclose(model, expected_model)

    # Test case 3: Test case with normal distribution
    X_data_failures = [10, 20, 30, 40, 50]
    Y_dist = Normal_Distribution(mu=50, sigma=10)
    expected_model = [1.3232401964337186, 1.2778210803854542, 1.665367588436365]
    fig, model = QQ_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist)
    assert isinstance(fig, plt.Figure)
    assert np.allclose(model, expected_model)

    # Test case 4: Test case with downsampling
    np.random.seed(0)
    X_data_failures = np.random.randint(1, 100, size=10000)
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    expected_model = [1.733973949688778, 1.5768901869332173, 10.371020288250993]
    fig, model = QQ_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist, downsample_scatterplot=True)
    assert isinstance(fig, plt.Figure)
    assert np.allclose(model, expected_model)

    # Test case 5: Test case with invalid input
    with pytest.raises(ValueError):
        QQ_plot_semiparametric(X_data_failures=None, Y_dist=Y_dist)

    with pytest.raises(ValueError):
        QQ_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=None)

    with pytest.raises(ValueError):
        QQ_plot_semiparametric(X_data_failures=[], Y_dist=Y_dist)

    with pytest.raises(ValueError):
        QQ_plot_semiparametric(X_data_failures=X_data_failures, Y_dist="invalid distribution")

    with pytest.raises(ValueError):
        QQ_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Weibull_Distribution(alpha=100, beta=2), method="invalid method")

def test_PP_plot_semiparametric():
    # Test case 1: Basic test case with Weibull distribution
    X_data_failures = [10, 20, 30, 40, 50]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    fig = PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist)
    assert isinstance(fig, plt.Figure)

    # Test case 2: Test case with right censored data
    X_data_failures = [10, 20, 30, 40, 50]
    X_data_right_censored = [25, 35]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    fig = PP_plot_semiparametric(X_data_failures=X_data_failures, X_data_right_censored=X_data_right_censored, Y_dist=Y_dist)
    assert isinstance(fig, plt.Figure)

    # Test case 3: Test case with different method
    X_data_failures = [10, 20, 30, 40, 50]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    fig = PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist, method="NA")
    assert isinstance(fig, plt.Figure)

    # Test case 4: Test case with show_diagonal_line set to False
    X_data_failures = [10, 20, 30, 40, 50]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    fig = PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist, show_diagonal_line=False)
    assert isinstance(fig, plt.Figure)

    # Test case 5: Test case with downsample_scatterplot set to True
    X_data_failures = [10, 20, 30, 40, 50]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    fig = PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist, downsample_scatterplot=True)
    assert isinstance(fig, plt.Figure)

    # Test case 6: Test case with additional kwargs
    X_data_failures = [10, 20, 30, 40, 50]
    Y_dist = Weibull_Distribution(alpha=100, beta=2)
    fig = PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist, color="r", marker="o")
    assert isinstance(fig, plt.Figure)

    # Test case 7: Test case with invalid input
    with pytest.raises(ValueError):
        PP_plot_semiparametric(X_data_failures=None, Y_dist=Y_dist)

    with pytest.raises(ValueError):
        PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=None)

    with pytest.raises(ValueError):
        PP_plot_semiparametric(X_data_failures=[], Y_dist=Y_dist)

    with pytest.raises(ValueError):
        PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist, method="invalid method")

    with pytest.raises(ValueError):
        PP_plot_semiparametric(X_data_failures=X_data_failures, Y_dist=Y_dist, downsample_scatterplot="invalid value") # type: ignore


def test_Lognormal_probability_plot():
    # Test case 1: Basic test case with Lognormal_2P distribution
    failures = [10, 20, 30, 40, 50]
    fig = Lognormal_probability_plot(failures=failures)
    assert isinstance(fig, plt.Figure)

    # Test case 2: Test case with right censored data
    failures = [10, 20, 30, 40, 50]
    right_censored = [25, 35]
    fig = Lognormal_probability_plot(failures=failures, right_censored=right_censored)
    assert isinstance(fig, plt.Figure)

    # Test case 3: Test case with fitted distribution
    failures = [10, 20, 30, 40, 50]
    fitted_dist_params = Lognormal_Distribution(mu=25, sigma=5)
    fig = Lognormal_probability_plot(failures=failures, _fitted_dist_params=fitted_dist_params)
    assert isinstance(fig, plt.Figure)

    # Test case 4: Test case with custom parameters
    failures = [10, 20, 30, 40, 50]
    fig = Lognormal_probability_plot(failures=failures, show_fitted_distribution=False, show_scatter_points=False)
    assert isinstance(fig, plt.Figure)

    # Test case 5: Test case with downsampling
    failures = np.random.randint(1, 100, size=10000)
    fig = Lognormal_probability_plot(failures=failures, downsample_scatterplot=True)
    assert isinstance(fig, plt.Figure)

    # Test case 6: Test case with invalid input
    with pytest.raises(ValueError):
        Lognormal_probability_plot(failures=[], show_fitted_distribution=True)

    with pytest.raises(ValueError):
        Lognormal_probability_plot(failures="invalid input")

    with pytest.raises(ValueError):
        Lognormal_probability_plot(failures=[10], _fitted_dist_params=None)

    with pytest.raises(ValueError):
        Lognormal_probability_plot(failures=[10, 20], CI=1.5)

    with pytest.raises(ValueError):
        Lognormal_probability_plot(failures=[10, 20], CI_type="invalid type")

def test_QQ_plot_parametric():
    # test Case 1 Basic
    X_dist = Weibull_Distribution(alpha=100, beta=2)
    Y_dist = Weibull_Distribution(alpha=120, beta=3)
    expected_model = [1.1371382215319796, 0.839195193693172, 33.01171610495985]
    fig, model = QQ_plot_parametric(X_dist=X_dist, Y_dist=Y_dist)
    assert isinstance(fig, plt.Figure)
    assert np.allclose(model, expected_model)

    # test Case 2 No X input
    with pytest.raises(ValueError):
        QQ_plot_parametric(X_dist=X_dist, Y_dist=None)

    # Test Case 3 No Y Input
    with pytest.raises(ValueError):
        QQ_plot_parametric(X_dist=None, Y_dist=Y_dist)

    # Test Case 4 with keyword arguments
    fig, model = QQ_plot_parametric(X_dist=X_dist, Y_dist=Y_dist, show_fitted_lines=False, show_diagonal_line=True)
    assert isinstance(fig, plt.Figure)
    assert np.allclose(model, expected_model)

    # Test Case 5 with downsampling
    fig, model = QQ_plot_parametric(X_dist=X_dist, Y_dist=Y_dist, downsample_scatterplot=True)
    assert isinstance(fig, plt.Figure)
    assert np.allclose(model, expected_model)

def test_Beta_probability_plot():
    # Test case 1: Basic test case with failure data
    dist = Beta_Distribution(alpha=5, beta=4)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fig = Beta_probability_plot(failures=data.failures)
    assert isinstance(fig, plt.Figure)

    # Test case 2: Test case with right censored data
    fig = Beta_probability_plot(failures=data.failures, right_censored=data.right_censored)
    assert isinstance(fig, plt.Figure)

    # Test case 3: Test case with fitted distribution
    fig = Beta_probability_plot(failures=data.failures, _fitted_dist_params=dist)
    assert isinstance(fig, plt.Figure)

    # Test case 4: Test case with custom parameters
    fig = Beta_probability_plot(failures=data.failures, show_fitted_distribution=False, show_scatter_points=False)
    assert isinstance(fig, plt.Figure)

    # Test case 5: Test case with downsampling
    dist = Beta_Distribution(alpha=5, beta=4)
    rawdata = dist.random_samples(1000, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    fig = Beta_probability_plot(failures=data.failures, downsample_scatterplot=True)
    assert isinstance(fig, plt.Figure)

    # Test case 6: Test case with invalid input
    with pytest.raises(ValueError):
        Beta_probability_plot(failures=[], show_fitted_distribution=True)

    with pytest.raises(ValueError):
        Beta_probability_plot(failures="invalid input")

    with pytest.raises(ValueError):
        Beta_probability_plot(failures=[10], _fitted_dist_params=None)

    with pytest.raises(ValueError):
        Beta_probability_plot(failures=[10, 20], CI=1.5)

    with pytest.raises(ValueError):
        Beta_probability_plot(failures=[10, 20], CI_type="invalid type")


def test_Gamma_probability_plot():
    # Test case 1: Basic test case with failure data
    failures = [10, 20, 30, 40, 50]
    fig = Gamma_probability_plot(failures=failures)
    assert isinstance(fig, plt.Figure)

    # Test case 2: Test case with right censored data
    failures = [10, 20, 30, 40, 50]
    right_censored = [25, 35]
    fig = Gamma_probability_plot(failures=failures, right_censored=right_censored)
    assert isinstance(fig, plt.Figure)

    # Test case 3: Test case with fitted distribution
    failures = [10, 20, 30, 40, 50]
    fitted_dist_params = Gamma_Distribution(alpha=100, beta=2)
    fig = Gamma_probability_plot(failures=failures, _fitted_dist_params=fitted_dist_params)
    assert isinstance(fig, plt.Figure)

    # Test case 4: Test case with custom parameters
    failures = [10, 20, 30, 40, 50]
    fig = Gamma_probability_plot(failures=failures, show_fitted_distribution=False, show_scatter_points=False)
    assert isinstance(fig, plt.Figure)

    # Test case 5: Test case with downsampling
    failures = np.random.randint(1, 100, size=10000)
    fig = Gamma_probability_plot(failures=failures, downsample_scatterplot=True)
    assert isinstance(fig, plt.Figure)

    # Test case 6: Test case with invalid input
    with pytest.raises(ValueError):
        Gamma_probability_plot(failures=[], show_fitted_distribution=True)

    with pytest.raises(ValueError):
        Gamma_probability_plot(failures="invalid input")

    with pytest.raises(ValueError):
        Gamma_probability_plot(failures=[10], _fitted_dist_params=None)

    with pytest.raises(ValueError):
        Gamma_probability_plot(failures=[10, 20], CI=1.5)

    with pytest.raises(ValueError):
        Gamma_probability_plot(failures=[10, 20], CI_type="invalid type")

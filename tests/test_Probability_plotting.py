import matplotlib.pyplot as plt
import numpy as np

from reliability.Distributions import (
    Exponential_Distribution,
    Loglogistic_Distribution,
    Normal_Distribution,
    Weibull_Distribution,
)
from reliability.Probability_plotting import (
    Exponential_probability_plot,
    Exponential_probability_plot_Weibull_Scale,
    Loglogistic_probability_plot,
    Normal_probability_plot,
    PP_plot_parametric,
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


def test_Exponential():
    data1 = Exponential_Distribution(Lambda=1 / 10).random_samples(
        50, seed=42
    )  # should give Exponential Lambda = 0.01 OR Weibull alpha = 10
    data2 = Exponential_Distribution(Lambda=1 / 100).random_samples(
        50, seed=42
    )  # should give Exponential Lambda = 0.001 OR Weibull alpha = 100
    Exponential_probability_plot(failures=data1)
    Exponential_probability_plot(failures=data2)


def test_PP_plot_parametric():
    Field = Normal_Distribution(mu=100, sigma=30)
    Lab = Weibull_Distribution(alpha=120, beta=3)
    PP_plot_parametric(X_dist=Field, Y_dist=Lab, x_quantile_lines=[0.3, 0.6], y_quantile_lines=[0.1, 0.6])


def test_Exponential_probability_plot_Weibull_Scale():
    data1 = Exponential_Distribution(Lambda=1 / 10).random_samples(
        50, seed=42
    )  # should give Exponential Lambda = 0.01 OR Weibull alpha = 10
    data2 = Exponential_Distribution(Lambda=1 / 100).random_samples(
        50, seed=42
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

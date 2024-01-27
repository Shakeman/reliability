import matplotlib.pyplot as plt

from reliability.Distributions import (
    Exponential_Distribution,
    Loglogistic_Distribution,
    Normal_Distribution,
    Weibull_Distribution,
)
from reliability.Probability_plotting import (
    Exponential_probability_plot,
    Loglogistic_probability_plot,
    Normal_probability_plot,
    Weibull_probability_plot,
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
    data = Exponential_Distribution(Lambda=0.2, gamma=10).random_samples(100, seed=1)
    plt.subplot(121)
    Weibull_probability_plot(failures=data)
    plt.title("Example of a good fit")
    plt.subplot(122)
    Exponential_probability_plot(failures=data, fit_gamma=True)
    plt.title("Example of a bad fit")
    plt.subplots_adjust(bottom=0.1, right=0.94, top=0.93, wspace=0.34)  # adjust the formatting

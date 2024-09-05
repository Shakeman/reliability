import warnings

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Gamma_Distribution,
)
from reliability.Fitters import (
    Fit_Gamma_2P,
    Fit_Gamma_3P,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
rtol = 1e-3


def test_Fit_Gamma_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Gamma_Distribution(alpha=50, beta=2)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Gamma_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha, 23.703139876038513, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 2.953402789321292, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 168.68322949686174, rtol=rtol, atol=atol)

    LS = Fit_Gamma_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 33.88787165992774, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 2.1496080508234647, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 169.49240784971596, rtol=rtol, atol=atol)


def test_Fit_Gamma_3P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Gamma_Distribution(alpha=50, beta=2, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Gamma_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha, 45.81587614140445, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 1.2705118741470818, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 514.9479934874031, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 170.1665016824922, rtol=rtol, atol=atol)

    LS = Fit_Gamma_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 53.29489695285618, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 1.1483566629865927, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 515.9054428511635, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 171.77457192367393, rtol=rtol, atol=atol)

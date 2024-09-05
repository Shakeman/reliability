import warnings

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Lognormal_Distribution,
)
from reliability.Fitters import (
    Fit_Lognormal_2P,
    Fit_Lognormal_3P,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
rtol = 1e-3


def test_Fit_Lognormal_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Lognormal_Distribution(mu=1, sigma=0.5)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Lognormal_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.mu, 0.8731320121429662, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 0.42136785757105316, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 49.68970223379, rtol=rtol, atol=atol)

    LS = Fit_Lognormal_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.mu, 0.8555959707789712, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 0.43075117372957444, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 49.74057946473653, rtol=rtol, atol=atol)


def test_Fit_Lognormal_3P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Lognormal_Distribution(mu=1, sigma=0.5, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Lognormal_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.mu, 0.45594171221474994, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 0.7585570251643191, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 500.75432966733763, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 51.29215447929426, rtol=rtol, atol=atol)

    LS = Fit_Lognormal_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.mu, 0.45594171221474994, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 0.7585570251643191, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 500.75432966733763, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 51.29215447929426, rtol=rtol, atol=atol)

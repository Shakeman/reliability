import warnings

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Loglogistic_Distribution,
)
from reliability.Fitters import (
    Fit_Loglogistic_2P,
    Fit_Loglogistic_3P,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
rtol = 1e-3


def test_Fit_Loglogistic_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Loglogistic_Distribution(alpha=50, beta=8)
    rawdata = dist.random_samples(200, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Loglogistic_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha, 49.79833124820633, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 7.816504949027389, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 984.6062308361478, rtol=rtol, atol=atol)

    LS = Fit_Loglogistic_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 50.7411720102561, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 6.749365426499312, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 987.9588689736962, rtol=rtol, atol=atol)


def test_Fit_Loglogistic_3P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Loglogistic_Distribution(alpha=50, beta=8, gamma=500)
    rawdata = dist.random_samples(200, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Loglogistic_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha, 549.0484413959788, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 99.83877818228838, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0.6655499876777405, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 983.188881240264, rtol=rtol, atol=atol)

    LS = Fit_Loglogistic_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 549.7140322741434, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 99.96019733772289, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 983.1888117425344, rtol=rtol, atol=atol)

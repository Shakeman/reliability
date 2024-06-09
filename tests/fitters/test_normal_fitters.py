import warnings

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Normal_Distribution,
)
from reliability.Fitters import (
    Fit_Normal_2P,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
rtol = 1e-3


def test_Fit_Normal_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Normal_Distribution(mu=50, sigma=8)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    MLE = Fit_Normal_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.mu, 49.01641649924297, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 6.653242350482225, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 91.15205546551952, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 92.43763765968633, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -43.223086556289175, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 63.64069171746617, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_mu_sigma, 1.0395705891908218, rtol=rtol, atol=atol)

    LS = Fit_Normal_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.mu, 48.90984235374872, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 6.990098677785364, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 91.21601631804141, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 92.50159851220822, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -43.25506698255012, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 63.657853523044515, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_mu_sigma, 1.0973540350799618, rtol=rtol, atol=atol)

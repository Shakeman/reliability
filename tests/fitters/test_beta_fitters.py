import warnings

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Beta_Distribution,
)
from reliability.Fitters import (
    Fit_Beta_2P,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
rtol = 1e-3


def test_Fit_Beta_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Beta_Distribution(alpha=5, beta=4)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Beta_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha, 6.822122125676898, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 5.387491122895962, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 9.955947979821396, rtol=rtol, atol=atol)

    LS = Fit_Beta_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 12.488183674295271, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 11.955159940824455, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 13.245846240703889, rtol=rtol, atol=atol)

import warnings

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Gumbel_Distribution,
)
from reliability.Fitters import (
    Fit_Gumbel_2P,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
rtol = 1e-3


def test_Fit_Gumbel_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Gumbel_Distribution(mu=50, sigma=8)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Gumbel_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.mu, 49.99907074514885, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 8.701414612335977, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 91.44499349436482, rtol=rtol, atol=atol)

    LS = Fit_Gumbel_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.mu, 49.123499991427174, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 8.654895777687841, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 91.53396436597629, rtol=rtol, atol=atol)

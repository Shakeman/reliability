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
    assert_allclose(MLE.mu, 47.97813932110471, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 5.487155810067562, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 83.17550426530995, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 84.46108645947676, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -39.23481095618439, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 76.43706903015115, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_mu_sigma, 1.8549915988421086, rtol=rtol, atol=atol)

    LS = Fit_Gumbel_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.mu, 46.43212585298994, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 4.827795060868229, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 83.88382894786476, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 85.16941114203156, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -39.58897329746179, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 76.44737853267988, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_mu_sigma, 0.32622575178078633, rtol=rtol, atol=atol)

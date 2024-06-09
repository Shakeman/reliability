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
    assert_allclose(MLE.mu, 0.9494190246173423, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 0.4267323457212804, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 49.69392320890687, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 50.979505403073674, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -22.494020427982846, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 46.91678130009629, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_mu_sigma, 0.002505454567167978, rtol=rtol, atol=atol)

    LS = Fit_Lognormal_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.mu, 0.9427890879489974, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 0.4475312141445822, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 49.757609068995194, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 51.043191263162, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -22.52586335802701, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 46.93509652892565, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_mu_sigma, 0.0025640250120794526, rtol=rtol, atol=atol)


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
    assert_allclose(MLE.mu, 0.5608879850309877, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 0.7396271168422542, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 500.79568888668746, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 52.067948767151364, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 53.555145587813335, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -22.283974383575682, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 46.95299490218758, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_mu_sigma, 0.007500058692172027, rtol=rtol, atol=atol)

    LS = Fit_Lognormal_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.mu, 0.976088004545536, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 0.4340076639560259, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 499.9229609896007, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 52.60637160294965, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 54.09356842361162, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -22.553185801474825, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 46.93164376455629, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_mu_sigma, 0.0025619981036203664, rtol=rtol, atol=atol)

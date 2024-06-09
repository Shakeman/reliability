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
    assert_allclose(MLE.alpha, 7.429048118107467, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 6.519338516778177, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 4.947836247236739, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 6.233418441403544, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -0.12097694714778129, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 63.64510718930826, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_alpha_beta, 9.993273704064205, rtol=rtol, atol=atol)

    LS = Fit_Beta_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 6.699688942917093, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 5.9477941734033575, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 5.02116420233583, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 6.306746396502635, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -0.1576409246973265, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 63.661784208694066, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_alpha_beta, 8.194012965628652, rtol=rtol, atol=atol)

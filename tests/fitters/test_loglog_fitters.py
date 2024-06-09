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
    assert_allclose(MLE.alpha, 50.25178370302894, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 7.869851191923439, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 941.9461734708389, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 948.4818944983512, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -468.94262988262756, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 582.5464625675626, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_alpha_beta, -0.14731273967044273, rtol=rtol, atol=atol)

    LS = Fit_Loglogistic_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 50.657493341191135, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 7.389285094946194, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 942.5623765547977, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 949.09809758231, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -469.25073142460695, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 582.5637861880587, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_alpha_beta, -0.1828511494829605, rtol=rtol, atol=atol)


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
    assert_allclose(MLE.alpha, 62.33043394760128, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 10.10583310083407, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 487.89067581835127, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 943.8128236583473, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 953.5853267783996, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -468.84518733937773, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 582.5424432369575, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_alpha_beta, -0.18172619136971793, rtol=rtol, atol=atol)

    LS = Fit_Loglogistic_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 62.35642581139214, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 10.033527117974446, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 487.90705692706877, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 943.8204938114803, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 953.5929969315326, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -468.84902241594426, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 582.5422083349323, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_alpha_beta, -0.18647186488112505, rtol=rtol, atol=atol)

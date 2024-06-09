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
    assert_allclose(MLE.alpha, 30.895317427895733, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 2.5300452519936405, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 154.33194705093553, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 155.61752924510233, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -74.81303234899717, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 38.004356262808585, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_alpha_beta, -11.610946543514364, rtol=rtol, atol=atol)

    LS = Fit_Gamma_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 25.803340662553182, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 2.8344248030280284, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 154.55898223226797, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 155.84456442643477, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -74.92654993966339, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 38.01670664187149, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_alpha_beta, -5.761109354575602, rtol=rtol, atol=atol)


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
    assert_allclose(MLE.alpha, 161.8637212853173, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 0.5429184966902371, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 515.4451173341464, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 150.0135606540687, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 151.50075747473068, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -71.25678032703435, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 38.63647775048046, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_alpha_beta, -11.302538880460721, rtol=rtol, atol=atol)

    LS = Fit_Gamma_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(LS.alpha, 15.52387782496473, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 6.379102526634475, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 471.0728464561921, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 158.76750225090194, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 160.25469907156392, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -75.63375112545097, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 38.025029148894646, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_alpha_beta, -5.605059720113306, rtol=rtol, atol=atol)

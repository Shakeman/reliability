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
    )
    assert_allclose(MLE.mu, 47.80488376, rtol=rtol, atol=atol)
    assert_allclose(MLE.sigma, 6.560501525359938, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 97.12216494401913, rtol=rtol, atol=atol)

    LS = Fit_Normal_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
    )
    assert_allclose(LS.mu, 47.16714285744403, rtol=rtol, atol=atol)
    assert_allclose(LS.sigma, 6.2953497068956965, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 97.28829409387366, rtol=rtol, atol=atol)

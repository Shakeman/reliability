import warnings

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose

from reliability.Distributions import (
    Exponential_Distribution,
)
from reliability.Fitters import (
    Fit_Exponential_1P,
    Fit_Exponential_2P,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
rtol = 1e-3


def test_Fit_Exponential_1P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Exponential_Distribution(Lambda=5)
    rawdata: npt.NDArray[np.float64] = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Exponential_1P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
    )
    assert_allclose(MLE.Lambda, 6.101198944227536, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, -22.032339191099148, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, -21.25882913976738, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, 12.127280706660684, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 29.59913306667145, rtol=rtol, atol=atol)

    LS = Fit_Exponential_1P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
    )
    assert_allclose(LS.Lambda, 5.776959885774546, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, -21.988412212242917, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, -21.214902160911148, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, 12.10531721723257, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 29.52124203457833, rtol=rtol, atol=atol)


def test_Fit_Exponential_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Exponential_Distribution(Lambda=5, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Exponential_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
    )
    assert_allclose(MLE.Lambda, 7.062867654421206, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 500.016737532126, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, -23.939665128347745, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, -22.65408293418094, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, 14.322773740644461, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 29.413655089419287, rtol=rtol, atol=atol)

    LS = Fit_Exponential_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
    )
    assert_allclose(LS.Lambda, 6.4445633542175, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 500.01368943066706, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, -23.031777273560103, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, -21.7461950793933, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, 13.86882981325064, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 29.33840933641424, rtol=rtol, atol=atol)

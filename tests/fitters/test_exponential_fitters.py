import warnings
from typing import TYPE_CHECKING

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Exponential_Distribution,
)
from reliability.Fitters import (
    Fit_Exponential_1P,
    Fit_Exponential_2P,
)
from reliability.Other_functions import make_right_censored_data

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

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
    MLE.print_results()
    MLE.plot()
    assert_allclose(MLE.Lambda, 4.840717199880608, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, -11.627214606819548, rtol=rtol, atol=atol)

    LS = Fit_Exponential_1P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
    )
    assert_allclose(LS.Lambda, 5.392088065738381, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, -11.482416561720566, rtol=rtol, atol=atol)


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
    MLE.print_results()
    MLE.plot()
    assert_allclose(MLE.Lambda, 5.08394828467135, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 500.00593053745575, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, -10.320253751764174, rtol=rtol, atol=atol)

    LS = Fit_Exponential_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
    )
    assert_allclose(LS.Lambda, 4.8653099226517025, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 499.9932589011877, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, -7.831344030677088, rtol=rtol, atol=atol)

import warnings
from typing import TYPE_CHECKING

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Beta_Distribution,
)
from reliability.Fitters import (
    Fit_Everything,
)
from reliability.Other_functions import make_right_censored_data

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
atol_big = 0  # 0 means it will not look at the absolute difference
rtol = 1e-3
rtol_big = 0.1  # 10% variation


def test_Fit_Everything():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Beta_Distribution(alpha=5, beta=4)
    rawdata: npt.NDArray[np.float64] = dist.random_samples(200, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    MLE = Fit_Everything(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
    )
    MLE.print_results()
    MLE.histogram_plot()
    MLE.p_p_plot()
    MLE.probability_plot()

    LS = Fit_Everything(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
    )
    LS.print_results()

    assert_allclose(
        MLE.best_distribution.alpha,
        0.08027422,
        rtol=rtol,
        atol=atol,
    )  # best fit here is a Beta distribution
    assert_allclose(MLE.best_distribution.beta, 7.331150443368852, rtol=rtol, atol=atol)

    assert_allclose(MLE.Weibull_2P_AICc, 73.31877158451951, rtol=rtol, atol=atol)

    assert_allclose(MLE._Fit_Everything__Weibull_3P_params.AICc, 73.63442576749442, rtol=rtol, atol=atol)

    assert_allclose(MLE.Gamma_2P_AICc, 72.12722744848621, rtol=rtol, atol=atol)

    assert_allclose(MLE.Gamma_3P_AICc, 74.18876272249429, rtol=rtol, atol=atol)

    assert_allclose(MLE.Loglogistic_2P_AICc, 72.40385182784487, rtol=rtol, atol=atol)

    assert_allclose(MLE.Loglogistic_3P_AICc, 74.19180237463885, rtol=rtol, atol=atol)

    assert_allclose(MLE.Lognormal_2P_AICc, 73.24509706417747, rtol=rtol, atol=atol)

    assert_allclose(MLE.Lognormal_3P_AICc, 75.30663233818555, rtol=rtol, atol=atol)

    assert_allclose(MLE.Normal_2P_AICc, 75.06409770011372, rtol=rtol, atol=atol)

    assert_allclose(MLE.Gumbel_2P_AICc, 84.62993257089313, rtol=rtol, atol=atol)

    assert_allclose(MLE.Beta_2P_AICc, 72.6561082208045, rtol=rtol, atol=atol)

    assert_allclose(MLE.Exponential_2P_AICc, 120.4878182745115, rtol=rtol, atol=atol)

    assert_allclose(MLE.Exponential_1P_AICc, 196.9534118830507, rtol=rtol, atol=atol)

    assert_allclose(
        LS.best_distribution.mu,
        -2.50996611,
        rtol=rtol,
        atol=atol,
    )  # best fit here is a Gamma 2P distribution
    assert_allclose(LS.best_distribution.beta, 7.27079628330181, rtol=rtol, atol=atol)

    assert_allclose(LS.Beta_2P_AICc, 72.8542452513679, rtol=rtol, atol=atol)

    assert_allclose(LS.Exponential_1P_AICc, 198.36545944087035, rtol=rtol, atol=atol)

    assert_allclose(LS.Exponential_2P_AICc, 125.6586727637188, rtol=rtol, atol=atol)

    assert_allclose(LS.Gamma_2P_AICc, 72.13986225107138, rtol=rtol, atol=atol)

    assert_allclose(LS.Gamma_3P_AICc, 74.18876272249429, rtol=rtol, atol=atol)

    assert_allclose(LS.Gumbel_2P_AICc, 95.73437648849374, rtol=rtol, atol=atol)

    assert_allclose(LS.Loglogistic_2P_AICc, 72.41566243584722, rtol=rtol, atol=atol)

    assert_allclose(LS.Loglogistic_3P_AICc, 74.34178321439887, rtol=rtol, atol=atol)

    assert_allclose(LS.Lognormal_2P_AICc, 73.958018643704, rtol=rtol, atol=atol)

    assert_allclose(LS.Lognormal_3P_AICc, 75.30663233818555, rtol=rtol, atol=atol)

    assert_allclose(LS.Normal_2P_AICc, 75.35221899116522, rtol=rtol, atol=atol)

    assert_allclose(LS.Weibull_2P_AICc, 74.07799768375571, rtol=rtol, atol=atol)

    assert_allclose(LS._Fit_Everything__Weibull_3P_params.AICc, 73.82580699484612, rtol=rtol, atol=atol)

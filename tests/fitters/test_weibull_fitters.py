import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from reliability.Datasets import defective_sample, electronics
from reliability.Distributions import Competing_Risks_Model, DSZI_Model, Mixture_Model, Weibull_Distribution
from reliability.Fitters import (
    Fit_Weibull_2P,
    Fit_Weibull_2P_grouped,
    Fit_Weibull_3P,
    Fit_Weibull_CR,
    Fit_Weibull_DS,
    Fit_Weibull_DSZI,
    Fit_Weibull_Mixture,
    Fit_Weibull_ZI,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
atol_big = 0  # 0 means it will not look at the absolute difference
rtol = 1e-3
rtol_big = 0.1  # 10% variation


def test_Fit_Weibull_2P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Weibull_Distribution(alpha=50, beta=2)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Weibull_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha, 47.0217247918312, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 1.982318106214522, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 120.41667818523239, rtol=rtol, atol=atol)

    MLE_beta = Fit_Weibull_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=False,
        print_results=False,
        force_beta=0.7,
        quantiles=True,
    )
    assert_allclose(MLE_beta.alpha, 69.06525502566299, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.beta, 0.7, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.AICc, 128.85522919807656, rtol=rtol, atol=atol)

    LS = Fit_Weibull_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=False,
        print_results=False,
    )
    assert_allclose(LS.alpha, 47.43515443864693, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 1.8458742548969898, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 120.4881112461033, rtol=rtol, atol=atol)


def test_Fit_Weibull_3P():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Weibull_Distribution(alpha=50, beta=2, gamma=500)
    rawdata = dist.random_samples(20, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)

    MLE = Fit_Weibull_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha, 53.47204088338224, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 0.5155304947981104, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 510.7623510663467, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 112.03506795155228, rtol=rtol, atol=atol)

    LS = Fit_Weibull_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=False,
        print_results=False,
    )
    assert_allclose(LS.alpha, 545.2235196821472, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 39.21654896663356, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 125.3766843880421, rtol=rtol, atol=atol)


def test_Fit_Weibull_Mixture():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    d1 = Weibull_Distribution(alpha=10, beta=3)
    d2 = Weibull_Distribution(alpha=40, beta=4)
    dist = Mixture_Model(distributions=[d1, d2], proportions=[0.2, 0.8])
    raw_data = dist.random_samples(100, seed=2)
    data = make_right_censored_data(data=raw_data, threshold=dist.mean)

    MLE = Fit_Weibull_Mixture(
        failures=data.failures,
        right_censored=data.right_censored,
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha_1, 10.62279389, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_1, 2.752678473152489, rtol=rtol, atol=atol)
    assert_allclose(MLE.alpha_2, 37.98235528150885, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_2, 6.104783916543841, rtol=rtol, atol=atol)
    assert_allclose(MLE.proportion_1, 0.2614784966586089, rtol=rtol, atol=atol)
    assert_allclose(MLE.proportion_2, 0.7385215033413911, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 433.0648387794518, rtol=rtol, atol=atol)


def test_Fit_Weibull_CR():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    d1 = Weibull_Distribution(alpha=50, beta=2)
    d2 = Weibull_Distribution(alpha=40, beta=10)
    CR_model = Competing_Risks_Model(distributions=[d1, d2])
    raw_data = CR_model.random_samples(100, seed=2)
    data = make_right_censored_data(data=raw_data, threshold=40)
    MLE = Fit_Weibull_CR(
        failures=data.failures,
        right_censored=data.right_censored,
        show_probability_plot=True,
        print_results=True,
    )
    assert_allclose(MLE.alpha_1, 52.38879644, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_1, 1.775887008836593, rtol=rtol, atol=atol)
    assert_allclose(MLE.alpha_2, 39.44857208719343, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_2, 9.740661609822869, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 656.9054812126976, rtol=rtol, atol=atol)


def test_fit_weibull_dzi_init():
    model = DSZI_Model(distribution=Weibull_Distribution(alpha=1200, beta=3), DS=0.7, ZI=0.2)
    failures, right_censored = model.random_samples(100, seed=5, right_censored_time=3000)
    fit = Fit_Weibull_DSZI(
        failures=failures,
        right_censored=right_censored,
        show_probability_plot=True,
        print_results=True,
    )
    assert isinstance(fit, Fit_Weibull_DSZI)
    assert fit.alpha > 0
    assert fit.beta > 0
    assert fit.DS > 0
    assert fit.ZI > 0


def test_fit_weibull_dzi_failures():
    with pytest.raises(ValueError):
        Fit_Weibull_DSZI(failures=[], right_censored=[60, 70, 80], show_probability_plot=False, print_results=False)


def test_Fit_Weibull_ZI_init():
    data = Weibull_Distribution(alpha=200, beta=5).random_samples(70, seed=1)
    zeros = np.zeros(30)
    failures = np.hstack([zeros, data])
    fit = Fit_Weibull_ZI(failures=failures)
    assert_allclose(actual=fit.alpha, desired=203.686397752733, rtol=rtol, atol=atol)
    assert_allclose(fit.beta, 5.472841068783268, rtol=rtol, atol=atol)
    assert_allclose(fit.AICc, 842.3158548968776, rtol=rtol, atol=atol)
    # Test case with invalid input
    with pytest.raises(ValueError):
        Fit_Weibull_ZI(failures=[], right_censored=[4, 5])


def test_Fit_Weibull_DS_init():
    failures = defective_sample().failures
    right_censored = defective_sample().right_censored
    fit_DS = Fit_Weibull_DS(
        failures=failures,
        right_censored=right_censored,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        downsample_scatterplot=True,
    )
    assert_allclose(actual=fit_DS.alpha, desired=170.983, rtol=rtol, atol=atol)
    assert_allclose(fit_DS.beta, 1.30109, rtol=rtol, atol=atol)
    assert_allclose(fit_DS.AICc, 23961.3, rtol=rtol, atol=atol)
    # Test case with invalid input
    with pytest.raises(ValueError):
        Fit_Weibull_DS(failures=[], right_censored=[4, 5])


def test_Fit_Weibull_2P_Grouped():
    fit = Fit_Weibull_2P_grouped(dataframe=electronics().dataframe)
    assert_allclose(actual=fit.alpha, desired=6.191922756866386e21, rtol=rtol, atol=atol)
    assert_allclose(fit.beta, 0.15374388796935232, rtol=rtol, atol=atol)
    assert_allclose(fit.AICc, 293.2364591402575, rtol=rtol, atol=atol)
    # Test case with invalid input
    with pytest.raises(ValueError):
        Fit_Weibull_2P_grouped(None)

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from reliability.Datasets import defective_sample
from reliability.Distributions import Competing_Risks_Model, DSZI_Model, Mixture_Model, Weibull_Distribution
from reliability.Fitters import (
    Fit_Weibull_2P,
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
    assert_allclose(MLE.alpha, 45.099010886086354, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 2.7827531773597984, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 115.66971887883678, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 116.95530107300358, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -55.4819182629478, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 55.60004028891652, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_alpha_beta, -0.9178064889295378, rtol=rtol, atol=atol)

    MLE_beta = Fit_Weibull_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=False,
        print_results=False,
        force_beta=0.7,
        quantiles=True,
    )
    assert_allclose(MLE_beta.alpha, 72.46689970922911, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.beta, 0.7, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.AICc, 130.7785604380564, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.BIC, 131.55207048938814, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.loglik, -64.27816910791708, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.AD, 56.535966829249546, rtol=rtol, atol=atol)
    assert_allclose(MLE_beta.Cov_alpha_beta, 0, rtol=rtol, atol=atol)

    LS = Fit_Weibull_2P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=False,
        print_results=False,
    )
    assert_allclose(LS.alpha, 42.91333312142757, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 2.9657153686461033, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 115.93668384456019, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 117.222266038727, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -55.61540074580951, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 55.62807482958476, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_alpha_beta, -0.1119680481788733, rtol=rtol, atol=atol)


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
    assert_allclose(MLE.alpha, 33.0123537701021, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta, 1.327313848890964, rtol=rtol, atol=atol)
    assert_allclose(MLE.gamma, 513.7220829514334, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 116.5327581203968, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 118.01995494105878, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -54.5163790601984, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 55.606805028079016, rtol=rtol, atol=atol)
    assert_allclose(MLE.Cov_alpha_beta, -0.7687781958569139, rtol=rtol, atol=atol)

    LS = Fit_Weibull_3P(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=False,
        print_results=False,
    )
    assert_allclose(LS.alpha, 32.639290779819824, rtol=rtol, atol=atol)
    assert_allclose(LS.beta, 1.2701961119432184, rtol=rtol, atol=atol)
    assert_allclose(LS.gamma, 514.5065549826453, rtol=rtol, atol=atol)
    assert_allclose(LS.AICc, 119.47369772704523, rtol=rtol, atol=atol)
    assert_allclose(LS.BIC, 120.96089454770721, rtol=rtol, atol=atol)
    assert_allclose(LS.loglik, -55.98684886352262, rtol=rtol, atol=atol)
    assert_allclose(LS.AD, 55.70853682331155, rtol=rtol, atol=atol)
    assert_allclose(LS.Cov_alpha_beta, -0.8435523816679948, rtol=rtol, atol=atol)


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
    assert_allclose(MLE.alpha_1, 11.06604639424718, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_1, 2.735078296796997, rtol=rtol, atol=atol)
    assert_allclose(MLE.alpha_2, 34.325433665495346, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_2, 7.60238532821206, rtol=rtol, atol=atol)
    assert_allclose(MLE.proportion_1, 0.23640116719132157, rtol=rtol, atol=atol)
    assert_allclose(MLE.proportion_2, 0.7635988328086785, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 471.97390405380236, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 484.3614571114024, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -230.66780309073096, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 320.1963544647712, rtol=rtol, atol=atol)


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
    assert_allclose(MLE.alpha_1, 53.05674752263902, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_1, 1.9411091317375062, rtol=rtol, atol=atol)
    assert_allclose(MLE.alpha_2, 38.026383998212154, rtol=rtol, atol=atol)
    assert_allclose(MLE.beta_2, 9.033349805988692, rtol=rtol, atol=atol)
    assert_allclose(MLE.AICc, 665.5311523940719, rtol=rtol, atol=atol)
    assert_allclose(MLE.BIC, 675.5307805064454, rtol=rtol, atol=atol)
    assert_allclose(MLE.loglik, -328.5550498812465, rtol=rtol, atol=atol)
    assert_allclose(MLE.AD, 34.0918038201449, rtol=rtol, atol=atol)


def test_fit_weibull_dzi_init():
    model = DSZI_Model(distribution=Weibull_Distribution(alpha=1200, beta=3), DS=0.7, ZI=0.2)
    failures, right_censored = model.random_samples(100, seed=5, right_censored_time=3000)
    fit = Fit_Weibull_DSZI(
        failures=failures, right_censored=right_censored, show_probability_plot=True, print_results=True
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
    assert_allclose(actual=fit.alpha, desired=192.93112941119827, rtol=rtol, atol=atol)
    assert_allclose(fit.beta, 4.531770672312745, rtol=rtol, atol=atol)
    assert_allclose(fit.AICc, 859.2589334451959, rtol=rtol, atol=atol)
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

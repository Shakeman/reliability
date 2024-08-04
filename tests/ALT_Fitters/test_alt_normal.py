import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from reliability.ALT_Fitters import (
    Fit_Normal_Dual_Exponential,
    Fit_Normal_Dual_Power,
    Fit_Normal_Exponential,
    Fit_Normal_Eyring,
    Fit_Normal_Power,
    Fit_Normal_Power_Exponential,
)
from reliability.Other_functions import make_ALT_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 0  # setting this as 0 means it will not look at the absolute tolerance
rtol = 0.01  # 1% variation allowed in relative tolerance for most things


def test_Fit_Normal_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Normal",
        life_stress_model="Exponential",
        a=500,
        b=1000,
        sigma=500,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Normal_Exponential(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
    )
    plt.close("all")
    assert_allclose(model.a, 510.32806900630544, rtol=rtol, atol=atol)
    assert_allclose(model.b, 973.8223647399388, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 486.1365917592639, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3670.508736811669, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3681.5390031545567, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1832.213827865294, rtol=rtol, atol=atol)


def test_Fit_Normal_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Normal",
        life_stress_model="Eyring",
        a=90,
        c=-14,
        sigma=500,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Normal_Eyring(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
    )
    plt.close("all")
    assert_allclose(model.a, 101.41052353, rtol=rtol, atol=atol)
    assert_allclose(model.c, -13.97387421375034, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 486.12929211552824, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3669.8593109070534, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3680.889577249941, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1831.889114912986, rtol=rtol, atol=atol)


def test_Fit_Normal_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Normal",
        life_stress_model="Power",
        a=6e6,
        n=-1.2,
        sigma=500,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Normal_Power(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
    )
    plt.close("all")
    assert_allclose(model.a, 6889372.65785002, rtol=rtol, atol=atol)
    assert_allclose(
        model.n,
        -1.224764224452152,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(
        model.sigma,
        479.4939976887211,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.AICc, 3670.0660652180977, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3681.096331560985, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1831.9924920685082, rtol=rtol, atol=atol)


def test_Fit_Normal_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    c = 1e15
    m = -4
    n = -2
    data = make_ALT_data(
        distribution="Normal",
        life_stress_model="Dual_Power",
        c=1e15,
        m=-4,
        n=-2,
        sigma=300,
        stress_1=[500, 400, 350, 300, 200, 180, 390, 250, 540],
        stress_2=[0.9, 0.8, 0.7, 0.6, 0.3, 0.3, 0.2, 0.7, 0.5],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Normal_Dual_Power(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=np.array([100, 0.2]),
    )
    plt.close("all")
    assert_allclose(model.c, c, rtol=rtol, atol=atol)
    assert_allclose(model.n, n, rtol=rtol, atol=atol)
    assert_allclose(model.m, m, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 278.58120575, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6494.966403925203, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6514.13129024107, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3243.460855593886, rtol=rtol, atol=atol)


def test_Fit_Normal_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Normal",
        life_stress_model="Dual_Exponential",
        a=60,
        b=0.1,
        c=5000,
        sigma=300,
        stress_1=[500, 400, 350, 300, 200, 180, 390, 250, 540],
        stress_2=[0.9, 0.8, 0.7, 0.6, 0.3, 0.3, 0.2, 0.7, 0.5],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Normal_Dual_Exponential(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=np.array([100, 0.2]),
    )
    plt.close("all")
    assert_allclose(model.a, 59.71344103606326, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.10065937394073277, rtol=rtol, atol=atol)
    assert_allclose(model.c, 5006.556618243661, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 281.9484045101182, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6394.529082362259, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6413.693968678126, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3193.2421948124143, rtol=rtol, atol=atol)


def test_Fit_Normal_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Normal",
        life_stress_model="Power_Exponential",
        a=70,
        c=2500,
        n=-0.25,
        sigma=100,
        stress_1=[500, 400, 350, 420, 245],
        stress_2=[12, 8, 6, 9, 10],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Normal_Power_Exponential(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=np.array([200, 5]),
    )
    plt.close("all")
    assert_allclose(model.a, 70.61416267, rtol=rtol, atol=atol)
    assert_allclose(model.c, 2498.268588097067, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.24817878201877347, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 94.48228285331875, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3015.3322425662163, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3032.1098668790974, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1503.6257172427042, rtol=rtol, atol=atol)

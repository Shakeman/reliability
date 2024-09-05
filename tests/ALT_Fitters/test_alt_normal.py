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
    assert_allclose(model.a, 471.5886888220982, rtol=rtol, atol=atol)
    assert_allclose(model.b, 1074.8553859511792, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 475.2491894375077, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3659.049006187109, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 63.482008886395526, rtol=rtol, atol=atol)
    assert_allclose(model.c, -14.067233086592458, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 476.1953014933615, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3658.3450263993263, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 5102112.669522883, rtol=rtol, atol=atol)
    assert_allclose(
        model.n,
        -1.1744095201869316,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(
        model.sigma,
        473.72057660881865,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.AICc, 3658.354925308246, rtol=rtol, atol=atol)


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
    assert_allclose(model.sigma, 297.4973325818998, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 10394.338276272189, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 61.80377588697216, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.09909070001811665, rtol=rtol, atol=atol)
    assert_allclose(model.c, 4971.944882781889, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 297.76993218383944, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6443.85659885724, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 74.67460177266736, rtol=rtol, atol=atol)
    assert_allclose(model.c, 2465.7741191592963, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.2501604387432519, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 99.32315101818536, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3043.220659587977, rtol=rtol, atol=atol)

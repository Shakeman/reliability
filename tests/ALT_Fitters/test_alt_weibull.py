import warnings

import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from reliability.ALT_Fitters import (
    Fit_Weibull_Dual_Exponential,
    Fit_Weibull_Dual_Power,
    Fit_Weibull_Exponential,
    Fit_Weibull_Eyring,
    Fit_Weibull_Power,
    Fit_Weibull_Power_Exponential,
)
from reliability.Other_functions import make_ALT_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 0  # setting this as 0 means it will not look at the absolute tolerance
rtol = 0.01  # 1% variation allowed in relative tolerance for most things


def test_Fit_Weibull_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Weibull",
        life_stress_model="Exponential",
        a=2000,
        b=10,
        beta=2.5,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Weibull_Exponential(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
    )
    plt.close("all")
    assert_allclose(model.a, 2153.752541872851, rtol=rtol, atol=atol)
    assert_allclose(model.b, 7.0806871632850354, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.6084287722032555, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3693.8395259640993, rtol=rtol, atol=atol)


def test_Fit_Weibull_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Weibull",
        life_stress_model="Eyring",
        a=1500,
        c=-10,
        beta=1,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Weibull_Eyring(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
    )
    plt.close("all")
    assert_allclose(model.a, 2063.9299881256084, rtol=rtol, atol=atol)
    assert_allclose(model.c, -8.746906466816322, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 1.0416258330443826, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4217.808596358866, rtol=rtol, atol=atol)


def test_Fit_Weibull_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Weibull",
        life_stress_model="Power",
        a=5e15,
        n=-4,
        beta=2.5,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Weibull_Power(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
    )
    plt.close("all")
    assert_allclose(model.a, 1.123013751762961e17, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.n, -4.511429406626488, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.beta, 2.5998128091856803, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6024.964778108171, rtol=rtol, atol=atol)


def test_Fit_Weibull_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Weibull",
        life_stress_model="Dual_Exponential",
        a=50,
        b=0.1,
        c=500,
        beta=2.5,
        stress_1=[500, 400, 350, 300, 200, 180, 390, 250, 540],
        stress_2=[0.9, 0.8, 0.7, 0.6, 0.3, 0.3, 0.2, 0.7, 0.5],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Weibull_Dual_Exponential(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=[100, 0.2],
    )
    plt.close("all")
    assert_allclose(model.a, 62.215172831404445, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.08066684667618673, rtol=rtol, atol=atol)
    assert_allclose(model.c, 564.1123255104022, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.7627192146927757, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6564.886720768444, rtol=rtol, atol=atol)


def test_Fit_Weibull_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Weibull",
        life_stress_model="Dual_Power",
        c=1e15,
        m=-4,
        n=-2,
        beta=2.5,
        stress_1=[500, 400, 350, 420, 245],
        stress_2=[12, 8, 6, 9, 10],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Weibull_Dual_Power(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=[250, 7],
    )
    plt.close("all")
    assert_allclose(model.c, 3674526569650209.5, rtol=rtol, atol=atol)
    assert_allclose(model.m, -4.12561554753957, rtol=rtol, atol=atol)
    assert_allclose(model.n, -2.197453348501671, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.875946232407607, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3590.722079245432, rtol=rtol, atol=atol)


def test_Fit_Weibull_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Weibull",
        life_stress_model="Power_Exponential",
        a=22,
        c=400,
        n=-0.25,
        beta=2.5,
        stress_1=[500, 400, 350, 420, 245],
        stress_2=[12, 8, 6, 9, 10],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Weibull_Power_Exponential(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=[200, 5],
    )
    plt.close("all")
    assert_allclose(model.a, 87.65242092877672, rtol=rtol, atol=atol)
    assert_allclose(model.c, 385.7441374514881, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.2582500455023094, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.8546609205024525, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3094.0022423730807, rtol=rtol, atol=atol)

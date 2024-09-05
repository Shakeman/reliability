import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from reliability.ALT_Fitters import (
    Fit_Exponential_Dual_Exponential,
    Fit_Exponential_Dual_Power,
    Fit_Exponential_Exponential,
    Fit_Exponential_Eyring,
    Fit_Exponential_Power,
    Fit_Exponential_Power_Exponential,
)
from reliability.Other_functions import make_ALT_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 0  # setting this as 0 means it will not look at the absolute tolerance
rtol = 0.01  # 1% variation allowed in relative tolerance for most things


def test_Fit_Exponential_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Exponential",
        life_stress_model="Exponential",
        a=2000,
        b=10,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Exponential_Exponential(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.a, 2183.9542738377386, rtol=rtol, atol=atol)
    assert_allclose(model.b, 6.71607189415336, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3969.030181703984, rtol=rtol, atol=atol)


def test_Fit_Exponential_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Exponential",
        life_stress_model="Eyring",
        a=1500,
        c=-10,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Exponential_Eyring(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.a, 1648.5495576916214, rtol=rtol, atol=atol)
    assert_allclose(model.c, -9.688671487388788, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4184.940125162059, rtol=rtol, atol=atol)


def test_Fit_Exponential_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Exponential",
        life_stress_model="Power",
        a=5e15,
        n=-4,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Exponential_Power(
        failures=data.failures,
        failure_stress=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress=data.right_censored_stresses,
        use_level_stress=300,
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.a, 4.372374521948593e16, rtol=rtol, atol=atol)
    assert_allclose(model.n, -4.351340863695483, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6299.630714623121, rtol=rtol, atol=atol)


def test_Fit_Exponential_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Exponential",
        life_stress_model="Dual_Exponential",
        a=50,
        b=0.2,
        c=500,
        stress_1=[500, 400, 350, 300, 200, 180, 390, 250, 540],
        stress_2=[0.9, 0.8, 0.7, 0.6, 0.3, 0.3, 0.2, 0.7, 0.5],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Exponential_Dual_Exponential(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=np.array([100, 0.2]),
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.a, 165.836346180442, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.1956918991810676, rtol=rtol, atol=atol)
    assert_allclose(model.c, 383.1651005769499, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 11466.452786105183, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 11480.833184680869, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -5730.213000195448, rtol=rtol, atol=atol)


def test_Fit_Exponential_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Exponential",
        life_stress_model="Dual_Power",
        c=1e15,
        m=-4,
        n=-2,
        stress_1=[500, 400, 350, 420, 245],
        stress_2=[12, 8, 6, 9, 10],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Exponential_Dual_Power(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=np.array([100, 0.2]),
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.c, 3615679816389494.5, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.m, -4.16050484405005, rtol=rtol, atol=atol)
    assert_allclose(model.n, -2.1141362526119827, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6084.88808416752, rtol=rtol, atol=atol)


def test_Fit_Exponential_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Exponential",
        life_stress_model="Power_Exponential",
        a=50,
        c=10000,
        n=-1.5,
        stress_1=[500, 400, 350, 420, 245],
        stress_2=[12, 8, 6, 9, 10],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Exponential_Power_Exponential(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=np.array([200, 5]),
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.a, 142.64430884632105, rtol=rtol, atol=atol)
    assert_allclose(model.c, 5458.049304201528, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.3010951396698511, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 5751.192602293334, rtol=rtol, atol=atol)

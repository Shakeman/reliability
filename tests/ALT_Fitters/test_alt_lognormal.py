import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from reliability.ALT_Fitters import (
    Fit_Lognormal_Dual_Exponential,
    Fit_Lognormal_Dual_Power,
    Fit_Lognormal_Exponential,
    Fit_Lognormal_Eyring,
    Fit_Lognormal_Power,
    Fit_Lognormal_Power_Exponential,
)
from reliability.Other_functions import make_ALT_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 0  # setting this as 0 means it will not look at the absolute tolerance
rtol = 0.01  # 1% variation allowed in relative tolerance for most things


def test_Fit_Lognormal_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Lognormal",
        life_stress_model="Exponential",
        a=2000,
        b=10,
        sigma=0.5,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Lognormal_Exponential(
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
    assert_allclose(model.a, 2033.15392013689, rtol=rtol, atol=atol)
    assert_allclose(model.b, 9.380682049168223, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4901664124825419, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3824.906495666694, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3835.9367620095813, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1909.4127072928063, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Eyring():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Lognormal",
        life_stress_model="Eyring",
        a=1500,
        c=-10,
        sigma=0.5,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Lognormal_Eyring(
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
    assert_allclose(model.a, 1533.14724753, rtol=rtol, atol=atol)
    assert_allclose(model.c, -9.98431861625691, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.49016660648069477, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4040.853294458127, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4051.8835608010145, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2017.386106688523, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Lognormal",
        life_stress_model="Power",
        a=5e15,
        n=-4,
        sigma=0.5,
        stress_1=[500, 400, 350],
        number_of_samples=100,
        fraction_censored=0.2,
        seed=1,
    )
    model = Fit_Lognormal_Power(
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
    assert_allclose(
        model.a,
        6986752555048811.0,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.n, -4.052657513810853, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.sigma, 0.49023468735812314, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6155.617784732405, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6166.648051075293, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3074.768351825662, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Dual_Power():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Lognormal",
        life_stress_model="Dual_Power",
        c=1e15,
        m=-4,
        n=-2,
        sigma=0.5,
        stress_1=[500, 400, 350, 420, 245],
        stress_2=[12, 8, 6, 9, 10],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Lognormal_Dual_Power(
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
    assert_allclose(model.c, 1041948435048137.1, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.m, -3.9879794497580847, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(
        model.n,
        -1.9979114185818097,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.sigma, 0.4910883211257458, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3727.3363565359323, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3744.0959378437265, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1859.6187527250188, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Power_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Lognormal",
        life_stress_model="Power_Exponential",
        a=200,
        c=400,
        n=-0.5,
        sigma=0.5,
        stress_1=[500, 400, 350, 420, 245],
        stress_2=[12, 8, 6, 9, 10],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Lognormal_Power_Exponential(
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
    assert_allclose(model.a, 203.64166629, rtol=rtol, atol=atol)
    assert_allclose(model.c, 391.30870096, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.44133854, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4910516937147718, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3200.6626325485195, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3217.4402568614005, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1596.2909122338558, rtol=rtol, atol=atol)


def test_Fit_Lognormal_Dual_Exponential():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    data = make_ALT_data(
        distribution="Lognormal",
        life_stress_model="Dual_Exponential",
        a=50,
        b=0.1,
        c=500,
        sigma=0.5,
        stress_1=[500, 400, 350, 300, 200, 180, 390, 250, 540],
        stress_2=[0.9, 0.8, 0.7, 0.6, 0.3, 0.3, 0.2, 0.7, 0.5],
        number_of_samples=100,
        fraction_censored=0.5,
        seed=1,
    )
    model = Fit_Lognormal_Dual_Exponential(
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
    assert_allclose(model.a, 73.25301902, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.11880547, rtol=rtol, atol=atol)
    assert_allclose(model.c, 499.62393884, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4837723906057129, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6786.541219570215, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6805.706105886082, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3389.248263416392, rtol=rtol, atol=atol)

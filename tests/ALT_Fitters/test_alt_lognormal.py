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
    )
    plt.close("all")
    assert_allclose(model.a, 1908.048447899532, rtol=rtol, atol=atol)
    assert_allclose(model.b, 12.802133247235542, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.4797308001925457, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3817.426209093163, rtol=rtol, atol=atol)


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
    )
    plt.close("all")
    assert_allclose(model.a, 1408.0357162993603, rtol=rtol, atol=atol)
    assert_allclose(model.c, -10.247060908350143, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.47975465111948656, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4033.3730693667667, rtol=rtol, atol=atol)


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
    )
    plt.close("all")
    assert_allclose(
        model.a,
        1386857914593527.0,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.n, -3.7834766874541224, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.sigma, 0.47980567226965753, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6148.1595131243, rtol=rtol, atol=atol)


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
    assert_allclose(model.c, 2025001594073569.8, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.m, -4.10306765883972, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(
        model.n,
        -1.9771667948254903,
        rtol=rtol,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.sigma, 0.5206885479789575, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3770.4702143474196, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 242.68553531905303, rtol=rtol, atol=atol)
    assert_allclose(model.c, 342.6394573810117, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.4170107712496197, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.5206867352489463, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3244.202696675515, rtol=rtol, atol=atol)


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
    )
    plt.close("all")
    assert_allclose(model.a, 77.72107829490336, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.08922384502386554, rtol=rtol, atol=atol)
    assert_allclose(model.c, 531.2400298122011, rtol=rtol, atol=atol)
    assert_allclose(model.sigma, 0.5191985969241459, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6855.920520053332, rtol=rtol, atol=atol)

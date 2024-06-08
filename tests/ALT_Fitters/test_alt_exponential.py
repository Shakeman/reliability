import warnings

import matplotlib.pyplot as plt
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
    assert_allclose(model.a, 1928.4687332654944, rtol=rtol, atol=atol)
    assert_allclose(model.b, 12.96779155174335, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3984.1086002100037, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3991.475761118912, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1990.0340980847998, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 1428.4686331863793, rtol=rtol, atol=atol)
    assert_allclose(model.c, -10.259884009475353, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4200.055398999253, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4207.422559908162, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2098.0074974794247, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 2.11708023e15, rtol=rtol, atol=atol)
    assert_allclose(model.n, -3.831313136385626, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6314.7161417145035, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6322.083302623412, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3155.33786883705, rtol=rtol, atol=atol)


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
        use_level_stress=[100, 0.2],
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.a, 152.843982, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.17461001, rtol=rtol, atol=atol)
    assert_allclose(model.c, 421.42393619, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 11467.269078434578, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 11481.649477010264, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -5730.621146360146, rtol=rtol, atol=atol)


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
        use_level_stress=[100, 0.2],
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.c, 1.37833563e15, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.m, -4.12999326, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.72377458, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6136.997370988174, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6149.592808186666, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3065.4744919457, rtol=rtol, atol=atol)


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
        use_level_stress=[200, 5],
        show_life_stress_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.a, 118.31010472, rtol=rtol, atol=atol)
    assert_allclose(model.c, 3645.02571372, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.05426527, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 5800.788764570015, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 5813.3842017685065, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2897.37018873662, rtol=rtol, atol=atol)

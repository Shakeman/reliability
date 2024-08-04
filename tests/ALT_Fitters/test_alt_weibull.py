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
    assert_allclose(model.a, 1965.7797395338112, rtol=rtol, atol=atol)
    assert_allclose(model.b, 11.0113385296826, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.3990457903278615, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3710.5717742652996, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3721.602040608187, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1852.2453465921092, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 1509.2873266741617, rtol=rtol, atol=atol)
    assert_allclose(model.c, -10.051886457997458, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 0.9696541010302344, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 4201.877589122597, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 4212.907855465484, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -2097.898254020758, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 3.54549542e15, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.n, -3.916925628937264, rtol=rtol, atol=atol)  # larger due to variation in python versions
    assert_allclose(model.beta, 2.399407397407449, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6041.16703767533, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6052.197304018217, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3017.5429782971246, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 109.3385636, rtol=rtol, atol=atol)
    assert_allclose(model.b, 0.09291729, rtol=rtol, atol=atol)
    assert_allclose(model.c, 456.09622917, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.53216375, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 6584.301151215161, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 6603.466037531028, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -3288.128229238865, rtol=rtol, atol=atol)


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
    assert_allclose(model.c, 1.17720289e15, rtol=rtol, atol=atol)
    assert_allclose(model.m, -4.08880812, rtol=rtol, atol=atol)
    assert_allclose(model.n, -1.7890763670187908, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.570622042328891, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3624.015790007103, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3640.793414319984, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1807.9674909631476, rtol=rtol, atol=atol)


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
    assert_allclose(model.a, 54.68972069, rtol=rtol, atol=atol)
    assert_allclose(model.c, 253.34403063, rtol=rtol, atol=atol)
    assert_allclose(model.n, -0.03048628, rtol=rtol, atol=atol)
    assert_allclose(model.beta, 2.571207071526618, rtol=rtol, atol=atol)
    assert_allclose(model.AICc, 3122.643924859823, rtol=rtol, atol=atol)
    assert_allclose(model.BIC, 3139.421549172704, rtol=rtol, atol=atol)
    assert_allclose(model.loglik, -1557.2815583895076, rtol=rtol, atol=atol)

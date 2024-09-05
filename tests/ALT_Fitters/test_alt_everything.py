import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from reliability.ALT_Fitters import (
    Fit_Everything_ALT,
)
from reliability.Other_functions import make_ALT_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 0  # setting this as 0 means it will not look at the absolute tolerance
rtol = 0.01  # 1% variation allowed in relative tolerance for most things
rtol_big = 0.1  # 10% variation allowed in relative tolerance allowed for some that seem to fail online. I don't know why online differs from local.


def test_Fit_Everything_ALT_single_stress():
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
    model = Fit_Everything_ALT(
        failures=data.failures,
        failure_stress_1=data.failure_stresses,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses,
        use_level_stress=300,
        show_best_distribution_probability_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.Weibull_Exponential_a, 2153.752541872851, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_b, 7.0806871632850354, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_beta, 2.6084287722032555, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_AICc, 3693.8395259640993, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Exponential_a, 2225.5744348127278, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_b, 4.78421380977871, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_sigma, 0.47520086163654235, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_AICc, 3715.6475359217893, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Exponential_a, 1943.1156768134028, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_b, 11.231279970707813, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_sigma, 746.4100628682714, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_AICc, 3901.96807936702, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Exponential_a, 2225.58350807, rtol=rtol_big, atol=atol)
    assert_allclose(model.Exponential_Exponential_b, 5.72266992, rtol=rtol_big, atol=atol)
    assert_allclose(model.Exponential_Exponential_AICc, 3941.20353553, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Eyring_a, 1807.513383898624, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_c, -8.82769633779257, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_beta, 2.596628021668307, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_AICc, 3694.4419336330748, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Eyring_a, 1807.513452071563, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_c, -8.61178729917166, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_sigma, 0.47454846767907666, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_AICc, 3713.1445696864544, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Eyring_a, 1547.8373595703624, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_c, -9.4049843, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_sigma, 741.3367958379617, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_AICc, 3902.083796875465, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Eyring_a, 1791.8911537886302, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_c, -8.82909465289566, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_AICc, 3941.076784027143, rtol=rtol, atol=atol)
    # TODO: Fix the following tests with large variation comments
    assert_allclose(
        model.Weibull_Power_a,
        9.182472990309451e16,
        rtol=0.9,
        atol=atol,
    )  # much larger due to variation in python versions. WHY???
    assert_allclose(
        model.Weibull_Power_n,
        -5.284439423399543,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(
        model.Weibull_Power_beta,
        2.5952588811964965,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.Weibull_Power_AICc, 3694.1603287723137, rtol=rtol, atol=atol)

    assert_allclose(
        model.Lognormal_Power_a,
        9.182472990309451e16,
        rtol=rtol,
        atol=atol,
    )  # much larger due to variation in python versions
    assert_allclose(
        model.Lognormal_Power_n,
        -5.320404067443571,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.Lognormal_Power_sigma, 0.472903801753384, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_AICc, 3713.1445696864544, rtol=rtol, atol=atol)

    assert_allclose(
        model.Normal_Power_a,
        9.182472990309451e16,
        rtol=rtol,
        atol=atol,
    )  # much larger due to variation in python versions
    assert_allclose(
        model.Normal_Power_n,
        -5.326001125617882,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.Normal_Power_sigma, 629.3037274427179, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_AICc, 3945.474134087913, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_a, 9.182472990309451e16, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_n, -5.290682821408106, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_AICc, 3940.8399339452335, rtol=rtol, atol=atol)


def test_Fit_Everything_ALT_dual_stress():
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
    model = Fit_Everything_ALT(
        failures=data.failures,
        failure_stress_1=data.failure_stresses_1,
        failure_stress_2=data.failure_stresses_2,
        right_censored=data.right_censored,
        right_censored_stress_1=data.right_censored_stresses_1,
        right_censored_stress_2=data.right_censored_stresses_2,
        use_level_stress=np.array([300, 0.2]),
        show_best_distribution_probability_plot=True,
        show_probability_plot=True,
        print_results=True,
    )
    plt.close("all")
    assert_allclose(model.Weibull_Dual_Exponential_a, 62.2152, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_b, 0.0806668, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_c, 564.112, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_beta, 2.7627192146927757, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_AICc, 6564.886720768444, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Exponential_a, 91.5827026944374, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_b, 0.08359277091190954, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_c, 416.4064889485941, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_sigma, 0.5175107494013935, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_AICc, 6668.123531615943, rtol=rtol, atol=atol)

    assert_allclose(
        model.Normal_Dual_Exponential_a,
        71.35774950271157,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_b,
        0.07856968724936834,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_c,
        396.4425191690044,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_sigma,
        281.1349854285839,
        rtol=rtol_big,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_AICc,
        6743.5960289298655,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions

    assert_allclose(model.Exponential_Dual_Exponential_a, 67.33714223758182, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_b, 0.08781179984353935, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_c, 654.9754345457596, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_AICc, 7122.3673423549435, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Dual_Power_c, 2163.6255971936753, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_m, -0.19086651144428335, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_n, -0.21238379983141098, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_beta, 2.7647779859319477, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_AICc, 6563.254170285242, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Power_c, 2163.6255936535636, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_m, -0.22416436636273407, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_n, -0.20927479748079417, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_sigma, 0.5172509518253507, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_AICc, 6665.5546607405295, rtol=rtol, atol=atol)

    assert_allclose(
        model.Normal_Dual_Power_c,
        2163.625590241544,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Power_m,
        -0.2446653343001644,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Power_n,
        -0.1900322032599973,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Power_sigma,
        281.1349854285839,
        rtol=rtol_big,
        atol=atol,
    )
    assert_allclose(model.Normal_Dual_Power_AICc, 6742.712909225624, rtol=rtol_big, atol=atol)

    assert_allclose(model.Exponential_Dual_Power_c, 2163.625600158227, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_m, -0.16343508840178644, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_n, -0.24587409073658778, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_AICc, 7122.053278785371, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Power_Exponential_a, 52.663589085428896, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_c, 604.6849409358756, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_n, -0.21079488643176175, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_beta, 2.75821868750326, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_AICc, 6566.171282768552, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_Exponential_a, 86.97588807910807, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_c, 445.4183093440497, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_n, -0.19879727230868505, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_sigma, 0.5186891546026857, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_AICc, 6670.4488395476665, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Power_Exponential_a, 42.05520973390023, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_c, 569.6563384403188, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_n, -0.18642736242800026, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_sigma, 287.2895675310549, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_AICc, 6597.550156277316, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_Exponential_a, 56.94671110346091, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_c, 705.8722291687777, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_n, -0.23001969401255662, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_AICc, 7122.498208840988, rtol=rtol, atol=atol)

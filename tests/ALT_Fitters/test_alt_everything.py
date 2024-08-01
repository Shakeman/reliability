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
    assert_allclose(model.Weibull_Exponential_a, 1965.7797395338112, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_b, 11.0113385296826, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_beta, 2.3990457903278615, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_AICc, 3710.5717742652996, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_BIC, 3721.602040608187, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Exponential_loglik, -1852.2453465921092, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Exponential_a, 2009.7501864638946, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_b, 7.92564035, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_sigma, 0.5166917673351094, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_AICc, 3735.234452757792, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_BIC, 3746.2647191006795, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Exponential_loglik, -1864.5766858383554, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Exponential_a, 1993.9519931456607, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_b, 8.93233575, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_sigma, 746.4100628682714, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_AICc, 3901.96807936702, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_BIC, 3912.9983457099074, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Exponential_loglik, -1947.9434991429694, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Exponential_a, 1990.2598136542852, rtol=rtol_big, atol=atol)
    assert_allclose(model.Exponential_Exponential_b, 9.884663513057722, rtol=rtol_big, atol=atol)
    assert_allclose(model.Exponential_Exponential_AICc, 3926.5330483709013, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_BIC, 3933.9002092798096, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Exponential_loglik, -1961.2463221652486, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Eyring_a, 1548.0749958573679, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_c, -9.445096203462972, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_beta, 2.3958314887542222, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_AICc, 3711.331305215902, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_BIC, 3722.3615715587894, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Eyring_loglik, -1852.6251120674103, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Eyring_a, 1591.6554769675936, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_c, -9.1020583345059, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_sigma, 0.517659675042165, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_AICc, 3736.1791910066286, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_BIC, 3747.209457349516, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Eyring_loglik, -1865.0490549627737, rtol=rtol, atol=atol)

    assert_allclose(model.Normal_Eyring_a, 1602.0465747399508, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_c, -9.196438650244044, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_sigma, 747.0850651825152, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_AICc, 3902.5447879728936, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_BIC, 3913.575054315781, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Eyring_loglik, -1948.2318534459062, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Eyring_a, 1572.4214513890167, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_c, -9.337426924177604, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_AICc, 3926.729095178802, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_BIC, 3934.09625608771, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Eyring_loglik, -1961.3443455691988, rtol=rtol, atol=atol)

    assert_allclose(
        model.Weibull_Power_a,
        2645794073306709.0,
        rtol=0.9,
        atol=atol,
    )  # much larger due to variation in python versions. WHY???
    assert_allclose(
        model.Weibull_Power_n,
        -4.698158834438999,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(
        model.Weibull_Power_beta,
        2.3785671118139122,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.Weibull_Power_AICc, 3715.2055323609407, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_BIC, 3726.235798703828, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_loglik, -1854.5622256399297, rtol=rtol, atol=atol)

    assert_allclose(
        model.Lognormal_Power_a,
        2899022021518504.5,
        rtol=rtol,
        atol=atol,
    )  # much larger due to variation in python versions
    assert_allclose(
        model.Lognormal_Power_n,
        -4.752882880383393,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.Lognormal_Power_sigma, 0.522183419683184, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_AICc, 3740.5903647388977, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_BIC, 3751.620631081785, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_loglik, -1867.2546418289082, rtol=rtol, atol=atol)

    assert_allclose(
        model.Normal_Power_a,
        2.89902202e15,
        rtol=rtol,
        atol=atol,
    )  # much larger due to variation in python versions
    assert_allclose(
        model.Normal_Power_n,
        -4.968632318615027,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions
    assert_allclose(model.Normal_Power_sigma, 644.64709162, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_AICc, 3963.26763092, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_BIC, 3974.29789726, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_loglik, -1978.59327492, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_a, 2899022021518504.5, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_n, -4.721375201743889, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_AICc, 3927.669127417165, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_BIC, 3935.036288326073, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_loglik, -1961.8143616883804, rtol=rtol, atol=atol)


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
    assert_allclose(model.Weibull_Dual_Exponential_a, 109.3385636, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_b, 0.09291729, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_c, 456.09622917, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_beta, 2.53216375, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_AICc, 6584.301151215161, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_BIC, 6603.466037531028, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Exponential_loglik, -3288.128229238865, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Exponential_a, 53.03066564, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_b, 0.10968515, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_c, 428.90741626, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_sigma, 0.5313705214283074, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_AICc, 6652.254628391293, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_BIC, 6671.4195147071605, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Exponential_loglik, -3322.1049678269314, rtol=rtol, atol=atol)

    assert_allclose(
        model.Normal_Dual_Exponential_a,
        49.31674140692864,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_b,
        0.10546747994283766,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_c,
        384.7807038273309,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_sigma,
        289.9438341852921,
        rtol=rtol_big,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_AICc,
        6787.86606886861,
        rtol=rtol_big,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_BIC,
        6807.030955184477,
        rtol=rtol_big,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Exponential_loglik,
        -3390.886396880605,
        rtol=rtol_big,
        atol=atol,
    )  # larger due to variation in python versions

    assert_allclose(model.Exponential_Dual_Exponential_a, 136.1545794, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_b, 0.10203765, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_c, 478.81825966, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_AICc, 7089.224587107275, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_BIC, 7103.604985682962, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Exponential_loglik, -3541.5989006964946, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Dual_Power_c, 914.3759566, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_m, -0.05470473, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_n, -0.28962859, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_beta, 2.5815730622862336, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_AICc, 6583.673537008596, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_BIC, 6602.838423324463, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Dual_Power_loglik, -3287.814422135583, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Dual_Power_c, 914.37590837, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_m, -0.09154364, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_n, -0.29140721, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_sigma, 0.5306077545728608, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_AICc, 6650.71962135737, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_BIC, 6669.884507673237, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Dual_Power_loglik, -3321.3374643099696, rtol=rtol, atol=atol)

    assert_allclose(
        model.Normal_Dual_Power_c,
        914.3759033056451,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Power_m,
        -0.11465369510079437,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Power_n,
        -0.28844941648459693,
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Power_sigma,
        289.9438341852921,
        rtol=rtol_big,
        atol=atol,
    )
    assert_allclose(model.Normal_Dual_Power_AICc, 6787.86606886861, rtol=rtol_big, atol=atol)
    assert_allclose(
        model.Normal_Dual_Power_BIC,
        6807.030955184477,
        rtol=rtol_big,
        atol=atol,
    )
    assert_allclose(
        model.Normal_Dual_Power_loglik,
        -3389.9106880655895,
        rtol=rtol_big,
        atol=atol,
    )

    assert_allclose(model.Exponential_Dual_Power_c, 914.37593082, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_m, -0.02871604, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_n, -0.31093619, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_AICc, 7088.926283426964, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_BIC, 7103.3066820026515, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Dual_Power_loglik, -3541.4497488563393, rtol=rtol, atol=atol)

    assert_allclose(model.Weibull_Power_Exponential_a, 42.780811253238745, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_c, 595.5272158322398, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_n, -0.24874378554058516, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_beta, 2.5871119030281675, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_AICc, 6582.118937551177, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_BIC, 6601.283823867044, rtol=rtol, atol=atol)
    assert_allclose(model.Weibull_Power_Exponential_loglik, -3287.037122406873, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_Exponential_a, 61.77652017, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_c, 444.28631096, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_n, -0.26714273, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_sigma, 0.5304232436388364, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_AICc, 6650.403864331276, rtol=rtol, atol=atol)
    assert_allclose(model.Lognormal_Power_Exponential_BIC, 6669.568750647143, rtol=rtol, atol=atol)

    assert_allclose(model.Lognormal_Power_Exponential_loglik, -3321.1795857969228, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_a, 37.59712219841342, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_c, 548.919567681429, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_n, -0.2262443694448219, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_sigma, 296.6005867790367, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_AICc, 6633.1976944950475, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_BIC, 6652.362580810915, rtol=rtol, atol=atol)
    assert_allclose(model.Normal_Power_Exponential_loglik, -3312.5765008788085, rtol=rtol, atol=atol)

    assert_allclose(model.Exponential_Power_Exponential_a, 42.937623270069224, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_c, 695.3534426950991, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_n, -0.26413696165231737, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_AICc, 7088.800993682692, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_BIC, 7103.18139225838, rtol=rtol, atol=atol)
    assert_allclose(model.Exponential_Power_Exponential_loglik, -3541.3871039842033, rtol=rtol, atol=atol)

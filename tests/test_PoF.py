import pytest
from numpy.testing import assert_allclose

from reliability.PoF import (
    SN_diagram,
    acceleration_factor,
    creep_failure_time,
    creep_rupture_curves,
    fracture_mechanics_crack_growth,
    fracture_mechanics_crack_initiation,
    palmgren_miner_linear_damage,
    strain_life_diagram,
    stress_strain_diagram,
    stress_strain_life_parameters_from_data,
)

atol = 1e-8
rtol = 1e-7


@pytest.mark.flaky(reruns=3)
def test_SN_diagram():
    stress: list[int] = [340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210]
    cycles: list[int] = [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000]
    stress_runout: list[int] = [210, 210, 205, 205, 205]
    cycles_runout: list[int] = [10**7, 10**7, 10**7, 10**7, 10**7]
    SN_diagram(
        stress=stress,
        cycles=cycles,
        stress_runout=stress_runout,
        cycles_runout=cycles_runout,
        method_for_bounds="residual",
        cycles_trace=[5 * 10**5],
        stress_trace=[260],
    )


def test_stress_strain_diagram():
    strain_data = [0.02, 0.015, 0.01, 0.006, 0.0035, 0.002]
    stress_data = [650, 625, 555, 480, 395, 330]
    cycles_data = [200, 350, 1100, 4600, 26000, 560000]
    params = stress_strain_life_parameters_from_data(
        stress=stress_data,
        strain=strain_data,
        cycles=cycles_data,
        E=216000,
    )
    params.plot()
    params.print_results()
    stress_strain = stress_strain_diagram(E=216000, n=params.n, K=params.K, max_strain=0.006)
    stress_strain.print_results()
    assert_allclose(stress_strain.max_strain, 0.006, rtol=rtol, atol=atol)
    assert_allclose(stress_strain.min_strain, -0.006, rtol=rtol, atol=atol)
    assert_allclose(stress_strain.max_stress, 483.8581623940639, rtol=rtol, atol=atol)
    assert_allclose(stress_strain.min_stress, -483.8581623940639, rtol=rtol, atol=atol)


def test_strain_life_diagram():
    results = strain_life_diagram(
        E=210000,
        sigma_f=1000,
        epsilon_f=1.1,
        b=-0.1,
        c=-0.6,
        K=1200,
        n=0.2,
        max_strain=0.0049,
        min_strain=-0.0029,
    )
    assert_allclose(results.cycles_to_failure, 13771.39230726717, rtol=rtol, atol=atol)
    assert_allclose(results.max_strain, 0.0049, rtol=rtol, atol=atol)
    assert_allclose(results.min_strain, -0.0029, rtol=rtol, atol=atol)
    assert_allclose(results.max_stress, 377.9702002307027, rtol=rtol, atol=atol)
    assert_allclose(results.min_stress, -321.06700330271457, rtol=rtol, atol=atol)


def test_fracture_mechanics_crack_initiation():
    results = fracture_mechanics_crack_initiation(
        P=0.15,
        A=5 * 80,
        Kt=2.41,
        q=0.9857,
        Sy=690,
        E=210000,
        K=1060,
        n=0.14,
        b=-0.081,
        c=-0.65,
        sigma_f=1160,
        epsilon_f=1.1,
        mean_stress_correction_method="SWT",
    )
    results.print_results()
    assert_allclose(results.cycles_to_failure, 2919.911371962644, rtol=rtol, atol=atol)
    assert_allclose(results.epsilon_max, 0.007547514721969089, rtol=rtol, atol=atol)
    assert_allclose(results.epsilon_mean, 8.673617379884035e-19, rtol=rtol, atol=atol)
    assert_allclose(results.epsilon_min, -0.007547514721969087, rtol=rtol, atol=atol)
    assert_allclose(results.sigma_max, 506.7290859876517, rtol=rtol, atol=atol)
    assert_allclose(results.sigma_mean, -5.684341886080802e-14, rtol=rtol, atol=atol)
    assert_allclose(results.sigma_min, -506.7290859876518, rtol=rtol, atol=atol)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fracture_mechanics():
    results = fracture_mechanics_crack_growth(Kc=66, C=6.91 * 10**-12, m=3, P=0.15, W=100, t=5, Kt=2.41, D=10)
    results.print_results()
    results.plot()
    assert_allclose(results.Nf_stage_1_iterative, 7576, rtol=rtol, atol=atol)
    assert_allclose(results.Nf_stage_1_simplified, 6802.128636224042, rtol=rtol, atol=atol)
    assert_allclose(results.Nf_stage_2_iterative, 671, rtol=rtol, atol=atol)
    assert_allclose(results.Nf_stage_2_simplified, 1133.6504736416755, rtol=rtol, atol=atol)
    assert_allclose(results.Nf_total_iterative, 8247, rtol=rtol, atol=atol)
    assert_allclose(results.Nf_total_simplified, 7935.779109865717, rtol=rtol, atol=atol)
    assert_allclose(results.final_crack_length_iterative, 6.389099561491157, rtol=rtol, atol=atol)
    assert_allclose(results.final_crack_length_simplified, 7.8603053527017686, rtol=rtol, atol=atol)
    assert_allclose(results.transition_length_iterative, 2.4520443881041274, rtol=rtol, atol=atol)
    assert_allclose(results.transition_length_simplified, 2.0798236309560947, rtol=rtol, atol=atol)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_creep():
    TEMP = [
        900,
        900,
        900,
        900,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1100,
        1100,
        1100,
        1100,
        1100,
        1200,
        1200,
        1200,
        1200,
        1350,
        1350,
        1350,
    ]
    STRESS = [90, 82, 78, 70, 80, 75, 68, 60, 56, 49, 43, 38, 60.5, 50, 40, 29, 22, 40, 30, 25, 20, 20, 15, 10]
    TTF = [
        37,
        975,
        3581,
        9878,
        7,
        17,
        213,
        1493,
        2491,
        5108,
        7390,
        10447,
        18,
        167,
        615,
        2220,
        6637,
        19,
        102,
        125,
        331,
        3.7,
        8.9,
        31.8,
    ]
    creep_rupture_curves(temp_array=TEMP, stress_array=STRESS, TTF_array=TTF, stress_trace=70, temp_trace=1100)


def test_creep_failure_time():
    results = creep_failure_time(temp_low=900, temp_high=1100, time_low=9878, print_results=True)
    assert_allclose(results, 8.27520045913433, rtol=rtol, atol=atol)


def test_palmgren_miner_linear_damage():
    stress = [1, 2, 4]
    results = palmgren_miner_linear_damage(
        rated_life=[50000, 6500, 1000],
        time_at_stress=[40 / 60, 15 / 60, 5 / 60],
        stress=stress,
    )
    assert_allclose(results[0], 0.00013512820512820512, rtol=rtol, atol=atol)
    assert_allclose(results[1], 7400.379506641367, rtol=rtol, atol=atol)
    assert len(stress) == len(results[2])
    assert all(a == b for a, b in zip(stress, results[2]))


def test_acceleration_factor():
    results = acceleration_factor(T_use=60, T_acc=100, Ea=1.2)
    results.print_results()
    assert_allclose(results.AF, 88.29574588463338, rtol=rtol, atol=atol)
    assert_allclose(results.Ea, 1.2, rtol=rtol, atol=atol)
    assert_allclose(results.T_acc, 100, rtol=rtol, atol=atol)
    assert_allclose(results.T_use, 60, rtol=rtol, atol=atol)

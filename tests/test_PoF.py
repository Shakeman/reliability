from numpy.testing import assert_allclose

from reliability.PoF import (
    SN_diagram,
    stress_strain_diagram,
    stress_strain_life_parameters_from_data,
)

atol = 1e-8
rtol = 1e-7


def test_SN_diagram():
    stress = [340, 300, 290, 275, 260, 255, 250, 235, 230, 220, 215, 210]
    cycles = [15000, 24000, 36000, 80000, 177000, 162000, 301000, 290000, 361000, 881000, 1300000, 2500000]
    stress_runout = [210, 210, 205, 205, 205]
    cycles_runout = [10**7, 10**7, 10**7, 10**7, 10**7]
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
        stress=stress_data, strain=strain_data, cycles=cycles_data, E=216000, show_plot=False, print_results=False
    )
    stress_strain = stress_strain_diagram(E=216000, n=params.n, K=params.K, max_strain=0.006)
    assert_allclose(stress_strain.max_strain, 0.006, rtol=rtol, atol=atol)
    assert_allclose(stress_strain.min_strain, -0.006, rtol=rtol, atol=atol)
    assert_allclose(stress_strain.max_stress, 483.8581623940639, rtol=rtol, atol=atol)
    assert_allclose(stress_strain.min_stress, -483.8581623940639, rtol=rtol, atol=atol)

import matplotlib.pyplot as plt

from reliability.PoF import stress_strain_diagram, stress_strain_life_parameters_from_data

strain_data = [0.02, 0.015, 0.01, 0.006, 0.0035, 0.002]
stress_data = [650, 625, 555, 480, 395, 330]
cycles_data = [200, 350, 1100, 4600, 26000, 560000]
params = stress_strain_life_parameters_from_data(
    stress=stress_data, strain=strain_data, cycles=cycles_data, E=216000, show_plot=False, print_results=False
)
a = stress_strain_diagram(E=216000, n=params.n, K=params.K, max_strain=0.006)
plt.show()

"""
Results from stress_strain_diagram:
Max stress: 483.8581623940648
Min stress: -483.8581623940648
Max strain: 0.006
Min strain: -0.006
"""

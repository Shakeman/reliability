import matplotlib.pyplot as plt

from reliability.Reliability_testing import likelihood_plot

old_design = [
    2,
    9,
    23,
    38,
    67,
    2,
    11,
    28,
    40,
    76,
    3,
    17,
    33,
    45,
    90,
    4,
    17,
    34,
    55,
    115,
    6,
    19,
    34,
    56,
    126,
    9,
    21,
    37,
    57,
    197,
]
new_design = [15, 116, 32, 148, 61, 178, 67, 181, 75, 183]
likelihood_plot(distribution="Weibull", failures=old_design, CI=[0.9, 0.95])
likelihood_plot(distribution="Weibull", failures=new_design, CI=[0.9, 0.95])
plt.show()

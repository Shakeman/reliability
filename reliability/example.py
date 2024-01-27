import matplotlib.pyplot as plt

from reliability.Distributions import Normal_Distribution, Weibull_Distribution
from reliability.Probability_plotting import PP_plot_parametric

Field = Normal_Distribution(mu=100, sigma=30)
Lab = Weibull_Distribution(alpha=120, beta=3)
PP_plot_parametric(X_dist=Field, Y_dist=Lab, x_quantile_lines=[0.3, 0.6], y_quantile_lines=[0.1, 0.6])
plt.show()

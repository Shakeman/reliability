import matplotlib.pyplot as plt

from reliability.Distributions import DSZI_Model, Lognormal_Distribution
from reliability.Probability_plotting import plot_points

model = DSZI_Model(distribution = Lognormal_Distribution(mu=2,sigma=0.5), DS= 0.75)
failures, right_censored = model.random_samples(50,seed=7, right_censored_time = 50)
model.SF()
plot_points(failures = failures, right_censored = right_censored, func="SF")
plt.show()

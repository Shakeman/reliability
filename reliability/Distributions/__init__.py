"""Probability Distributions Module

Standard distributions are:
    Weibull_Distribution
    Normal_Distribution
    Lognormal_Distribution
    Exponential_Distribution
    Gamma_Distribution
    Beta_Distribution
    Loglogistic_Distribution
    Gumbel_Distribution

Mixture distributions are:
    Mixture_Model - this must be created using 2 or more of the above standard distributions
    Competing_Risks_Model - this must be created using 2 or more of the above standard distributions

Example usage:
dist = Weibull_Distribution(alpha = 8, beta = 1.2)
print(dist.mean)
    >> 7.525246866054174
print(dist.quantile(0.05))
    >> 0.6731943793488804
print(dist.mean_residual_life(15))
    >> 5.556500198354015
dist.plot()
    >> A figure of 5 plots and descriptive statistics will be displayed
dist.CHF()
    >> Cumulative Hazard Function plot will be displayed
values = dist.random_samples(number_of_samples=10000)
    >> random values will be generated from the distribution
"""

from reliability.Distributions._beta_dist import Beta_Distribution
from reliability.Distributions._competing_risks import Competing_Risks_Model
from reliability.Distributions._dszi_model import DSZI_Model
from reliability.Distributions._exponential_dist import Exponential_Distribution
from reliability.Distributions._gamma_dist import Gamma_Distribution
from reliability.Distributions._gumbel_dist import Gumbel_Distribution
from reliability.Distributions._loglogistic_dist import Loglogistic_Distribution
from reliability.Distributions._lognormal_dist import Lognormal_Distribution
from reliability.Distributions._mixture_model import Mixture_Model
from reliability.Distributions._normal_dist import Normal_Distribution
from reliability.Distributions._weibull_dist import Weibull_Distribution

__all__ = [
    "Beta_Distribution",
    "Exponential_Distribution",
    "Gamma_Distribution",
    "Gumbel_Distribution",
    "Loglogistic_Distribution",
    "Lognormal_Distribution",
    "Weibull_Distribution",
    "Normal_Distribution",
    "Competing_Risks_Model",
    "Mixture_Model",
    "DSZI_Model",
]

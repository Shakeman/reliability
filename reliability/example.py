from reliability.Datasets import mixture
from reliability.Fitters import Fit_Weibull_Mixture

Fit_Weibull_Mixture(failures=mixture().failures, right_censored=mixture().right_censored)

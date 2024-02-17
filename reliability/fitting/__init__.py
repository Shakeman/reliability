"""Fitters.

This module contains custom fitting functions for parametric distributions which
support complete and right censored data.
The included functions are:

- Fit_Weibull_2P
- Fit_Weibull_3P
- Fit_Exponential_1P
- Fit_Exponential_2P
- Fit_Gamma_2P
- Fit_Gamma_3P
- Fit_Lognormal_2P
- Fit_Lognormal_3P
- Fit_Normal_2P
- Fit_Gumbel_2P
- Fit_Beta_2P
- Fit_Loglogistic_2P
- Fit_Loglogistic_3P
- Fit_Weibull_Mixture
- Fit_Weibull_CR
- Fit_Weibull_DS
- Fit_Weibull_ZI
- Fit_Weibull_DSZI

Note that the Beta distribution is only for data in the range 0 < t < 1.
There is also a Fit_Everything function which will fit all distributions (except
the Weibull_Mixture and Weibull_CR models) and will provide plots and a table of
values.

All functions in this module work using autograd to find the derivative of the
log-likelihood function. In this way, the code only needs to specify the log PDF
and log SF in order to obtain the fitted parameters. Initial guesses of the
parameters are essential for autograd and are obtained using least squares or
non-linear least squares (depending on the function). If the distribution is an
extremely bad fit or is heavily censored (>99%) then these guesses may be poor
and the fit might not be successful. Generally the fit achieved by autograd is
highly successful, and whenever it fails the initial guess will be used and a
warning will be displayed.
"""
from reliability.fitting._fit_basic import Fit_Beta_2P, Fit_Gumbel_2P, Fit_Normal_2P
from reliability.fitting._fit_everything import Fit_Everything
from reliability.fitting._fit_exponential import Fit_Exponential_1P, Fit_Exponential_2P
from reliability.fitting._fit_gamma import Fit_Gamma_2P, Fit_Gamma_3P
from reliability.fitting._fit_loglog import Fit_Loglogistic_2P, Fit_Loglogistic_3P
from reliability.fitting._fit_lognorm import Fit_Lognormal_2P, Fit_Lognormal_3P
from reliability.fitting._fit_weibull import (
    Fit_Weibull_2P,
    Fit_Weibull_2P_grouped,
    Fit_Weibull_3P,
    Fit_Weibull_CR,
    Fit_Weibull_DS,
    Fit_Weibull_DSZI,
    Fit_Weibull_Mixture,
    Fit_Weibull_ZI,
)

__all__ = [
    "Fit_Everything",
    "Fit_Weibull_2P",
    "Fit_Weibull_2P_grouped",
    "Fit_Weibull_3P",
    "Fit_Weibull_CR",
    "Fit_Weibull_DS",
    "Fit_Weibull_DSZI",
    "Fit_Weibull_Mixture",
    "Fit_Weibull_ZI",
    "Fit_Exponential_1P",
    "Fit_Exponential_2P",
    "Fit_Lognormal_2P",
    "Fit_Lognormal_3P",
    "Fit_Loglogistic_2P",
    "Fit_Loglogistic_3P",
    "Fit_Gamma_2P",
    "Fit_Gamma_3P",
    "Fit_Beta_2P",
    "Fit_Gumbel_2P",
    "Fit_Normal_2P",
]

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
from __future__ import annotations

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from autograd.differential_operators import hessian
from autograd.scipy.special import beta as abeta
from autograd.scipy.special import erf
from autograd.scipy.special import gamma as agamma
from autograd_gamma import betainc, gammaincc
from numpy.linalg import LinAlgError

from reliability.Distributions import (
    Beta_Distribution,
    Exponential_Distribution,
    Gamma_Distribution,
    Gumbel_Distribution,
    Loglogistic_Distribution,
    Lognormal_Distribution,
    Normal_Distribution,
)
from reliability.Probability_plotting import plotting_positions
from reliability.Utils import (
    LS_optimization,
    MLE_optimization,
    anderson_darling,
    colorprint,
    extract_CI,
    fitters_input_checking,
    round_and_string,
)

anp.seterr("ignore")
dec = 3  # number of decimals to use when rounding fitted parameters in labels

# change pandas display options
pd.options.display.float_format = "{:g}".format  # improves formatting of numbers in dataframe
pd.options.display.max_columns = 9  # shows the dataframe without ... truncation
pd.options.display.width = 200  # prevents wrapping after default 80 characters


class Fit_Exponential_1P:
    """Fits a one parameter Exponential distribution (Lambda) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for the model's parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    Lambda : float
        the fitted Exponential_1P Lambda parameter
    Lambda_inv : float
        the inverse of the fitted Exponential_1P Lambda parameter
    Lambda_SE : float
        the standard error (sqrt(variance)) of the parameter
    Lambda_SE_inv : float
        the standard error (sqrt(variance)) of the inverse of the parameter
    Lambda_upper : float
        the upper CI estimate of the parameter
    Lambda_lower : float
        the lower CI estimate of the parameter
    Lambda_upper_inv : float
        the upper CI estimate of the inverse of the parameter
    Lambda_lower_inv : float
        the lower CI estimate of the inverse of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Exponential_Distribution object with the parameter of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles. This is only produced if
        quantiles is not None. Since quantiles defaults to None, this output
        is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is a one parameter distribution, but the results provide both the
    parameter (Lambda) as well as the inverse (1/Lambda). This is provided for
    convenience as some other software (Minitab and scipy.stats) use 1/Lambda
    instead of Lambda. Lambda_SE_inv, Lambda_upper_inv, and Lambda_lower_inv are
    also provided for convenience.

    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Exponential_1P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        self.gamma = 0

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Exponential_1P",
            LL_func=Fit_Exponential_1P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.Lambda = LS_results.guess[0]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Exponential_1P",
                LL_func=Fit_Exponential_1P.LL,
                initial_guess=[LS_results.guess[0]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.Lambda = MLE_results.scale
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.Lambda]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
            self.Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = abs(1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z)
            self.Lambda_lower_inv = 1 / self.Lambda_upper
            self.Lambda_upper_inv = 1 / self.Lambda_lower
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_1P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.Lambda_SE = 0
            self.Lambda_upper = self.Lambda
            self.Lambda_lower = self.Lambda
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = 0
            self.Lambda_lower_inv = 1 / self.Lambda
            self.Lambda_upper_inv = 1 / self.Lambda

        results_data = {
            "Parameter": ["Lambda", "1/Lambda"],
            "Point Estimate": [self.Lambda, self.Lambda_inv],
            "Standard Error": [self.Lambda_SE, self.Lambda_SE_inv],
            "Lower CI": [self.Lambda_lower, self.Lambda_lower_inv],
            "Upper CI": [self.Lambda_upper, self.Lambda_upper_inv],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Exponential_Distribution(Lambda=self.Lambda, Lambda_SE=self.Lambda_SE, CI=CI)

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 1
        LL2 = 2 * Fit_Exponential_1P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_1P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import (
                Exponential_probability_plot_Weibull_Scale,
            )

            rc = None if len(right_censored) == 0 else right_censored
            Exponential_probability_plot_Weibull_Scale(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, L):  # Log PDF (1 parameter Expon)
        return anp.log(L) - L * t

    @staticmethod
    def logR(t, L):  # Log SF (1 parameter Expon)
        return -(L * t)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (1 parameter Expon)
        LL_f = Fit_Exponential_1P.logf(T_f, params[0]).sum()
        LL_rc = Fit_Exponential_1P.logR(T_rc, params[0]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_2P:
    """Fits a two parameter Exponential distribution (Lambda, gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 1 element.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for the model's parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    Lambda : float
        the fitted Exponential_1P Lambda parameter
    Lambda_inv : float
        the inverse of the fitted Exponential_1P Lambda parameter
    gamma : float
        the fitted Exponential_2P gamma parameter
    Lambda_SE : float
        the standard error (sqrt(variance)) of the parameter
    Lambda_SE_inv : float
        the standard error (sqrt(variance)) of the inverse of the parameter
    gamma_SE : float
        the standard error (sqrt(variance)) of the parameter
    Lambda_upper : float
        the upper CI estimate of the parameter
    Lambda_lower : float
        the lower CI estimate of the parameter
    Lambda_upper_inv : float
        the upper CI estimate of the inverse of the parameter
    Lambda_lower_inv : float
        the lower CI estimate of the inverse of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Exponential_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles. This is only produced if
        quantiles is not None. Since quantiles defaults to None, this output
        is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is a two parameter distribution (Lambda, gamma), but the results
    provide both Lambda as well as the inverse (1/Lambda). This is provided for
    convenience as some other software (Minitab and scipy.stats) use 1/Lambda
    instead of Lambda. Lambda_SE_inv, Lambda_upper_inv, and Lambda_lower_inv are
    also provided for convenience.

    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        # To obtain the confidence intervals of the parameters, the gamma parameter is estimated by optimizing the log-likelihood function but
        # it is assumed as fixed because the variance-covariance matrix of the estimated parameters cannot be determined numerically. By assuming
        # the standard error in gamma is zero, we can use Exponential_1P to obtain the confidence intervals for Lambda. This is the same procedure
        # performed by both Reliasoft and Minitab. You may find the results are slightly different to Minitab and this is because the optimization
        # of gamma is done more efficiently here than Minitab does it. This is evidenced by comparing the log-likelihood for the same data input.

        inputs = fitters_input_checking(
            dist="Exponential_2P",
            failures=failures,
            right_censored=right_censored,
            CI=CI,
            quantiles=quantiles,
            method=method,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        quantiles = inputs.quantiles
        method = inputs.method
        optimizer = inputs.optimizer

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Exponential_2P",
            LL_func=Fit_Exponential_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.Lambda = LS_results.guess[0]
            self.gamma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            if (
                LS_results.guess[0] < 1
            ):  # The reason for having an inverted and non-inverted cases is due to the gradient being too shallow in some cases. If Lambda<1 we invert it so it's bigger. This prevents the gradient getting too shallow for the optimizer to find the correct minimum.
                MLE_results = MLE_optimization(
                    func_name="Exponential_2P",
                    LL_func=Fit_Exponential_2P.LL_inv,
                    initial_guess=[1 / LS_results.guess[0], LS_results.guess[1]],
                    failures=failures,
                    right_censored=right_censored,
                    optimizer=optimizer,
                )
                self.Lambda = 1 / MLE_results.scale
            else:
                MLE_results = MLE_optimization(
                    func_name="Exponential_2P",
                    LL_func=Fit_Exponential_2P.LL,
                    initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                    failures=failures,
                    right_censored=right_censored,
                    optimizer=optimizer,
                )
                self.Lambda = MLE_results.scale
            self.gamma = MLE_results.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. Uses Exponential_1P because gamma (while optimized) cannot be used in the MLE solution as the solution is unbounded. This is why there are no CI limits on gamma.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params_1P = [self.Lambda]
        params_2P = [self.Lambda, self.gamma]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(
            np.array(tuple(params_1P)),
            np.array(tuple(failures - self.gamma)),
            np.array(tuple(right_censored - self.gamma)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.Lambda_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.gamma_SE = 0
            self.Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
            self.Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))
            self.gamma_upper = self.gamma
            self.gamma_lower = self.gamma
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = abs(1 / self.Lambda * np.log(self.Lambda / self.Lambda_upper) / Z)
            self.Lambda_lower_inv = 1 / self.Lambda_upper
            self.Lambda_upper_inv = 1 / self.Lambda_lower
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.Lambda_SE = 0
            self.gamma_SE = 0
            self.Lambda_upper = self.Lambda
            self.Lambda_lower = self.Lambda
            self.gamma_upper = self.gamma
            self.gamma_lower = self.gamma
            self.Lambda_inv = 1 / self.Lambda
            self.Lambda_SE_inv = 0
            self.Lambda_lower_inv = 1 / self.Lambda
            self.Lambda_upper_inv = 1 / self.Lambda

        results_data = {
            "Parameter": ["Lambda", "1/Lambda", "Gamma"],
            "Point Estimate": [self.Lambda, self.Lambda_inv, self.gamma],
            "Standard Error": [self.Lambda_SE, self.Lambda_SE_inv, self.gamma_SE],
            "Lower CI": [self.Lambda_lower, self.Lambda_lower_inv, self.gamma_lower],
            "Upper CI": [self.Lambda_upper, self.Lambda_upper_inv, self.gamma_upper],
        }

        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Exponential_Distribution(
            Lambda=self.Lambda,
            gamma=self.gamma,
            Lambda_SE=self.Lambda_SE,
            CI=CI,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Exponential_2P.LL(params_2P, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import (
                Exponential_probability_plot_Weibull_Scale,
            )

            rc = None if len(right_censored) == 0 else right_censored
            Exponential_probability_plot_Weibull_Scale(
                failures=failures,
                right_censored=rc,
                CI=CI,
                __fitted_dist_params=self,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, L, g):  # Log PDF (2 parameter Expon)
        return anp.log(L) - L * (t - g)

    @staticmethod
    def logR(t, L, g):  # Log SF (2 parameter Expon)
        return -(L * (t - g))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter Expon)
        LL_f = Fit_Exponential_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Exponential_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    # #this is the inverted forms of the above functions. It simply changes Lambda to be 1/Lambda which is necessary when Lambda<<1
    @staticmethod
    def LL_inv(params, T_f, T_rc):
        # log likelihood function (2 parameter Expon)
        LL_f = Fit_Exponential_2P.logf(T_f, 1 / params[0], params[1]).sum()
        LL_rc = Fit_Exponential_2P.logR(T_rc, 1 / params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_2P:
    """Fits a two parameter Normal distribution (mu,sigma) to the data provided.
    Note that it will return a fit that may be partially in the negative domain
    (x<0). If you need an entirely positive distribution that is similar to
    Normal then consider using Weibull.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements if force_sigma is not
        specified or at least 1 element if force_sigma is specified.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    force_sigma : float, int, optional
        Used to specify the beta value if you need to force sigma to be a
        certain value. Used in ALT probability plotting. Optional input. If
        specified it must be > 0.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    mu : float
        the fitted Normal_2P mu parameter
    sigma : float
        the fitted Normal_2P sigma parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Normal_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        optimizer=None,
        CI_type="time",
        method="MLE",
        force_sigma=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Normal_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_sigma=force_sigma,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_sigma = inputs.force_sigma
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Normal_2P",
            LL_func=Fit_Normal_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
            force_shape=force_sigma,
            LL_func_force=Fit_Normal_2P.LL_fs,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Normal_2P",
                LL_func=Fit_Normal_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
                force_shape=force_sigma,
                LL_func_force=Fit_Normal_2P.LL_fs,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Normal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.Cov_mu_sigma = covariance_matrix[0][1]
                self.mu_upper = self.mu + (Z * self.mu_SE)  # these are unique to normal and lognormal mu params
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
                self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Normal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer.",
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma

        else:
            hessian_matrix = hessian(Fit_Normal_2P.LL_fs)(
                np.array((self.mu,)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array((force_sigma,)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu + (Z * self.mu_SE)  # these are unique to normal and lognormal mu params
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Normal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer.",
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma

        results_data = {
            "Parameter": ["Mu", "Sigma"],
            "Point Estimate": [self.mu, self.sigma],
            "Standard Error": [self.mu_SE, self.sigma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Normal_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        if force_sigma is None:
            k = 2
            LL2 = 2 * Fit_Normal_2P.LL(params, failures, right_censored)
        else:
            k = 1
            LL2 = 2 * Fit_Normal_2P.LL_fs([self.mu], failures, right_censored, force_sigma)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Normal_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Normal_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Normal_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma):  # Log PDF (Normal)
        return anp.log(anp.exp(-0.5 * (((t - mu) / sigma) ** 2))) - anp.log(sigma * (2 * anp.pi) ** 0.5)

    @staticmethod
    def logR(t, mu, sigma):  # Log SF (Normal)
        return anp.log((1 + erf(((mu - t) / sigma) / 2**0.5)) / 2)

    @staticmethod
    def LL(params, T_f, T_rc):  # log likelihood function (2 parameter Normal)
        LL_f = Fit_Normal_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Normal_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fs(params, T_f, T_rc, force_sigma):
        # log likelihood function (2 parameter Normal) FORCED SIGMA
        LL_f = Fit_Normal_2P.logf(T_f, params[0], force_sigma).sum()
        LL_rc = Fit_Normal_2P.logR(T_rc, params[0], force_sigma).sum()
        return -(LL_f + LL_rc)


class Fit_Gumbel_2P:
    """Fits a two parameter Gumbel distribution (mu,sigma) to the data provided.
    Note that it will return a fit that may be partially in the negative domain
    (x<0). If you need an entirely positive distribution that is similar to
    Gumbel then consider using Weibull.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle).

    Returns
    -------
    mu : float
        the fitted Gumbel_2P mu parameter
    sigma : float
        the fitted Gumbel_2P sigma parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Gumbel_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    The Gumbel Distribution is similar to the Normal Distribution, with mu
    controlling the peak of the distribution between -inf < mu < inf.

    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Gumbel_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Gumbel_2P",
            LL_func=Fit_Gumbel_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Gumbel_2P",
                LL_func=Fit_Gumbel_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        hessian_matrix = hessian(Fit_Gumbel_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_mu_sigma = covariance_matrix[0][1]
            self.mu_upper = self.mu + (Z * self.mu_SE)  # these are unique to gumbel, normal and lognormal mu params
            self.mu_lower = self.mu + (-Z * self.mu_SE)
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Gumbel_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.mu_SE = 0
            self.sigma_SE = 0
            self.Cov_mu_sigma = 0
            self.mu_upper = self.mu
            self.mu_lower = self.mu
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        results_data = {
            "Parameter": ["Mu", "Sigma"],
            "Point Estimate": [self.mu, self.sigma],
            "Standard Error": [self.mu_SE, self.sigma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Gumbel_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Gumbel_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Gumbel_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gumbel_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Gumbel_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma):  # Log PDF (Gumbel)
        return -anp.log(sigma) + (t - mu) / sigma - anp.exp((t - mu) / sigma)

    @staticmethod
    def logR(t, mu, sigma):  # Log SF (Gumbel)
        return -anp.exp((t - mu) / sigma)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter Gumbel)
        LL_f = Fit_Gumbel_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Gumbel_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Lognormal_2P:
    """Fits a two parameter Lognormal distribution (mu,sigma) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements if force_sigma is not
        specified or at least 1 element if force_sigma is specified.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    force_sigma : float, int, optional
        Used to specify the sigma value if you need to force sigma to be a
        certain value. Used in ALT probability plotting. Optional input. If
        specified it must be > 0.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle).

    Returns
    -------
    mu : float
        the fitted Lognormal_2P alpha parameter
    sigma : float
        the fitted Lognormal_2P beta parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Lognormal_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        optimizer=None,
        CI_type="time",
        method="MLE",
        force_sigma=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Lognormal_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_sigma=force_sigma,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_sigma = inputs.force_sigma
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Lognormal_2P",
            LL_func=Fit_Lognormal_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
            force_shape=force_sigma,
            LL_func_force=Fit_Lognormal_2P.LL_fs,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Lognormal_2P",
                LL_func=Fit_Lognormal_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
                force_shape=force_sigma,
                LL_func_force=Fit_Lognormal_2P.LL_fs,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.mu, self.sigma]
        if force_sigma is None:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.Cov_mu_sigma = covariance_matrix[0][1]
                self.mu_upper = self.mu + (Z * self.mu_SE)  # mu is positive or negative
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))  # sigma is strictly positive
                self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Lognormal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer.",
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
        else:
            hessian_matrix = hessian(Fit_Lognormal_2P.LL_fs)(
                np.array((self.mu,)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array((force_sigma,)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu + (Z * self.mu_SE)  # mu is positive or negative
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Lognormal_2P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer.",
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma

        results_data = {
            "Parameter": ["Mu", "Sigma"],
            "Point Estimate": [self.mu, self.sigma],
            "Standard Error": [self.mu_SE, self.sigma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Lognormal_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        if force_sigma is None:
            k = 2
            LL2 = 2 * Fit_Lognormal_2P.LL(params, failures, right_censored)
        else:
            k = 1
            LL2 = 2 * Fit_Lognormal_2P.LL_fs([self.mu], failures, right_censored, force_sigma)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Lognormal_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Lognormal_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Lognormal_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma):  # Log PDF (Lognormal)
        return anp.log(anp.exp(-0.5 * (((anp.log(t) - mu) / sigma) ** 2)) / (t * sigma * (2 * anp.pi) ** 0.5))

    @staticmethod
    def logR(t, mu, sigma):  # Log SF (Lognormal)
        return anp.log(0.5 - 0.5 * erf((anp.log(t) - mu) / (sigma * 2**0.5)))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter lognormal)
        LL_f = Fit_Lognormal_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Lognormal_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fs(params, T_f, T_rc, force_sigma):
        # log likelihood function (2 parameter lognormal) FORCED SIGMA
        LL_f = Fit_Lognormal_2P.logf(T_f, params[0], force_sigma).sum()
        LL_rc = Fit_Lognormal_2P.logR(T_rc, params[0], force_sigma).sum()
        return -(LL_f + LL_rc)


class Fit_Lognormal_3P:
    """Fits a three parameter Lognormal distribution (mu,sigma,gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), or 'LS' (least squares estimation).
        Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle).

    Returns
    -------
    mu : float
        the fitted Lognormal_3P mu parameter
    sigma : float
        the fitted Lognormal_3P sigma parameter
    gamma : float
        the fitted Lognormal_3P gamma parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    sigma_SE :float
        the standard error (sqrt(variance)) of the parameter
    gamma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_mu_sigma : float
        the covariance between the parameters
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    sigma_upper : float
        the upper CI estimate of the parameter
    sigma_lower : float
        the lower CI estimate of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Lognormal_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    If the fitted gamma parameter is less than 0.01, the Lognormal_3P results
    will be discarded and the Lognormal_2P distribution will be fitted. The
    returned values for gamma and gamma_SE will be 0.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        optimizer=None,
        method="MLE",
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Lognormal_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Lognormal_3P",
            LL_func=Fit_Lognormal_3P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.mu = LS_results.guess[0]
            self.sigma = LS_results.guess[1]
            self.gamma = LS_results.guess[2]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Lognormal_3P",
                LL_func=Fit_Lognormal_3P.LL,
                initial_guess=[
                    LS_results.guess[0],
                    LS_results.guess[1],
                    LS_results.guess[2],
                ],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.mu = MLE_results.scale
            self.sigma = MLE_results.shape
            self.gamma = MLE_results.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        if (
            self.gamma < 0.01
        ):  # If the solver finds that gamma is very near zero then we should have used a Lognormal_2P distribution. Can't proceed with Lognormal_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Lognormal_3P
            lognormal_2P_results = Fit_Lognormal_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.mu = lognormal_2P_results.mu
            self.sigma = lognormal_2P_results.sigma
            self.gamma = 0
            self.mu_SE = lognormal_2P_results.mu_SE
            self.sigma_SE = lognormal_2P_results.sigma_SE
            self.gamma_SE = 0
            self.Cov_mu_sigma = lognormal_2P_results.Cov_mu_sigma
            self.mu_upper = lognormal_2P_results.mu_upper
            self.mu_lower = lognormal_2P_results.mu_lower
            self.sigma_upper = lognormal_2P_results.sigma_upper
            self.sigma_lower = lognormal_2P_results.sigma_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P = [self.mu, self.sigma, self.gamma]

        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P = [self.mu, self.sigma]
            params_3P = [self.mu, self.sigma, self.gamma]
            # here we need to get mu_SE and sigma_SE from the Lognormal_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                # this is to get the gamma_SE. Unfortunately this approach for mu_SE and sigma_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Lognormal_3P.LL)(
                    np.array(tuple(params_3P)),
                    np.array(tuple(failures)),
                    np.array(tuple(right_censored)),
                )
                covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)
                self.mu_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.sigma_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
                self.Cov_mu_sigma = covariance_matrix[0][1]
                self.mu_upper = self.mu + (Z * self.mu_SE)  # Mu can be positive or negative.
                self.mu_lower = self.mu + (-Z * self.mu_SE)
                self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))  # sigma is strictly positive
                self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
                self.gamma_upper = (
                    self.gamma * (np.exp(Z * (self.gamma_SE / self.gamma)))
                )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
                self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Lognormal_3P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer.",
                    ),
                    text_color="red",
                )
                self.mu_SE = 0
                self.sigma_SE = 0
                self.gamma_SE = 0
                self.Cov_mu_sigma = 0
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.sigma_upper = self.sigma
                self.sigma_lower = self.sigma
                self.gamma_upper = self.gamma
                self.gamma_lower = self.gamma

        results_data = {
            "Parameter": ["Mu", "Sigma", "Gamma"],
            "Point Estimate": [self.mu, self.sigma, self.gamma],
            "Standard Error": [self.mu_SE, self.sigma_SE, self.gamma_SE],
            "Lower CI": [self.mu_lower, self.sigma_lower, self.gamma_lower],
            "Upper CI": [self.mu_upper, self.sigma_upper, self.gamma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Lognormal_Distribution(
            mu=self.mu,
            sigma=self.sigma,
            gamma=self.gamma,
            mu_SE=self.mu_SE,
            sigma_SE=self.sigma_SE,
            Cov_mu_sigma=self.Cov_mu_sigma,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Lognormal_3P.LL(params_3P, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Lognormal_3P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Lognormal_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            fig = Lognormal_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01:
                # manually change the legend to reflect that Lognormal_3P was fitted. The default legend in the probability plot thinks Lognormal_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Lognormal_3P\n(μ="
                        + round_and_string(self.mu, dec)
                        + ", σ="
                        + round_and_string(self.sigma, dec)
                        + ", γ="
                        + round_and_string(self.gamma, dec)
                        + ")",
                    ),
                )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, mu, sigma, gamma):  # Log PDF (3 parameter Lognormal)
        return anp.log(
            anp.exp(-0.5 * (((anp.log(t - gamma) - mu) / sigma) ** 2)) / ((t - gamma) * sigma * (2 * anp.pi) ** 0.5),
        )

    @staticmethod
    def logR(t, mu, sigma, gamma):  # Log SF (3 parameter Lognormal)
        return anp.log(0.5 - 0.5 * erf((anp.log(t - gamma) - mu) / (sigma * 2**0.5)))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (3 parameter Lognormal)
        LL_f = Fit_Lognormal_3P.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Lognormal_3P.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Gamma_2P:
    """Fits a two parameter Gamma distribution (alpha,beta) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Gamma_2P alpha parameter
    beta : float
        the fitted Gamma_2P beta parameter
    mu : float
        mu = ln(alpha). Alternate parametrisation (mu, beta) used for the
        confidence intervals.
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    Cov_mu_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Gamma_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    This is a two parameter distribution but it has two parametrisations. These
    are alpha,beta and mu,beta. The alpha,beta parametrisation is reported in
    the results table while the mu,beta parametrisation is accessible from the
    results by name. The reason for this is because the most common
    parametrisation (alpha,beta) should be reported while the less common
    parametrisation (mu,beta) is used by some other software so is provided
    for convenience of comparison. The mu = ln(alpha) relationship is simple
    but this relationship does not extend to the variances or covariances so
    additional calculations are required to find both solutions. The mu,beta
    parametrisation is used for the confidence intervals as it is more stable.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        method="MLE",
        optimizer=None,
        quantiles=None,
        CI_type="time",
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Gamma_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            CI_type=CI_type,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        LS_results = LS_optimization(
            func_name="Gamma_2P",
            LL_func=Fit_Gamma_2P.LL_ab,
            failures=failures,
            right_censored=right_censored,
            method="LS",
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.mu = np.log(self.alpha)
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results_ab = MLE_optimization(
                func_name="Gamma_2P",
                LL_func=Fit_Gamma_2P.LL_ab,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results_ab.scale
            self.mu = np.log(MLE_results_ab.scale)
            self.beta = MLE_results_ab.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results_ab.optimizer

        # confidence interval estimates of parameters
        # this needs to be done in terms of alpha beta (ab) parametrisation
        # and mu beta (mb) parametrisation
        Z = -ss.norm.ppf((1 - CI) / 2)
        params_ab = [self.alpha, self.beta]
        hessian_matrix_ab = hessian(Fit_Gamma_2P.LL_ab)(
            np.array(tuple(params_ab)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix_ab = np.linalg.inv(hessian_matrix_ab)
            self.alpha_SE = abs(covariance_matrix_ab[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix_ab[1][1]) ** 0.5
            self.Cov_alpha_beta = covariance_matrix_ab[0][1]
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

            params_mb = [self.mu, self.beta]
            hessian_matrix_mb = hessian(Fit_Gamma_2P.LL_mb)(
                np.array(tuple(params_mb)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
            )
            covariance_matrix_mb = np.linalg.inv(hessian_matrix_mb)
            self.mu_SE = abs(covariance_matrix_mb[0][0]) ** 0.5
            self.Cov_mu_beta = covariance_matrix_mb[0][1]
            self.mu_upper = self.mu * (np.exp(Z * (self.mu_SE / self.mu)))
            self.mu_lower = self.mu * (np.exp(-Z * (self.mu_SE / self.mu)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Gamma_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.mu_SE = 0
            self.Cov_alpha_beta = 0
            self.Cov_mu_beta = 0
            self.alpha_upper = self.alpha
            self.alpha_lower = self.alpha
            self.beta_upper = self.beta
            self.beta_lower = self.beta
            self.mu_upper = self.mu
            self.mu_lower = self.mu

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Gamma_Distribution(
            alpha=self.alpha,
            mu=self.mu,
            beta=self.beta,
            alpha_SE=self.alpha_SE,
            mu_SE=self.mu_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            Cov_mu_beta=self.Cov_mu_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Gamma_2P.LL_ab(params_ab, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Gamma_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gamma_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Gamma_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI_type=CI_type,
                CI=CI,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf_ab(t, a, b):  # Log PDF (2 parameter Gamma) - alpha, beta parametrisation
        return anp.log(t ** (b - 1)) - anp.log((a**b) * agamma(b)) - (t / a)

    @staticmethod
    def logR_ab(t, a, b):  # Log SF (2 parameter Gamma) - alpha, beta parametrisation
        return anp.log(gammaincc(b, t / a))

    @staticmethod
    def LL_ab(params, T_f, T_rc):
        # log likelihood function (2 parameter Gamma) - alpha, beta parametrisation
        LL_f = Fit_Gamma_2P.logf_ab(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Gamma_2P.logR_ab(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def logf_mb(t, m, b):  # Log PDF (2 parameter Gamma) - mu, beta parametrisation
        return anp.log(t ** (b - 1)) - anp.log((anp.exp(m) ** b) * agamma(b)) - (t / anp.exp(m))

    @staticmethod
    def logR_mb(t, m, b):  # Log SF (2 parameter Gamma) - mu, beta parametrisation
        return anp.log(gammaincc(b, t / anp.exp(m)))

    @staticmethod
    def LL_mb(params, T_f, T_rc):
        # log likelihood function (2 parameter Gamma) - mu, beta parametrisation
        LL_f = Fit_Gamma_2P.logf_mb(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Gamma_2P.logR_mb(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Gamma_3P:
    """Fits a three parameter Gamma distribution (alpha,beta,gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), or 'LS' (least squares estimation).
        Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Gamma_3P alpha parameter
    beta : float
        the fitted Gamma_3P beta parameter
    mu : float
        mu = ln(alpha). Alternate parametrisation (mu, beta) used for the
        confidence intervals.
    gamma : float
        the fitted Gamma_3P gamma parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    mu_SE : float
        the standard error (sqrt(variance)) of the parameter
    gamma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    Cov_mu_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    mu_upper : float
        the upper CI estimate of the parameter
    mu_lower : float
        the lower CI estimate of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Gamma_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    If the fitted gamma parameter is less than 0.01, the Gamma_3P results will
    be discarded and the Gamma_2P distribution will be fitted. The returned
    values for gamma and gamma_SE will be 0.

    This is a three parameter distribution but it has two parametrisations.
    These are alpha,beta,gamma and mu,beta,gamma. The alpha,beta,gamma
    parametrisation is reported in the results table while the mu,beta,gamma
    parametrisation is accessible from the results by name. The reason for this
    is because the most common parametrisation (alpha,beta,gamma) should be
    reported while the less common parametrisation (mu,beta,gamma) is used by
    some other software so is provided for convenience of comparison. The
    mu = ln(alpha) relationship is simple but this relationship does not extend
    to the variances or covariances so additional calculations are required to
    find both solutions. The mu,beta,gamma parametrisation is used for the
    confidence intervals as it is more stable.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        method="MLE",
        quantiles=None,
        CI_type="time",
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Gamma_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            CI_type=CI_type,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Gamma_3P",
            LL_func=Fit_Gamma_3P.LL_abg,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.mu = np.log(self.alpha)
            self.beta = LS_results.guess[1]
            self.gamma = LS_results.guess[2]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results_abg = MLE_optimization(
                func_name="Gamma_3P",
                LL_func=Fit_Gamma_3P.LL_abg,
                initial_guess=[
                    LS_results.guess[0],
                    LS_results.guess[1],
                    LS_results.guess[2],
                ],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results_abg.scale
            self.mu = np.log(MLE_results_abg.scale)
            self.beta = MLE_results_abg.shape
            self.gamma = MLE_results_abg.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results_abg.optimizer

        if (
            self.gamma < 0.01
        ):  # If the solver finds that gamma is very near zero then we should have used a Gamma_2P distribution. Can't proceed with Gamma_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Gamma_3P
            gamma_2P_results = Fit_Gamma_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.alpha = gamma_2P_results.alpha
            self.beta = gamma_2P_results.beta
            self.mu = gamma_2P_results.mu
            self.gamma = 0
            self.alpha_SE = gamma_2P_results.alpha_SE
            self.beta_SE = gamma_2P_results.beta_SE
            self.mu_SE = gamma_2P_results.mu_SE
            self.gamma_SE = 0
            self.Cov_alpha_beta = gamma_2P_results.Cov_alpha_beta
            self.Cov_mu_beta = gamma_2P_results.Cov_mu_beta
            self.alpha_upper = gamma_2P_results.alpha_upper
            self.alpha_lower = gamma_2P_results.alpha_lower
            self.beta_upper = gamma_2P_results.beta_upper
            self.beta_lower = gamma_2P_results.beta_lower
            self.mu_upper = gamma_2P_results.mu_upper
            self.mu_lower = gamma_2P_results.mu_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P_abg = [self.alpha, self.beta, self.gamma]
        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P_ab = [self.alpha, self.beta]
            params_2P_mb = [self.mu, self.beta]
            params_3P_abg = [self.alpha, self.beta, self.gamma]
            # here we need to get alpha_SE and beta_SE from the Gamma_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix_ab = hessian(Fit_Gamma_2P.LL_ab)(
                np.array(tuple(params_2P_ab)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix_ab = np.linalg.inv(hessian_matrix_ab)
                hessian_matrix_mb = hessian(Fit_Gamma_2P.LL_mb)(
                    np.array(tuple(params_2P_mb)),
                    np.array(tuple(failures - self.gamma)),
                    np.array(tuple(right_censored - self.gamma)),
                )
                covariance_matrix_mb = np.linalg.inv(hessian_matrix_mb)

                # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Gamma_3P.LL_abg)(
                    np.array(tuple(params_3P_abg)),
                    np.array(tuple(failures)),
                    np.array(tuple(right_censored)),
                )
                covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)

                self.alpha_SE = abs(covariance_matrix_ab[0][0]) ** 0.5
                self.beta_SE = abs(covariance_matrix_ab[1][1]) ** 0.5
                self.mu_SE = abs(covariance_matrix_mb[0][0]) ** 0.5
                self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
                self.Cov_alpha_beta = covariance_matrix_ab[0][1]
                self.Cov_mu_beta = covariance_matrix_mb[0][1]
                self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
                self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
                self.mu_upper = self.mu * (np.exp(Z * (self.mu_SE / self.mu)))
                self.mu_lower = self.mu * (np.exp(-Z * (self.mu_SE / self.mu)))
                self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
                self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
                self.gamma_upper = self.gamma * (np.exp(Z * (self.gamma_SE / self.gamma)))
                self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Gamma_3P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer.",
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.mu_SE = 0
                self.gamma_SE = 0
                self.Cov_alpha_beta = 0
                self.Cov_mu_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.mu_upper = self.mu
                self.mu_lower = self.mu
                self.beta_upper = self.beta
                self.beta_lower = self.beta
                self.gamma_upper = self.gamma
                self.gamma_lower = self.gamma

        results_data = {
            "Parameter": ["Alpha", "Beta", "Gamma"],
            "Point Estimate": [self.alpha, self.beta, self.gamma],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.gamma_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.gamma_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.gamma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Gamma_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            mu=self.mu,
            gamma=self.gamma,
            alpha_SE=self.alpha_SE,
            mu_SE=self.mu_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            Cov_mu_beta=self.Cov_mu_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Gamma_3P.LL_abg(params_3P_abg, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Gamma_3P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Gamma_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            fig = Gamma_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01:
                # manually change the legend to reflect that Gamma_3P was fitted. The default legend in the probability plot thinks Gamma_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Gamma_3P\n(α="
                        + round_and_string(self.alpha, dec)
                        + ", β="
                        + round_and_string(self.beta, dec)
                        + ", γ="
                        + round_and_string(self.gamma, dec)
                        + ")",
                    ),
                )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf_abg(t, a, b, g):  # Log PDF (3 parameter Gamma) - alpha,beta,gamma
        return anp.log((t - g) ** (b - 1)) - anp.log((a**b) * agamma(b)) - ((t - g) / a)

    @staticmethod
    def logR_abg(t, a, b, g):  # Log SF (3 parameter Gamma) - alpha,beta,gamma
        return anp.log(gammaincc(b, (t - g) / a))

    @staticmethod
    def LL_abg(params, T_f, T_rc):
        # log likelihood function (3 parameter Gamma) - alpha,beta,gamma
        LL_f = Fit_Gamma_3P.logf_abg(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Gamma_3P.logR_abg(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def logf_mbg(t, m, b, g):  # Log PDF (3 parameter Gamma) - mu,beta,gamma
        return anp.log((t - g) ** (b - 1)) - anp.log((anp.exp(m) ** b) * agamma(b)) - ((t - g) / anp.exp(m))

    @staticmethod
    def logR_mbg(t, m, b, g):  # Log SF (3 parameter Gamma) - mu,beta,gamma
        return anp.log(gammaincc(b, (t - g) / anp.exp(m)))

    @staticmethod
    def LL_mbg(params, T_f, T_rc):
        # log likelihood function (3 parameter Gamma) - mu,beta,gamma
        LL_f = Fit_Gamma_3P.logf_mbg(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Gamma_3P.logR_mbg(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Beta_2P:
    """Fits a two parameter Beta distribution (alpha,beta) to the data provided.
    All data must be in the range 0 < x < 1.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        point estimates. Default is None which results in no output. To use
        default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set quantiles as either
        'auto', True, 'default', 'on'. If an array or list is specified then it
        will be used instead of the default array. Any array or list specified
        must contain values between 0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Beta_2P alpha parameter
    beta : float
        the fitted Beta_2P beta parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Beta_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    Confidence intervals on the plots are not provided.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Beta_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Beta_2P",
            LL_func=Fit_Beta_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Beta_2P",
                LL_func=Fit_Beta_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            # for Beta_2P there are actually 2 shape parameters but this just uses the scale and shape nomenclature
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        hessian_matrix = hessian(Fit_Beta_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = covariance_matrix[0][1]
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Beta_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.Cov_alpha_beta = 0
            self.alpha_upper = self.alpha
            self.alpha_lower = self.alpha
            self.beta_upper = self.beta
            self.beta_lower = self.beta

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Beta_Distribution(
            alpha=self.alpha,
            beta=self.beta,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            quantile_data = {
                "Quantile": quantiles,
                "Point Estimate": point_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Point Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Beta_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Beta_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Beta_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Beta_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Beta)
        return anp.log((t ** (a - 1)) * ((1 - t) ** (b - 1))) - anp.log(abeta(a, b))

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Beta)
        return anp.log(1 - betainc(a, b, t))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter beta)
        LL_f = Fit_Beta_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Beta_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Loglogistic_2P:
    """Fits a two parameter Loglogistic distribution (alpha,beta) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), 'LS' (least squares estimation), 'RRX' (Rank
        regression on X), or 'RRY' (Rank regression on Y). LS will perform both
        RRX and RRY and return the better one. Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Loglogistic_2P alpha parameter
    beta : float
        the fitted Loglogistic_2P beta parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Loglogistic_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        method="MLE",
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Loglogistic_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Loglogistic_2P",
            LL_func=Fit_Loglogistic_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Loglogistic_2P",
                LL_func=Fit_Loglogistic_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        hessian_matrix = hessian(Fit_Loglogistic_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.Cov_alpha_beta = covariance_matrix[0][1]
            self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
            self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
            self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
            self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Loglogistic_2P model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.Cov_alpha_beta = 0
            self.alpha_upper = self.alpha
            self.alpha_lower = self.alpha
            self.beta_upper = self.beta
            self.beta_lower = self.beta

        results_data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Loglogistic_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            alpha_SE=self.alpha_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 2
        LL2 = 2 * Fit_Loglogistic_2P.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Loglogistic_2P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Loglogistic_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Loglogistic_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b):  # Log PDF (2 parameter Loglogistic)
        return anp.log(b / a) - (b + 1) * anp.log(t / a) - 2 * anp.log(1 + (t / a) ** -b)

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Loglogistic)
        return -anp.log(1 + (t / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter Loglogistic)
        LL_f = Fit_Loglogistic_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Loglogistic_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Loglogistic_3P:
    """Fits a three parameter Loglogistic distribution (alpha,beta,gamma) to the
    data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
    method : str, optional
        The method used to fit the distribution. Must be either 'MLE' (maximum
        likelihood estimation), or 'LS' (least squares estimation).
        Default is 'MLE'.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the `documentation
        <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [0.01, 0.05, 0.1,..., 0.95, 0.99] set
        quantiles as either 'auto', True, 'default', 'on'.
        If an array or list is specified then it will be used instead of the
        default array. Any array or list specified must contain values between
        0 and 1.
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        probability plot (e.g. color, label, linestyle)

    Returns
    -------
    alpha : float
        the fitted Loglogistic_3P alpha parameter
    beta : float
        the fitted Loglogistic_3P beta parameter
    gamma : float
        the fitted Loglogistic_3P gamma parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    gamma_SE :float
        the standard error (sqrt(variance)) of the parameter
    Cov_alpha_beta : float
        the covariance between the parameters
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    gamma_upper : float
        the upper CI estimate of the parameter
    gamma_lower : float
        the lower CI estimate of the parameter
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    loglik2 : float
        LogLikelihood*-2 (as used in JMP Pro)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    AD : float
        the Anderson Darling (corrected) statistic (as reported by Minitab)
    distribution : object
        a Loglogistic_Distribution object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    quantiles : dataframe
        a pandas dataframe of the quantiles with bounds on time. This is only
        produced if quantiles is not None. Since quantiles defaults to None,
        this output is not normally produced.
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    If the fitting process encounters a problem a warning will be printed. This
    may be caused by the chosen distribution being a very poor fit to the data
    or the data being heavily censored. If a warning is printed, consider trying
    a different optimizer.

    If the fitted gamma parameter is less than 0.01, the Loglogistic_3P results
    will be discarded and the Loglogistic_2P distribution will be fitted. The
    returned values for gamma and gamma_SE will be 0.

    """

    def __init__(
        self,
        failures=None,
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        CI_type="time",
        optimizer=None,
        method="MLE",
        quantiles=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Loglogistic_3P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        CI_type = inputs.CI_type

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Loglogistic_3P",
            LL_func=Fit_Lognormal_3P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
        )

        # least squares method
        if method in ["LS", "RRX", "RRY"]:
            self.alpha = LS_results.guess[0]
            self.beta = LS_results.guess[1]
            self.gamma = LS_results.guess[2]
            self.method = str("Least Squares Estimation (" + LS_results.method + ")")
            self.optimizer = None
        # maximum likelihood method
        elif method == "MLE":
            MLE_results = MLE_optimization(
                func_name="Loglogistic_3P",
                LL_func=Fit_Loglogistic_3P.LL,
                initial_guess=[
                    LS_results.guess[0],
                    LS_results.guess[1],
                    LS_results.guess[2],
                ],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.gamma = MLE_results.gamma
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        if (
            self.gamma < 0.01
        ):  # If the solver finds that gamma is very near zero then we should have used a Loglogistic_2P distribution. Can't proceed with Loglogistic_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Loglogistic_3P
            loglogistic_2P_results = Fit_Loglogistic_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.alpha = loglogistic_2P_results.alpha
            self.beta = loglogistic_2P_results.beta
            self.gamma = 0
            self.alpha_SE = loglogistic_2P_results.alpha_SE
            self.beta_SE = loglogistic_2P_results.beta_SE
            self.gamma_SE = 0
            self.Cov_alpha_beta = loglogistic_2P_results.Cov_alpha_beta
            self.alpha_upper = loglogistic_2P_results.alpha_upper
            self.alpha_lower = loglogistic_2P_results.alpha_lower
            self.beta_upper = loglogistic_2P_results.beta_upper
            self.beta_lower = loglogistic_2P_results.beta_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P = [self.alpha, self.beta, self.gamma]
        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P = [self.alpha, self.beta]
            params_3P = [self.alpha, self.beta, self.gamma]
            # here we need to get alpha_SE and beta_SE from the Loglogistic_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix = hessian(Fit_Loglogistic_2P.LL)(
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Loglogistic_3P.LL)(
                    np.array(tuple(params_3P)),
                    np.array(tuple(failures)),
                    np.array(tuple(right_censored)),
                )
                covariance_matrix_for_gamma = np.linalg.inv(hessian_matrix_for_gamma)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
                self.gamma_SE = abs(covariance_matrix_for_gamma[2][2]) ** 0.5
                self.Cov_alpha_beta = covariance_matrix[0][1]
                self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
                self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
                self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
                self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
                self.gamma_upper = (
                    self.gamma * (np.exp(Z * (self.gamma_SE / self.gamma)))
                )  # here we assume gamma can only be positive as there are bounds placed on it in the optimizer. Minitab assumes positive or negative so bounds are different
                self.gamma_lower = self.gamma * (np.exp(-Z * (self.gamma_SE / self.gamma)))
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + self.optimizer
                        + " optimizer is non-invertable for the Loglogistic_3P model.\n"
                        "Confidence interval estimates of the parameters could not be obtained.\n"
                        "You may want to try fitting the model using a different optimizer.",
                    ),
                    text_color="red",
                )
                self.alpha_SE = 0
                self.beta_SE = 0
                self.gamma_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha
                self.alpha_lower = self.alpha
                self.beta_upper = self.beta
                self.beta_lower = self.beta
                self.gamma_upper = self.gamma
                self.gamma_lower = self.gamma

        results_data = {
            "Parameter": ["Alpha", "Beta", "Gamma"],
            "Point Estimate": [self.alpha, self.beta, self.gamma],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.gamma_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.gamma_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.gamma_upper],
        }
        self.results = pd.DataFrame(
            results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )
        self.distribution = Loglogistic_Distribution(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            alpha_SE=self.alpha_SE,
            beta_SE=self.beta_SE,
            Cov_alpha_beta=self.Cov_alpha_beta,
            CI=CI,
            CI_type=CI_type,
        )

        if quantiles is not None:
            point_estimate = self.distribution.quantile(q=quantiles)
            lower_estimate, upper_estimate = extract_CI(
                dist=self.distribution,
                func="CDF",
                CI_type="time",
                CI=CI,
                CI_y=quantiles,
            )
            quantile_data = {
                "Quantile": quantiles,
                "Lower Estimate": lower_estimate,
                "Point Estimate": point_estimate,
                "Upper Estimate": upper_estimate,
            }
            self.quantiles = pd.DataFrame(
                quantile_data,
                columns=[
                    "Quantile",
                    "Lower Estimate",
                    "Point Estimate",
                    "Upper Estimate",
                ],
            )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Loglogistic_3P.LL(params_3P, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Loglogistic_3P (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Loglogistic_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            fig = Loglogistic_probability_plot(
                failures=failures,
                right_censored=rc,
                __fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01:
                # manually change the legend to reflect that Loglogistic_3P was fitted. The default legend in the probability plot thinks Loglogistic_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Loglogistic_3P\n(α="
                        + round_and_string(self.alpha, dec)
                        + ", β="
                        + round_and_string(self.beta, dec)
                        + ", γ="
                        + round_and_string(self.gamma, dec)
                        + ")",
                    ),
                )
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, g):  # Log PDF (3 parameter Loglogistic)
        return anp.log(b / a) - (b + 1) * anp.log((t - g) / a) - 2 * anp.log(1 + ((t - g) / a) ** -b)

    @staticmethod
    def logR(t, a, b, g):  # Log SF (3 parameter Loglogistic)
        return -anp.log(1 + ((t - g) / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (3 parameter Loglogistic)
        LL_f = Fit_Loglogistic_3P.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Loglogistic_3P.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)

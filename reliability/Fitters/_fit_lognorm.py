from __future__ import annotations

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss
from autograd.differential_operators import hessian
from autograd.scipy.special import erf
from numpy.linalg import LinAlgError

from reliability.Distributions import (
    Lognormal_Distribution,
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
        failures: npt.NDArray[np.float64] | list[float],
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        optimizer=None,
        CI_type: str | None = "time",
        method: str | None = "MLE",
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
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(  # type: ignore
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
                        + str(self.optimizer)
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
            hessian_matrix = hessian(Fit_Lognormal_2P.LL_fs)(  # type: ignore
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
                        + str(self.optimizer)
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
            self.AICc: np.float64 = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
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
            EPSILON = 1e-10
            if frac_censored % 1 < EPSILON:
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
                _fitted_dist_params=self,
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
        failures: npt.NDArray[np.float64] | list[float],
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type: str | None = "time",
        optimizer=None,
        method: str | None = "MLE",
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
        MIN_GAMMA = 0.01
        if (
            self.gamma < MIN_GAMMA
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
            hessian_matrix = hessian(Fit_Lognormal_2P.LL)(  # type: ignore
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                # this is to get the gamma_SE. Unfortunately this approach for mu_SE and sigma_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Lognormal_3P.LL)(  # type: ignore
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
                        + str(self.optimizer)
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
            self.AICc: np.float64 = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
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
            EPSILON = 1e-10
            if frac_censored % 1 < EPSILON:
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
                _fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            THRESHOLD = 0.01
            if self.gamma < THRESHOLD and fig.axes[0].legend_ is not None:
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

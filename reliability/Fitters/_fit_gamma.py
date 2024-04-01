from __future__ import annotations

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from autograd.differential_operators import hessian
from autograd.scipy.special import gamma as agamma
from autograd_gamma import gammaincc
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError

from reliability.Distributions import (
    Gamma_Distribution,
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
        method: str | None="MLE",
        optimizer=None,
        quantiles=None,
        CI_type: str | None="time",
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
        hessian_matrix_ab = hessian(Fit_Gamma_2P.LL_ab)( # type: ignore
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
            hessian_matrix_mb = hessian(Fit_Gamma_2P.LL_mb)( # type: ignore
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
            if self.optimizer is not None:
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
                _fitted_dist_params=self,
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
        method: str | None="MLE",
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
            hessian_matrix_ab = hessian(Fit_Gamma_2P.LL_ab)( # type: ignore
                np.array(tuple(params_2P_ab)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix_ab = np.linalg.inv(hessian_matrix_ab)
                hessian_matrix_mb = hessian(Fit_Gamma_2P.LL_mb)( # type: ignore
                    np.array(tuple(params_2P_mb)),
                    np.array(tuple(failures - self.gamma)),
                    np.array(tuple(right_censored - self.gamma)),
                )
                covariance_matrix_mb = np.linalg.inv(hessian_matrix_mb)

                # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Gamma_3P.LL_abg)( # type: ignore
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
                if self.optimizer is not None:
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
            fig: Figure = Gamma_probability_plot(
                failures=failures,
                right_censored=rc,
                _fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if self.gamma < 0.01 and fig.axes[0].legend_ is not None:
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

from __future__ import annotations

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss
from autograd.differential_operators import hessian
from numpy.linalg import LinAlgError

from reliability.Distributions import (
    Exponential_Distribution,
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
    method : str
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
        failures: npt.NDArray[np.float64],
        right_censored: npt.NDArray[np.float64] | None = None,
        CI=0.95,
        quantiles: bool | str | list | np.ndarray | None = None,
        method: str | None = "MLE",
        optimizer: str | None = None,
    ) -> None:
        inputs = fitters_input_checking(
            dist="Exponential_1P",
            failures=failures,
            method=method,
            right_censored=right_censored,
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
        LS_method: str | None = "LS" if method == "MLE" else method
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
            self.optimizer: str | None = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.Lambda]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(  # type: ignore
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
            if self.optimizer is not None:
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
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])
        self.__CI = CI
        self.__failures = failures
        self.__right_censored = right_censored
        self.__n = n
        self.__quantiles = quantiles

    def print_results(self) -> None:
        """Prints the results of the exponential fitting analysis.

        This method prints various results and statistics obtained from the exponential fitting analysis.
        It includes the confidence interval, analysis method, optimizer used (if applicable),
        number of failures and right censored data points, the results table, goodness of fit statistics,
        and the table of quantiles (if available).

        Returns
        -------
            None

        """
        CI_rounded = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored = len(self.__right_censored) / self.__n * 100
        EPSILON = 1e-10
        if frac_censored % 1 < EPSILON:
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
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.goodness_of_fit.to_string(index=False), "\n")

        if self.__quantiles is not None:
            print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
            print(self.quantiles.to_string(index=False), "\n")

    def plot(self, downsample_scatterplot=True, **kwargs) -> plt.Figure:
        """Generates a probability plot for the fitted exponential distribution.

        Args:
        ----
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

        Returns:
        -------
            plt.Figure: The generated probability plot figure.

        """
        from reliability.Probability_plotting import (
            Exponential_probability_plot_Weibull_Scale,
        )

        rc = None if len(self.__right_censored) == 0 else self.__right_censored
        Exponential_probability_plot_Weibull_Scale(
            failures=self.__failures,
            right_censored=rc,
            _fitted_dist_params=self,
            CI=self.__CI,
            downsample_scatterplot=downsample_scatterplot,
            **kwargs,
        )
        return plt.gcf()

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
        failures: npt.NDArray[np.float64] | list[float],
        right_censored=None,
        CI=0.95,
        quantiles=None,
        method: str | None = "MLE",
        optimizer=None,
    ) -> None:
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
            self.gamma: np.float64 = LS_results.guess[1]
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
            self.optimizer: str | None = MLE_results.optimizer

        # confidence interval estimates of parameters. Uses Exponential_1P because gamma (while optimized) cannot be used in the MLE solution as the solution is unbounded. This is why there are no CI limits on gamma.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params_1P = [self.Lambda]
        params_2P = [self.Lambda, self.gamma]
        hessian_matrix = hessian(Fit_Exponential_1P.LL)(  # type: ignore
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
            if self.optimizer is not None:
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
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])
        self.__CI = CI
        self.__failures = failures
        self.__right_censored = right_censored
        self.__n = n
        self.__quantiles = quantiles

    def print_results(self) -> None:
        """Prints the results of the exponential fitting analysis.

        This method prints various statistics and results obtained from the exponential fitting analysis.
        It includes the confidence interval, analysis method, optimizer (if applicable), number of failures
        and right censored data points, as well as the results and goodness of fit statistics.

        If quantiles are available, it also prints a table of quantiles with the confidence interval bounds.

        Returns
        -------
            None

        """
        CI_rounded = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored = len(self.__right_censored) / self.__n * 100
        EPSILON = 1e-10
        if frac_censored % 1 < EPSILON:
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
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.goodness_of_fit.to_string(index=False), "\n")

        if self.__quantiles is not None:
            print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
            print(self.quantiles.to_string(index=False), "\n")

    def plot(self, downsample_scatterplot=True, **kwargs) -> plt.Figure:
        """Plots the Exponential probability plot on a Weibull Scale.

        Args:
        ----
            downsample_scatterplot (bool, optional): Whether to downsample the scatterplot. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the Exponential_probability_plot_Weibull_Scale function.

        Returns:
        -------
            plt.Figure: The matplotlib Figure object containing the plot.

        """
        from reliability.Probability_plotting import (
            Exponential_probability_plot_Weibull_Scale,
        )

        rc = None if len(self.__right_censored) == 0 else self.__right_censored
        Exponential_probability_plot_Weibull_Scale(
            failures=self.__failures,
            right_censored=rc,
            CI=self.__CI,
            _fitted_dist_params=self,
            downsample_scatterplot=downsample_scatterplot,
            **kwargs,
        )
        return plt.gcf()

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

from __future__ import annotations

from typing import Literal

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss
from autograd import value_and_grad
from autograd.differential_operators import hessian
from numpy.linalg import LinAlgError
from scipy.optimize import minimize

from reliability.Distributions import (
    Competing_Risks_Model,
    DSZI_Model,
    Mixture_Model,
    Weibull_Distribution,
)
from reliability.Probability_plotting import plotting_positions
from reliability.Utils import (
    LS_optimization,
    MLE_optimization,
    anderson_darling,
    colorprint,
    extract_CI,
    fitters_input_checking,
    least_squares,
    round_and_string,
)

anp.seterr("ignore")
dec = 3  # number of decimals to use when rounding fitted parameters in labels

# change pandas display options
pd.options.display.float_format = "{:g}".format  # improves formatting of numbers in dataframe
pd.options.display.max_columns = 9  # shows the dataframe without ... truncation
pd.options.display.width = 200  # prevents wrapping after default 80 characters


class Fit_Weibull_2P:
    """Fits a two parameter Weibull distribution (alpha,beta) to the data provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements if force_beta is not
        specified or at least 1 element if force_beta is specified.
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
    force_beta : float, int, optional
        Used to specify the beta value if you need to force beta to be a certain
        value. Used in ALT probability plotting. Optional input. If specified it
        must be > 0.
    quantiles : bool, str, list, array, None, optional
        quantiles (y-values) to produce a table of quantiles failed with
        lower, point, and upper estimates. Default is None which results in no
        output. To use default array [1, 5, 10,..., 95, 99] set quantiles as
        either 'auto', True, 'default', 'on'.
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
        the fitted Weibull_2P alpha parameter
    beta : float
        the fitted Weibull_2P beta parameter
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
        a Weibull_Distribution object with the parameters of the fitted
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
        CI_type: str | None = "time",
        method: Literal["MLE", "RRX", "RRY", "LS", "NLLS"] = "MLE",
        optimizer=None,
        force_beta=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Weibull_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_beta=force_beta,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_beta = inputs.force_beta
        CI_type = inputs.CI_type
        self.gamma = 0

        # Obtain least squares estimates
        LS_method = "LS" if method == "MLE" else method
        LS_results = LS_optimization(
            func_name="Weibull_2P",
            LL_func=Fit_Weibull_2P.LL,
            failures=failures,
            right_censored=right_censored,
            method=LS_method,
            force_shape=force_beta,
            LL_func_force=Fit_Weibull_2P.LL_fb,
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
                func_name="Weibull_2P",
                LL_func=Fit_Weibull_2P.LL,
                initial_guess=[LS_results.guess[0], LS_results.guess[1]],
                failures=failures,
                right_censored=right_censored,
                optimizer=optimizer,
                force_shape=force_beta,
                LL_func_force=Fit_Weibull_2P.LL_fb,
            )
            self.alpha = MLE_results.scale
            self.beta = MLE_results.shape
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        if force_beta is None:
            hessian_matrix = hessian(Fit_Weibull_2P.LL)(  # type: ignore
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
                        + str(self.optimizer)
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
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

        else:  # this is for when force beta is specified
            hessian_matrix = hessian(Fit_Weibull_2P.LL_fb)(  # type: ignore
                np.array((self.alpha,)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),
                np.array((force_beta,)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
                self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
                self.beta_upper = self.beta
                self.beta_lower = self.beta
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + str(self.optimizer)
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
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
        self.distribution = Weibull_Distribution(
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
        if force_beta is None:
            k = 2
            LL2 = 2 * Fit_Weibull_2P.LL(params, failures, right_censored)
        else:
            k = 1
            LL2 = 2 * Fit_Weibull_2P.LL_fb(params, failures, right_censored, force_beta)
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
                str("Results from Fit_Weibull_2P (" + str(CI_rounded) + "% CI):"),
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
            from reliability.Probability_plotting import Weibull_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Weibull_probability_plot(
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
    def logf(t, a, b):  # Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Weibull)
        return -((t / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (2 parameter weibull)
        LL_f = Fit_Weibull_2P.logf(T_f, params[0], params[1]).sum()
        LL_rc = Fit_Weibull_2P.logR(T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fb(params, T_f, T_rc, force_beta):
        # log likelihood function (2 parameter weibull) FORCED BETA
        LL_f = Fit_Weibull_2P.logf(T_f, params[0], force_beta).sum()
        LL_rc = Fit_Weibull_2P.logR(T_rc, params[0], force_beta).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_2P_grouped:
    r"""Fits a two parameter Weibull distribution (alpha,beta) to the data provided.
    This function is similar to Fit_Weibull_2P however it accepts a dataframe
    which allows for efficient handling of grouped (repeated) data.

    Parameters
    ----------
    dataframe : dataframe
        a pandas dataframe of the appropriate format. See the example in Notes.
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
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. The default optimizer is
        'TNC'. The option to use all these optimizers is not available (as it is
        in all the other Fitters). If the optimizer fails, the initial guess
        will be returned.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    CI_type : str, optional
        This is the confidence bounds on time or reliability shown on the plot.
        Use 'none' to turn off the confidence intervals. Must be either 'time',
        'reliability', or 'none'. Default is 'time'. Some flexibility in names is
        allowed (eg. 't', 'time', 'r', 'rel', 'reliability' are all valid).
    force_beta : float, int, optional
        Used to specify the beta value if you need to force beta to be a certain
        value. Used in ALT probability plotting. Optional input. If specified it
        must be > 0.
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
        the fitted Weibull_2P alpha parameter
    beta : float
        the fitted Weibull_2P beta parameter
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
        a Weibull_Distribution object with the parameters of the fitted
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

    Requirements of the input dataframe:
    The column titles MUST be 'category', 'time', 'quantity'
    The category values MUST be 'F' for failure or 'C' for censored (right
    censored). The time values are the failure or right censored times.
    The quantity is the number of items at that time. This must be specified for
    all values even if the quantity is 1.

    Example of the input dataframe:

    +------------+------------+-----------+
    | category   | time       | quantity  |
    +============+============+===========+
    | F          | 24         | 1         |
    +------------+------------+-----------+
    | F          | 29         | 1         |
    +------------+------------+-----------+
    | F          | 34         | 1         |
    +------------+------------+-----------+
    | F          | 39         | 2         |
    +------------+------------+-----------+
    | F          | 40         | 1         |
    +------------+------------+-----------+
    | F          | 42         | 3         |
    +------------+------------+-----------+
    | F          | 44         | 1         |
    +------------+------------+-----------+
    | C          | 50         | 3         |
    +------------+------------+-----------+
    | C          | 55         | 5         |
    +------------+------------+-----------+
    | C          | 60         | 10        |
    +------------+------------+-----------+

    This is easiest to achieve by importing data from excel. An example of this
    is:

    .. code:: python

        import pandas as pd
        from reliability.Fitters import Fit_Weibull_2P_grouped
        filename = 'C:\\Users\\Current User\\Desktop\\data.xlsx'
        df = pd.read_excel(io=filename)
        Fit_Weibull_2P_grouped(dataframe=df)

    """

    def __init__(
        self,
        dataframe=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        force_beta=None,
        quantiles=None,
        method: Literal["MLE", "RRX", "RRY", "LS"] = "MLE",
        optimizer=None,
        CI_type="time",
        downsample_scatterplot=True,
        **kwargs,
    ):
        if dataframe is None or type(dataframe) is not pd.core.frame.DataFrame:
            raise ValueError(
                'dataframe must be a pandas dataframe with the columns "category" (F for failure or C for censored), "time" (the failure times), and "quantity" (the number of events at each time)',
            )
        for item in dataframe.columns.to_numpy():
            if item not in ["category", "time", "quantity"]:
                raise ValueError(
                    'The titles of the dataframe columns must be: "category" (F for failure or C for censored), "time" (the failure times), and "quantity" (the number of events at each time)',
                )
        categories = dataframe.category.unique()
        for item in categories:
            if item not in ["F", "C"]:
                raise ValueError(
                    'The category column must have values "F" or "C" for failure or censored (right censored) respectively. Other values were detected.',
                )

        # automatically filter out rows with zeros and print warning if zeros have been removed
        dataframe0 = dataframe
        dataframe = dataframe0[dataframe0["time"] > 0]
        if len(dataframe0.time.values) != len(dataframe.time.values):
            colorprint(
                "WARNING: dataframe contained zeros. These have been removed to enable fitting.",
                text_color="red",
            )

        # unpack the dataframe
        failures_df = dataframe[dataframe["category"] == "F"]
        right_censored_df = dataframe[dataframe["category"] == "C"]
        failure_times = failures_df.time.to_numpy()
        failure_qty = failures_df.quantity.to_numpy()
        right_censored_times = right_censored_df.time.to_numpy()
        right_censored_qty = right_censored_df.quantity.to_numpy()

        # recompile the data to get the plotting positions for the initial guess
        failures = np.repeat(failure_times, failure_qty)
        right_censored = np.repeat(right_censored_times, right_censored_qty)

        # perform input error checking for the rest of the inputs
        inputs = fitters_input_checking(
            dist="Weibull_2P",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
            CI=CI,
            quantiles=quantiles,
            force_beta=force_beta,
            CI_type=CI_type,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        method = inputs.method
        optimizer = inputs.optimizer
        quantiles = inputs.quantiles
        force_beta = inputs.force_beta
        CI_type = inputs.CI_type
        self.gamma = 0
        if optimizer not in ["L-BFGS-B", "TNC", "powell", "nelder-mead"]:
            optimizer = "TNC"  # temporary correction for "best" and "all"

        if method == "RRX":
            guess = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRX",
                force_shape=force_beta,
            )
            LS_method = "RRX"
        elif method == "RRY":
            guess = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRY",
                force_shape=force_beta,
            )
            LS_method = "RRY"
        elif method in ["LS", "MLE"]:
            guess_RRX = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRX",
                force_shape=force_beta,
            )
            guess_RRY = least_squares(
                dist="Weibull_2P",
                failures=failures,
                right_censored=right_censored,
                method="RRY",
                force_shape=force_beta,
            )
            if force_beta is not None:
                loglik_RRX = -Fit_Weibull_2P_grouped.LL_fb(
                    guess_RRX,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                    force_beta,
                )
                loglik_RRY = -Fit_Weibull_2P_grouped.LL_fb(
                    guess_RRY,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                    force_beta,
                )
            else:
                loglik_RRX = -Fit_Weibull_2P_grouped.LL(
                    guess_RRX,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                )
                loglik_RRY = -Fit_Weibull_2P_grouped.LL(
                    guess_RRY,
                    failure_times,
                    right_censored_times,
                    failure_qty,
                    right_censored_qty,
                )
            # take the best one
            if abs(loglik_RRX) < abs(loglik_RRY):  # RRX is best
                LS_method = "RRX"
                guess = guess_RRX
            else:  # RRY is best
                LS_method = "RRY"
                guess = guess_RRY

        if method in ["LS", "RRX", "RRY"]:
            self.alpha = guess[0]
            self.beta = guess[1]
            self.method = str("Least Squares Estimation (" + LS_method + ")")
            self.optimizer = None
        elif method == "MLE":
            self.method = "Maximum Likelihood Estimation (MLE)"
            self.optimizer = optimizer
            n = sum(failure_qty) + sum(right_censored_qty)
            k = len(guess)
            initial_guess = guess
            EPSILON = 0.001
            MAX_ITERATIONS = 10
            if force_beta is None:
                bnds = [
                    (0, None),
                    (0, None),
                ]  # bounds on the solution. Helps a lot with stability
                runs = 0
                delta_BIC = 1
                BIC_array = [1000000]

                while delta_BIC > EPSILON and runs < MAX_ITERATIONS:  # exits after BIC convergence or 10 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(Fit_Weibull_2P_grouped.LL),
                        guess,
                        args=(
                            failure_times,
                            right_censored_times,
                            failure_qty,
                            right_censored_qty,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bnds,
                        options={"maxiter": 300},
                    )  # this includes maxiter as TNC often exceeds the default limit of 100
                    params = result.x
                    guess = [params[0], params[1]]
                    LL2 = 2 * Fit_Weibull_2P_grouped.LL(
                        guess,
                        failure_times,
                        right_censored_times,
                        failure_qty,
                        right_censored_qty,
                    )
                    BIC_array.append(np.log(n) * k + LL2)
                    delta_BIC = abs(BIC_array[-1] - BIC_array[-2])
            else:  # force beta is True
                bnds = [(0, None)]  # bounds on the solution. Helps a lot with stability
                runs = 0
                delta_BIC = 1
                BIC_array = [1000000]
                guess = [guess[0]]
                k = len(guess)
                while delta_BIC > EPSILON and runs < MAX_ITERATIONS:  # exits after BIC convergence or 5 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(Fit_Weibull_2P_grouped.LL_fb),
                        guess,
                        args=(
                            failure_times,
                            right_censored_times,
                            failure_qty,
                            right_censored_qty,
                            force_beta,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bnds,
                        options={"maxiter": 300},
                    )
                    params = result.x
                    guess = [params[0]]
                    LL2 = 2 * Fit_Weibull_2P_grouped.LL_fb(
                        guess,
                        failure_times,
                        right_censored_times,
                        failure_qty,
                        right_censored_qty,
                        force_beta,
                    )
                    BIC_array.append(np.log(n) * k + LL2)
                    delta_BIC = abs(BIC_array[-1] - BIC_array[-2])

            # check if the optimizer was successful. If it failed then return the initial guess with a warning
            if result.success is True:
                params = result.x
                if force_beta is None:
                    self.alpha = params[0]
                    self.beta = params[1]
                else:
                    self.alpha = params[0]
                    self.beta = force_beta
            else:  # return the initial guess with a warning
                colorprint(
                    "WARNING: MLE estimates failed for Fit_Weibull_2P_grouped. The least squares estimates have been returned. These results may not be as accurate as MLE. You may want to try another optimzer from 'L-BFGS-B','TNC','powell','nelder-mead'.",
                    text_color="red",
                )
                if force_beta is None:
                    self.alpha = initial_guess[0]
                    self.beta = initial_guess[1]
                else:
                    self.alpha = initial_guess[0]
                    self.beta = force_beta

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta]
        if force_beta is None:
            hessian_matrix = hessian(Fit_Weibull_2P_grouped.LL)(  # type: ignore
                np.array(tuple(params)),
                np.array(tuple(failure_times)),
                np.array(tuple(right_censored_times)),
                np.array(tuple(failure_qty)),
                np.array(tuple(right_censored_qty)),
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
                        + str(self.optimizer)
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
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

        else:  # this is for when force beta is specified
            hessian_matrix = hessian(Fit_Weibull_2P_grouped.LL_fb)(  # type: ignore
                np.array((self.alpha,)),
                np.array(tuple(failure_times)),
                np.array(tuple(right_censored_times)),
                np.array(tuple(failure_qty)),
                np.array(tuple(right_censored_qty)),
                np.array((force_beta,)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
                self.beta_SE = 0
                self.Cov_alpha_beta = 0
                self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
                self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
                self.beta_upper = self.beta
                self.beta_lower = self.beta
            except LinAlgError:
                # this exception is rare but can occur with some optimizers
                colorprint(
                    str(
                        "WARNING: The hessian matrix obtained using the "
                        + str(self.optimizer)
                        + " optimizer is non-invertable for the Weibull_2P model.\n"
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
        self.distribution = Weibull_Distribution(
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
        n = sum(failure_qty) + sum(right_censored_qty)
        if force_beta is None:
            k = 2
            LL2 = 2 * Fit_Weibull_2P_grouped.LL(
                params,
                failure_times,
                right_censored_times,
                failure_qty,
                right_censored_qty,
            )
        else:
            k = 1
            LL2 = 2 * Fit_Weibull_2P_grouped.LL_fb(
                params,
                failure_times,
                right_censored_times,
                failure_qty,
                right_censored_qty,
                force_beta,
            )
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
            frac_censored = sum(right_censored_qty) / n * 100
            EPSILON = 1e-10
            if frac_censored % 1 < EPSILON:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_2P_grouped (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method:", self.method)
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(sum(failure_qty)) + "/" + str(sum(right_censored_qty))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.goodness_of_fit.to_string(index=False), "\n")

            if quantiles is not None:
                print(str("Table of quantiles (" + str(CI_rounded) + "% CI bounds on time):"))
                print(self.quantiles.to_string(index=False), "\n")

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Weibull_probability_plot(
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
    def logf(t, a, b):  # Log PDF (2 parameter Weibull)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b

    @staticmethod
    def logR(t, a, b):  # Log SF (2 parameter Weibull)
        return -((t / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc, Q_f, Q_rc):
        # log likelihood function (2 parameter weibull) ==> T is for time, Q is for quantity
        LL_f = (Fit_Weibull_2P_grouped.logf(T_f, params[0], params[1]) * Q_f).sum()
        LL_rc = (Fit_Weibull_2P_grouped.logR(T_rc, params[0], params[1]) * Q_rc).sum()
        return -(LL_f + LL_rc)

    @staticmethod
    def LL_fb(params, T_f, T_rc, Q_f, Q_rc, force_beta):
        # log likelihood function (2 parameter weibull) FORCED BETA  ==> T is for time, Q is for quantity
        LL_f = (Fit_Weibull_2P_grouped.logf(T_f, params[0], force_beta) * Q_f).sum()
        LL_rc = (Fit_Weibull_2P_grouped.logR(T_rc, params[0], force_beta) * Q_rc).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_3P:
    """Fits a three parameter Weibull distribution (alpha,beta,gamma) to the data
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 3 elements
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
        the fitted Weibull_3P alpha parameter
    beta : float
        the fitted Weibull_3P beta parameter
    gamma : float
        the fitted Weibull_3P gamma parameter
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
        a Weibull_Distribution object with the parameters of the fitted
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

    If the fitted gamma parameter is less than 0.01, the Weibull_3P results will
    be discarded and the Weibull_2P distribution will be fitted. The returned
    values for gamma and gamma_SE will be 0.

    """

    def __init__(
        self,
        failures: npt.NDArray[np.float64] | list[float],
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        quantiles=None,
        CI_type="time",
        optimizer=None,
        method: Literal["MLE", "RRX", "RRY", "LS"] = "MLE",
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Weibull_3P",
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
            func_name="Weibull_3P",
            LL_func=Fit_Weibull_3P.LL,
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
                func_name="Weibull_3P",
                LL_func=Fit_Weibull_3P.LL,
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
        MIN_GAMMA = 0.01
        if (
            self.gamma < MIN_GAMMA
        ):  # If the solver finds that gamma is very near zero then we should have used a Weibull_2P distribution. Can't proceed with Weibull_3P as the confidence interval calculations for gamma result in nan (Zero division error). Need to recalculate everything as the SE values will be incorrect for Weibull_3P
            weibull_2P_results = Fit_Weibull_2P(
                failures=failures,
                right_censored=right_censored,
                show_probability_plot=False,
                print_results=False,
                CI=CI,
            )
            self.alpha = weibull_2P_results.alpha
            self.beta = weibull_2P_results.beta
            self.gamma = 0
            self.alpha_SE = weibull_2P_results.alpha_SE
            self.beta_SE = weibull_2P_results.beta_SE
            self.gamma_SE = 0
            self.Cov_alpha_beta = weibull_2P_results.Cov_alpha_beta
            self.alpha_upper = weibull_2P_results.alpha_upper
            self.alpha_lower = weibull_2P_results.alpha_lower
            self.beta_upper = weibull_2P_results.beta_upper
            self.beta_lower = weibull_2P_results.beta_lower
            self.gamma_upper = 0
            self.gamma_lower = 0
            params_3P = [self.alpha, self.beta, self.gamma]
        else:
            # confidence interval estimates of parameters
            Z = -ss.norm.ppf((1 - CI) / 2)
            params_2P = [self.alpha, self.beta]
            params_3P = [self.alpha, self.beta, self.gamma]
            # here we need to get alpha_SE and beta_SE from the Weibull_2P by providing an adjusted dataset (adjusted for gamma)
            hessian_matrix = hessian(Fit_Weibull_2P.LL)(  # type: ignore
                np.array(tuple(params_2P)),
                np.array(tuple(failures - self.gamma)),
                np.array(tuple(right_censored - self.gamma)),
            )
            try:
                covariance_matrix = np.linalg.inv(hessian_matrix)
                # this is to get the gamma_SE. Unfortunately this approach for alpha_SE and beta_SE give SE values that are very large resulting in incorrect CI plots. This is the same method used by Reliasoft
                hessian_matrix_for_gamma = hessian(Fit_Weibull_3P.LL)(  # type: ignore
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
                        + str(self.optimizer)
                        + " optimizer is non-invertable for the Weibull_3P model.\n"
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
        self.distribution = Weibull_Distribution(
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
        LL2 = 2 * Fit_Weibull_3P.LL(params_3P, failures, right_censored)
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
                str("Results from Fit_Weibull_3P (" + str(CI_rounded) + "% CI):"),
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
            from reliability.Probability_plotting import Weibull_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            fig = Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                _fitted_dist_params=self,
                CI=CI,
                CI_type=CI_type,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            MIN_GAMMA = 0.01
            if self.gamma < MIN_GAMMA and fig.axes[0].legend_ is not None:
                # manually change the legend to reflect that Weibull_3P was fitted. The default legend in the probability plot thinks Weibull_2P was fitted when gamma=0
                fig.axes[0].legend_.get_texts()[0].set_text(
                    str(
                        "Fitted Weibull_3P\n(α="
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
    def logf(t, a, b, g):  # Log PDF (3 parameter Weibull)
        return (b - 1) * anp.log((t - g) / a) + anp.log(b / a) - ((t - g) / a) ** b

    @staticmethod
    def logR(t, a, b, g):  # Log SF (3 parameter Weibull)
        return -(((t - g) / a) ** b)

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (3 parameter Weibull)
        LL_f = Fit_Weibull_3P.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Weibull_3P.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_Mixture:
    """Fits a mixture of two Weibull_2P distributions (this does not fit the gamma
    parameter). Right censoring is supported, though care should be taken to
    ensure that there still appears to be two groups when plotting only the
    failure data. A second group cannot be made from a mostly or totally
    censored set of samples. Use this model when you think there are multiple
    failure modes acting to create the failure data.

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 4 failures,
        but it is highly recommended to use another model if you have less than
        20 failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
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
    alpha_1 : float
        the fitted Weibull_2P alpha parameter for the first (left) group
    beta_1 : float
        the fitted Weibull_2P beta parameter for the first (left) group
    alpha_2 : float
        the fitted Weibull_2P alpha parameter for the second (right) group
    beta_2 : float
        the fitted Weibull_2P beta parameter for the second (right) group
    proportion_1 : float
        the fitted proportion of the first (left) group
    proportion_2 : float
        the fitted proportion of the second (right) group. Same as
        1-proportion_1
    alpha_1_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_1_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_2_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_2_SE :float
        the standard error (sqrt(variance)) of the parameter
    proportion_1_SE : float
        the standard error (sqrt(variance)) of the parameter
    alpha_1_upper : float
        the upper CI estimate of the parameter
    alpha_1_lower : float
        the lower CI estimate of the parameter
    alpha_2_upper : float
        the upper CI estimate of the parameter
    alpha_2_lower : float
        the lower CI estimate of the parameter
    beta_1_upper : float
        the upper CI estimate of the parameter
    beta_1_lower : float
        the lower CI estimate of the parameter
    beta_2_upper : float
        the upper CI estimate of the parameter
    beta_2_lower : float
        the lower CI estimate of the parameter
    proportion_1_upper : float
        the upper CI estimate of the parameter
    proportion_1_lower : float
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
        a Mixture_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is different to the Weibull Competing Risks as the overall Survival
    Function is the sum of the individual Survival Functions multiplied by a
    proportion rather than being the product as is the case in the Weibull
    Competing Risks Model.

    Mixture Model: :math:`SF_{model} = (proportion_1 x SF_1) +
    ((1-proportion_1) x SF_2)`

    Competing Risks Model: :math:`SF_{model} = SF_1 x SF_2`

    Similar to the competing risks model, you can use this model when you think
    there are multiple failure modes acting to create the failure data.

    Whilst some failure modes may not be fitted as well by a Weibull
    distribution as they may be by another distribution, it is unlikely that a
    mixture of data from two distributions (particularly if they are
    overlapping) will be fitted noticeably better by other types of mixtures
    than would be achieved by a Weibull mixture. For this reason, other types
    of mixtures are not implemented.

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
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Weibull_Mixture",
            failures=failures,
            right_censored=right_censored,
            CI=CI,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        n = len(failures) + len(right_censored)
        _, y = plotting_positions(failures=failures, right_censored=right_censored)  # this is only used to find AD

        # this algorithm is used to estimate the dividing line between the two groups
        # firstly it fits a gaussian kde to the histogram
        # then it draws two straight lines from the highest peak of the kde down to the lower and upper bounds of the failures
        # the dividing line is the point where the difference between the kde and the straight lines is greatest
        max_failures = max(failures)
        min_failures = min(failures)
        gkde = ss.gaussian_kde(failures)
        delta = max_failures - min_failures
        x_kde = np.linspace(min_failures - delta / 5, max_failures + delta / 5, 100)
        y_kde = gkde.evaluate(x_kde)
        peak_y = max(y_kde)
        peak_x = x_kde[np.where(y_kde == peak_y)][0]

        left_x = min_failures
        left_y = gkde.evaluate(left_x)
        left_m = (peak_y - left_y) / (peak_x - left_x)
        left_c = -left_m * left_x + left_y
        left_line_x = np.linspace(left_x, peak_x, 100)
        left_line_y = left_m * left_line_x + left_c  # y=mx+c
        left_kde = gkde.evaluate(left_line_x)
        left_diff = abs(left_line_y - left_kde)
        left_diff_max = max(left_diff)
        left_div_line = left_line_x[np.where(left_diff == left_diff_max)][0]

        right_x = max_failures
        right_y = gkde.evaluate(right_x)
        right_m = (right_y - peak_y) / (right_x - peak_x)
        right_c = -right_m * right_x + right_y
        right_line_x = np.linspace(peak_x, right_x, 100)
        right_line_y = right_m * right_line_x + right_c  # y=mx+c
        right_kde = gkde.evaluate(right_line_x)
        right_diff = abs(right_line_y - right_kde)
        right_diff_max = max(right_diff)
        right_div_line = right_line_x[np.where(right_diff == right_diff_max)][0]

        dividing_line = left_div_line if left_diff_max > right_diff_max else right_div_line

        number_of_items_in_group_1 = len(np.where(failures < dividing_line)[0])
        number_of_items_in_group_2 = len(failures) - number_of_items_in_group_1
        MIN_GROUP_ITEMS = 2
        if number_of_items_in_group_1 < MIN_GROUP_ITEMS:
            failures_sorted = np.sort(failures)
            dividing_line = (
                failures_sorted[1] + failures_sorted[2]
            ) / 2  # adjusts the dividing line in case there aren't enough failures in the first group

        if number_of_items_in_group_2 < MIN_GROUP_ITEMS:
            failures_sorted = np.sort(failures)
            dividing_line = (
                failures_sorted[-2] + failures_sorted[-3]
            ) / 2  # adjusts the dividing line in case there aren't enough failures in the second group

        # this is the point at which data is assigned to one group or another for the purpose of generating the initial guess
        GROUP_1_failures = []
        GROUP_2_failures = []
        GROUP_1_right_cens = []
        GROUP_2_right_cens = []
        for item in failures:
            if item < dividing_line:
                GROUP_1_failures.append(item)
            else:
                GROUP_2_failures.append(item)
        for item in right_censored:
            if item < dividing_line:
                GROUP_1_right_cens.append(item)
            else:
                GROUP_2_right_cens.append(item)

        # get inputs for the guess by fitting a weibull to each of the groups with their respective censored data
        group_1_estimates = Fit_Weibull_2P(
            failures=GROUP_1_failures,
            right_censored=GROUP_1_right_cens,
            show_probability_plot=False,
            print_results=False,
            optimizer=optimizer,
        )
        group_2_estimates = Fit_Weibull_2P(
            failures=GROUP_2_failures,
            right_censored=GROUP_2_right_cens,
            show_probability_plot=False,
            print_results=False,
            optimizer=optimizer,
        )
        # proportion guess
        p_guess = (len(GROUP_1_failures) + len(GROUP_1_right_cens)) / n
        guess = [
            group_1_estimates.alpha,
            group_1_estimates.beta,
            group_2_estimates.alpha,
            group_2_estimates.beta,
            p_guess,
        ]  # A1,B1,A2,B2,P

        # solve it
        MLE_results = MLE_optimization(
            func_name="Weibull_mixture",
            LL_func=Fit_Weibull_Mixture.LL,
            initial_guess=guess,
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha_1 = MLE_results.alpha_1
        self.beta_1 = MLE_results.beta_1
        self.alpha_2 = MLE_results.alpha_2
        self.beta_2 = MLE_results.beta_2
        self.proportion_1 = MLE_results.proportion_1
        self.proportion_2 = MLE_results.proportion_2
        self.optimizer = MLE_results.optimizer
        dist_1 = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1)
        dist_2 = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2)
        self.distribution = Mixture_Model(
            distributions=[dist_1, dist_2],
            proportions=[self.proportion_1, self.proportion_2],
        )

        params = [
            self.alpha_1,
            self.beta_1,
            self.alpha_2,
            self.beta_2,
            self.proportion_1,
        ]
        LL2 = 2 * Fit_Weibull_Mixture.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        k = 5
        if n - k - 1 > 0:
            self.AICc: np.float64 = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_Mixture.LL)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_1_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_1_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.alpha_2_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.beta_2_SE = abs(covariance_matrix[3][3]) ** 0.5
            self.proportion_1_SE = abs(covariance_matrix[4][4]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + str(self.optimizer)
                    + " optimizer is non-invertable for the Weibull_Mixture model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_1_SE = 0
            self.beta_1_SE = 0
            self.alpha_2_SE = 0
            self.beta_2_SE = 0
            self.proportion_1_SE = 0

        self.alpha_1_upper = self.alpha_1 * (np.exp(Z * (self.alpha_1_SE / self.alpha_1)))
        self.alpha_1_lower = self.alpha_1 * (np.exp(-Z * (self.alpha_1_SE / self.alpha_1)))
        self.beta_1_upper = self.beta_1 * (np.exp(Z * (self.beta_1_SE / self.beta_1)))
        self.beta_1_lower = self.beta_1 * (np.exp(-Z * (self.beta_1_SE / self.beta_1)))
        self.alpha_2_upper = self.alpha_2 * (np.exp(Z * (self.alpha_2_SE / self.alpha_2)))
        self.alpha_2_lower = self.alpha_2 * (np.exp(-Z * (self.alpha_2_SE / self.alpha_2)))
        self.beta_2_upper = self.beta_2 * (np.exp(Z * (self.beta_2_SE / self.beta_2)))
        self.beta_2_lower = self.beta_2 * (np.exp(-Z * (self.beta_2_SE / self.beta_2)))
        self.proportion_1_upper = self.proportion_1 / (
            self.proportion_1
            + (1 - self.proportion_1)
            * (np.exp(-Z * self.proportion_1_SE / (self.proportion_1 * (1 - self.proportion_1))))
        )
        # ref: http://reliawiki.org/index.php/The_Mixed_Weibull_Distribution
        self.proportion_1_lower = self.proportion_1 / (
            self.proportion_1
            + (1 - self.proportion_1)
            * (np.exp(Z * self.proportion_1_SE / (self.proportion_1 * (1 - self.proportion_1))))
        )

        Data = {
            "Parameter": ["Alpha 1", "Beta 1", "Alpha 2", "Beta 2", "Proportion 1"],
            "Point Estimate": [
                self.alpha_1,
                self.beta_1,
                self.alpha_2,
                self.beta_2,
                self.proportion_1,
            ],
            "Standard Error": [
                self.alpha_1_SE,
                self.beta_1_SE,
                self.alpha_2_SE,
                self.beta_2_SE,
                self.proportion_1_SE,
            ],
            "Lower CI": [
                self.alpha_1_lower,
                self.beta_1_lower,
                self.alpha_2_lower,
                self.beta_2_lower,
                self.proportion_1_lower,
            ],
            "Upper CI": [
                self.alpha_1_upper,
                self.beta_1_upper,
                self.alpha_2_upper,
                self.beta_2_upper,
                self.proportion_1_upper,
            ],
        }
        self.results = pd.DataFrame(
            Data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        self.AD: float = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])

        if print_results is True:
            CI_rounded: float = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored: float = len(right_censored) / n * 100
            EPSILON = 1e-10
            if frac_censored % 1 < EPSILON:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Weibull_Mixture (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
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

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                show_fitted_distribution=False,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull MM "
                    + round_and_string(self.proportion_1, dec)
                    + r" ($\alpha_1=$"
                    + round_and_string(self.alpha_1, dec)
                    + r", $\beta_1=$"
                    + round_and_string(self.beta_1, dec)
                    + ")+\n                             "
                    + round_and_string(self.proportion_2, dec)
                    + r" ($\alpha_2=$"
                    + round_and_string(self.alpha_2, dec)
                    + r", $\beta_2=$"
                    + round_and_string(self.beta_2, dec)
                    + ")",
                )
            xvals = np.logspace(np.log10(min(failures)) - 3, np.log10(max(failures)) + 1, 1000)
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title("Probability Plot\nWeibull Mixture CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a1, b1, a2, b2, p):  # Log Mixture PDF (2 parameter Weibull)
        return anp.log(
            p * ((b1 * t ** (b1 - 1)) / (a1**b1)) * anp.exp(-((t / a1) ** b1))
            + (1 - p) * ((b2 * t ** (b2 - 1)) / (a2**b2)) * anp.exp(-((t / a2) ** b2)),
        )

    @staticmethod
    def logR(t, a1, b1, a2, b2, p):  # Log Mixture SF (2 parameter Weibull)
        return anp.log(p * anp.exp(-((t / a1) ** b1)) + (1 - p) * anp.exp(-((t / a2) ** b2)))

    @staticmethod
    def LL(params, T_f, T_rc):
        # Log Mixture Likelihood function (2 parameter weibull)
        LL_f = Fit_Weibull_Mixture.logf(T_f, params[0], params[1], params[2], params[3], params[4]).sum()
        LL_rc = Fit_Weibull_Mixture.logR(T_rc, params[0], params[1], params[2], params[3], params[4]).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_CR:
    """Fits a Weibull Competing Risks Model consisting of two Weibull_2P
    distributions (this does not fit the gamma parameter). Similar to the
    mixture model, you can use this model when you think there are multiple
    failure modes acting to create the failure data.

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 4 failures,
        but it is highly recommended to use another model if you have less than
        20 failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
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
    alpha_1 : float
        the fitted Weibull_2P alpha parameter for the first distribution
    beta_1 : float
        the fitted Weibull_2P beta parameter for the first distribution
    alpha_2 : float
        the fitted Weibull_2P alpha parameter for the second distribution
    beta_2 : float
        the fitted Weibull_2P beta parameter for the second distribution
    alpha_1_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_1_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_2_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_2_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_1_upper : float
        the upper CI estimate of the parameter
    alpha_1_lower : float
        the lower CI estimate of the parameter
    alpha_2_upper : float
        the upper CI estimate of the parameter
    alpha_2_lower : float
        the lower CI estimate of the parameter
    beta_1_upper : float
        the upper CI estimate of the parameter
    beta_1_lower : float
        the lower CI estimate of the parameter
    beta_2_upper : float
        the upper CI estimate of the parameter
    beta_2_lower : float
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
        a Competing_Risks_Model object with the parameters of the fitted
        distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
    probability_plot : object
        the axes handle for the probability plot. This is only returned if
        show_probability_plot = True

    Notes
    -----
    This is different to the Weibull Mixture model as the overall Survival
    Function is the product of the individual Survival Functions rather than
    being the sum as is the case in the Weibull Mixture Model.

    Mixture Model: :math:`SF_{model} = (proportion_1 x SF_1) +
    ((1-proportion_1) x SF_2)`

    Competing Risks Model: :math:`SF_{model} = SF_1 x SF_2`

    Whilst some failure modes may not be fitted as well by a Weibull
    distribution as they may be by another distribution, it is unlikely that
    data from a competing risks model will be fitted noticeably better by other
    types of competing risks models than would be achieved by a Weibull
    Competing Risks model. For this reason, other types of competing risks
    models are not implemented.

    """

    def __init__(
        self,
        failures: npt.NDArray[np.float64] | list[float],
        right_censored=None,
        show_probability_plot=True,
        print_results=True,
        CI=0.95,
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Weibull_CR",
            failures=failures,
            right_censored=right_censored,
            CI=CI,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        n = len(failures) + len(right_censored)
        _, y = plotting_positions(failures=failures, right_censored=right_censored)  # this is only used to find AD

        # this algorithm is used to estimate the dividing line between the two groups
        # firstly it fits a gaussian kde to the histogram
        # then it draws two straight lines from the highest peak of the kde down to the lower and upper bounds of the failures
        # the dividing line is the point where the difference between the kde and the straight lines is greatest
        max_failures = max(failures)
        min_failures = min(failures)
        gkde = ss.gaussian_kde(failures)
        delta = max_failures - min_failures
        x_kde = np.linspace(min_failures - delta / 5, max_failures + delta / 5, 100)
        y_kde = gkde.evaluate(x_kde)
        peak_y = max(y_kde)
        peak_x = x_kde[np.where(y_kde == peak_y)][0]

        left_x = min_failures
        left_y = gkde.evaluate(left_x)
        left_m = (peak_y - left_y) / (peak_x - left_x)
        left_c = -left_m * left_x + left_y
        left_line_x = np.linspace(left_x, peak_x, 100)
        left_line_y = left_m * left_line_x + left_c  # y=mx+c
        left_kde = gkde.evaluate(left_line_x)
        left_diff = abs(left_line_y - left_kde)
        left_diff_max = max(left_diff)
        left_div_line = left_line_x[np.where(left_diff == left_diff_max)][0]

        right_x = max_failures
        right_y = gkde.evaluate(right_x)
        right_m = (right_y - peak_y) / (right_x - peak_x)
        right_c = -right_m * right_x + right_y
        right_line_x = np.linspace(peak_x, right_x, 100)
        right_line_y = right_m * right_line_x + right_c  # y=mx+c
        right_kde = gkde.evaluate(right_line_x)
        right_diff = abs(right_line_y - right_kde)
        right_diff_max = max(right_diff)
        right_div_line = right_line_x[np.where(right_diff == right_diff_max)][0]

        dividing_line = left_div_line if left_diff_max > right_diff_max else right_div_line

        number_of_items_in_group_1 = len(np.where(failures < dividing_line)[0])
        number_of_items_in_group_2 = len(failures) - number_of_items_in_group_1
        MIN_GROUP_ITEMS = 2
        if number_of_items_in_group_1 < MIN_GROUP_ITEMS:
            failures_sorted = np.sort(failures)
            dividing_line = (
                failures_sorted[1] + failures_sorted[2]
            ) / 2  # adjusts the dividing line in case there aren't enough failures in the first group
        if number_of_items_in_group_2 < MIN_GROUP_ITEMS:
            failures_sorted = np.sort(failures)
            dividing_line = (
                failures_sorted[-2] + failures_sorted[-3]
            ) / 2  # adjusts the dividing line in case there aren't enough failures in the second group

        # this is the point at which data is assigned to one group or another for the purpose of generating the initial guess
        GROUP_1_failures = []
        GROUP_2_failures = []
        GROUP_1_right_cens = []
        GROUP_2_right_cens = []
        for item in failures:
            if item < dividing_line:
                GROUP_1_failures.append(item)
            else:
                GROUP_2_failures.append(item)
        for item in right_censored:
            if item < dividing_line:
                GROUP_1_right_cens.append(item)
            else:
                GROUP_2_right_cens.append(item)

        # get inputs for the guess by fitting a weibull to each of the groups with their respective censored data
        group_1_estimates = Fit_Weibull_2P(
            failures=GROUP_1_failures,
            right_censored=GROUP_1_right_cens,
            show_probability_plot=False,
            print_results=False,
        )
        group_2_estimates = Fit_Weibull_2P(
            failures=GROUP_2_failures,
            right_censored=GROUP_2_right_cens,
            show_probability_plot=False,
            print_results=False,
        )
        guess = [
            group_1_estimates.alpha,
            group_1_estimates.beta,
            group_2_estimates.alpha,
            group_2_estimates.beta,
        ]  # A1,B1,A2,B2

        # solve it
        MLE_results = MLE_optimization(
            func_name="Weibull_CR",
            LL_func=Fit_Weibull_CR.LL,
            initial_guess=guess,
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha_1 = MLE_results.alpha_1
        self.beta_1 = MLE_results.beta_1
        self.alpha_2 = MLE_results.alpha_2
        self.beta_2 = MLE_results.beta_2
        self.optimizer = MLE_results.optimizer
        dist_1 = Weibull_Distribution(alpha=self.alpha_1, beta=self.beta_1)
        dist_2 = Weibull_Distribution(alpha=self.alpha_2, beta=self.beta_2)
        self.distribution = Competing_Risks_Model(distributions=[dist_1, dist_2])

        params = [self.alpha_1, self.beta_1, self.alpha_2, self.beta_2]
        k = 4
        LL2 = 2 * Fit_Weibull_CR.LL(params, failures, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc: np.float64 = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        hessian_matrix = hessian(Fit_Weibull_CR.LL)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_1_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_1_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.alpha_2_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.beta_2_SE = abs(covariance_matrix[3][3]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + str(self.optimizer)
                    + " optimizer is non-invertable for the Weibull_CR model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_1_SE = 0
            self.beta_1_SE = 0
            self.alpha_2_SE = 0
            self.beta_2_SE = 0

        self.alpha_1_upper = self.alpha_1 * (np.exp(Z * (self.alpha_1_SE / self.alpha_1)))
        self.alpha_1_lower = self.alpha_1 * (np.exp(-Z * (self.alpha_1_SE / self.alpha_1)))
        self.beta_1_upper = self.beta_1 * (np.exp(Z * (self.beta_1_SE / self.beta_1)))
        self.beta_1_lower = self.beta_1 * (np.exp(-Z * (self.beta_1_SE / self.beta_1)))
        self.alpha_2_upper = self.alpha_2 * (np.exp(Z * (self.alpha_2_SE / self.alpha_2)))
        self.alpha_2_lower = self.alpha_2 * (np.exp(-Z * (self.alpha_2_SE / self.alpha_2)))
        self.beta_2_upper = self.beta_2 * (np.exp(Z * (self.beta_2_SE / self.beta_2)))
        self.beta_2_lower = self.beta_2 * (np.exp(-Z * (self.beta_2_SE / self.beta_2)))

        Data = {
            "Parameter": ["Alpha 1", "Beta 1", "Alpha 2", "Beta 2"],
            "Point Estimate": [self.alpha_1, self.beta_1, self.alpha_2, self.beta_2],
            "Standard Error": [
                self.alpha_1_SE,
                self.beta_1_SE,
                self.alpha_2_SE,
                self.beta_2_SE,
            ],
            "Lower CI": [
                self.alpha_1_lower,
                self.beta_1_lower,
                self.alpha_2_lower,
                self.beta_2_lower,
            ],
            "Upper CI": [
                self.alpha_1_upper,
                self.beta_1_upper,
                self.alpha_2_upper,
                self.beta_2_upper,
            ],
        }
        self.results = pd.DataFrame(
            Data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

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
                str("Results from Fit_Weibull_CR (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
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

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                show_fitted_distribution=False,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull CR "
                    + r" ($\alpha_1=$"
                    + round_and_string(self.alpha_1, dec)
                    + r", $\beta_1=$"
                    + round_and_string(self.beta_1, dec)
                    + ") x\n                            "
                    + r" ($\alpha_2=$"
                    + round_and_string(self.alpha_2, dec)
                    + r", $\beta_2=$"
                    + round_and_string(self.beta_2, dec)
                    + ")",
                )
            xvals = np.logspace(np.log10(min(failures)) - 3, np.log10(max(failures)) + 1, 1000)
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            plt.title("Probability Plot\nWeibull Competing Risks CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a1, b1, a2, b2):  # Log PDF (Competing Risks)
        return anp.log(
            -(-(b2 * (t / a2) ** b2) / t - (b1 * (t / a1) ** b1) / t) * anp.exp(-((t / a2) ** b2) - (t / a1) ** b1),
        )

    @staticmethod
    def logR(t, a1, b1, a2, b2):  # Log SF (Competing Risks)
        return -((t / a1) ** b1) - ((t / a2) ** b2)

    @staticmethod
    def LL(params, T_f, T_rc):
        # Log Likelihood function (Competing Risks)
        LL_f = Fit_Weibull_CR.logf(T_f, params[0], params[1], params[2], params[3]).sum()
        LL_rc = Fit_Weibull_CR.logR(T_rc, params[0], params[1], params[2], params[3]).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_DSZI:
    """Fits a Weibull Defective Subpopulation Zero Inflated (DSZI) distribution to
    the data provided. This is a 4 parameter distribution (alpha, beta, DS, ZI).

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 2 non-zero
        failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
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
        the fitted Weibull_DSZI alpha parameter
    beta : float
        the fitted Weibull_DSZI beta parameter
    DS : float
        the fitted Weibull_DSZI DS parameter
    ZI : float
        the fitted Weibull_DSZI ZI parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    DS_SE :float
        the standard error (sqrt(variance)) of the parameter
    ZI_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    DS_upper : float
        the upper CI estimate of the parameter
    DS_lower : float
        the lower CI estimate of the parameter
    ZI_upper : float
        the upper CI estimate of the parameter
    ZI_lower : float
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
        a DSZI_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
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
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        # need to remove zeros before passing to fitters input checking
        failures = np.asarray(failures)
        failures_no_zeros = failures[failures != 0]
        failures_zeros = failures[failures == 0]

        inputs = fitters_input_checking(
            dist="Weibull_DSZI",
            failures=failures_no_zeros,
            right_censored=right_censored,
            optimizer=optimizer,
            CI=CI,
        )
        failures_no_zeros = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        # obtain initial estimates of the parameters
        _, y_pts = plotting_positions(failures=failures, right_censored=right_censored)
        DS_guess = max(y_pts)

        weibull_2P_fit = Fit_Weibull_2P(
            failures=failures_no_zeros,
            right_censored=None,
            print_results=False,
            show_probability_plot=False,
            optimizer=optimizer,
        )
        alpha_guess = weibull_2P_fit.alpha
        beta_guess = weibull_2P_fit.beta
        ZI_guess = len(failures_zeros) / (len(failures) + len(right_censored))

        # maximum likelihood method
        MLE_results = MLE_optimization(
            func_name="Weibull_DSZI",
            LL_func=Fit_Weibull_DSZI.LL,
            initial_guess=[alpha_guess, beta_guess, DS_guess, ZI_guess],
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha = MLE_results.alpha
        self.beta = MLE_results.beta
        self.DS = MLE_results.DS
        self.ZI = MLE_results.ZI
        self.method = "Maximum Likelihood Estimation (MLE)"
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta, self.DS, self.ZI]
        hessian_matrix = hessian(Fit_Weibull_DSZI.LL)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures_zeros)),
            np.array(tuple(failures_no_zeros)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.DS_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.ZI_SE = abs(covariance_matrix[3][3]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + str(self.optimizer)
                    + " optimizer is non-invertable for the Weibull_DSZI Model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.DS_SE = 0
            self.ZI_SE = 0

        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        if self.DS == 1:
            self.DS_lower = 1  # DS=1 causes a divide by zero error for CIs
            self.DS_upper = 1
        else:
            self.DS_upper = self.DS / (self.DS + (1 - self.DS) * (np.exp(-Z * self.DS_SE / (self.DS * (1 - self.DS)))))
            self.DS_lower = self.DS / (self.DS + (1 - self.DS) * (np.exp(Z * self.DS_SE / (self.DS * (1 - self.DS)))))
        if self.ZI == 0:
            self.ZI_upper = 0  # ZI = 0 causes a divide by zero error for CIs
            self.ZI_lower = 0
        else:
            self.ZI_upper = self.ZI / (self.ZI + (1 - self.ZI) * (np.exp(-Z * self.ZI_SE / (self.ZI * (1 - self.ZI)))))
            self.ZI_lower = self.ZI / (self.ZI + (1 - self.ZI) * (np.exp(Z * self.ZI_SE / (self.ZI * (1 - self.ZI)))))

        results_data = {
            "Parameter": ["Alpha", "Beta", "DS", "ZI"],
            "Point Estimate": [self.alpha, self.beta, self.DS, self.ZI],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.DS_SE, self.ZI_SE],
            "Lower CI": [
                self.alpha_lower,
                self.beta_lower,
                self.DS_lower,
                self.ZI_lower,
            ],
            "Upper CI": [
                self.alpha_upper,
                self.beta_upper,
                self.DS_upper,
                self.ZI_upper,
            ],
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
        self.distribution = DSZI_Model(
            distribution=Weibull_Distribution(alpha=self.alpha, beta=self.beta),
            DS=self.DS,
            ZI=self.ZI,
        )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 4
        LL2 = 2 * Fit_Weibull_DSZI.LL(params, failures_zeros, failures_no_zeros, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        # moves all the y values for the x=0 points to be equal to the value of ZI.
        y = np.where(x == 0, self.ZI, y)
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
                str("Results from Fit_Weibull_DSZI (" + str(CI_rounded) + "% CI):"),
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

        if show_probability_plot is True:
            from reliability.Probability_plotting import (
                Weibull_probability_plot,
                plot_points,
            )

            rc = None if len(right_censored) == 0 else right_censored
            Weibull_probability_plot(
                failures=failures_no_zeros,
                right_censored=rc,
                show_fitted_distribution=False,
                show_scatter_points=False,
                **kwargs,
            )

            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull_DSZI"
                    + r" ($\alpha=$"
                    + round_and_string(self.alpha, dec)
                    + r", $\beta=$"
                    + round_and_string(self.beta, dec)
                    + r", $DS=$"
                    + round_and_string(self.DS, dec)
                    + r", $ZI=$"
                    + round_and_string(self.ZI, dec)
                    + ")",
                )
            plot_points(
                failures=failures,
                right_censored=right_censored,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )

            xvals = np.logspace(
                np.log10(min(failures_no_zeros)) - 3,
                np.log10(max(failures_no_zeros)) + 1,
                1000,
            )
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title("Probability Plot\nWeibull Defective Subpopulation Zero Inflated CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, ds, zi):  # Log PDF (Weibull DSZI)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b + anp.log(ds - zi)

    @staticmethod
    def logR(t, a, b, ds, zi):  # Log SF (Weibull DSZI)
        return anp.log(1 - ((1 - anp.exp(-((t / a) ** b))) * (ds - zi) + zi))

    @staticmethod
    def LL(params, T_0, T_f, T_rc):
        # log likelihood function (Weibull DSZI)
        LL_0 = (anp.log(params[3]) * len(T_0)) if params[3] > 0 else 0
        # deals with t=0
        # enables fitting when ZI = 0 to avoid log(0) error
        LL_f = Fit_Weibull_DSZI.logf(T_f, params[0], params[1], params[2], params[3]).sum()
        LL_rc = Fit_Weibull_DSZI.logR(T_rc, params[0], params[1], params[2], params[3]).sum()
        return -(LL_0 + LL_f + LL_rc)


class Fit_Weibull_DS:
    """Fits a Weibull Defective Subpopulation (DS) distribution to the data
    provided. This is a 3 parameter distribution (alpha, beta, DS).

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 2 failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
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
        the fitted Weibull_DS alpha parameter
    beta : float
        the fitted Weibull_DS beta parameter
    DS : float
        the fitted Weibull_DS DS parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    DS_SE :float
        the standard error (sqrt(variance)) of the parameter
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    DS_upper : float
        the upper CI estimate of the parameter
    DS_lower : float
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
        a DSZI_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
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
        optimizer=None,
        downsample_scatterplot=True,
        **kwargs,
    ):
        inputs = fitters_input_checking(
            dist="Weibull_DS",
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
            CI=CI,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        # obtain initial estimates of the parameters
        _, y_pts = plotting_positions(failures=failures, right_censored=right_censored)

        DS_guess = max(y_pts)
        weibull_2P_fit = Fit_Weibull_2P(
            failures=failures,
            right_censored=None,
            print_results=False,
            show_probability_plot=False,
            optimizer=optimizer,
        )
        alpha_guess = weibull_2P_fit.alpha
        beta_guess = weibull_2P_fit.beta

        # maximum likelihood method
        MLE_results = MLE_optimization(
            func_name="Weibull_DS",
            LL_func=Fit_Weibull_DS.LL,
            initial_guess=[alpha_guess, beta_guess, DS_guess],
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha = MLE_results.alpha
        self.beta = MLE_results.beta
        self.DS = MLE_results.DS
        self.method = "Maximum Likelihood Estimation (MLE)"
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta, self.DS]
        hessian_matrix = hessian(Fit_Weibull_DS.LL)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.DS_SE = abs(covariance_matrix[2][2]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + str(self.optimizer)
                    + " optimizer is non-invertable for the Weibull_DS Model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.DS_SE = 0

        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        if self.DS == 1:
            self.DS_lower = 1  # DS=1 causes a divide by zero error for CIs
            self.DS_upper = 1
        else:
            self.DS_upper = self.DS / (self.DS + (1 - self.DS) * (np.exp(-Z * self.DS_SE / (self.DS * (1 - self.DS)))))
            self.DS_lower = self.DS / (self.DS + (1 - self.DS) * (np.exp(Z * self.DS_SE / (self.DS * (1 - self.DS)))))

        results_data = {
            "Parameter": ["Alpha", "Beta", "DS"],
            "Point Estimate": [self.alpha, self.beta, self.DS],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.DS_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.DS_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.DS_upper],
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
        self.distribution = DSZI_Model(
            distribution=Weibull_Distribution(alpha=self.alpha, beta=self.beta),
            DS=self.DS,
        )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Weibull_DS.LL(params, failures, right_censored)
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
                str("Results from Fit_Weibull_DS (" + str(CI_rounded) + "% CI):"),
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

        if show_probability_plot is True:
            from reliability.Probability_plotting import Weibull_probability_plot

            rc = None if len(right_censored) == 0 else right_censored
            Weibull_probability_plot(
                failures=failures,
                right_censored=rc,
                show_fitted_distribution=False,
                downsample_scatterplot=downsample_scatterplot,
                **kwargs,
            )
            if "label" in kwargs:
                label_str = kwargs.pop("label")
            else:
                label_str = str(
                    r"Fitted Weibull_DS"
                    + r" ($\alpha=$"
                    + round_and_string(self.alpha, dec)
                    + r", $\beta=$"
                    + round_and_string(self.beta, dec)
                    + r", $DS=$"
                    + round_and_string(self.DS, dec)
                    + ")",
                )
            xvals = np.logspace(np.log10(min(failures)) - 3, np.log10(max(failures)) + 1, 1000)
            self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
            # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            plt.title("Probability Plot\nWeibull Defective Subpopulation CDF")
            self.probability_plot = plt.gca()

    @staticmethod
    def logf(t, a, b, ds):  # Log PDF (Weibull DS)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b + anp.log(ds)

    @staticmethod
    def logR(t, a, b, ds):  # Log SF (Weibull DS)
        return anp.log(1 - ((1 - anp.exp(-((t / a) ** b))) * ds))

    @staticmethod
    def LL(params, T_f, T_rc):
        # log likelihood function (Weibull DS)
        LL_f = Fit_Weibull_DS.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Weibull_DS.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Weibull_ZI:
    """Fits a Weibull Zero Inflated (ZI) distribution to the data
    provided. This is a 3 parameter distribution (alpha, beta, ZI).

    Parameters
    ----------
    failures : array, list
        An array or list of the failure data. There must be at least 2 non-zero
        failures.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_probability_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the point estimate, standard error, Lower CI and
        Upper CI for each parameter. True or False. Default = True
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
        the fitted Weibull_ZI alpha parameter
    beta : float
        the fitted Weibull_ZI beta parameter
    ZI : float
        the fitted Weibull_ZI ZI parameter
    alpha_SE : float
        the standard error (sqrt(variance)) of the parameter
    beta_SE :float
        the standard error (sqrt(variance)) of the parameter
    ZI_SE :float
        the standard error (sqrt(variance)) of the parameter.
    alpha_upper : float
        the upper CI estimate of the parameter
    alpha_lower : float
        the lower CI estimate of the parameter
    beta_upper : float
        the upper CI estimate of the parameter
    beta_lower : float
        the lower CI estimate of the parameter
    ZI_upper : float
        the upper CI estimate of the parameter.
    ZI_lower : float
        the lower CI estimate of the parameter.
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
        a DSZI_Model object with the parameters of the fitted distribution
    results : dataframe
        a pandas dataframe of the results (point estimate, standard error,2
        lower CI and upper CI for each parameter)
    goodness_of_fit : dataframe
        a pandas dataframe of the goodness of fit values (Log-likelihood, AICc,
        BIC, AD).
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
        CI=0.95,
        optimizer=None,
    ):
        # need to remove zeros before passing to fitters input checking
        failures = np.asarray(failures)
        failures_no_zeros = failures[failures != 0]
        failures_zeros = failures[failures == 0]

        inputs = fitters_input_checking(
            dist="Weibull_ZI",
            failures=failures_no_zeros,
            right_censored=right_censored,
            optimizer=optimizer,
            CI=CI,
        )
        failures_no_zeros = inputs.failures
        right_censored = inputs.right_censored
        CI = inputs.CI
        optimizer = inputs.optimizer

        # obtain initial estimates of the parameters
        weibull_2P_fit = Fit_Weibull_2P(
            failures=failures_no_zeros,
            right_censored=right_censored,
            print_results=False,
            show_probability_plot=False,
            optimizer=optimizer,
            CI=CI,
        )
        alpha_guess = weibull_2P_fit.alpha
        beta_guess = weibull_2P_fit.beta
        ZI_guess = len(failures_zeros) / (len(failures) + len(right_censored))

        # maximum likelihood method
        MLE_results = MLE_optimization(
            func_name="Weibull_ZI",
            LL_func=Fit_Weibull_ZI.LL,
            initial_guess=[alpha_guess, beta_guess, ZI_guess],
            failures=failures,
            right_censored=right_censored,
            optimizer=optimizer,
        )
        self.alpha = MLE_results.alpha
        self.beta = MLE_results.beta
        self.ZI = MLE_results.ZI
        self.method = "Maximum Likelihood Estimation (MLE)"
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters. This uses the Fisher Matrix so it can be applied to both MLE and LS estimates.
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.alpha, self.beta, self.ZI]
        hessian_matrix = hessian(Fit_Weibull_ZI.LL)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures_zeros)),
            np.array(tuple(failures_no_zeros)),
            np.array(tuple(right_censored)),
        )

        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.alpha_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.beta_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.ZI_SE = abs(covariance_matrix[2][2]) ** 0.5
        except LinAlgError:
            # this exception is rare but can occur with some optimizers
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + str(self.optimizer)
                    + " optimizer is non-invertable for the Weibull_ZI Model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.alpha_SE = 0
            self.beta_SE = 0
            self.ZI_SE = 0

        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        if self.ZI == 0:
            self.ZI_upper = 0  # ZI = 0 causes a divide by zero error for CIs
            self.ZI_lower = 0
        else:
            self.ZI_upper = self.ZI / (self.ZI + (1 - self.ZI) * (np.exp(-Z * self.ZI_SE / (self.ZI * (1 - self.ZI)))))
            self.ZI_lower = self.ZI / (self.ZI + (1 - self.ZI) * (np.exp(Z * self.ZI_SE / (self.ZI * (1 - self.ZI)))))

        results_data = {
            "Parameter": ["Alpha", "Beta", "ZI"],
            "Point Estimate": [self.alpha, self.beta, self.ZI],
            "Standard Error": [self.alpha_SE, self.beta_SE, self.ZI_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower, self.ZI_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper, self.ZI_upper],
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
        self.distribution = DSZI_Model(
            distribution=Weibull_Distribution(alpha=self.alpha, beta=self.beta),
            ZI=self.ZI,
        )

        # goodness of fit measures
        n = len(failures) + len(right_censored)
        k = 3
        LL2 = 2 * Fit_Weibull_ZI.LL(params, failures_zeros, failures_no_zeros, right_censored)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc: np.float64 = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2

        x, y = plotting_positions(failures=failures, right_censored=right_censored)
        # moves all the y values for the x=0 points to be equal to the value of ZI.
        y = np.where(x == 0, self.ZI, y)
        self.AD = anderson_darling(fitted_cdf=self.distribution.CDF(xvals=x, show_plot=False), empirical_cdf=y)
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC", "AD"],
            "Value": [self.loglik, self.AICc, self.BIC, self.AD],
        }
        self.goodness_of_fit = pd.DataFrame(GoF_data, columns=["Goodness of fit", "Value"])
        self.__CI: float = CI
        self.__failures = failures
        self.__right_censored = right_censored
        self.__failures_no_zeros = failures_no_zeros
        self.__n = n

    def print_results(self) -> None:
        """Prints the results of the Weibull fitting analysis.

        This method prints various statistics and results obtained from the Weibull fitting analysis,
        including the confidence interval, analysis method, optimizer used, number of failures and right censored data points,
        as well as the results and goodness of fit statistics.

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
            str("Results from Fit_Weibull_ZI (" + str(CI_rounded) + "% CI):"),
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

    def plot(self, downsample_scatterplot=True, **kwargs) -> plt.Axes:
        """Plots the Weibull probability plot and scatter points.

        Args:
        ----
            downsample_scatterplot (bool, optional): Whether to downsample the scatter plot. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the plotting functions.

        Returns:
        -------
            matplotlib.axes.Axes: The current axes instance.

        """
        from reliability.Probability_plotting import (
            Weibull_probability_plot,
            plot_points,
        )

        rc = None if len(self.__right_censored) == 0 else self.__right_censored
        Weibull_probability_plot(
            failures=self.__failures_no_zeros,
            right_censored=rc,
            show_fitted_distribution=False,
            show_scatter_points=False,
            **kwargs,
        )
        if "label" in kwargs:
            label_str = kwargs.pop("label")
        else:
            label_str = str(
                r"Fitted Weibull_ZI"
                + r" ($\alpha=$"
                + round_and_string(self.alpha, dec)
                + r", $\beta=$"
                + round_and_string(self.beta, dec)
                + r", $ZI=$"
                + round_and_string(self.ZI, dec)
                + ")",
            )
        plot_points(
            failures=self.__failures,
            right_censored=self.__right_censored,
            downsample_scatterplot=downsample_scatterplot,
            **kwargs,
        )
        xvals = np.logspace(
            np.log10(min(self.__failures_no_zeros)) - 3,
            np.log10(max(self.__failures_no_zeros)) + 1,
            1000,
        )
        self.distribution.CDF(xvals=xvals, label=label_str, **kwargs)
        # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
        plt.title("Probability Plot\nWeibull Zero Inflated CDF")
        return plt.gca()

    @staticmethod
    def logf(t, a, b, zi):  # Log PDF (Weibull ZI)
        return (b - 1) * anp.log(t / a) + anp.log(b / a) - (t / a) ** b + anp.log(1 - zi)

    @staticmethod
    def logR(t, a, b, zi):  # Log SF (Weibull ZI)
        return anp.log(1 - ((1 - anp.exp(-((t / a) ** b))) * (1 - zi) + zi))

    @staticmethod
    def LL(params, T_0, T_f, T_rc):
        # log likelihood function (Weibull ZI)
        LL_0 = (anp.log(params[2]) * len(T_0)) if params[2] > 0 else 0
        # deals with t=0
        # enables fitting when ZI = 0 to avoid log(0) error
        LL_f = Fit_Weibull_ZI.logf(T_f, params[0], params[1], params[2]).sum()
        LL_rc = Fit_Weibull_ZI.logR(T_rc, params[0], params[1], params[2]).sum()
        return -(LL_0 + LL_f + LL_rc)

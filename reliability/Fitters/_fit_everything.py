from __future__ import annotations

import contextlib

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reliability.Distributions import (
    Beta_Distribution,
    Competing_Risks_Model,
    DSZI_Model,
    Exponential_Distribution,
    Gamma_Distribution,
    Gumbel_Distribution,
    Loglogistic_Distribution,
    Lognormal_Distribution,
    Mixture_Model,
    Normal_Distribution,
    Weibull_Distribution,
)
from reliability.Nonparametric import KaplanMeier
from reliability.Probability_plotting import plotting_positions
from reliability.Utils import (
    colorprint,
    fitters_input_checking,
    round_and_string,
    xy_downsample,
)

anp.seterr("ignore")
dec = 3  # number of decimals to use when rounding fitted parameters in labels

# change pandas display options
pd.options.display.float_format = "{:g}".format  # improves formatting of numbers in dataframe
pd.options.display.max_columns = 9  # shows the dataframe without ... truncation
pd.options.display.width = 200  # prevents wrapping after default 80 characters


class Fit_Everything:
    """This function will fit all available distributions to the data provided.
    The only distributions not fitted are Weibull_DSZI and Weibull_ZI. The
    Beta_2P distribution will only be fitted if the data are between 0 and 1.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements for all the 2 parameter
        distributions to be fitted and 3 elements for all distributions to be
        fitted.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    sort_by : str
        Goodness of fit test to sort results by. Must be 'BIC','AICc','AD', or
        'Log-likelihood'. Default is BIC.
    show_probability_plot : bool, optional
        Provides a probability plot of each of the fitted distributions. True or
        False. Default = True
    show_histogram_plot : bool, optional
        True or False. Default = True. Will show a histogram (scaled to account
        for censored data) with the PDF and CDF of each fitted distribution.
    show_PP_plot : bool, optional
        Provides a comparison of parametric vs non-parametric fit using
        Probability-Probability (PP) plot. True or False. Default = True.
    show_best_distribution_probability_plot : bool, optional
        Provides a probability plot in a new figure of the best fitting
        distribution. True or False. Default = True.
    exclude : list, array, optional
        List or array of strings specifying which distributions to exclude.
        Default is None. Options are Weibull_2P, Weibull_3P, Weibull_CR,
        Weibull_Mixture, Weibull_DS, Normal_2P, Gamma_2P, Loglogistic_2P,
        Gamma_3P, Lognormal_2P, Lognormal_3P, Loglogistic_3P, Gumbel_2P,
        Exponential_2P, Exponential_1P, Beta_2P.
    print_results : bool, optional
        Will show the results of the fitted parameters and the goodness of fit
        tests in a dataframe. True/False. Defaults to True.
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
    downsample_scatterplot : bool, int, optional
        If True or None, and there are over 1000 points, then the scatterplot
        will be downsampled by a factor. The default downsample factor will seek
        to produce between 500 and 1000 points. If a number is specified, it
        will be used as the downsample factor. Default is True. This
        functionality makes plotting faster when there are very large numbers of
        points. It only affects the scatterplot not the calculations.

    Returns
    -------
    results : dataframe
        a pandas dataframe of results. Fitted parameters in this dataframe may
        be accessed by name. See below example in Notes.
    best_distribution : object
        a reliability.Distributions object created based on the parameters of
        the best fitting distribution.
    best_distribution_name : str
        the name of the best fitting distribution. E.g. 'Weibull_3P'
    parameters and goodness of fit results : float
        This is provided for each fitted distribution. For example, the
        Weibull_3P distribution values are Weibull_3P_alpha, Weibull_3P_beta,
        Weibull_3P_gamma, Weibull_3P_BIC, Weibull_3P_AICc, Weibull_3P_AD,
        Weibull_3P_loglik
    excluded_distributions : list
        a list of strings of the excluded distributions.
    probability_plot : object
        The figure handle from the probability plot (only provided if
        show_probability_plot is True).
    best_distribution_probability_plot : object
        The figure handle from the best distribution probability plot (only
        provided if show_best_distribution_probability_plot is True).
    histogram_plot : object
        The figure handle from the histogram plot (only provided if
        show_histogram_plot is True).
    PP_plot : object
        The figure handle from the probability-probability plot (only provided
        if show_PP_plot is True).

    Notes
    -----
    All parametric models have the number of parameters in the name. For
    example, Weibull_2P uses alpha and beta, whereas Weibull_3P uses alpha,
    beta, and gamma. This is applied even for Normal_2P for consistency in
    naming conventions. From the results, the distributions are sorted based on
    their goodness of fit test results, where the smaller the goodness of fit
    value, the better the fit of the distribution to the data.

    If the data provided contains only 2 failures, the three parameter
    distributions will automatically be excluded.

    Example Usage:

    .. code:: python

        X = [5,3,8,6,7,4,5,4,2]
        output = Fit_Everything(X)
        print('Weibull Alpha =',output.Weibull_2P_alpha)

    """
    def __init__(
        self,
        failures =None,
        right_censored=None,
        exclude=None,
        sort_by="BIC",
        method: str | None ="MLE",
        optimizer=None,
        print_results=True,
        show_histogram_plot=True,
        show_PP_plot=True,
        show_probability_plot=True,
        show_best_distribution_probability_plot=True,
        downsample_scatterplot=True,
    ):
        inputs = fitters_input_checking(
            dist="Everything",
            failures=failures,
            right_censored=right_censored,
            method=method,
            optimizer=optimizer,
        )
        failures = inputs.failures
        right_censored = inputs.right_censored
        method = inputs.method
        optimizer = inputs.optimizer

        if method in ["RRX", "RRY", "LS", "NLLS"]:
            method = "LS"

        if show_histogram_plot not in [True, False]:
            raise ValueError("show_histogram_plot must be either True or False. Defaults to True.")
        if print_results not in [True, False]:
            raise ValueError("print_results must be either True or False. Defaults to True.")
        if show_PP_plot not in [True, False]:
            raise ValueError("show_PP_plot must be either True or False. Defaults to True.")
        if show_probability_plot not in [True, False]:
            raise ValueError("show_probability_plot must be either True or False. Defaults to True.")
        if show_best_distribution_probability_plot not in [True, False]:
            raise ValueError("show_best_distribution_probability_plot must be either True or False. Defaults to True.")

        self.failures = failures
        self.right_censored = right_censored
        self._all_data = np.hstack([failures, right_censored])
        # This is used for scaling the histogram when there is censored data
        self._frac_fail = len(failures) / len(self._all_data)
        # This is used for reporting the fraction censored in the printed output
        self._frac_cens = len(right_censored) / len(self._all_data)
        # sorting the failure data is necessary for plotting quantiles in order
        d = sorted(self._all_data)
        self.__downsample_scatterplot = downsample_scatterplot

        if exclude is None:
            exclude = []
        if type(exclude) == np.ndarray:
            exclude = list(exclude)
        if type(exclude) not in [list, np.ndarray]:
            raise ValueError(
                'exclude must be a list or array or strings that match the names of the distributions to be excluded. eg "Weibull_2P".',
            )
        if len(failures) < 3:
            exclude.extend(
                [
                    "Weibull_3P",
                    "Gamma_3P",
                    "Loglogistic_3P",
                    "Lognormal_3P",
                    "Weibull_Mixture",
                    "Weibull_CR",
                ],
            )  # do not fit the 3P distributions if there are only 2 failures
        # flexible name checking for excluded distributions
        excluded_distributions = []
        unknown_exclusions = []
        for item in exclude:
            if type(item) not in [str, np.str_]:
                raise ValueError(
                    "exclude must be a list or array of strings that specified the distributions to be excluded from fitting. Available strings are:"
                    "\nWeibull_2P\nWeibull_3P\nNormal_2P\nGamma_2P\nLoglogistic_2P\nGamma_3P\nLognormal_2P\nLognormal_3P\nLoglogistic_3P\nGumbel_2P\nExponential_2P\nExponential_1P\nBeta_2P\nWeibull_Mixture\nWeibull_CR\nWeibull_DS",
                )
            if item.upper() in ["WEIBULL_2P", "WEIBULL2P", "WEIBULL2"]:
                excluded_distributions.append("Weibull_2P")
            elif item.upper() in ["WEIBULL_3P", "WEIBULL3P", "WEIBULL3"]:
                excluded_distributions.append("Weibull_3P")
            elif item.upper() in ["GAMMA_2P", "GAMMA2P", "GAMMA2"]:
                excluded_distributions.append("Gamma_2P")
            elif item.upper() in ["GAMMA_3P", "GAMMA3P", "GAMMA3"]:
                excluded_distributions.append("Gamma_3P")
            elif item.upper() in ["LOGNORMAL_2P", "LOGNORMAL2P", "LOGNORMAL2"]:
                excluded_distributions.append("Lognormal_2P")
            elif item.upper() in ["LOGNORMAL_3P", "LOGNORMAL3P", "LOGNORMAL3"]:
                excluded_distributions.append("Lognormal_3P")
            elif item.upper() in [
                "EXPONENTIAL_1P",
                "EXPONENTIAL1P",
                "EXPONENTIAL1",
                "EXPON_1P",
                "EXPON1P",
                "EXPON1",
            ]:
                excluded_distributions.append("Exponential_1P")
            elif item.upper() in [
                "EXPONENTIAL_2P",
                "EXPONENTIAL2P",
                "EXPONENTIAL2",
                "EXPON_2P",
                "EXPON2P",
                "EXPON2",
            ]:
                excluded_distributions.append("Exponential_2P")
            elif item.upper() in ["NORMAL_2P", "NORMAL2P", "NORMAL2"]:
                excluded_distributions.append("Normal_2P")
            elif item.upper() in ["GUMBEL_2P", "GUMBEL2P", "GUMBEL2"]:
                excluded_distributions.append("Gumbel_2P")
            elif item.upper() in ["LOGLOGISTIC_2P", "LOGLOGISTIC2P", "LOGLOGISTIC2"]:
                excluded_distributions.append("Loglogistic_2P")
            elif item.upper() in ["LOGLOGISTIC_3P", "LOGLOGISTIC3P", "LOGLOGISTIC3"]:
                excluded_distributions.append("Loglogistic_3P")
            elif item.upper() in ["BETA_2P", "BETA2P", "BETA2"]:
                excluded_distributions.append("Beta_2P")
            elif item.upper() in [
                "WEIBULL MIXTURE",
                "WEIBULLMIXTURE",
                "WEIBULL_MIXTURE",
                "MIXTURE",
                "WEIBULLMIX",
                "WEIBULL_MIX",
                "MIX",
            ]:
                excluded_distributions.append("Weibull_Mixture")
            elif item.upper() in [
                "WEIBULL CR",
                "WEIBULLCR",
                "WEIBULL_CR",
                "WEIBULL_COMPETING_RISKS",
                "WEIBULL_COMPETINGRISKS",
                "WEIBULLCOMPETINGRISKS",
                "WEIBULL COMPETING RISKS",
                "WEIBULL COMPETINGRISKS",
                "COMPETINGRISKS",
                "COMPETING RISKS",
                "CR",
            ]:
                excluded_distributions.append("Weibull_CR")
            elif item.upper() in [
                "WEIBULLDS",
                "WEIBULL_DS",
                "WEIBULL DS",
                "WEIBULL_DEFECTIVE_SUBPOPULATION",
                "WEIBULL_DEFECTIVESUBPOPULATION",
                "WEIBULLDEFECTIVESUBPOPULATION",
                "WEIBULL DEFECTIVE SUBPOPULATION",
                "DEFECTIVE SUBPOPULATION",
                "DEFECTIVESUBPOPULATION",
                "DS",
            ]:
                excluded_distributions.append("Weibull_DS")
            else:
                unknown_exclusions.append(item)
        if len(unknown_exclusions) > 0:
            colorprint(
                str(
                    "WARNING: The following items were not recognised distributions to exclude: "
                    + str(unknown_exclusions),
                ),
                text_color="red",
            )
            colorprint(
                "Available distributions to exclude are: Weibull_2P, Weibull_3P, Normal_2P, Gamma_2P, Loglogistic_2P, Gamma_3P, Lognormal_2P, Lognormal_3P, Loglogistic_3P, Gumbel_2P, Exponential_2P, Exponential_1P, Beta_2P, Weibull_Mixture, Weibull_CR, Weibull_DS",
                text_color="red",
            )
        if "Beta_2P" not in excluded_distributions and (
            max(self._all_data) >= 1
        ):  # if Beta wasn't manually excluded, check if is needs to be automatically excluded based on data above 1
            excluded_distributions.append("Beta_2P")
        self.excluded_distributions = excluded_distributions

        # create an empty dataframe to append the data from the fitted distributions
        df = pd.DataFrame(
            columns=[
                "Distribution",
                "Alpha",
                "Beta",
                "Gamma",
                "Alpha 1",
                "Beta 1",
                "Alpha 2",
                "Beta 2",
                "Proportion 1",
                "DS",
                "Mu",
                "Sigma",
                "Lambda",
                "Log-likelihood",
                "AICc",
                "BIC",
                "AD",
                "optimizer",
            ],
        )
        # Fit the parametric models and extract the fitted parameters
        if "Weibull_3P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Weibull_3P
            self.__Weibull_3P_params = Fit_Weibull_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_3P_alpha = self.__Weibull_3P_params.alpha
            self.Weibull_3P_beta = self.__Weibull_3P_params.beta
            self.Weibull_3P_gamma = self.__Weibull_3P_params.gamma
            self.Weibull_3P_loglik = self.__Weibull_3P_params.loglik
            self.Weibull_3P_BIC = self.__Weibull_3P_params.BIC
            self.Weibull_3P_AICc = self.__Weibull_3P_params.AICc
            self.Weibull_3P_AD = self.__Weibull_3P_params.AD
            self.Weibull_3P_optimizer = self.__Weibull_3P_params.optimizer
            self._parametric_CDF_Weibull_3P = self.__Weibull_3P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_3P"],
                            "Alpha": [self.Weibull_3P_alpha],
                            "Beta": [self.Weibull_3P_beta],
                            "Gamma": [self.Weibull_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_3P_loglik],
                            "AICc": [self.Weibull_3P_AICc],
                            "BIC": [self.Weibull_3P_BIC],
                            "AD": [self.Weibull_3P_AD],
                            "optimizer": [self.Weibull_3P_optimizer],
                        },
                    ),
                ],
            )

        if "Gamma_3P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Gamma_3P
            self.__Gamma_3P_params = Fit_Gamma_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Gamma_3P_alpha = self.__Gamma_3P_params.alpha
            self.Gamma_3P_beta = self.__Gamma_3P_params.beta
            self.Gamma_3P_mu = self.__Gamma_3P_params.mu
            self.Gamma_3P_gamma = self.__Gamma_3P_params.gamma
            self.Gamma_3P_loglik = self.__Gamma_3P_params.loglik
            self.Gamma_3P_BIC = self.__Gamma_3P_params.BIC
            self.Gamma_3P_AICc = self.__Gamma_3P_params.AICc
            self.Gamma_3P_AD = self.__Gamma_3P_params.AD
            self.Gamma_3P_optimizer = self.__Gamma_3P_params.optimizer
            self._parametric_CDF_Gamma_3P = self.__Gamma_3P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Gamma_3P"],
                            "Alpha": [self.Gamma_3P_alpha],
                            "Beta": [self.Gamma_3P_beta],
                            "Gamma": [self.Gamma_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Gamma_3P_loglik],
                            "AICc": [self.Gamma_3P_AICc],
                            "BIC": [self.Gamma_3P_BIC],
                            "AD": [self.Gamma_3P_AD],
                            "optimizer": [self.Gamma_3P_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Exponential_2P
            self.__Exponential_2P_params = Fit_Exponential_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Exponential_2P_lambda = self.__Exponential_2P_params.Lambda
            self.Exponential_2P_gamma = self.__Exponential_2P_params.gamma
            self.Exponential_2P_loglik = self.__Exponential_2P_params.loglik
            self.Exponential_2P_BIC = self.__Exponential_2P_params.BIC
            self.Exponential_2P_AICc = self.__Exponential_2P_params.AICc
            self.Exponential_2P_AD = self.__Exponential_2P_params.AD
            self.Exponential_2P_optimizer = self.__Exponential_2P_params.optimizer
            self._parametric_CDF_Exponential_2P = self.__Exponential_2P_params.distribution.CDF(
                xvals=d,
                show_plot=False,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Exponential_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [self.Exponential_2P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [self.Exponential_2P_lambda],
                            "Log-likelihood": [self.Exponential_2P_loglik],
                            "AICc": [self.Exponential_2P_AICc],
                            "BIC": [self.Exponential_2P_BIC],
                            "AD": [self.Exponential_2P_AD],
                            "optimizer": [self.Exponential_2P_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_3P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Lognormal_3P
            self.__Lognormal_3P_params = Fit_Lognormal_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Lognormal_3P_mu = self.__Lognormal_3P_params.mu
            self.Lognormal_3P_sigma = self.__Lognormal_3P_params.sigma
            self.Lognormal_3P_gamma = self.__Lognormal_3P_params.gamma
            self.Lognormal_3P_loglik = self.__Lognormal_3P_params.loglik
            self.Lognormal_3P_BIC = self.__Lognormal_3P_params.BIC
            self.Lognormal_3P_AICc = self.__Lognormal_3P_params.AICc
            self.Lognormal_3P_AD = self.__Lognormal_3P_params.AD
            self.Lognormal_3P_optimizer = self.__Lognormal_3P_params.optimizer
            self._parametric_CDF_Lognormal_3P = self.__Lognormal_3P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Lognormal_3P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [self.Lognormal_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Lognormal_3P_mu],
                            "Sigma": [self.Lognormal_3P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Lognormal_3P_loglik],
                            "AICc": [self.Lognormal_3P_AICc],
                            "BIC": [self.Lognormal_3P_BIC],
                            "AD": [self.Lognormal_3P_AD],
                            "optimizer": [self.Lognormal_3P_optimizer],
                        },
                    ),
                ],
            )

        if "Normal_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Normal_2P
            self.__Normal_2P_params = Fit_Normal_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Normal_2P_mu = self.__Normal_2P_params.mu
            self.Normal_2P_sigma = self.__Normal_2P_params.sigma
            self.Normal_2P_loglik = self.__Normal_2P_params.loglik
            self.Normal_2P_BIC = self.__Normal_2P_params.BIC
            self.Normal_2P_AICc = self.__Normal_2P_params.AICc
            self.Normal_2P_AD = self.__Normal_2P_params.AD
            self.Normal_2P_optimizer = self.__Normal_2P_params.optimizer
            self._parametric_CDF_Normal_2P = self.__Normal_2P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Normal_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Normal_2P_mu],
                            "Sigma": [self.Normal_2P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Normal_2P_loglik],
                            "AICc": [self.Normal_2P_AICc],
                            "BIC": [self.Normal_2P_BIC],
                            "AD": [self.Normal_2P_AD],
                            "optimizer": [self.Normal_2P_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Lognormal_2P
            self.__Lognormal_2P_params = Fit_Lognormal_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Lognormal_2P_mu = self.__Lognormal_2P_params.mu
            self.Lognormal_2P_sigma = self.__Lognormal_2P_params.sigma
            self.Lognormal_2P_gamma = 0
            self.Lognormal_2P_loglik = self.__Lognormal_2P_params.loglik
            self.Lognormal_2P_BIC = self.__Lognormal_2P_params.BIC
            self.Lognormal_2P_AICc = self.__Lognormal_2P_params.AICc
            self.Lognormal_2P_AD = self.__Lognormal_2P_params.AD
            self.Lognormal_2P_optimizer = self.__Lognormal_2P_params.optimizer
            self._parametric_CDF_Lognormal_2P = self.__Lognormal_2P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Lognormal_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Lognormal_2P_mu],
                            "Sigma": [self.Lognormal_2P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Lognormal_2P_loglik],
                            "AICc": [self.Lognormal_2P_AICc],
                            "BIC": [self.Lognormal_2P_BIC],
                            "AD": [self.Lognormal_2P_AD],
                            "optimizer": [self.Lognormal_2P_optimizer],
                        },
                    ),
                ],
            )

        if "Gumbel_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Gumbel_2P
            self.__Gumbel_2P_params = Fit_Gumbel_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Gumbel_2P_mu = self.__Gumbel_2P_params.mu
            self.Gumbel_2P_sigma = self.__Gumbel_2P_params.sigma
            self.Gumbel_2P_loglik = self.__Gumbel_2P_params.loglik
            self.Gumbel_2P_BIC = self.__Gumbel_2P_params.BIC
            self.Gumbel_2P_AICc = self.__Gumbel_2P_params.AICc
            self.Gumbel_2P_AD = self.__Gumbel_2P_params.AD
            self.Gumbel_2P_optimizer = self.__Gumbel_2P_params.optimizer
            self._parametric_CDF_Gumbel_2P = self.__Gumbel_2P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Gumbel_2P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [self.Gumbel_2P_mu],
                            "Sigma": [self.Gumbel_2P_sigma],
                            "Lambda": [""],
                            "Log-likelihood": [self.Gumbel_2P_loglik],
                            "AICc": [self.Gumbel_2P_AICc],
                            "BIC": [self.Gumbel_2P_BIC],
                            "AD": [self.Gumbel_2P_AD],
                            "optimizer": [self.Gumbel_2P_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Weibull_2P
            self.__Weibull_2P_params = Fit_Weibull_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_2P_alpha = self.__Weibull_2P_params.alpha
            self.Weibull_2P_beta = self.__Weibull_2P_params.beta
            self.Weibull_2P_gamma = 0
            self.Weibull_2P_loglik = self.__Weibull_2P_params.loglik
            self.Weibull_2P_BIC = self.__Weibull_2P_params.BIC
            self.Weibull_2P_AICc = self.__Weibull_2P_params.AICc
            self.Weibull_2P_AD = self.__Weibull_2P_params.AD
            self.Weibull_2P_optimizer = self.__Weibull_2P_params.optimizer
            self._parametric_CDF_Weibull_2P = self.__Weibull_2P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_2P"],
                            "Alpha": [self.Weibull_2P_alpha],
                            "Beta": [self.Weibull_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_2P_loglik],
                            "AICc": [self.Weibull_2P_AICc],
                            "BIC": [self.Weibull_2P_BIC],
                            "AD": [self.Weibull_2P_AD],
                            "optimizer": [self.Weibull_2P_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_Mixture" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Weibull_Mixture
            self.__Weibull_Mixture_params = Fit_Weibull_Mixture(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_Mixture_alpha_1 = self.__Weibull_Mixture_params.alpha_1
            self.Weibull_Mixture_beta_1 = self.__Weibull_Mixture_params.beta_1
            self.Weibull_Mixture_alpha_2 = self.__Weibull_Mixture_params.alpha_2
            self.Weibull_Mixture_beta_2 = self.__Weibull_Mixture_params.beta_2
            self.Weibull_Mixture_proportion_1 = self.__Weibull_Mixture_params.proportion_1
            self.Weibull_Mixture_loglik = self.__Weibull_Mixture_params.loglik
            self.Weibull_Mixture_BIC = self.__Weibull_Mixture_params.BIC
            self.Weibull_Mixture_AICc = self.__Weibull_Mixture_params.AICc
            self.Weibull_Mixture_AD = self.__Weibull_Mixture_params.AD
            self.Weibull_Mixture_optimizer = self.__Weibull_Mixture_params.optimizer
            self._parametric_CDF_Weibull_Mixture = self.__Weibull_Mixture_params.distribution.CDF(
                xvals=d,
                show_plot=False,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_Mixture"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [self.Weibull_Mixture_alpha_1],
                            "Beta 1": [self.Weibull_Mixture_beta_1],
                            "Alpha 2": [self.Weibull_Mixture_alpha_2],
                            "Beta 2": [self.Weibull_Mixture_beta_2],
                            "Proportion 1": [self.Weibull_Mixture_proportion_1],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_Mixture_loglik],
                            "AICc": [self.Weibull_Mixture_AICc],
                            "BIC": [self.Weibull_Mixture_BIC],
                            "AD": [self.Weibull_Mixture_AD],
                            "optimizer": [self.Weibull_Mixture_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_CR" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Weibull_CR
            self.__Weibull_CR_params = Fit_Weibull_CR(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_CR_alpha_1 = self.__Weibull_CR_params.alpha_1
            self.Weibull_CR_beta_1 = self.__Weibull_CR_params.beta_1
            self.Weibull_CR_alpha_2 = self.__Weibull_CR_params.alpha_2
            self.Weibull_CR_beta_2 = self.__Weibull_CR_params.beta_2
            self.Weibull_CR_loglik = self.__Weibull_CR_params.loglik
            self.Weibull_CR_BIC = self.__Weibull_CR_params.BIC
            self.Weibull_CR_AICc = self.__Weibull_CR_params.AICc
            self.Weibull_CR_AD = self.__Weibull_CR_params.AD
            self.Weibull_CR_optimizer = self.__Weibull_CR_params.optimizer
            self._parametric_CDF_Weibull_CR = self.__Weibull_CR_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_CR"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [self.Weibull_CR_alpha_1],
                            "Beta 1": [self.Weibull_CR_beta_1],
                            "Alpha 2": [self.Weibull_CR_alpha_2],
                            "Beta 2": [self.Weibull_CR_beta_2],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_CR_loglik],
                            "AICc": [self.Weibull_CR_AICc],
                            "BIC": [self.Weibull_CR_BIC],
                            "AD": [self.Weibull_CR_AD],
                            "optimizer": [self.Weibull_CR_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_DS" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Weibull_DS
            self.__Weibull_DS_params = Fit_Weibull_DS(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Weibull_DS_alpha = self.__Weibull_DS_params.alpha
            self.Weibull_DS_beta = self.__Weibull_DS_params.beta
            self.Weibull_DS_DS = self.__Weibull_DS_params.DS
            self.Weibull_DS_loglik = self.__Weibull_DS_params.loglik
            self.Weibull_DS_BIC = self.__Weibull_DS_params.BIC
            self.Weibull_DS_AICc = self.__Weibull_DS_params.AICc
            self.Weibull_DS_AD = self.__Weibull_DS_params.AD
            self.Weibull_DS_optimizer = self.__Weibull_DS_params.optimizer
            self._parametric_CDF_Weibull_DS = self.__Weibull_DS_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Weibull_DS"],
                            "Alpha": [self.Weibull_DS_alpha],
                            "Beta": [self.Weibull_DS_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [self.Weibull_DS_DS],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Weibull_DS_loglik],
                            "AICc": [self.Weibull_DS_AICc],
                            "BIC": [self.Weibull_DS_BIC],
                            "AD": [self.Weibull_DS_AD],
                            "optimizer": [self.Weibull_DS_optimizer],
                        },
                    ),
                ],
            )

        if "Gamma_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Gamma_2P
            self.__Gamma_2P_params = Fit_Gamma_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Gamma_2P_alpha = self.__Gamma_2P_params.alpha
            self.Gamma_2P_beta = self.__Gamma_2P_params.beta
            self.Gamma_2P_mu = self.__Gamma_2P_params.mu
            self.Gamma_2P_gamma = 0
            self.Gamma_2P_loglik = self.__Gamma_2P_params.loglik
            self.Gamma_2P_BIC = self.__Gamma_2P_params.BIC
            self.Gamma_2P_AICc = self.__Gamma_2P_params.AICc
            self.Gamma_2P_AD = self.__Gamma_2P_params.AD
            self.Gamma_2P_optimizer = self.__Gamma_2P_params.optimizer
            self._parametric_CDF_Gamma_2P = self.__Gamma_2P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Gamma_2P"],
                            "Alpha": [self.Gamma_2P_alpha],
                            "Beta": [self.Gamma_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Gamma_2P_loglik],
                            "AICc": [self.Gamma_2P_AICc],
                            "BIC": [self.Gamma_2P_BIC],
                            "AD": [self.Gamma_2P_AD],
                            "optimizer": [self.Gamma_2P_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_1P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Exponential_1P
            self.__Exponential_1P_params = Fit_Exponential_1P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Exponential_1P_lambda = self.__Exponential_1P_params.Lambda
            self.Exponential_1P_gamma = 0
            self.Exponential_1P_loglik = self.__Exponential_1P_params.loglik
            self.Exponential_1P_BIC = self.__Exponential_1P_params.BIC
            self.Exponential_1P_AICc = self.__Exponential_1P_params.AICc
            self.Exponential_1P_AD = self.__Exponential_1P_params.AD
            self.Exponential_1P_optimizer = self.__Exponential_1P_params.optimizer
            self._parametric_CDF_Exponential_1P = self.__Exponential_1P_params.distribution.CDF(
                xvals=d,
                show_plot=False,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Exponential_1P"],
                            "Alpha": [""],
                            "Beta": [""],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [self.Exponential_1P_lambda],
                            "Log-likelihood": [self.Exponential_1P_loglik],
                            "AICc": [self.Exponential_1P_AICc],
                            "BIC": [self.Exponential_1P_BIC],
                            "AD": [self.Exponential_1P_AD],
                            "optimizer": [self.Exponential_1P_optimizer],
                        },
                    ),
                ],
            )

        if "Loglogistic_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Loglogistic_2P
            self.__Loglogistic_2P_params = Fit_Loglogistic_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Loglogistic_2P_alpha = self.__Loglogistic_2P_params.alpha
            self.Loglogistic_2P_beta = self.__Loglogistic_2P_params.beta
            self.Loglogistic_2P_gamma = 0
            self.Loglogistic_2P_loglik = self.__Loglogistic_2P_params.loglik
            self.Loglogistic_2P_BIC = self.__Loglogistic_2P_params.BIC
            self.Loglogistic_2P_AICc = self.__Loglogistic_2P_params.AICc
            self.Loglogistic_2P_AD = self.__Loglogistic_2P_params.AD
            self.Loglogistic_2P_optimizer = self.__Loglogistic_2P_params.optimizer
            self._parametric_CDF_Loglogistic_2P = self.__Loglogistic_2P_params.distribution.CDF(
                xvals=d,
                show_plot=False,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Loglogistic_2P"],
                            "Alpha": [self.Loglogistic_2P_alpha],
                            "Beta": [self.Loglogistic_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Loglogistic_2P_loglik],
                            "AICc": [self.Loglogistic_2P_AICc],
                            "BIC": [self.Loglogistic_2P_BIC],
                            "AD": [self.Loglogistic_2P_AD],
                            "optimizer": [self.Loglogistic_2P_optimizer],
                        },
                    ),
                ],
            )

        if "Loglogistic_3P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Loglogistic_3P
            self.__Loglogistic_3P_params = Fit_Loglogistic_3P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Loglogistic_3P_alpha = self.__Loglogistic_3P_params.alpha
            self.Loglogistic_3P_beta = self.__Loglogistic_3P_params.beta
            self.Loglogistic_3P_gamma = self.__Loglogistic_3P_params.gamma
            self.Loglogistic_3P_loglik = self.__Loglogistic_3P_params.loglik
            self.Loglogistic_3P_BIC = self.__Loglogistic_3P_params.BIC
            self.Loglogistic_3P_AICc = self.__Loglogistic_3P_params.AICc
            self.Loglogistic_3P_AD = self.__Loglogistic_3P_params.AD
            self.Loglogistic_3P_optimizer = self.__Loglogistic_3P_params.optimizer
            self._parametric_CDF_Loglogistic_3P = self.__Loglogistic_3P_params.distribution.CDF(
                xvals=d,
                show_plot=False,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Loglogistic_3P"],
                            "Alpha": [self.Loglogistic_3P_alpha],
                            "Beta": [self.Loglogistic_3P_beta],
                            "Gamma": [self.Loglogistic_3P_gamma],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Loglogistic_3P_loglik],
                            "AICc": [self.Loglogistic_3P_AICc],
                            "BIC": [self.Loglogistic_3P_BIC],
                            "AD": [self.Loglogistic_3P_AD],
                            "optimizer": [self.Loglogistic_3P_optimizer],
                        },
                    ),
                ],
            )

        if "Beta_2P" not in self.excluded_distributions:
            from reliability.Fitters import Fit_Beta_2P
            self.__Beta_2P_params = Fit_Beta_2P(
                failures=failures,
                right_censored=right_censored,
                method=method,
                optimizer=optimizer,
                show_probability_plot=False,
                print_results=False,
            )
            self.Beta_2P_alpha = self.__Beta_2P_params.alpha
            self.Beta_2P_beta = self.__Beta_2P_params.beta
            self.Beta_2P_loglik = self.__Beta_2P_params.loglik
            self.Beta_2P_BIC = self.__Beta_2P_params.BIC
            self.Beta_2P_AICc = self.__Beta_2P_params.AICc
            self.Beta_2P_AD = self.__Beta_2P_params.AD
            self.Beta_2P_optimizer = self.__Beta_2P_params.optimizer
            self._parametric_CDF_Beta_2P = self.__Beta_2P_params.distribution.CDF(xvals=d, show_plot=False)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "Distribution": ["Beta_2P"],
                            "Alpha": [self.Beta_2P_alpha],
                            "Beta": [self.Beta_2P_beta],
                            "Gamma": [""],
                            "Alpha 1": [""],
                            "Beta 1": [""],
                            "Alpha 2": [""],
                            "Beta 2": [""],
                            "Proportion 1": [""],
                            "DS": [""],
                            "Mu": [""],
                            "Sigma": [""],
                            "Lambda": [""],
                            "Log-likelihood": [self.Beta_2P_loglik],
                            "AICc": [self.Beta_2P_AICc],
                            "BIC": [self.Beta_2P_BIC],
                            "AD": [self.Beta_2P_AD],
                            "optimizer": [self.Beta_2P_optimizer],
                        },
                    ),
                ],
            )

        # change to sorting by BIC if there is insufficient data to get the AICc for everything that was fitted
        if sort_by in ["AIC", "aic", "aicc", "AICc"] and "Insufficient data" in df["AICc"].values:
            sort_by = "BIC"
        # sort the dataframe by BIC, AICc, or AD. Smallest AICc, BIC, AD is better fit
        if not isinstance(sort_by, str):
            raise ValueError(
                "Invalid input to sort_by. Options are 'BIC', 'AICc', 'AD', or 'Log-likelihood'. Default is 'BIC'.",
            )
        if sort_by.upper() == "BIC":
            df2 = df.sort_values(by="BIC")
        elif sort_by.upper() in ["AICC", "AIC"]:
            df2 = df.sort_values(by="AICc")
        elif sort_by.upper() == "AD":
            df2 = df.sort_values(by="AD")
        elif sort_by.upper() in [
            "LOGLIK",
            "LOG LIK",
            "LOG-LIKELIHOOD",
            "LL",
            "LOGLIKELIHOOD",
            "LOG LIKELIHOOD",
        ]:
            df["LLabs"] = abs(df["Log-likelihood"])  # need to create a new column for the absolute value before sorting
            df2 = df.sort_values(by="LLabs")
            df2.drop("LLabs", axis=1, inplace=True)  # remove the column created just for sorting
        else:
            raise ValueError(
                "Invalid input to sort_by. Options are 'BIC', 'AICc', 'AD', or 'Log-likelihood'. Default is 'BIC'.",
            )
        if len(df2.index.values) == 0:
            raise ValueError("You have excluded all available distributions")
        self.results = df2

        # creates a distribution object of the best fitting distribution and assigns its name
        best_dist = self.results["Distribution"].values[0]
        self.best_distribution_name = best_dist
        if best_dist == "Weibull_2P":
            self.best_distribution = Weibull_Distribution(alpha=self.Weibull_2P_alpha, beta=self.Weibull_2P_beta)
        elif best_dist == "Weibull_3P":
            self.best_distribution = Weibull_Distribution(
                alpha=self.Weibull_3P_alpha,
                beta=self.Weibull_3P_beta,
                gamma=self.Weibull_3P_gamma,
            )
        elif best_dist == "Weibull_Mixture":
            d1 = Weibull_Distribution(alpha=self.Weibull_Mixture_alpha_1, beta=self.Weibull_Mixture_beta_1)
            d2 = Weibull_Distribution(alpha=self.Weibull_Mixture_alpha_2, beta=self.Weibull_Mixture_beta_2)
            self.best_distribution = Mixture_Model(
                distributions=[d1, d2],
                proportions=[
                    self.Weibull_Mixture_proportion_1,
                    1 - self.Weibull_Mixture_proportion_1,
                ],
            )
        elif best_dist == "Weibull_CR":
            d1 = Weibull_Distribution(alpha=self.Weibull_CR_alpha_1, beta=self.Weibull_CR_beta_1)
            d2 = Weibull_Distribution(alpha=self.Weibull_CR_alpha_2, beta=self.Weibull_CR_beta_2)
            self.best_distribution = Competing_Risks_Model(distributions=[d1, d2])
        if best_dist == "Weibull_DS":
            d1 = Weibull_Distribution(alpha=self.Weibull_DS_alpha, beta=self.Weibull_DS_beta)
            self.best_distribution = DSZI_Model(distribution=d1, DS=self.Weibull_DS_DS)
        elif best_dist == "Gamma_2P":
            self.best_distribution = Gamma_Distribution(alpha=self.Gamma_2P_alpha, beta=self.Gamma_2P_beta)
        elif best_dist == "Gamma_3P":
            self.best_distribution = Gamma_Distribution(
                alpha=self.Gamma_3P_alpha,
                beta=self.Gamma_3P_beta,
                gamma=self.Gamma_3P_gamma,
            )
        elif best_dist == "Lognormal_2P":
            self.best_distribution = Lognormal_Distribution(mu=self.Lognormal_2P_mu, sigma=self.Lognormal_2P_sigma)
        elif best_dist == "Lognormal_3P":
            self.best_distribution = Lognormal_Distribution(
                mu=self.Lognormal_3P_mu,
                sigma=self.Lognormal_3P_sigma,
                gamma=self.Lognormal_3P_gamma,
            )
        elif best_dist == "Exponential_1P":
            self.best_distribution = Exponential_Distribution(Lambda=self.Exponential_1P_lambda)
        elif best_dist == "Exponential_2P":
            self.best_distribution = Exponential_Distribution(
                Lambda=self.Exponential_2P_lambda,
                gamma=self.Exponential_2P_gamma,
            )
        elif best_dist == "Normal_2P":
            self.best_distribution = Normal_Distribution(mu=self.Normal_2P_mu, sigma=self.Normal_2P_sigma)
        elif best_dist == "Beta_2P":
            self.best_distribution = Beta_Distribution(alpha=self.Beta_2P_alpha, beta=self.Beta_2P_beta)
        elif best_dist == "Loglogistic_2P":
            self.best_distribution = Loglogistic_Distribution(
                alpha=self.Loglogistic_2P_alpha,
                beta=self.Loglogistic_2P_beta,
            )
        elif best_dist == "Loglogistic_3P":
            self.best_distribution = Loglogistic_Distribution(
                alpha=self.Loglogistic_3P_alpha,
                beta=self.Loglogistic_3P_beta,
                gamma=self.Loglogistic_3P_gamma,
            )
        elif best_dist == "Gumbel_2P":
            self.best_distribution = Gumbel_Distribution(mu=self.Gumbel_2P_mu, sigma=self.Gumbel_2P_sigma)

        # print the results
        if print_results is True:  # printing occurs by default
            frac_censored = self._frac_cens * 100
            colorprint("Results from Fit_Everything:", bold=True, underline=True)
            print("Analysis method:", method)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")

        if show_histogram_plot is True:
            # plotting enabled by default
            self.histogram_plot = Fit_Everything.__histogram_plot(self)

        if show_PP_plot is True:
            # plotting enabled by default
            self.PP_plot = Fit_Everything.__P_P_plot(self)

        if show_probability_plot is True:
            # plotting enabled by default
            self.probability_plot = Fit_Everything.__probability_plot(self)

        if show_best_distribution_probability_plot is True:
            # plotting enabled by default
            self.best_distribution_probability_plot = Fit_Everything.__probability_plot(self, best_only=True)

        if (
            show_histogram_plot is True
            or show_PP_plot is True
            or show_probability_plot is True
            or show_best_distribution_probability_plot is True
        ):
            plt.show()

    def __probplot_layout(self):
        """Internal function to provide layout formatting of the plots.
        """
        items = len(self.results.index.values)  # number of items fitted
        xx1, yy1 = 2.5, 2  # multipliers for easy adjustment of window sizes
        xx2, yy2 = 0.5, 0.5
        if items == 16:
            # figsizes are in (w,h) format using the above multipliers
            cols, rows, figsize, figsizePP = (
                6,
                3,
                (xx1 * 8, yy1 * 4),
                (xx2 * 23, yy2 * 15),
            )
        elif items in [13, 14, 15]:
            cols, rows, figsize, figsizePP = (
                5,
                3,
                (xx1 * 7, yy1 * 4),
                (xx2 * 20, yy2 * 15),
            )
        elif items in [10, 11, 12]:
            cols, rows, figsize, figsizePP = (
                4,
                3,
                (xx1 * 6, yy1 * 4),
                (xx2 * 17, yy2 * 15),
            )
        elif items in [7, 8, 9]:
            cols, rows, figsize, figsizePP = (
                3,
                3,
                (xx1 * 5, yy1 * 4),
                (xx2 * 14, yy2 * 15),
            )
        elif items in [5, 6]:
            cols, rows, figsize, figsizePP = (
                3,
                2,
                (xx1 * 5, yy1 * 3),
                (xx2 * 13, yy2 * 11),
            )
        elif items == 4:
            cols, rows, figsize, figsizePP = (
                2,
                2,
                (xx1 * 4, yy1 * 3),
                (xx2 * 12, yy2 * 11),
            )
        elif items == 3:
            cols, rows, figsize, figsizePP = (
                3,
                1,
                (xx1 * 5, yy1 * 2.5),
                (xx2 * 20, yy2 * 8),
            )
        elif items == 2:
            cols, rows, figsize, figsizePP = (
                2,
                1,
                (xx1 * 4, yy1 * 2),
                (xx2 * 12, yy2 * 8),
            )
        elif items == 1:
            cols, rows, figsize, figsizePP = (
                1,
                1,
                (xx1 * 3, yy1 * 2),
                (xx2 * 12, yy2 * 8),
            )
        return cols, rows, figsize, figsizePP

    def __histogram_plot(self):
        """Generates a histogram plot of PDF and CDF of the fitted distributions.
        """
        X = self.failures
        # define plotting limits
        xmin = 0
        xmax = max(X) * 1.2

        plt.figure(figsize=(12, 6))
        # this is the order to plot things so that the legend matches the results dataframe
        plotting_order = self.results["Distribution"].values
        iqr = np.subtract(*np.percentile(X, [75, 25]))  # interquartile range
        # Freedman-Diaconis rule ==> https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        bin_width = 2 * iqr * len(X) ** -(1 / 3)
        num_bins = int(np.ceil((max(X) - min(X)) / bin_width))
        # we need to make the histogram manually (can't use plt.hist) due to need to scale the heights when there's censored data
        hist, bins = np.histogram(X, bins=num_bins, density=True)
        hist_cumulative = np.cumsum(hist) / sum(hist)
        width = np.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2

        # Probability Density Functions
        plt.subplot(132)
        plt.bar(
            center,
            hist * self._frac_fail,
            align="center",
            width=width,
            color="lightgrey",
            edgecolor="k",
            linewidth=0.5,
        )
        ls = "-"
        for counter, item in enumerate(plotting_order):
            if counter > 10:
                ls = "--"
            if item == "Weibull_2P":
                self.__Weibull_2P_params.distribution.PDF(label=r"Weibull_2P ($\alpha , \beta$)", linestyle=ls)
            elif item == "Weibull_3P":
                self.__Weibull_3P_params.distribution.PDF(label=r"Weibull_3P ($\alpha , \beta , \gamma$)", linestyle=ls)
            elif item == "Weibull_Mixture":
                self.__Weibull_Mixture_params.distribution.PDF(
                    label=r"Weibull_Mixture ($\alpha_1 , \beta_1 , \alpha_2 , \beta_2 , p_1$)",
                    linestyle=ls,
                    xmax=xmax * 2,
                )
            elif item == "Weibull_CR":
                self.__Weibull_CR_params.distribution.PDF(
                    label=r"Weibull_CR ($\alpha_1 , \beta_1 , \alpha_2 , \beta_2$)",
                    linestyle=ls,
                    xmax=xmax * 2,
                )
            elif item == "Weibull_DS":
                self.__Weibull_DS_params.distribution.PDF(
                    label=r"Weibull_DS ($\alpha , \beta , DS$)",
                    linestyle=ls,
                    xmax=xmax * 2,
                )
            elif item == "Gamma_2P":
                self.__Gamma_2P_params.distribution.PDF(label=r"Gamma_2P ($\alpha , \beta$)", linestyle=ls)
            elif item == "Gamma_3P":
                self.__Gamma_3P_params.distribution.PDF(label=r"Gamma_3P ($\alpha , \beta , \gamma$)", linestyle=ls)
            elif item == "Exponential_1P":
                self.__Exponential_1P_params.distribution.PDF(label=r"Exponential_1P ($\lambda$)", linestyle=ls)
            elif item == "Exponential_2P":
                self.__Exponential_2P_params.distribution.PDF(
                    label=r"Exponential_2P ($\lambda , \gamma$)",
                    linestyle=ls,
                )
            elif item == "Lognormal_2P":
                self.__Lognormal_2P_params.distribution.PDF(label=r"Lognormal_2P ($\mu , \sigma$)", linestyle=ls)
            elif item == "Lognormal_3P":
                self.__Lognormal_3P_params.distribution.PDF(
                    label=r"Lognormal_3P ($\mu , \sigma , \gamma$)",
                    linestyle=ls,
                )
            elif item == "Normal_2P":
                self.__Normal_2P_params.distribution.PDF(label=r"Normal_2P ($\mu , \sigma$)", linestyle=ls)
            elif item == "Gumbel_2P":
                self.__Gumbel_2P_params.distribution.PDF(label=r"Gumbel_2P ($\mu , \sigma$)", linestyle=ls)
            elif item == "Loglogistic_2P":
                self.__Loglogistic_2P_params.distribution.PDF(label=r"Loglogistic_2P ($\alpha , \beta$)", linestyle=ls)
            elif item == "Loglogistic_3P":
                self.__Loglogistic_3P_params.distribution.PDF(
                    label=r"Loglogistic_3P ($\alpha , \beta , \gamma$)",
                    linestyle=ls,
                )
            elif item == "Beta_2P":
                self.__Beta_2P_params.distribution.PDF(label=r"Beta_2P ($\alpha , \beta$)", linestyle=ls)
        handles, labels = plt.gca().get_legend_handles_labels()
        lgd = plt.gca().legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(-1.1, 1),
            frameon=False,
            title="Distribution Fitted\n",
        )
        lgd._legend_box.align = "left"
        plt.xlim(xmin, xmax)
        plt.ylim(0, max(hist * self._frac_fail) * 1.2)
        plt.title("Probability Density Function")
        plt.xlabel("Data")
        plt.ylabel("Probability density")

        # Cumulative Distribution Functions
        plt.subplot(133)
        _, ecdf_y = plotting_positions(failures=self.failures, right_censored=self.right_censored)
        plt.bar(
            center,
            hist_cumulative * max(ecdf_y),
            align="center",
            width=width,
            color="lightgrey",
            edgecolor="k",
            linewidth=0.5,
        )

        counter = 0
        ls = "-"
        for item in plotting_order:
            counter += 1
            if counter > 10:
                ls = "--"
            if item == "Weibull_2P":
                self.__Weibull_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Weibull_3P":
                self.__Weibull_3P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Weibull_Mixture":
                self.__Weibull_Mixture_params.distribution.CDF(linestyle=ls, xmax=xmax * 2)
            elif item == "Weibull_CR":
                self.__Weibull_CR_params.distribution.CDF(linestyle=ls, xmax=xmax * 2)
            elif item == "Weibull_DS":
                self.__Weibull_DS_params.distribution.CDF(linestyle=ls, xmax=xmax * 2)
            elif item == "Gamma_2P":
                self.__Gamma_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Gamma_3P":
                self.__Gamma_3P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Exponential_1P":
                self.__Exponential_1P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Exponential_2P":
                self.__Exponential_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Lognormal_2P":
                self.__Lognormal_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Lognormal_3P":
                self.__Lognormal_3P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Normal_2P":
                self.__Normal_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Gumbel_2P":
                self.__Gumbel_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Loglogistic_2P":
                self.__Loglogistic_2P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Loglogistic_3P":
                self.__Loglogistic_3P_params.distribution.CDF(plot_CI=False, linestyle=ls)
            elif item == "Beta_2P":
                self.__Beta_2P_params.distribution.CDF(linestyle=ls)
        plt.xlim(xmin, xmax)
        plt.ylim(0, max(ecdf_y) * 1.2)
        plt.title("Cumulative Distribution Function")
        plt.xlabel("Data")
        plt.ylabel("Cumulative probability density")
        plt.suptitle("Histogram plot of each fitted distribution")
        plt.subplots_adjust(left=0, bottom=0.10, right=0.97, top=0.88, wspace=0.18)
        return plt.gcf()

    def __P_P_plot(self):
        """Generates a subplot of Probability-Probability plots to compare the
        parametric vs non-parametric plots of the fitted distributions.
        """
        # Kaplan-Meier estimate of quantiles. Used in P-P plot.
        nonparametric = KaplanMeier(
            failures=self.failures,
            right_censored=self.right_censored,
            print_results=False,
            show_plot=False,
        )
        nonparametric_CDF = 1 - nonparametric.KM  # change SF into CDF

        cols, rows, _, figsizePP = Fit_Everything.__probplot_layout(self)
        # this is the order to plot things which matches the results dataframe
        plotting_order = self.results["Distribution"].values
        plt.figure(figsize=figsizePP)
        plt.suptitle(
            "Semi-parametric Probability-Probability plots of each fitted distribution\nParametric (x-axis) vs Non-Parametric (y-axis)\n",
        )
        subplot_counter = 1
        for item in plotting_order:
            plt.subplot(rows, cols, subplot_counter)

            xx = nonparametric_CDF
            plotlim = max(xx)
            if item == "Exponential_1P":
                yy = self._parametric_CDF_Exponential_1P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Exponential_2P":
                yy = self._parametric_CDF_Exponential_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Lognormal_2P":
                yy = self._parametric_CDF_Lognormal_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Lognormal_3P":
                yy = self._parametric_CDF_Lognormal_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_2P":
                yy = self._parametric_CDF_Weibull_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_3P":
                yy = self._parametric_CDF_Weibull_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_Mixture":
                yy = self._parametric_CDF_Weibull_Mixture
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_CR":
                yy = self._parametric_CDF_Weibull_CR
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Weibull_DS":
                yy = self._parametric_CDF_Weibull_DS
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Loglogistic_2P":
                yy = self._parametric_CDF_Loglogistic_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Loglogistic_3P":
                yy = self._parametric_CDF_Loglogistic_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Gamma_2P":
                yy = self._parametric_CDF_Gamma_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Gamma_3P":
                yy = self._parametric_CDF_Gamma_3P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Normal_2P":
                yy = self._parametric_CDF_Normal_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Gumbel_2P":
                yy = self._parametric_CDF_Gumbel_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy
            elif item == "Beta_2P":
                yy = self._parametric_CDF_Beta_2P
                max_yy = max(yy)
                if max_yy > plotlim:
                    plotlim = max_yy

            # downsample if necessary
            x_scatter, y_scatter = xy_downsample(xx, yy, downsample_factor=self.__downsample_scatterplot)
            # plot the scatterplot
            plt.scatter(x_scatter, y_scatter, marker=".", color="k")
            plt.title(item)
            plt.plot([-1, 2], [-1, 2], "r", alpha=0.7)  # red diagonal line
            plt.axis("square")
            plt.yticks([])
            plt.xticks([])
            plt.xlim(-plotlim * 0.05, plotlim * 1.05)
            plt.ylim(-plotlim * 0.05, plotlim * 1.05)
            subplot_counter += 1
        plt.tight_layout()
        return plt.gcf()

    def __probability_plot(self, best_only=False):
        """Generates a subplot of all the probability plots
        """
        from reliability.Probability_plotting import (
            Beta_probability_plot,
            Exponential_probability_plot_Weibull_Scale,
            Gamma_probability_plot,
            Gumbel_probability_plot,
            Loglogistic_probability_plot,
            Lognormal_probability_plot,
            Normal_probability_plot,
            Weibull_probability_plot,
        )

        plt.figure()
        if best_only is False:
            cols, rows, figsize, _ = Fit_Everything.__probplot_layout(self)
            # this is the order to plot to match the results dataframe
            plotting_order = self.results["Distribution"].values
            plt.suptitle("Probability plots of each fitted distribution\n\n")
            subplot_counter = 1
        else:
            plotting_order = [self.results["Distribution"].values[0]]

        # xvals is used by Weibull_Mixture, Weibull_CR, and Weibull_DS
        xvals = np.logspace(np.log10(min(self.failures)) - 3, np.log10(max(self.failures)) + 1, 1000)
        for item in plotting_order:
            if best_only is False:
                plt.subplot(rows, cols, subplot_counter)
            if item == "Exponential_1P":
                Exponential_probability_plot_Weibull_Scale(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Exponential_1P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Exponential_2P":
                Exponential_probability_plot_Weibull_Scale(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Exponential_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Lognormal_2P":
                Lognormal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Lognormal_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Lognormal_3P":
                Lognormal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Lognormal_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Weibull_2P":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Weibull_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Weibull_3P":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Weibull_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Weibull_Mixture":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    show_fitted_distribution=False,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
                self.__Weibull_Mixture_params.distribution.CDF(xvals=xvals)
                # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            elif item == "Weibull_CR":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    show_fitted_distribution=False,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
                self.__Weibull_CR_params.distribution.CDF(xvals=xvals)
                # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            elif item == "Weibull_DS":
                Weibull_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    show_fitted_distribution=False,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
                self.__Weibull_DS_params.distribution.CDF(xvals=xvals)
                # need to add this manually as Weibull_probability_plot can only add Weibull_2P and Weibull_3P using __fitted_dist_params
            elif item == "Loglogistic_2P":
                Loglogistic_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Loglogistic_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Loglogistic_3P":
                Loglogistic_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Loglogistic_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Gamma_2P":
                Gamma_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Gamma_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Gamma_3P":
                Gamma_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Gamma_3P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Normal_2P":
                Normal_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Normal_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Gumbel_2P":
                Gumbel_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Gumbel_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )
            elif item == "Beta_2P":
                Beta_probability_plot(
                    failures=self.failures,
                    right_censored=self.right_censored,
                    _fitted_dist_params=self.__Beta_2P_params,
                    downsample_scatterplot=self.__downsample_scatterplot,
                )

            if best_only is False:
                plt.title(item)
                ax = plt.gca()
                ax.set_yticklabels([], minor=False)
                ax.set_xticklabels([], minor=False)
                ax.set_yticklabels([], minor=True)
                ax.set_xticklabels([], minor=True)
                ax.set_ylabel("")
                ax.set_xlabel("")
                with contextlib.suppress(AttributeError):
                    ax.get_legend().remove()
                    # some plots don't have a legend added so this exception ignores them when trying to remove the legend
                subplot_counter += 1
            else:
                if self.best_distribution_name == "Weibull_Mixture":
                    title_detail = "Weibull Mixture Model"
                elif self.best_distribution_name == "Weibull_CR":
                    title_detail = "Weibull Competing Risks Model"
                elif self.best_distribution_name == "Weibull_DS":
                    title_detail = "Weibull Defective Subpopulation Model"
                else:
                    title_detail = self.best_distribution.param_title_long
                plt.title(str("Probability plot of best distribution\n" + title_detail))
        if best_only is False:
            plt.tight_layout()
            plt.gcf().set_size_inches(figsize)
        return plt.gcf()

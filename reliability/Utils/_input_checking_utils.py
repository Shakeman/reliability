import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss

from reliability.Utils._ancillary_utils import colorprint
from reliability.Utils._array_utils import (
    generate_X_array,
)


def distributions_input_checking(
    self,
    func,
    xvals,
    xmin,
    xmax,
    show_plot=None,
    plot_CI=None,
    CI_type=None,
    CI=None,
    CI_y=None,
    CI_x=None,
):
    """Performs checks and sets default values for the inputs to distributions
    sub function (PDF, CDF, SF, HF, CHF)

    Parameters
    ----------
    self : object
        Distribution object created by reliability.Distributions
    func : str
        Must be either 'PDF','CDF', 'SF', 'HF', 'CHF'
    xvals : array, list
        x-values for plotting.
    xmin : int, float
        minimum x-value for plotting.
    xmax : int, float
        maximum x-value for plotting.
    show_plot : bool
        Whether the plot is to be shown.
    plot_CI : bool, optional
        Whether the confidence intervals are to be shown on the plot. Default is
        None.
    CI_type : str, optional
        If specified, it must be "time" or "reliability". Default is None
    CI : float, optional
        The confidence intervals. If specified, it must be between 0 and 1.
        Default is None.
    CI_y : list, array, optional
        The confidence interval y-values to trace. Default is None.
    CI_x : list, array, optional
        The confidence interval x-values to trace. Default is None.

    Returns
    -------
    X : array
        An array of the x-values for the plot. Created using generate_X_array
    xvals : array, list
        x-values for plotting.
    xmin : int, float
        minimum x-value for plotting.
    xmax : int, float
        maximum x-value for plotting.
    show_plot : bool
        Whether the plot is to be shown. Default is True. Only returned if func
        is 'PDF','CDF', 'SF', 'HF, or 'CHF'
    plot_CI : bool
        Whether the confidence intervals are to be shown on the plot. Default is
        True. Only returned if func is 'CDF', 'SF', or 'CHF' and self.name
        !='Beta'.
    CI_type : str
        The type of confidence interval. Will be either "time", "reliability", or "none".
        Default is "time". Only returned if func is 'CDF', 'SF', or 'CHF'.
        If self.name =='Beta' or self.name=='Exponential' it will return 'none'. If
        self.CI_type is specified and CI_type is not specified then self.CI_type will be
        used for CI_type.
    CI : float
        The confidence intervals between 0 and 1. Default is 0.95. Only returned
        if func is 'CDF', 'SF', or 'CHF' and self.name !='Beta'. If self.CI is
        specified and CI is None then self.CI will be used for CI.
    CI_y : list, array, float, int
        The confidence interval y-values to trace. Default is None. Only
        returned if func is 'CDF', 'SF', or 'CHF' and self.name !='Beta'.
    CI_x : list, array, float, int
        The confidence interval x-values to trace. Default is None. Only
        returned if func is 'CDF', 'SF', or 'CHF' and self.name !='Beta'.

    """
    if func not in ["PDF", "CDF", "SF", "HF", "CHF", "ALL"]:
        raise ValueError("func must be either 'PDF','CDF', 'SF', 'HF', 'CHF', 'ALL'")

    # type checking
    if type(xvals) not in [type(None), list, np.ndarray, int, float, np.float64]:
        raise ValueError(
            "xvals must be an int, float, list, or array. Default is None. Value of xvals is:" + str(xvals),
        )
    if type(xmin) not in [type(None), int, float]:
        raise ValueError("xmin must be an int or float. Default is None")
    if type(xmax) not in [type(None), int, float]:
        raise ValueError("xmax must be an int or float. Default is None")
    if type(show_plot) not in [type(None), bool]:
        raise ValueError("show_plot must be True or False. Default is True")
    if type(plot_CI) not in [type(None), bool]:
        raise ValueError(
            "plot_CI must be True or False. Default is True. Only used if the distribution object was created by Fitters.",
        )
    if type(CI_type) not in [type(None), str]:
        raise ValueError(
            'CI_type must be "time" or "reliability". Default is "time". Only used if the distribution object was created by Fitters.',
        )
    if CI is True:
        CI = 0.95
    if CI is False:
        CI = 0.95
        plot_CI = False
    if type(CI) not in [type(None), float]:
        raise ValueError(
            "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval. Only used if the distribution object was created by Fitters.",
        )
    if type(CI_y) not in [type(None), list, np.ndarray, float, int]:
        raise ValueError(
            'CI_y must be a list, array, float, or int. Default is None. Only used if the distribution object was created by Fitters anc CI_type="time".',
        )
    if type(CI_x) not in [type(None), list, np.ndarray, float, int]:
        raise ValueError(
            'CI_x must be a list, array, float, or int. Default is None. Only used if the distribution object was created by Fitters anc CI_type="reliability".',
        )

    # default values
    if xmin is None and xmax is None and type(xvals) not in [list, np.ndarray, type(None)]:
        X = xvals
        show_plot = False
    else:
        X = generate_X_array(dist=self, xvals=xvals, xmin=xmin, xmax=xmax)

    if CI is None and self.Z is None:
        CI = 0.95
    elif CI is not None:  # CI takes precedence over Z
        if CI <= 0 or CI >= 1:
            raise ValueError("CI must be between 0 and 1")
    else:  # CI is None and Z is not None
        CI = 1 - ss.norm.cdf(-self.Z) * 2  # converts Z to CI

    if show_plot is None:
        show_plot = True

    no_CI_array = ["None", "NONE", "none", "OFF", "Off", "off"]
    if self.name == "Exponential":
        if CI_type not in no_CI_array and CI_type is not None:
            colorprint(
                "WARNING: CI_type is not required for the Exponential distribution since the confidence intervals of time and reliability are identical",
                text_color="red",
            )
        CI_type = None
    elif self.name == "Beta":
        if CI_type not in no_CI_array and CI_type is not None:
            colorprint(
                "WARNING: CI_type is not used for the Beta distribution since the confidence intervals are not implemented",
                text_color="red",
            )
        CI_type = None
    else:
        if CI_type is None:
            CI_type = None if self.CI_type in no_CI_array or self.CI_type is None else self.CI_type
        elif CI_type in no_CI_array:
            CI_type = None

        if isinstance(CI_type, str):
            if CI_type.upper() in ["T", "TIME"]:
                CI_type = "time"
            elif CI_type.upper() in ["R", "REL", "RELIABILITY"]:
                CI_type = "reliability"
            else:
                colorprint(
                    "WARNING: CI_type is not recognised. Accepted values are 'time', 'reliability' and 'none'",
                    text_color="red",
                )
                CI_type = None

    if CI_x is not None and CI_y is not None:
        if CI_type == "reliability":
            colorprint(
                "WARNING: CI_x and CI_y can not be specified at the same time. CI_y has been reset to None and the results for CI_x will be provided.",
                text_color="red",
            )
            CI_y = None
        else:
            colorprint(
                "WARNING: CI_x and CI_y can not be specified at the same time. CI_x has been reset to None and the results for CI_y will be provided.",
                text_color="red",
            )
            CI_x = None

    if CI_x is not None:
        if type(CI_x) in [float, int]:
            if CI_x <= 0 and self.name not in ["Normal", "Gumbel"]:
                raise ValueError("CI_x must be greater than 0")
            CI_x = np.array([CI_x])  # package as array. Will be unpacked later
        else:
            CI_x = np.asarray(CI_x)
            if min(CI_x) <= 0 and self.name not in ["Normal", "Gumbel"]:
                raise ValueError("CI_x values must all be greater than 0")

    if CI_y is not None:
        if type(CI_y) in [float, int]:
            if CI_y <= 0:
                raise ValueError("CI_y must be greater than 0")
            if CI_y >= 1 and func in ["CDF", "SF"]:
                raise ValueError("CI_y must be less than 1")
            CI_y = np.array([CI_y])  # package as array. Will be unpacked later
        else:
            CI_y = np.asarray(CI_y)
            if min(CI_y) <= 0:
                raise ValueError("CI_y values must all be above 0")
            if max(CI_y) >= 1 and func in ["CDF", "SF"]:
                raise ValueError("CI_y values must all be below 1")

    if self.name == "Beta":
        if func in ["PDF", "CDF", "SF", "HF", "CHF"]:
            return X, xvals, xmin, xmax, show_plot
        else:  # func ='ALL' which is used for the .plot() method
            return X, xvals, xmin, xmax
    elif func in ["PDF", "HF"]:
        return X, xvals, xmin, xmax, show_plot
    elif func in ["CDF", "SF", "CHF"]:
        return X, xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x
    else:  # func ='ALL' which is used for the .plot() method
        return X, xvals, xmin, xmax


def validate_CI_params(*args):
    """Returns False if any of the args is None or Nan, else returns True.
    This function is different to using all() because it performs the checks
    using np.isfinite(arg).

    Parameters
    ----------
    *args : bool
        Any number of boolean arguments

    Returns
    -------
    is_valid : bool
        False if any of the args is None or Nan else returns True.

    """
    is_valid = True
    for arg in args:
        if arg is None or np.isfinite(arg) is np.False_:
            is_valid = False
    return is_valid


class fitters_input_checking:
    """This function performs error checking and some basic default operations for
    all the inputs given to each of the fitters.

    Parameters
    ----------
    dist : str
        Must be one of "Everything", "Weibull_2P", "Weibull_3P", "Gamma_2P",
        "Gamma_3P", "Exponential_1P", "Exponential_2P", "Gumbel_2P",
        "Normal_2P", "Lognormal_2P", "Lognormal_3P", "Loglogistic_2P",
        "Loglogistic_3P", "Beta_2P", "Weibull_Mixture", "Weibull_CR",
        "Weibull_DSZI", "Weibull_DS", "Weibull_ZI".
    failures : array, list
        The failure data
    right_censored : array, list, optional
        The right censored data
    method : str
        Must be either "MLE","LS","RRX", or "RRY". Some flexibility in input is
        tolerated. eg "LS", "LEAST SQUARES", "LSQ", "NLRR", "NLLS" will all be
        recogsised as "LS". Default is MLE
    optimizer : str, optional
        Must be one of "TNC", "L-BFGS-B", "nelder-mead", "powell", "best".
        Default is None which will result in each being tried until one
        succeeds. For more detail see the `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default is 0.95 for 95%
        confidence interval (2 sided).
    quantiles : array, list, bool, optional
        An array or list of the quantiles to calculate. If True then the
        default array will be used. Default array is [0.01, 0.05, 0.1, 0.2, 0.25,
        0.5, 0.75, 0.8, 0.9, 0.95, 0.99].
        If False then no quantiles will be calculated. Default is False.
    force_beta : float, int, optional
        Used to force beta for the Weibull_2P distribution. Default is None
        which will not force beta.
    force_sigma : float, int, optional
        Used to force sigma for the Normal_2P and Lognormal_2P distributions.
        Default is None which will not force sigma.
    CI_type : str, optional
        Must be either "time" or "reliability". Default is None which results in
        "time" being used (controlled in Fitters). Some flexibility is strings
        is allowed. eg. "r", "R", "rel", "REL", "reliability", "RELIABILITY"
        will all be recognized as "reliability".

    Returns
    -------
    failures : array
        The failure times
    right_censored : array
        The right censored times. This will be an empty array if the input was
        None.
    CI : float
        The confidence interval (between 0 and 1)
    method : str, None
        This will return "MLE", "LS", "RRX", "RRY" or None.
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None.
    quantiles : array, None
        The quantiles or None.
    force_beta : float, None
        The beta parameter to be forced in Weibull_2P
    force_sigma : float, None
            The sigma parameter to be forced in Normal_2P, or Lognormal_2P
    CI_type : str, None
        "time", "reliability", 'None' or None

    Notes
    -----
    For full detail on what is checked and the errors produced, you should read
    the source code.

    Some returns are None if the input is None. How None affects the behavior
    is governed by other functions such as the individual fitters and other
    Utils.

    """

    def __init__(
        self,
        dist: str,
        failures,
        method: str | None = None,
        right_censored=None,
        optimizer: str | None = None,
        CI: float = 0.95,
        quantiles: bool | str | list | np.ndarray | None = False,
        force_beta: float | None = None,
        force_sigma: float | None = None,
        CI_type: str | None = None,
    ):
        if dist not in [
            "Everything",
            "Weibull_2P",
            "Weibull_3P",
            "Gamma_2P",
            "Gamma_3P",
            "Exponential_1P",
            "Exponential_2P",
            "Gumbel_2P",
            "Normal_2P",
            "Lognormal_2P",
            "Lognormal_3P",
            "Loglogistic_2P",
            "Loglogistic_3P",
            "Beta_2P",
            "Weibull_Mixture",
            "Weibull_CR",
            "Weibull_DSZI",
            "Weibull_DS",
            "Weibull_ZI",
        ]:
            raise ValueError("incorrect dist specified. Use the correct name. eg. Weibull_2P")

        # fill right_censored with empty list if not specified
        if right_censored is None:
            right_censored = []

        # type checking and converting to arrays for failures and right_censored
        if type(failures) not in [list, np.ndarray]:
            raise ValueError("failures must be a list or array of failure data")
        if type(right_censored) not in [list, np.ndarray]:
            raise ValueError("right_censored must be a list or array of right censored failure data")
        failures = np.asarray(failures).astype(float)
        right_censored = np.asarray(right_censored).astype(float)

        # check failures and right_censored are in the right range for the distribution
        if dist not in ["Normal_2P", "Gumbel_2P"]:
            # raise an error for values below zero
            all_data = np.hstack([failures, right_censored])
            if dist == "Beta_2P" and (min(all_data) < 0 or max(all_data) > 1):
                raise ValueError("All failure and censoring times for the beta distribution must be between 0 and 1.")
            elif min(all_data) < 0:
                raise ValueError("All failure and censoring times must be greater than zero.")
            # remove zeros and issue a warning. These are impossible since the pdf should be 0 at t=0. Leaving them in causes an error.
            rc0 = right_censored
            f0 = failures
            right_censored = rc0[rc0 != 0]
            failures = f0[f0 != 0]
            if len(failures) != len(f0):
                if dist == "Everything":
                    colorprint(
                        "WARNING: failures contained zeros. These have been removed to enable fitting of all distributions. Consider using Fit_Weibull_ZI or Fit_Weibull_DSZI if you need to include the zero inflation in the models.",
                        text_color="red",
                    )
                else:
                    colorprint(
                        str(
                            "WARNING: failures contained zeros. These have been removed to enable fitting of the "
                            + dist
                            + " distribution. Consider using Fit_Weibull_ZI or Fit_Weibull_DSZI if you need to include the zero inflation in the model.",
                        ),
                        text_color="red",
                    )

            if len(right_censored) != len(rc0):
                if dist == "Everything":
                    colorprint(
                        "WARNING: right_censored contained zeros. These have been removed to enable fitting of all distributions.",
                        text_color="red",
                    )
                else:
                    colorprint(
                        str(
                            "WARNING: right_censored contained zeros. These have been removed to enable fitting of the "
                            + dist
                            + " distribution.",
                        ),
                        text_color="red",
                    )
            if dist == "Beta_2P":
                rc1 = right_censored
                f1 = failures
                right_censored = rc1[rc1 != 1]
                failures = f1[f1 != 1]
                if len(failures) != len(f1):
                    colorprint(
                        "WARNING: failures contained ones. These have been removed to enable fitting of the Beta_2P distribution.",
                        text_color="red",
                    )
                if len(right_censored) != len(rc1):
                    colorprint(
                        "WARNING: right_censored contained ones. These have been removed to enable fitting of the Beta_2P distribution.",
                        text_color="red",
                    )

        # type and value checking for CI
        if type(CI) not in [float, np.float64]:
            raise ValueError("CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.")
        if CI <= 0 or CI >= 1:
            raise ValueError("CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.")

        # error checking for optimizer
        if optimizer is not None:
            if not isinstance(optimizer, str):
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html',
                )
            if optimizer.upper() == "TNC":
                optimizer = "TNC"
            elif optimizer.upper() == "POWELL":
                optimizer = "powell"
            elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
                optimizer = "L-BFGS-B"
            elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD", "NM"]:
                optimizer = "nelder-mead"
            elif optimizer.upper() in ["ALL", "BEST"]:
                optimizer = "best"
            else:
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html',
                )

        # error checking for method
        if method is not None:
            if not isinstance(method, str):
                raise ValueError(
                    'method must be either "MLE" (maximum likelihood estimation), "LS" (least squares), "RRX" (rank regression on X), or "RRY" (rank regression on Y).',
                )
            if method.upper() == "RRX":
                method = "RRX"
            elif method.upper() == "RRY":
                method = "RRY"
            elif method.upper() in ["LS", "LEAST SQUARES", "LSQ", "NLRR", "NLLS"]:
                method = "LS"
            elif method.upper() in [
                "MLE",
                "ML",
                "MAXIMUM LIKELIHOOD ESTIMATION",
                "MAXIMUM LIKELIHOOD",
                "MAX LIKELIHOOD",
            ]:
                method = "MLE"
            else:
                raise ValueError(
                    'method must be either "MLE" (maximum likelihood estimation), "LS" (least squares), "RRX" (rank regression on X), or "RRY" (rank regression on Y).',
                )

        # quantiles error checking
        if type(quantiles) in [str, bool]:
            if quantiles in ["auto", True, "default", "on"]:
                # quantiles to be used as the defaults in the table of quantiles #
                quantiles = np.array([0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99])
        elif quantiles is not None:
            if type(quantiles) not in [list, np.ndarray]:
                raise ValueError("quantiles must be a list or array")
            quantiles = np.asarray(quantiles)
            if max(quantiles) >= 1 or min(quantiles) <= 0:
                raise ValueError("quantiles must be between 0 and 1")

        # force_beta and force_sigma error checking
        if force_beta is not None:
            if force_beta <= 0:
                raise ValueError("force_beta must be greater than 0.")
            if isinstance(force_beta, int):
                force_beta = float(force_beta)  # autograd needs floats. crashes with ints
        if force_sigma is not None:
            if force_sigma <= 0:
                raise ValueError("force_sigma must be greater than 0.")
            if isinstance(force_sigma, int):
                force_sigma = float(force_sigma)  # autograd needs floats. crashes with ints

        # minimum number of failures checking
        if dist in ["Weibull_3P", "Gamma_3P", "Lognormal_3P", "Loglogistic_3P"]:
            min_failures = 3
        elif dist in [
            "Weibull_2P",
            "Gamma_2P",
            "Normal_2P",
            "Lognormal_2P",
            "Gumbel_2P",
            "Loglogistic_2P",
            "Beta_2P",
            "Exponential_2P",
            "Everything",
            "Weibull_ZI",
            "Weibull_DS",
            "Weibull_DSZI",
        ]:
            min_failures = 2 if force_sigma is None and force_beta is None else 1
        elif dist == "Exponential_1P":
            min_failures = 1
        elif dist in ["Weibull_Mixture", "Weibull_CR"]:
            min_failures = 4

        number_of_unique_failures = len(
            np.unique(failures),
        )  # failures need to be unique. ie. [4,4] counts as 1 distinct failure
        if number_of_unique_failures < min_failures:
            if force_beta is not None:
                raise ValueError(
                    str(
                        "The minimum number of distinct failures required for a "
                        + dist
                        + " distribution with force_beta specified is "
                        + str(min_failures)
                        + ".",
                    ),
                )
            elif force_sigma is not None:
                raise ValueError(
                    str(
                        "The minimum number of distinct failures required for a "
                        + dist
                        + " distribution with force_sigma specified is "
                        + str(min_failures)
                        + ".",
                    ),
                )
            elif dist == "Everything":
                raise ValueError(
                    "The minimum number of distinct failures required to fit everything is " + str(min_failures) + ".",
                )
            else:
                raise ValueError(
                    str(
                        "The minimum number of distinct failures required for a "
                        + dist
                        + " distribution is "
                        + str(min_failures)
                        + ".",
                    ),
                )

        # error checking for CI_type
        if type(CI_type) not in [str, type(None)]:
            raise ValueError('CI_type must be "time", "reliability", or "none"')
        if CI_type is not None:
            if CI_type.upper() in ["T", "TIME"]:
                CI_type = "time"
            elif CI_type.upper() in ["R", "REL", "RELIABILITY"]:
                CI_type = "reliability"
            elif CI_type.upper() in ["NONE", "OFF"]:
                CI_type = "none"
            else:
                raise ValueError('CI_type must be "time", "reliability", or "none"')

        # return everything
        self.failures = failures
        self.right_censored = right_censored
        self.CI: float = CI
        self.method = method
        self.optimizer: str | None = optimizer
        self.quantiles = quantiles
        self.force_beta = force_beta
        self.force_sigma = force_sigma
        self.CI_type: str | None = CI_type


class ALT_fitters_input_checking:
    """This function performs error checking and some basic default operations for
    all the inputs given to each of the ALT_fitters.

    Parameters
    ----------
    dist : str
        Must be one of "Exponential", "Weibull", "Lognormal", "Normal",
        "Everything".
    life_stress_model : str
        Must be one of "Exponential", "Eyring", "Power","Dual_Exponential",
        "Power_Exponential", "Dual_Power", "Everything".
    failures : array, list
        The failure data
    failure_stress_1 : array, list
        The stresses corresponding to the failure data
    failure_stress_2 : array, list, optional
        The second stresses corresponding to the failure data. Only required for
        dual stress models. Default is None.
    right_censored : array, list, optional
        The right censored data. Default is None.
    right_censored_stress_1 : array, list, optional
        The stresses corresponding to the right censored data. Default is None.
    right_censored_stress_2 : array, list, optional
        The second stresses corresponding to the right censored data. Only
        required for dual stress models. Default is None.
    CI : float, optional
        The confidence interval (between 0 and 1). Default is 0.95 for 95%
        confidence interval (two sided).
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None. Default is None.
    use_level_stress : float, int, list, array, optional
        The use level stress. Must be float or int for single stress models.
        Must be array or list [stress_1, stress_2] for dual stress models.
        Default is None.

    Returns
    -------
    failures : array
        The failure times
    failure_stress_1 : array
        The failure stresses
    failure_stress_2 : array
        The second failure stresses. This will be an empty array if the input
        was None.
    right_censored : array
        The right censored times. This will be an empty array if the input was
        None.
    right_censored_stress_1 : array
        The right censored failure stresses. This will be an empty array if the
        input was None.
    right_censored_stress_2 : array
        The right censored second failure stresses. This will be an empty array
        if the input was None.
    CI : float
        The confidence interval (between 0 and 1)
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None.
    use_level_stress : float, array, None
        The use level stress. This will be a float for single stress models, or
        an array for dual stress models. This will be None if the input was
        None.
    failure_groups : array
        An array of arrays. This is the failure data grouped by failure
        stresses.
    right_censored_groups : array
        An array of arrays. This is the right censored data grouped by right
        censored stresses.
    stresses_for_groups : array
        An array of arrays. These are the stresses for each of the groups.

    Notes
    -----
    For full detail on what is checked and the errors produced, you should read
    the source code.

    Some returns are None if the input is None. How None affects the behavior
    is governed by other functions such as the individual ALT fitters and other
    Utils.

    """

    def __init__(
        self,
        dist,
        life_stress_model,
        failures,
        failure_stress_1,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        CI=0.95,
        use_level_stress=None,
        optimizer=None,
    ):
        if dist not in ["Exponential", "Weibull", "Lognormal", "Normal", "Everything"]:
            raise ValueError("dist must be one of Exponential, Weibull, Lognormal, Normal, Everything.")
        if life_stress_model not in [
            "Exponential",
            "Eyring",
            "Power",
            "Dual_Exponential",
            "Power_Exponential",
            "Dual_Power",
            "Everything",
        ]:
            raise ValueError(
                "life_stess_model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power, Everything.",
            )

        if life_stress_model == "Everything":
            if failure_stress_2 is not None:
                is_dual_stress = True
                min_failures_reqd = 4
            else:
                is_dual_stress = False
                min_failures_reqd = 3
        elif life_stress_model in [
            "Dual_Exponential",
            "Power_Exponential",
            "Dual_Power",
        ]:
            is_dual_stress = True
            min_failures_reqd = 4
        else:
            is_dual_stress = False
            min_failures_reqd = 3

        # failure checks
        if is_dual_stress is True and (failure_stress_1 is None or failure_stress_2 is None):
            raise ValueError("failure_stress_1 and failure_stress_2 must be provided for dual stress models.")
        if is_dual_stress is False:
            if failure_stress_1 is None:
                raise ValueError("failure_stress_1 must be provided")
            if failure_stress_2 is not None:
                colorprint(
                    str(
                        "WARNING: failure_stress_2 is not being used as "
                        + life_stress_model
                        + " is a single stress model.",
                    ),
                    text_color="red",
                )
            failure_stress_2 = []

        # right_censored checks
        if right_censored is None:
            if right_censored_stress_1 is not None:
                colorprint(
                    "WARNING: right_censored_stress_1 is not being used as right_censored was not provided.",
                    text_color="red",
                )
            if right_censored_stress_2 is not None:
                colorprint(
                    "WARNING: right_censored_stress_2 is not being used as right_censored was not provided.",
                    text_color="red",
                )
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []
        else:
            if is_dual_stress is True and (right_censored_stress_1 is None or right_censored_stress_2 is None):
                raise ValueError(
                    "right_censored_stress_1 and right_censored_stress_2 must be provided for dual stress models.",
                )
            if is_dual_stress is False:
                if right_censored_stress_1 is None:
                    raise ValueError("right_censored_stress_1 must be provided")
                if right_censored_stress_2 is not None:
                    colorprint(
                        str(
                            "WARNING: right_censored_stress_2 is not being used as "
                            + life_stress_model
                            + " is a single stress model.",
                        ),
                        text_color="red",
                    )
                right_censored_stress_2 = []

        # type checking and converting to arrays for failures and right_censored
        if type(failures) not in [list, np.ndarray]:
            raise ValueError("failures must be a list or array of failure data")
        if type(failure_stress_1) not in [list, np.ndarray]:
            raise ValueError("failure_stress_1 must be a list or array of failure stress data")
        if type(failure_stress_2) not in [list, np.ndarray]:
            raise ValueError("failure_stress_2 must be a list or array of failure stress data")

        if type(right_censored) not in [list, np.ndarray]:
            raise ValueError("right_censored must be a list or array of right censored failure data")
        if type(right_censored_stress_1) not in [list, np.ndarray]:
            raise ValueError("right_censored_stress_1 must be a list or array of right censored failure stress data")
        if type(right_censored_stress_2) not in [list, np.ndarray]:
            raise ValueError("right_censored_stress_2 must be a list or array of right censored failure stress data")

        failures = np.asarray(failures).astype(float)
        failure_stress_1 = np.asarray(failure_stress_1).astype(float)
        failure_stress_2 = np.asarray(failure_stress_2).astype(float)
        right_censored = np.asarray(right_censored).astype(float)
        right_censored_stress_1 = np.asarray(right_censored_stress_1).astype(float)
        right_censored_stress_2 = np.asarray(right_censored_stress_2).astype(float)

        # check that list lengths match
        if is_dual_stress is False:
            if len(failures) != len(failure_stress_1):
                raise ValueError("failures must have the same number of elements as failure_stress_1")
            if len(right_censored) != len(right_censored_stress_1):
                raise ValueError("right_censored must have the same number of elements as right_censored_stress_1")
        else:
            if len(failures) != len(failure_stress_1) or len(failures) != len(failure_stress_2):
                raise ValueError(
                    "failures must have the same number of elements as failure_stress_1 and failure_stress_2",
                )
            if len(right_censored) != len(right_censored_stress_1) or len(right_censored) != len(
                right_censored_stress_2,
            ):
                raise ValueError(
                    "right_censored must have the same number of elements as right_censored_stress_1 and right_censored_stress_2",
                )

        # raise an error for values <= 0. Not even the Normal Distribution is allowed to have failures at negative life.
        if min(np.hstack([failures, right_censored])) <= 0:
            raise ValueError("All failure and right censored values must be greater than zero.")

        # type and value checking for CI
        if type(CI) not in [float, np.float64]:
            raise ValueError("CI must be between 0 and 1. Default is 0.95 for 95% confidence interval.")
        if CI <= 0 or CI >= 1:
            raise ValueError("CI must be between 0 and 1. Default is 0.95 for 95% confidence interval.")

        # error checking for optimizer
        if optimizer is not None:
            if not isinstance(optimizer, str):
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html',
                )
            if optimizer.upper() == "TNC":
                optimizer = "TNC"
            elif optimizer.upper() == "POWELL":
                optimizer = "powell"
            elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
                optimizer = "L-BFGS-B"
            elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD", "NM"]:
                optimizer = "nelder-mead"
            elif optimizer.upper() in ["ALL", "BEST"]:
                optimizer = "best"
            else:
                raise ValueError(
                    'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html',
                )

        # check the number of unique stresses
        unique_stresses_1 = np.unique(failure_stress_1)
        if len(unique_stresses_1) < 2:
            raise ValueError("failure_stress_1 must have at least 2 unique stresses.")
        if is_dual_stress is True:
            unique_stresses_2 = np.unique(failure_stress_2)
            if len(unique_stresses_2) < 2:
                raise ValueError(
                    "failure_stress_2 must have at least 2 unique stresses when using a dual stress model.",
                )

        # group the failures into their failure_stresses and then check there are enough to fit the model
        if is_dual_stress is False:
            failure_df_ungrouped = pd.DataFrame(
                data={"failures": failures, "failure_stress_1": failure_stress_1},
                columns=["failures", "failure_stress_1"],
            )
            failure_groups = []
            unique_failure_stresses = []
            for key, items in failure_df_ungrouped.groupby(["failure_stress_1"]):
                key = key[0]
                values = list(items.iloc[:, 0].values)
                failure_groups.append(values)
                unique_failure_stresses.append(key)
            # Check that there are enough failures to fit the model.
            # This does not mean 2 failures at each stress.
            # All we need is as many failures as there are parameters in the model.
            total_unique_failures = 0
            for _, failure_group in enumerate(failure_groups):
                total_unique_failures += len(np.unique(failure_group))
            if total_unique_failures < min_failures_reqd:
                if life_stress_model == "Everything":
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for all ALT models to be fitted.",
                        ),
                    )
                else:
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for the "
                            + dist
                            + "-"
                            + life_stress_model
                            + " model to be fitted.",
                        ),
                    )

            if len(right_censored) > 0:
                right_censored_df_ungrouped = pd.DataFrame(
                    data={
                        "right_censored": right_censored,
                        "right_censored_stress_1": right_censored_stress_1,
                    },
                    columns=["right_censored", "right_censored_stress_1"],
                )
                right_censored_groups = []
                unique_right_censored_stresses = []
                for key, items in right_censored_df_ungrouped.groupby(["right_censored_stress_1"]):
                    key = key[0]
                    values = list(items.iloc[:, 0].values)
                    right_censored_groups.append(values)
                    unique_right_censored_stresses.append(key)
                    if key not in unique_failure_stresses:
                        raise ValueError(
                            str("The right censored stress " + str(key) + " does not appear in failure stresses."),
                        )

                # add in empty lists for stresses which appear in failure_stress_1 but not in right_censored_stress_1
                for i, stress in enumerate(unique_failure_stresses):
                    if stress not in unique_right_censored_stresses:
                        right_censored_groups.insert(i, [])

            else:
                right_censored_groups = None
        else:  # This is for dual stress cases
            # concatenate the stresses to deal with them as a pair
            failure_stress_pairs = []
            for i in range(len(failure_stress_1)):
                failure_stress_pairs.append(str(failure_stress_1[i]) + "_" + str(failure_stress_2[i]))

            failure_df_ungrouped = pd.DataFrame(
                data={
                    "failures": failures,
                    "failure_stress_pairs": failure_stress_pairs,
                },
                columns=["failures", "failure_stress_pairs"],
            )
            failure_groups = []
            unique_failure_stresses_str = []
            for key, items in failure_df_ungrouped.groupby(["failure_stress_pairs"]):
                key = key[0]
                values = list(items.iloc[:, 0].values)
                failure_groups.append(values)
                unique_failure_stresses_str.append(key)
            # Check that there are enough failures to fit the model.
            # This does not mean 2 failures at each stress.
            # All we need is as many failures as there are parameters in the model.
            total_unique_failures = 0
            for _, failure_group in enumerate(failure_groups):
                total_unique_failures += len(np.unique(failure_group))
            if total_unique_failures < min_failures_reqd:
                if life_stress_model == "Everything":
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for all ALT models to be fitted.",
                        ),
                    )
                else:
                    raise ValueError(
                        str(
                            "There must be at least "
                            + str(min_failures_reqd)
                            + " unique failures for the "
                            + dist
                            + "-"
                            + life_stress_model
                            + "model to be fitted.",
                        ),
                    )

            # unpack the concatenated string for dual stresses ==> ['10.0_1000.0','20.0_2000.0','5.0_500.0'] should be [[10.0,1000.0],[20.0,2000.0],[5.0,500.0]]
            unique_failure_stresses = []
            for item in unique_failure_stresses_str:
                stress_pair = [float(x) for x in list(item.split("_"))]
                unique_failure_stresses.append(stress_pair)

            if len(right_censored) > 0:
                # concatenate the right censored stresses to deal with them as a pair
                right_censored_stress_pairs = []
                for i in range(len(right_censored_stress_1)):
                    right_censored_stress_pairs.append(
                        str(right_censored_stress_1[i]) + "_" + str(right_censored_stress_2[i]),
                    )

                right_censored_df_ungrouped = pd.DataFrame(
                    data={
                        "right_censored": right_censored,
                        "right_censored_stress_pairs": right_censored_stress_pairs,
                    },
                    columns=["right_censored", "right_censored_stress_pairs"],
                )
                right_censored_groups = []
                unique_right_censored_stresses_str = []
                for key, items in right_censored_df_ungrouped.groupby(["right_censored_stress_pairs"]):
                    key = key[0]
                    values = list(items.iloc[:, 0].values)
                    right_censored_groups.append(values)
                    unique_right_censored_stresses_str.append(key)
                    if key not in unique_failure_stresses_str:
                        raise ValueError(
                            str(
                                "The right censored stress pair "
                                + str([float(x) for x in list(key.split("_"))])
                                + " does not appear in failure stresses.",
                            ),
                        )

                # add in empty lists for stresses which appear in failure_stress but not in right_censored_stress
                for i, stress in enumerate(unique_failure_stresses_str):
                    if stress not in unique_right_censored_stresses_str:
                        right_censored_groups.insert(i, [])
            else:
                right_censored_groups = None

        # check that use level stress is the correct type
        if is_dual_stress is False and use_level_stress is not None:
            if type(use_level_stress) in [list, tuple, np.ndarray, str, bool, dict]:
                raise ValueError("use_level_stress must be a number")
            use_level_stress = float(use_level_stress)
        elif is_dual_stress is True and use_level_stress is not None:
            if type(use_level_stress) not in [list, np.ndarray]:
                raise ValueError(
                    "use_level_stress must be an array or list of the use level stresses. eg. use_level_stress = [stress_1, stress_2].",
                )
            if len(use_level_stress) != 2:
                raise ValueError(
                    "use_level_stress must be an array or list of length 2 with the use level stresses. eg. use_level_stress = [stress_1, stress_2].",
                )
            use_level_stress = np.asarray(use_level_stress)

        # return everything
        self.failures = failures
        self.failure_stress_1 = failure_stress_1
        self.failure_stress_2 = failure_stress_2
        self.right_censored = right_censored
        self.right_censored_stress_1 = right_censored_stress_1
        self.right_censored_stress_2 = right_censored_stress_2
        self.CI = CI
        self.optimizer = optimizer
        self.use_level_stress: float | npt.NDArray[np.float64] = use_level_stress
        self.failure_groups = failure_groups[::-1]
        if right_censored_groups is None:
            self.right_censored_groups = right_censored_groups
        else:
            self.right_censored_groups = right_censored_groups[::-1]
        self.stresses_for_groups = unique_failure_stresses[::-1]

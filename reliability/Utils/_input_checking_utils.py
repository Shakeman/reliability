from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss

from reliability.Utils._ancillary_utils import colorprint
from reliability.Utils._array_utils import (
    generate_X_array,
)


class distributions_input_checking:
    """Performs checks and sets default values for the inputs to distributions
    sub function (PDF, CDF, SF, HF, CHF).

    Parameters
    ----------
    dist : object
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

    def __init__(
        self,
        dist,
        func: str,
        xvals: list[float] | npt.NDArray[np.float64] | float | None,
        xmin: None | float | np.float64,
        xmax: None | float | np.float64,
        show_plot: None | bool = None,
        plot_CI: None | bool = None,
        CI_type: None | Literal["time", "reliability"] = None,
        CI: None | float | np.float64 = None,
        CI_y: None | list[float] | npt.NDArray[np.float64] | float | np.float64 = None,
        CI_x: None | list[float] | npt.NDArray[np.float64] | float | np.float64 = None,
    ) -> None:
        """Initialize the object with the given parameters.

        Parameters
        ----------
        - dist: The distribution object.
        - func: The function type. Must be one of "PDF", "CDF", "SF", "HF", "CHF", "ALL".
        - xvals: The x-values. Can be None, int, float, list, or numpy array.
        - xmin: The minimum x-value. Can be None, int, or float.
        - xmax: The maximum x-value. Can be None, int, or float.
        - show_plot: Whether to show the plot. Can be None or bool. Default is None.
        - plot_CI: Whether to plot the confidence interval. Can be None or bool. Default is None.
        - CI_type: The type of confidence interval. Can be None or str. Default is None.
        - CI: The confidence interval. Can be None or float. Default is None.
        - CI_y: The y-values for the confidence interval. Can be None, float, int, list, or numpy array. Default is None.
        - CI_x: The x-values for the confidence interval. Can be None, float, int, list, or numpy array. Default is None.

        """
        # code implementation...
        if func not in ["PDF", "CDF", "SF", "HF", "CHF", "ALL"]:
            msg = "func must be either 'PDF','CDF', 'SF', 'HF', 'CHF', 'ALL'"
            raise ValueError(msg)

        # type checking
        if type(xvals) not in [type(None), list, np.ndarray, int, float, np.float64]:
            raise ValueError(
                "xvals must be an int, float, list, or array. Default is None. Value of xvals is:" + str(xvals),
            )
        self.xvals = xvals
        if type(xmin) not in [type(None), int, float]:
            msg = "xmin must be an int or float. Default is None"
            raise ValueError(msg)
        self.xmin = xmin
        if type(xmax) not in [type(None), int, float]:
            msg = "xmax must be an int or float. Default is None"
            raise ValueError(msg)
        self.xmax = xmax
        if type(show_plot) not in [type(None), bool]:
            msg = "show_plot must be True or False. Default is True"
            raise ValueError(msg)
        if type(plot_CI) not in [type(None), bool]:
            msg = "plot_CI must be True or False. Default is True. Only used if the distribution object was created by Fitters."
            raise ValueError(
                msg,
            )
        if type(CI_type) not in [type(None), str]:
            msg = 'CI_type must be "time" or "reliability". Default is "time". Only used if the distribution object was created by Fitters.'
            raise ValueError(
                msg,
            )
        if CI is True:
            CI = 0.95
        if CI is False:
            CI = 0.95
            plot_CI = False
        self.plot_CI: None | bool = plot_CI
        if type(CI) not in [type(None), float]:
            msg = "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval. Only used if the distribution object was created by Fitters."
            raise ValueError(
                msg,
            )
        if type(CI_y) not in [type(None), list, np.ndarray, float, int]:
            msg = 'CI_y must be a list, array, float, or int. Default is None. Only used if the distribution object was created by Fitters anc CI_type="time".'
            raise ValueError(
                msg,
            )
        if type(CI_x) not in [type(None), list, np.ndarray, float, int]:
            msg = 'CI_x must be a list, array, float, or int. Default is None. Only used if the distribution object was created by Fitters anc CI_type="reliability".'
            raise ValueError(
                msg,
            )

        # default values
        if xmin is None and xmax is None and xvals is not None and type(xvals) not in [list, np.ndarray, type(None)]:
            X = np.asarray(xvals)
            show_plot = False
        else:
            X: npt.NDArray[np.float64] = generate_X_array(dist=dist, xvals=xvals, xmin=xmin, xmax=xmax)
        self.X: npt.NDArray[np.float64] = X
        if CI is None and dist.Z is None:
            CI = 0.95
        elif CI is not None:  # CI takes precedence over Z
            if CI <= 0 or CI >= 1:
                msg = "CI must be between 0 and 1"
                raise ValueError(msg)
        else:  # CI is None and Z is not None
            CI = 1 - ss.norm.cdf(-dist.Z) * 2  # converts Z to CI
        self.CI: float = CI
        if show_plot is None:
            show_plot = True
        self.show_plot: bool = show_plot
        no_CI_array: list[str] = ["None", "NONE", "none", "OFF", "Off", "off"]
        if dist.name == "Exponential":
            if CI_type not in no_CI_array and CI_type is not None:
                colorprint(
                    "WARNING: CI_type is not required for the Exponential distribution since the confidence intervals of time and reliability are identical",
                    text_color="red",
                )
            CI_type = None
        elif dist.name == "Beta":
            if CI_type not in no_CI_array and CI_type is not None:
                colorprint(
                    "WARNING: CI_type is not used for the Beta distribution since the confidence intervals are not implemented",
                    text_color="red",
                )
            CI_type = None
        else:
            if CI_type is None:
                CI_type = None if dist.CI_type in no_CI_array or dist.CI_type is None else dist.CI_type
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
        self.CI_type: None | Literal["time", "reliability"] = CI_type
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
            if isinstance(CI_x, float | int):
                if CI_x <= 0 and dist.name not in ["Normal", "Gumbel"]:
                    msg = "CI_x must be greater than 0"
                    raise ValueError(msg)
                CI_x = np.array([np.float64(CI_x)])  # package as array. Will be unpacked later
            else:
                CI_x = np.asarray(CI_x)
                if min(CI_x) <= 0 and dist.name not in ["Normal", "Gumbel"]:
                    msg = "CI_x values must all be greater than 0"
                    raise ValueError(msg)
        self.CI_x: npt.NDArray[np.float64] = CI_x

        if CI_y is not None:
            if isinstance(CI_y, float | int):
                if CI_y <= 0:
                    msg = "CI_y must be greater than 0"
                    raise ValueError(msg)
                if CI_y >= 1 and func in ["CDF", "SF"]:
                    msg = "CI_y must be less than 1"
                    raise ValueError(msg)
                CI_y = np.array([np.float64(CI_y)])  # package as array. Will be unpacked later
            else:
                CI_y = np.asarray(CI_y)
                if min(CI_y) <= 0:
                    msg = "CI_y values must all be above 0"
                    raise ValueError(msg)
                if max(CI_y) >= 1 and func in ["CDF", "SF"]:
                    msg = "CI_y values must all be below 1"
                    raise ValueError(msg)
        self.CI_y: npt.NDArray[np.float64] = CI_y


def validate_CI_params(*args: bool):
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
        failures: npt.NDArray[np.float64] | list[float],
        method: Literal["MLE", "RRX", "RRY", "LS", "NLLS"] = "MLE",
        right_censored: npt.NDArray[np.float64] | list[float] | None = None,
        optimizer: str | None = None,
        CI: float = 0.95,
        quantiles: bool | str | list | np.ndarray | None = False,
        force_beta: float | None = None,
        force_sigma: float | None = None,
        CI_type: str | None = None,
    ) -> None:
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
            msg = "incorrect dist specified. Use the correct name. eg. Weibull_2P"
            raise ValueError(msg)

        # fill right_censored with empty list if not specified
        if right_censored is None:
            right_censored = np.asarray([]).astype(float)

        # type checking and converting to arrays for failures and right_censored
        if type(failures) not in [list, np.ndarray]:
            msg = "failures must be a list or array of failure data"
            raise ValueError(msg)
        if type(right_censored) not in [list, np.ndarray]:
            msg = "right_censored must be a list or array of right censored failure data"
            raise ValueError(msg)
        failures = np.asarray(failures).astype(float)
        right_censored = np.asarray(right_censored).astype(float)

        # check failures and right_censored are in the right range for the distribution
        if dist not in ["Normal_2P", "Gumbel_2P"]:
            # raise an error for values below zero
            all_data = np.hstack([failures, right_censored])
            if dist == "Beta_2P" and (min(all_data) < 0 or max(all_data) > 1):
                msg = "All failure and censoring times for the beta distribution must be between 0 and 1."
                raise ValueError(msg)
            if min(all_data) < 0:
                msg = "All failure and censoring times must be greater than zero."
                raise ValueError(msg)
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
            msg = "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            raise ValueError(msg)
        if CI <= 0 or CI >= 1:
            msg = "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            raise ValueError(msg)

        # error checking for optimizer
        if optimizer is not None:
            if not isinstance(optimizer, str):
                msg = 'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
                raise ValueError(
                    msg,
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
                msg = 'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
                raise ValueError(
                    msg,
                )

        # error checking for method
        if method is not None:
            if not isinstance(method, str):
                msg = 'method must be either "MLE" (maximum likelihood estimation), "LS" (least squares), "RRX" (rank regression on X), or "RRY" (rank regression on Y).'
                raise ValueError(
                    msg,
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
                msg = 'method must be either "MLE" (maximum likelihood estimation), "LS" (least squares), "RRX" (rank regression on X), or "RRY" (rank regression on Y).'
                raise ValueError(
                    msg,
                )

        # quantiles error checking
        if type(quantiles) in [str, bool]:
            if quantiles in ["auto", True, "default", "on"]:
                # quantiles to be used as the defaults in the table of quantiles #
                quantiles = np.array([0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99])
        elif quantiles is not None:
            if type(quantiles) not in [list, np.ndarray]:
                msg = "quantiles must be a list or array"
                raise ValueError(msg)
            quantiles = np.asarray(quantiles)
            if max(quantiles) >= 1 or min(quantiles) <= 0:
                msg = "quantiles must be between 0 and 1"
                raise ValueError(msg)

        # force_beta and force_sigma error checking
        if force_beta is not None:
            if force_beta <= 0:
                msg = "force_beta must be greater than 0."
                raise ValueError(msg)
            if isinstance(force_beta, int):
                force_beta = float(force_beta)  # autograd needs floats. crashes with ints
        if force_sigma is not None:
            if force_sigma <= 0:
                msg = "force_sigma must be greater than 0."
                raise ValueError(msg)
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
            if force_sigma is not None:
                raise ValueError(
                    str(
                        "The minimum number of distinct failures required for a "
                        + dist
                        + " distribution with force_sigma specified is "
                        + str(min_failures)
                        + ".",
                    ),
                )
            if dist == "Everything":
                raise ValueError(
                    "The minimum number of distinct failures required to fit everything is " + str(min_failures) + ".",
                )
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
            msg = 'CI_type must be "time", "reliability", or "none"'
            raise ValueError(msg)
        if CI_type is not None:
            if CI_type.upper() in ["T", "TIME"]:
                CI_type = "time"
            elif CI_type.upper() in ["R", "REL", "RELIABILITY"]:
                CI_type = "reliability"
            elif CI_type.upper() in ["NONE", "OFF"]:
                CI_type = "none"
            else:
                msg = 'CI_type must be "time", "reliability", or "none"'
                raise ValueError(msg)

        # return everything
        self.failures = failures
        self.right_censored = right_censored
        self.CI: float = CI
        self.method: Literal["MLE", "RRX", "RRY", "LS"] = method
        self.optimizer: str | None = optimizer
        self.quantiles: npt.NDArray[np.float64] = quantiles
        self.force_beta = force_beta
        self.force_sigma = force_sigma
        self.CI_type: str | None = CI_type


class ALT_fitters_input_checking:
    """This function performs error checking and some basic default operations for
    al@l the inputs given to each of the ALT_fitters.

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
        use_level_stress: float | npt.NDArray[np.float64] | None = None,
        optimizer: Literal["TNC", "L-BFGS-B", "nelder-mead", "powell", "best"] | None = None,
    ) -> None:
        if dist not in ["Exponential", "Weibull", "Lognormal", "Normal", "Everything"]:
            msg = "dist must be one of Exponential, Weibull, Lognormal, Normal, Everything."
            raise ValueError(msg)
        if life_stress_model not in [
            "Exponential",
            "Eyring",
            "Power",
            "Dual_Exponential",
            "Power_Exponential",
            "Dual_Power",
            "Everything",
        ]:
            msg = "life_stess_model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power, Everything."
            raise ValueError(
                msg,
            )

        if life_stress_model == "Everything":
            is_dual_stress = failure_stress_2 is not None
        elif life_stress_model in [
            "Dual_Exponential",
            "Power_Exponential",
            "Dual_Power",
        ]:
            is_dual_stress = True
        else:
            is_dual_stress = False

        # failure checks
        if is_dual_stress is True and (failure_stress_1 is None or failure_stress_2 is None):
            msg = "failure_stress_1 and failure_stress_2 must be provided for dual stress models."
            raise ValueError(msg)
        if is_dual_stress is False:
            if failure_stress_1 is None:
                msg = "failure_stress_1 must be provided"
                raise ValueError(msg)
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
        match is_dual_stress:
            case True:
                update = alt_fitters_dual_stress_input_checking(
                    dist,
                    life_stress_model,
                    failures,
                    failure_stress_1,
                    failure_stress_2,
                    right_censored,
                    right_censored_stress_1,
                    right_censored_stress_2,
                    CI,
                    use_level_stress,
                    optimizer,
                )
                self.failure_stress_2 = update.failure_stress_2
                self.right_censored_stress_2 = update.right_censored_stress_2
            case False:
                update = alt_single_stress_fitters_input_checking(
                    dist,
                    life_stress_model,
                    failures,
                    failure_stress_1,
                    right_censored,
                    right_censored_stress_1,
                    CI,
                    use_level_stress,
                    optimizer,
                )
                self.failure_stress_2 = np.array([])
                self.right_censored_stress_2 = np.array([])
        # check that use level stress is the correct type
        if is_dual_stress is False and update.use_level_stress is not None:
            if type(use_level_stress) in [list, tuple, np.ndarray, str, bool, dict]:
                msg = "use_level_stress must be a number"
                raise ValueError(msg)
            use_level_stress = float(update.use_level_stress)
        elif is_dual_stress is True and update.use_level_stress is not None:
            if type(use_level_stress) not in [list, np.ndarray]:
                msg = "use_level_stress must be an array or list of the use level stresses. eg. use_level_stress = [stress_1, stress_2]."
                raise ValueError(
                    msg,
                )
            EXPECTED_USE_LEVEL_STRESS_LENGTH = 2

            if len(update.use_level_stress) != EXPECTED_USE_LEVEL_STRESS_LENGTH:
                msg = "use_level_stress must be an array or list of length 2 with the use level stresses. eg. use_level_stress = [stress_1, stress_2]."
                raise ValueError(
                    msg,
                )
            use_level_stress = np.asarray(update.use_level_stress)

        # returns everything
        self.failures = update.failures
        self.failure_stress_1 = update.failure_stress_1
        self.right_censored = update.right_censored
        self.right_censored_stress_1 = update.right_censored_stress_1
        self.CI = update.CI
        self.optimizer: Literal["TNC", "L-BFGS-B", "nelder-mead", "powell", "best"] | None = update.optimizer
        self.use_level_stress: float | npt.NDArray[np.float64] | None = update.use_level_stress
        self.failure_groups = update.failure_groups
        if update.right_censored_groups is None:
            self.right_censored_groups = update.right_censored_groups
        else:
            self.right_censored_groups = update.right_censored_groups
        self.stresses_for_groups = update.stresses_for_groups


class alt_fitters_dual_stress_input_checking:
    """This function performs error checking and some basic default operations for
    al@l the inputs given to each of the ALT_fitters.

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
        The second failure stresses.
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
    use_level_stress : array, None
        The use level stress. This will be an array for dual stress models. This will be None if the input was
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
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        CI=0.95,
        use_level_stress=None,
        optimizer=None,
    ) -> None:
        if dist not in ["Exponential", "Weibull", "Lognormal", "Normal", "Everything"]:
            msg = "dist must be one of Exponential, Weibull, Lognormal, Normal."
            raise ValueError(msg)
        if life_stress_model not in ["Dual_Exponential", "Power_Exponential", "Dual_Power", "Everything"]:
            msg = "life_stess_model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power, Everything."
            raise ValueError(
                msg,
            )
        min_failures_reqd = 4

        # failure checks
        if failure_stress_1 is None or failure_stress_2 is None:
            msg = "failure_stress_1 and failure_stress_2 must be provided for dual stress models."
            raise ValueError(msg)

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
        elif right_censored_stress_1 is None or right_censored_stress_2 is None:
            msg = "right_censored_stress_1 and right_censored_stress_2 must be provided for dual stress models."
            raise ValueError(
                msg,
            )

        # type checking and converting to arrays for failures and right_censored
        if type(failures) not in [list, np.ndarray]:
            msg = "failures must be a list or array of failure data"
            raise ValueError(msg)
        if type(failure_stress_1) not in [list, np.ndarray]:
            msg = "failure_stress_1 must be a list or array of failure stress data"
            raise ValueError(msg)
        if type(failure_stress_2) not in [list, np.ndarray]:
            msg = "failure_stress_2 must be a list or array of failure stress data"
            raise ValueError(msg)
        if type(right_censored) not in [list, np.ndarray]:
            msg = "right_censored must be a list or array of right censored failure data"
            raise ValueError(msg)
        if type(right_censored_stress_1) not in [list, np.ndarray]:
            msg = "right_censored_stress_1 must be a list or array of right censored failure stress data"
            raise ValueError(msg)
        if type(right_censored_stress_2) not in [list, np.ndarray]:
            msg = "right_censored_stress_2 must be a list or array of right censored failure stress data"
            raise ValueError(msg)

        failures = np.asarray(failures).astype(float)
        failure_stress_1 = np.asarray(failure_stress_1).astype(float)
        failure_stress_2 = np.asarray(failure_stress_2).astype(float)
        right_censored = np.asarray(right_censored).astype(float)
        right_censored_stress_1 = np.asarray(right_censored_stress_1).astype(float)
        right_censored_stress_2 = np.asarray(right_censored_stress_2).astype(float)

        if len(failures) != len(failure_stress_1) or len(failures) != len(failure_stress_2):
            msg = "failures must have the same number of elements as failure_stress_1 and failure_stress_2"
            raise ValueError(
                msg,
            )
        if len(right_censored) != len(right_censored_stress_1) or len(right_censored) != len(
            right_censored_stress_2,
        ):
            msg = "right_censored must have the same number of elements as right_censored_stress_1 and right_censored_stress_2"
            raise ValueError(
                msg,
            )

        # raise an error for values <= 0. Not even the Normal Distribution is allowed to have failures at negative life.
        if min(np.hstack([failures, right_censored])) <= 0:
            msg = "All failure and right censored values must be greater than zero."
            raise ValueError(msg)

        CI = check_confidence_interval(CI)

        # error checking for optimizer
        if optimizer is not None:
            optimizer = check_optimizer(optimizer)

        # check the number of unique stresses
        unique_stresses_1 = np.unique(failure_stress_1)
        MIN_UNIQUE_STRESSES = 2

        if len(unique_stresses_1) < MIN_UNIQUE_STRESSES:
            msg = "failure_stress_1 must have at least 2 unique stresses."
            raise ValueError(msg)
        unique_stresses_2 = np.unique(failure_stress_2)
        if len(unique_stresses_2) < MIN_UNIQUE_STRESSES:
            msg = "failure_stress_2 must have at least 2 unique stresses when using a dual stress model."
            raise ValueError(
                msg,
            )

        # group the failures into their failure_stresses and then check there are enough to fit the model
        # concatenate the stresses to deal with them as a pair
        failure_stress_pairs: list[str] = [
            str(failure_stress_1[i]) + "_" + str(failure_stress_2[i]) for i in range(len(failure_stress_1))
        ]

        failure_df_ungrouped = pd.DataFrame(
            data={
                "failures": failures,
                "failure_stress_pairs": failure_stress_pairs,
            },
            columns=["failures", "failure_stress_pairs"],
        )
        failure_groups = []
        unique_failure_stresses_str = []
        for keys, items in failure_df_ungrouped.groupby(["failure_stress_pairs"]):
            key = keys[0]
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
            right_censored_stress_pairs: list[str] = [
                str(right_censored_stress_1[i]) + "_" + str(right_censored_stress_2[i])
                for i in range(len(right_censored_stress_1))
            ]

            right_censored_df_ungrouped = pd.DataFrame(
                data={
                    "right_censored": right_censored,
                    "right_censored_stress_pairs": right_censored_stress_pairs,
                },
                columns=["right_censored", "right_censored_stress_pairs"],
            )
            right_censored_groups = []
            unique_right_censored_stresses_str = []
            for keys, items in right_censored_df_ungrouped.groupby(["right_censored_stress_pairs"]):
                key = keys[0]
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

        if use_level_stress is not None:
            if type(use_level_stress) not in [list, np.ndarray]:
                msg = "use_level_stress must be an array or list of the use level stresses. eg. use_level_stress = [stress_1, stress_2]."
                raise ValueError(
                    msg,
                )
            EXPECTED_USE_LEVEL_STRESS_LENGTH = 2
            if len(use_level_stress) != EXPECTED_USE_LEVEL_STRESS_LENGTH:
                msg = "use_level_stress must be an array or list of length 2 with the use level stresses. eg. use_level_stress = [stress_1, stress_2]."
                raise ValueError(
                    msg,
                )
            use_level_stress = np.asarray(use_level_stress)

        # returns everything
        self.failures = failures
        self.failure_stress_1 = failure_stress_1
        self.failure_stress_2 = failure_stress_2
        self.right_censored = right_censored
        self.right_censored_stress_1 = right_censored_stress_1
        self.right_censored_stress_2 = right_censored_stress_2
        self.CI = CI
        self.optimizer = optimizer
        self.use_level_stress = use_level_stress
        self.failure_groups = failure_groups[::-1]
        if right_censored_groups is None:
            self.right_censored_groups = right_censored_groups
        else:
            self.right_censored_groups = right_censored_groups[::-1]
        self.stresses_for_groups = unique_failure_stresses[::-1]


class alt_single_stress_fitters_input_checking:
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
    right_censored : array, list, optional
        The right censored data. Default is None.
    right_censored_stress_1 : array, list, optional
        The stresses corresponding to the right censored data. Default is None.
    CI : float, optional
        The confidence interval (between 0 and 1). Default is 0.95 for 95%
        confidence interval (two sided).
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None. Default is None.
    use_level_stress : float, int, list, array, optional
        The use level stress. Must be float or int for single stress models.
        Default is None.

    Returns
    -------
    failures : array
        The failure times
    failure_stress_1 : array
        The failure stresses
    right_censored : array
        The right censored times. This will be an empty array if the input was
        None.
    right_censored_stress_1 : array
        The right censored failure stresses. This will be an empty array if the
        input was None.
    CI : float
        The confidence interval (between 0 and 1)
    optimizer : str, None
        This will return "TNC", "L-BFGS-B", "nelder-mead", "powell", "best", or
        None.
    use_level_stress : float
        The use level stress. This will be a float for single stress models.
        This will be None if the input was
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
        right_censored=None,
        right_censored_stress_1=None,
        CI: float = 0.95,
        use_level_stress=None,
        optimizer: str | None = None,
    ) -> None:
        if dist not in ["Exponential", "Weibull", "Lognormal", "Normal", "Everything"]:
            msg = "dist must be one of Exponential, Weibull, Lognormal, Normal."
            raise ValueError(msg)
        if life_stress_model not in ["Exponential", "Eyring", "Power", "Everything"]:
            msg = "life_stess_model must be one of Exponential, Eyring, Power, Everything"
            raise ValueError(
                msg,
            )
        min_failures_reqd = 3
        # failure checks
        if failure_stress_1 is None:
            msg = "failure_stress_1 must be provided"
            raise ValueError(msg)
        # right_censored checks
        if right_censored is None:
            if right_censored_stress_1 is not None:
                colorprint(
                    "WARNING: right_censored_stress_1 is not being used as right_censored was not provided.",
                    text_color="red",
                )
            right_censored = []
            right_censored_stress_1 = []
        elif right_censored_stress_1 is None:
            msg = "right_censored_stress_1 must be provided"
            raise ValueError(msg)

        # type checking and converting to arrays for failures
        if type(failures) not in [list, np.ndarray]:
            msg = "failures must be a list or array of failure data"
            raise ValueError(msg)
        if type(failure_stress_1) not in [list, np.ndarray]:
            msg = "failure_stress_1 must be a list or array of failure stress data"
            raise ValueError(msg)
        failures = np.asarray(failures).astype(float)
        failure_stress_1 = np.asarray(failure_stress_1).astype(float)
        # check that list lengths match
        if len(failures) != len(failure_stress_1):
            msg = "failures must have the same number of elements as failure_stress_1"
            raise ValueError(msg)

        # type checking and converting to arrays for failures right_censored
        if type(right_censored) not in [list, np.ndarray]:
            msg = "right_censored must be a list or array of right censored failure data"
            raise ValueError(msg)
        if type(right_censored_stress_1) not in [list, np.ndarray]:
            msg = "right_censored_stress_1 must be a list or array of right censored failure stress data"
            raise ValueError(msg)
        right_censored = np.asarray(right_censored).astype(float)
        right_censored_stress_1 = np.asarray(right_censored_stress_1).astype(float)
        # check that list lengths match
        if len(right_censored) != len(right_censored_stress_1):
            msg = "right_censored must have the same number of elements as right_censored_stress_1"
            raise ValueError(msg)

        # raise an error for values <= 0. Not even the Normal Distribution is allowed to have failures at negative life.
        if min(np.hstack([failures, right_censored])) <= 0:
            msg = "All failure and right censored values must be greater than zero."
            raise ValueError(msg)

        CI = check_confidence_interval(CI)

        # error checking for optimizer
        if optimizer is not None:
            optimizer = check_optimizer(optimizer)

        # check the number of unique stresses
        unique_stresses_1 = np.unique(failure_stress_1)
        MIN_UNIQUE_STRESSES = 2

        if len(unique_stresses_1) < MIN_UNIQUE_STRESSES:
            msg = "failure_stress_1 must have at least 2 unique stresses."
            raise ValueError(msg)

        error_msg = str(
            "There must be at least "
            + str(min_failures_reqd)
            + " unique failures for the "
            + dist
            + "-"
            + life_stress_model
            + " model to be fitted.",
        )
        # group the failures into their failure_stresses and then check there are enough to fit the model
        failure_groups, unique_failure_stresses = group_failure_stresses(
            failures,
            failure_stress_1,
            min_failures_reqd,
            error_msg,
        )

        right_censored_groups = group_right_censored(right_censored, right_censored_stress_1, unique_failure_stresses)

        # check that use level stress is the correct type
        if use_level_stress is not None:
            if type(use_level_stress) in [list, tuple, np.ndarray, str, bool, dict]:
                msg = "use_level_stress must be a number"
                raise ValueError(msg)
            use_level_stress = float(use_level_stress)

        # returns everything
        self.failures = failures
        self.failure_stress_1 = failure_stress_1
        self.right_censored = right_censored
        self.right_censored_stress_1 = right_censored_stress_1
        self.CI: float = CI
        self.optimizer: str | None = optimizer
        self.use_level_stress: float | None = use_level_stress
        self.failure_groups = failure_groups[::-1]
        if right_censored_groups is None:
            self.right_censored_groups = right_censored_groups
        else:
            self.right_censored_groups = right_censored_groups[::-1]
        self.stresses_for_groups = unique_failure_stresses[::-1]


def check_optimizer(optimizer) -> str:
    """Check if the given optimizer is valid and return the standardized optimizer name.

    Parameters
    ----------
    optimizer (str): The optimizer to be checked.

    Returns
    -------
    str: The standardized optimizer name.

    Raises
    ------
    ValueError: If the optimizer is not one of the valid options.

    """
    # error checking for optimizer
    if not isinstance(optimizer, str):
        msg = 'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
        raise TypeError(
            msg,
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
        msg = 'optimizer must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best" or None. For more detail see the documentation: https://reliability.readthedocs.io/en/latest/Optimizers.html'
        raise ValueError(
            msg,
        )
    return optimizer


def check_confidence_interval(CI: float) -> float:
    """Check if the given confidence interval is valid and return the standardized confidence interval.

    Parameters
    ----------
    CI :float
        The confidence interval to be checked.

    Returns
    -------
    float: The standardized confidence interval.

    Raises
    ------
    ValueError: If the CI is not between 0 and 1.

    """
    if type(CI) not in [float, np.float64]:
        msg = "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval."
        raise ValueError(msg)
    if CI <= 0 or CI >= 1:
        msg = "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval."
        raise ValueError(msg)
    return CI


def group_right_censored(right_censored, right_censored_stress_1, unique_failure_stresses):
    """Group the right-censored data and group them based on the right-censored stress values.

    Args:
    ----
        right_censored (list): List of right-censored data.
        right_censored_stress_1 (list): List of right-censored stress values.
        unique_failure_stresses (list): List of unique failure stresses.

    Returns:
    -------
        list: List of grouped right-censored data based on right-censored stress values.

    """
    if len(right_censored) > 0:
        right_censored_df_ungrouped = pd.DataFrame(
            data={
                "right_censored": right_censored,
                "right_censored_stress_1": right_censored_stress_1,
            },
            columns=["right_censored", "right_censored_stress_1"],
        )
        right_censored_df_grouped = right_censored_df_ungrouped.groupby(["right_censored_stress_1"])

        unique_right_censored_stresses_df = right_censored_df_grouped.nunique()

        unique_right_censored_stresses = unique_right_censored_stresses_df.index.tolist()

        right_censored_groups = right_censored_df_grouped["right_censored"].apply(list).to_numpy().tolist()

        for keys, _ in right_censored_df_ungrouped.groupby(["right_censored_stress_1"]):
            key = keys[0]
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
    return right_censored_groups


def group_failure_stresses(failures, failure_stress_1, min_failures_reqd, error_msg):
    """Groups the failures into their failure_stresses and checks if there are enough failures to fit the model.

    Args:
    ----
        failures (list): A list of failure values.
        failure_stress_1 (list): A list of failure stress values corresponding to the failures.
        min_failures_reqd (int): The minimum number of failures required to fit the model.
        error_msg (str): The error message to raise if there are not enough failures.

    Returns:
    -------
        tuple: A tuple containing two lists - `failure_groups` and `unique_failure_stresses`.
            - `failure_groups` (list): A list of lists, where each inner list contains the failures grouped by their failure_stress_1 value.
            - `unique_failure_stresses` (list): A list of unique failure_stress_1 values.

    Raises:
    ------
        ValueError: If the total number of unique failures is less than `min_failures_reqd`.

    """
    # group the failures into their failure_stresses and then check there are enough to fit the model
    failure_df_ungrouped = pd.DataFrame(
        data={"failures": failures, "failure_stress_1": failure_stress_1},
        columns=["failures", "failure_stress_1"],
    )

    failure_df_grouped = failure_df_ungrouped.groupby(["failure_stress_1"])

    unique_failures_df = failure_df_grouped.nunique()

    unique_failure_stresses = unique_failures_df.index.tolist()

    total_unique_failures = unique_failures_df.sum().sum()

    failure_groups = failure_df_grouped["failures"].apply(list).to_numpy().tolist()
    # Check that there are enough failures to fit the model.
    # This does not mean 2 failures at each stress.
    # All we need is as many failures as there are parameters in the model.
    if total_unique_failures < min_failures_reqd:
        raise ValueError(error_msg)
    return failure_groups, unique_failure_stresses


def group_df_data(array_1, array_2):
    df_ungrouped = pd.DataFrame(
        data={"array_1": array_1, "array_2": array_2},
        columns=["array_1", "array_2"],
    )

    df_grouped = df_ungrouped.groupby(["array_2"])

    unique_df = df_grouped.nunique()

    unique_failure_stresses = unique_df.index.tolist()

    total_uniques = unique_df.sum().sum()

    list_groups_in_df = df_grouped["array_1"].apply(list).to_numpy().tolist()

    return list_groups_in_df, unique_failure_stresses, total_uniques

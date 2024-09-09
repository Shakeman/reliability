"""Nonparametric.

Provides non-parametric estimates of survival function, cumulative distribution
function, and cumulative hazard function. Three estimation methods are
implemented:
KaplanMeier
NelsonAalen
RankAdjustment

These methods arrive at very similar results but are distinctly different in
their approach. Kaplan-Meier is more popular. All three methods support failures
and right censored data. Confidence intervals are provided using the Greenwood
formula with Normal approximation (as implemented in Minitab).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss

from reliability.Utils import colorprint

pd.set_option("display.width", 200)  # prevents wrapping after default 80 characters
pd.set_option("display.max_columns", 9)  # shows the dataframe without ... truncation


class KaplanMeier:
    """Uses the Kaplan-Meier estimation method to calculate the reliability from
    failure data. Right censoring is supported and confidence bounds are
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    show_plot : bool, optional
        True or False. Default = True
    print_results : bool, optional
        Prints a dataframe of the results. True or False. Default = True
    plot_type : str
        Must be either 'SF', 'CDF', or 'CHF'. Default is SF.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    plot_CI : bool
        Shades the upper and lower confidence interval. True or False. Default =
        True
    kwargs
        Plotting keywords that are passed directly to matplotlib for the
        plot (e.g. color, label, linestyle)

    Returns
    -------
    results : dataframe
        A pandas dataframe of results for the SF
    KM : array
        The Kaplan-Meier Estimate column from results dataframe. This column is
        the non-parametric estimate of the Survival Function (reliability
        function).
    xvals : array
        the x-values to plot the stepwise plot as seen when show_plot=True
    SF : array
        survival function stepwise values (these differ from the KM values as
        there are extra values added in to make the plot into a step plot)
    CDF : array
        cumulative distribution function stepwise values
    CHF : array
        cumulative hazard function stepwise values
    SF_lower : array
        survival function stepwise values for lower CI
    SF_upper : array
        survival function stepwise values for upper CI
    CDF_lower : array
        cumulative distribution function stepwise values for lower CI
    CDF_upper : array
        cumulative distribution function stepwise values for upper CI
    CHF_lower : array
        cumulative hazard function stepwise values for lower CI
    CHF_upper : array
        cumulative hazard function stepwise values for upper CI
    data : array
        the failures and right_censored values sorted. Same as 'Failure times'
        column from results dataframe
    censor_codes : array
        the censoring codes (0 or 1) from the sorted data. Same as 'Censoring
        code (censored=0)' column from results dataframe

    Notes
    -----
    The confidence bounds are calculated using the Greenwood formula with
    Normal approximation, which is the same as featured in Minitab.

    The Kaplan-Meier method provides the SF. The CDF and CHF are obtained from
    transformations of the SF. It is not possible to obtain a useful version of
    the PDF or HF as the derivative of a stepwise function produces
    discontinuous (jagged) functions.

    """

    def __init__(
        self,
        failures: npt.NDArray[np.float64] | list[int] | None = None,
        right_censored: npt.NDArray[np.float64] | list[int] | None = None,
        CI=0.95,
    ) -> None:
        np.seterr(divide="ignore")  # divide by zero occurs if last detapoint is a failure so risk set is zero

        if failures is None:
            msg = "failures must be provided to calculate non-parametric estimates."
            raise ValueError(msg)
        if right_censored is None:
            right_censored = []  # create empty array so it can be added in hstack
        if CI < 0 or CI > 1:
            msg = "CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals."
            raise ValueError(msg)
        MIN_FAILURES = 2
        if len(failures) < MIN_FAILURES:
            raise ValueError(
                str(
                    "failures has a length of "
                    + str(len(failures))
                    + ". The minimum acceptable number of failures is 2",
                ),
            )

        # turn the failures and right censored times into a two lists of times and censoring codes
        d, censor_code = times_and_censor_codes(failures, right_censored)

        self.data: npt.NDArray[np.float64] = d
        self.censor_codes: npt.NDArray[np.float64] = censor_code

        n: int = len(d)  # number of items
        failures_array = np.arange(1, n + 1)  # array of number of items (1 to n)
        remaining_array = failures_array[::-1]  # items remaining (n to 1)
        z: np.float64 = ss.norm.ppf(1 - (1 - CI) / 2)
        q = 1 - censor_code / (remaining_array)
        KM: npt.NDArray[np.float64] = np.cumprod(q)
        R2 = KM**2
        fracs = 1 / (remaining_array * (remaining_array - 1))
        fracs[censor_code == 0] = 0
        fracsums = np.cumsum(fracs)
        deltas = (fracsums * R2) ** 0.5 * z
        KM_lower = KM - deltas
        KM_upper = KM + deltas
        KM_lower = np.array(KM_lower)
        KM_upper = np.array(KM_upper)
        KM_upper[KM_upper > 1] = 1
        KM_lower[KM_lower < 0] = 0

        # assemble the pandas dataframe for the output
        DATA = {
            "Failure times": d,
            "Censoring code (censored=0)": censor_code,
            "Items remaining": remaining_array,
            "Kaplan-Meier Estimate": KM,
            "Lower CI bound": KM_lower,
            "Upper CI bound": KM_upper,
        }
        self.results = pd.DataFrame(
            DATA,
            columns=[
                "Failure times",
                "Censoring code (censored=0)",
                "Items remaining",
                "Kaplan-Meier Estimate",
                "Lower CI bound",
                "Upper CI bound",
            ],
        )

        KM_x: npt.NDArray[np.int32] = np.array(0)
        KM_y: npt.NDArray[np.int32] = np.array(1)  # adds a start point for 100% reliability at 0 time
        KM_y_upper = []
        KM_y_lower = []

        for i in failures_array:
            if i == 1:
                if censor_code[i - 1] == 0:  # if the first item is censored
                    KM_x = np.append(KM_x, (d[i - 1]))
                    KM_y = np.append(KM_y, 1)
                    KM_y_lower.append(1)
                    KM_y_upper.append(1)
                else:  # if the first item is a failure
                    KM_x = np.append(KM_x, (d[i - 1]))
                    KM_x = np.append(KM_x, (d[i - 1]))
                    KM_y = np.append(KM_y, 1)
                    KM_y = np.append(KM_y, KM[i - 1])
                    KM_y_lower.append(1)
                    KM_y_upper.append(1)
                    KM_y_lower.append(1)
                    KM_y_upper.append(1)
            elif KM[i - 2] == KM[i - 1]:  # if the next item is censored
                KM_x = np.append(KM_x, (d[i - 1]))
                KM_y = np.append(KM_y, KM[i - 1])
                KM_y_lower.append(KM_lower[i - 2])
                KM_y_upper.append(KM_upper[i - 2])
            else:  # if the next item is a failure
                KM_x = np.append(KM_x, (d[i - 1]))
                KM_y = np.append(KM_y, KM[i - 2])
                KM_y_lower.append(KM_lower[i - 2])
                KM_y_upper.append(KM_upper[i - 2])
                KM_x = np.append(KM_x, (d[i - 1]))
                KM_y = np.append(KM_y, KM[i - 1])
                KM_y_lower.append(KM_lower[i - 2])
                KM_y_upper.append(KM_upper[i - 2])
        KM_y_lower.append(KM_y_lower[-1])
        KM_y_upper.append(KM_y_upper[-1])
        self.KM = np.array(KM)
        self.xvals = np.array(KM_x)
        self.SF = np.array(KM_y)
        self.SF_lower = np.array(KM_y_lower)
        self.SF_upper = np.array(KM_y_upper)
        self.CDF = 1 - self.SF
        self.CDF_lower = 1 - self.SF_upper
        self.CDF_upper = 1 - self.SF_lower
        self.CHF = -np.log(self.SF)
        self.CHF_lower = -np.log(self.SF_upper)
        self.CHF_upper = -np.log(self.SF_lower)  # this will be inf when SF=0

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)
        self.__CI_rounded = CI_rounded
        self.__xmax = max(d)

    def plot(self, plot_type="SF", plot_CI=True, **kwargs):
        """Plot the Kaplan-Meier estimate for the specified plot type.

        Parameters
        ----------
            plot_type (str): The type of plot to generate. Valid options are "SF" (Survival Function),
                "CDF" (Cumulative Density Function), or "CHF" (Cumulative Hazard Function). Default is SF
            plot_CI (bool, optional): Whether to plot the confidence bounds. Defaults to True.
            **kwargs: Plotting keywords that are passed directly to matplotlib for the
                plot (e.g. color, label, linestyle)

        Raises
        ------
            ValueError: If the `plot_type` is not one of the valid options.

        Returns
        -------
            None

        """
        xlim_upper = plt.xlim(auto=None)[1]
        if plot_type not in ["CDF", "SF", "CHF", "cdf", "sf", "chf"]:
            msg = "plot_type must be CDF, SF, or CHF. Default is SF."
            raise ValueError(msg)
        if plot_type in ["SF", "sf"]:
            p = plt.plot(self.xvals, self.SF, **kwargs)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str("Kaplan-Meier SF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds")
                plt.fill_between(
                    self.xvals,
                    self.SF_lower,
                    self.SF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Kaplan-Meier estimate of Survival Function"
            plt.xlabel("Failure units")
            plt.ylabel("Reliability")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim([0, 1.1])
        elif plot_type in ["CDF", "cdf"]:
            p = plt.plot(self.xvals, self.CDF, **kwargs)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str("Kaplan-Meier CDF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds")
                plt.fill_between(
                    self.xvals,
                    self.CDF_lower,
                    self.CDF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Kaplan-Meier estimate of Cumulative Density Function"
            plt.xlabel("Failure units")
            plt.ylabel("Fraction Failing")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim([0, 1.1])
        elif plot_type in ["CHF", "chf"]:
            ylims = plt.ylim(
                auto=None,
            )  # get the existing ylims so other plots are considered when setting the limits
            p = plt.plot(self.xvals, self.CHF, **kwargs)
            CHF_upper = np.nan_to_num(self.CHF_upper, posinf=1e10)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str("Kaplan-Meier CHF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds")
                plt.fill_between(
                    self.xvals,
                    self.CHF_lower,
                    CHF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Kaplan-Meier estimate of Cumulative Hazard Function"
            plt.xlabel("Failure units")
            plt.ylabel("Cumulative Hazard")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim(
                [0, max(ylims[1], self.CHF[-2] * 1.2)],
            )  # set the limits for y. Need to do this because the upper CI bound is inf.
        else:
            msg = "plot_type must be CDF, SF, CHF"
            raise ValueError(msg)

    def print_results(self):
        """Prints the results from KaplanMeier analysis with confidence interval.

        This method prints the results of the KaplanMeier analysis, including the confidence interval,
        in a formatted manner.

        Args:
        ----
            None

        Returns:
        -------
            None

        """
        colorprint(
            str("Results from KaplanMeier (" + str(self.__CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print(self.results.to_string(index=False), "\n")


class NelsonAalen:
    """Uses the Nelson-Aalen estimation method to calculate the reliability from
    failure data. Right censoring is supported and confidence bounds are
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.

    Returns
    -------
    results : dataframe
        A pandas dataframe of results for the SF
    NA : array
        The Nelson-Aalen Estimate column from results dataframe. This column is
        the non-parametric estimate of the Survival Function (reliability
        function).
    xvals : array
        the x-values to plot the stepwise plot as seen when show_plot=True
    SF : array
        survival function stepwise values (these differ from the NA values as
        there are extra values added in to make the plot into a step plot)
    CDF : array
        cumulative distribution function stepwise values
    CHF : array
        cumulative hazard function stepwise values
    SF_lower : array
        survival function stepwise values for lower CI
    SF_upper : array
        survival function stepwise values for upper CI
    CDF_lower : array
        cumulative distribution function stepwise values for lower CI
    CDF_upper : array
        cumulative distribution function stepwise values for upper CI
    CHF_lower : array
        cumulative hazard function stepwise values for lower CI
    CHF_upper : array
        cumulative hazard function stepwise values for upper CI
    data : array
        the failures and right_censored values sorted. Same as 'Failure times'
        column from results dataframe
    censor_codes : array
        the censoring codes (0 or 1) from the sorted data. Same as 'Censoring
        code (censored=0)' column from results dataframe

    Notes
    -----
    The confidence bounds are calculated using the Greenwood formula with
    Normal approximation, which is the same as featured in Minitab.

    The Nelson-Aalen method provides the SF. The CDF and CHF are obtained from
    transformations of the SF. It is not possible to obtain a useful version of
    the PDF or HF as the derivative of a stepwise function produces
    discontinuous (jagged) functions. Nelson-Aalen does obtain the HF directly
    which is then used to obtain the CHF, but this function is not smooth and is
    of little use.

    """

    def __init__(
        self,
        failures: npt.NDArray[np.float64] | list[int] | None = None,
        right_censored: npt.NDArray[np.float64] | list[int] | None = None,
        CI=0.95,
    ) -> None:
        np.seterr(divide="ignore")  # divide by zero occurs if last detapoint is a failure so risk set is zero

        if failures is None:
            msg = "failures must be provided to calculate non-parametric estimates."
            raise ValueError(msg)
        if right_censored is None:
            right_censored = []  # create empty array so it can be added in hstack
        if CI < 0 or CI > 1:
            msg = "CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals."
            raise ValueError(msg)
        MIN_FAILURES = 2
        if len(failures) < MIN_FAILURES:
            raise ValueError(
                str(
                    "failures has a length of "
                    + str(len(failures))
                    + ". The minimum acceptable number of failures is 2",
                ),
            )

        # turn the failures and right censored times into a two lists of times and censoring codes
        d, censor_code = times_and_censor_codes(failures, right_censored)

        self.data = d
        self.censor_codes = censor_code

        n: int = len(d)  # number of items
        failures_array = np.arange(1, n + 1)  # array of number of items (1 to n)
        remaining_array = failures_array[::-1]  # items remaining (n to 1)
        z = ss.norm.ppf(1 - (1 - CI) / 2)
        h = censor_code / (remaining_array)  # obtain HF
        H = np.cumsum(h)  # obtain CHF
        NA = np.exp(-H)  # Survival function
        R2 = NA**2
        fracs = 1 / (remaining_array * (remaining_array - 1))
        fracs[censor_code == 0] = 0
        fracsums = np.cumsum(fracs)
        deltas = (fracsums * R2) ** 0.5 * z
        NA_lower = NA - deltas
        NA_upper = NA + deltas
        NA_lower = np.array(NA_lower)
        NA_upper = np.array(NA_upper)
        NA_upper[NA_upper > 1] = 1
        NA_lower[NA_lower < 0] = 0

        # assemble the pandas dataframe for the output
        DATA = {
            "Failure times": d,
            "Censoring code (censored=0)": censor_code,
            "Items remaining": remaining_array,
            "Nelson-Aalen Estimate": NA,
            "Lower CI bound": NA_lower,
            "Upper CI bound": NA_upper,
        }
        self.results = pd.DataFrame(
            DATA,
            columns=[
                "Failure times",
                "Censoring code (censored=0)",
                "Items remaining",
                "Nelson-Aalen Estimate",
                "Lower CI bound",
                "Upper CI bound",
            ],
        )

        NA_x: npt.NDArray[np.int32] = np.array(0)
        NA_y: npt.NDArray[np.int32] = np.array(1)  # adds a start point for 100% reliability at 0 time
        NA_y_upper = []
        NA_y_lower = []

        for i in failures_array:
            if i == 1:
                if censor_code[i - 1] == 0:  # if the first item is censored
                    NA_x = np.append(NA_x, (d[i - 1]))
                    NA_y = np.append(NA_y, 1)
                    NA_y_lower.append(1)
                    NA_y_upper.append(1)
                else:  # if the first item is a failure
                    NA_x = np.append(NA_x, (d[i - 1]))
                    NA_x = np.append(NA_x, (d[i - 1]))
                    NA_y = np.append(NA_y, 1)
                    NA_y = np.append(NA_y, NA[i - 1])
                    NA_y_lower.append(1)
                    NA_y_upper.append(1)
                    NA_y_lower.append(1)
                    NA_y_upper.append(1)
            elif NA[i - 2] == NA[i - 1]:  # if the next item is censored
                NA_x = np.append(NA_x, (d[i - 1]))
                NA_y = np.append(NA_y, NA[i - 1])
                NA_y_lower.append(NA_lower[i - 2])
                NA_y_upper.append(NA_upper[i - 2])
            else:  # if the next item is a failure
                NA_x = np.append(NA_x, (d[i - 1]))
                NA_y = np.append(NA_y, NA[i - 2])
                NA_y_lower.append(NA_lower[i - 2])
                NA_y_upper.append(NA_upper[i - 2])
                NA_x = np.append(NA_x, (d[i - 1]))
                NA_y = np.append(NA_y, NA[i - 1])
                NA_y_lower.append(NA_lower[i - 2])
                NA_y_upper.append(NA_upper[i - 2])
        NA_y_lower.append(NA_y_lower[-1])
        NA_y_upper.append(NA_y_upper[-1])
        self.xvals = np.array(NA_x)
        self.NA = np.array(NA)
        self.SF = np.array(NA_y)
        self.SF_lower = np.array(NA_y_lower)
        self.SF_upper = np.array(NA_y_upper)
        self.CDF = 1 - self.SF
        self.CDF_lower = 1 - self.SF_upper
        self.CDF_upper = 1 - self.SF_lower
        self.CHF = -np.log(self.SF)
        self.CHF_lower = -np.log(self.SF_upper)
        self.CHF_upper = -np.log(self.SF_lower)  # this will be inf when SF=0

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)
        self.__CI_rounded = CI_rounded
        self.__xmax = max(d)

    def plot(self, plot_type="SF", plot_CI=True, **kwargs):
        """Plot the Nelson-Aalen estimate for the specified plot type.

        Parameters
        ----------
            plot_type (str): The type of plot to generate. Valid options are "SF" (Survival Function),
                "CDF" (Cumulative Density Function), or "CHF" (Cumulative Hazard Function). Default is SF
            plot_CI (bool, optional): Whether to plot the confidence bounds. Defaults to True.
            **kwargs: Plotting keywords that are passed directly to matplotlib for the
                plot (e.g. color, label, linestyle)

        Raises
        ------
            ValueError: If the `plot_type` is not one of the valid options.

        Returns
        -------
            None

        """
        if plot_type not in ["CDF", "SF", "CHF", "cdf", "sf", "chf"]:
            msg = "plot_type must be CDF, SF, or CHF. Default is SF."
            raise ValueError(msg)
        xlim_upper = plt.xlim(auto=None)[1]
        if plot_type in ["SF", "sf"]:
            p = plt.plot(self.xvals, self.SF, **kwargs)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str("Nelson-Aalen SF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds")
                plt.fill_between(
                    self.xvals,
                    self.SF_lower,
                    self.SF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Nelson-Aalen estimate of Survival Function"
            plt.xlabel("Failure units")
            plt.ylabel("Reliability")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim([0, 1.1])
        elif plot_type in ["CDF", "cdf"]:
            p = plt.plot(self.xvals, self.CDF, **kwargs)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str("Nelson-Aalen CDF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds")
                plt.fill_between(
                    self.xvals,
                    self.CDF_lower,
                    self.CDF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Nelson-Aalen estimate of Cumulative Density Function"
            plt.xlabel("Failure units")
            plt.ylabel("Fraction Failing")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim([0, 1.1])
        elif plot_type in ["CHF", "chf"]:
            ylims = plt.ylim(
                auto=None,
            )  # get the existing ylims so other plots are considered when setting the limits
            p = plt.plot(self.xvals, self.CHF, **kwargs)
            CHF_upper = np.nan_to_num(self.CHF_upper, posinf=1e10)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str("Nelson-Aalen CHF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds")
                plt.fill_between(
                    self.xvals,
                    self.CHF_lower,
                    CHF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Nelson-Aalen estimate of Cumulative Hazard Function"
            plt.xlabel("Failure units")
            plt.ylabel("Cumulative Hazard")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim(
                [0, max(ylims[1], self.CHF[-2] * 1.2)],
            )  # set the limits for y. Need to do this because the upper CI bound is inf.
        else:
            msg = "plot_type must be CDF, SF, CHF"
            raise ValueError(msg)

    def print_results(self):
        """Prints the results from NelsonAalen analysis with confidence interval.

        This method prints the results of the NelsonAalen analysis, including the confidence interval,
        in a formatted manner.

        Args:
        ----
            None

        Returns:
        -------
            None

        """
        colorprint(
            str("Results from NelsonAalen (" + str(self.__CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print(self.results.to_string(index=False), "\n")


class RankAdjustment:
    """Uses the rank-adjustment estimation method to calculate the reliability from
    failure data. Right censoring is supported and confidence bounds are
    provided.

    Parameters
    ----------
    failures : array, list
        The failure data. Must have at least 2 elements.
    right_censored : array, list, optional
        The right censored data. Optional input. Default = None.
    CI : float, optional
        confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    a - int,float,optional
        The heuristic constant for plotting positions of the form
        (k-a)/(n+1-2a). Optional input. Default is a=0.3 which is the median
        rank method (same as the default in Minitab). Must be in the range 0 to
        1. For more heuristics, see:
        https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot#Heuristics


    Returns
    -------
    results : dataframe
        A pandas dataframe of results for the SF
    RA : array
        The Rank Adjustment Estimate column from results dataframe. This column
        is the non-parametric estimate of the Survival Function (reliability
        function).
    xvals : array
        the x-values to plot the stepwise plot as seen when show_plot=True
    SF : array
        survival function stepwise values (these differ from the RA values as
        there are extra values added in to make the plot into a step plot)
    CDF : array
        cumulative distribution function stepwise values
    CHF : array
        cumulative hazard function stepwise values
    SF_lower : array
        survival function stepwise values for lower CI
    SF_upper : array
        survival function stepwise values for upper CI
    CDF_lower : array
        cumulative distribution function stepwise values for lower CI
    CDF_upper : array
        cumulative distribution function stepwise values for upper CI
    CHF_lower : array
        cumulative hazard function stepwise values for lower CI
    CHF_upper : array
        cumulative hazard function stepwise values for upper CI
    data : array
        the failures and right_censored values sorted. Same as 'Failure times'
        column from results dataframe
    censor_codes : array
        the censoring codes (0 or 1) from the sorted data. Same as 'Censoring
        code (censored=0)' column from results dataframe

    Notes
    -----
    The confidence bounds are calculated using the Greenwood formula with
    Normal approximation, which is the same as featured in Minitab.

    The rank-adjustment method provides the SF. The CDF and CHF are obtained from
    transformations of the SF. It is not possible to obtain a useful version of
    the PDF or HF as the derivative of a stepwise function produces
    discontinuous (jagged) functions.

    The Rank-adjustment algorithm is the same as is used in
    Probability_plotting.plotting_positions to obtain y-values for the scatter
    plot. As with plotting_positions, the heuristic constant "a" is accepted,
    with the default being 0.3 for median ranks.

    """

    def __init__(
        self,
        failures: npt.NDArray[np.float64] | list[float] | list[int],
        right_censored: npt.NDArray[np.float64] | list[float] | list[int] | None,
        plotting_hueristic: float = 0.3,
        CI: float = 0.95,
    ) -> None:
        if right_censored is None:
            right_censored = []  # create empty array so it can be added in hstack
        if CI < 0 or CI > 1:
            msg = "CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals."
            raise ValueError(msg)
        MINIMUM_FAILURES = 2
        if len(failures) < MINIMUM_FAILURES:
            raise ValueError(
                str(
                    "failures has a length of "
                    + str(len(failures))
                    + ". The minimum acceptable number of failures is 2",
                ),
            )

        # turn the failures and right censored times into a two lists of times and censoring codes
        d, censor_code = times_and_censor_codes(failures, right_censored)
        n: int = len(d)  # number of items
        failures_array = np.arange(1, n + 1)  # array of number of items (1 to n)
        remaining_array = failures_array[::-1]  # items remaining (n to 1)

        # obtain the rank adjustment estimates
        from reliability.Probability_plotting import (
            plotting_positions,
        )

        x, y = plotting_positions(failures=failures, right_censored=right_censored, a=plotting_hueristic, sort=True)
        # create the stepwise plot using the plotting positions
        x_array = np.insert(x.repeat(2), 0, 0)
        y_array = [0]
        for i in range(len(x)):
            if i == 0:
                y_array.extend([0, y[i]])
            else:
                y_array.extend([y[i - 1], y[i]])
        if censor_code[-1] == 0:  # repeat the last value if censored
            x_array = np.append(x_array, (d[-1]))
            y_array.append(y_array[-1])

        # convert the plotting positions (which are only for the failures) into the full Rank Adjustment column by adding the values for the censored data
        y_extended = [0]
        y_extended = np.hstack(
            [0, y]
        )  # need to add 0 to the start of the plotting positions since the CDF always starts at 0
        fail_counts = np.cumsum(censor_code).astype(int)  # the number of failures up to each point
        RA = 1 - y_extended[fail_counts]
        z = ss.norm.ppf(1 - (1 - CI) / 2)
        R2 = RA**2
        fracs = 1 / (remaining_array * (remaining_array - 1))
        fracs[censor_code == 0] = 0
        fracsums = np.cumsum(fracs)
        deltas = (fracsums * R2) ** 0.5 * z
        RA_lower = RA - deltas
        RA_upper = RA + deltas
        RA_lower = np.array(RA_lower)
        RA_upper = np.array(RA_upper)
        RA_upper[RA_upper > 1] = 1
        RA_lower[RA_lower < 0] = 0

        # create the stepwise plot for the confidence intervals.
        # first we downsample the RA_lower and RA_upper. This converts the RA_upper and RA_lower to only arrays corresponding the the values where there are failures
        RA_lower_downsample = [1]  # reliability starts at 1
        RA_upper_downsample = [1]
        for i in range(len(RA)):
            if censor_code[i] != 0:  # this means the current item is a failure
                RA_lower_downsample.append(RA_lower[i])
                RA_upper_downsample.append(RA_upper[i])
        # then we upsample by converting to stepwise plot. Essentially this is just repeating each value twice in the downsampled arrays
        RA_y_lower = []
        RA_y_upper = []
        for i in range(len(RA_lower_downsample)):
            RA_y_lower.extend([RA_lower_downsample[i], RA_lower_downsample[i]])
            RA_y_upper.extend([RA_upper_downsample[i], RA_upper_downsample[i]])
        if (
            censor_code[-1] == 1
        ):  # if the last value is a failure we need to remove the last element as the plot ends in a vertical line not a horizontal line
            RA_y_lower = RA_y_lower[0:-1]
            RA_y_upper = RA_y_upper[0:-1]

        self.xvals = x_array
        # RA are the values from the dataframe. 1 value for each time (failure or right censored). RA is for "rank adjustment" just as KM is "Kaplan-Meier"
        self.RA = np.array(RA)
        self.SF = 1 - np.array(y_array)  # these are the stepwise values for the plot.
        self.SF_lower = np.array(RA_y_lower)
        self.SF_upper = np.array(RA_y_upper)
        self.CDF = np.array(y_array)
        self.CDF_lower = 1 - self.SF_upper
        self.CDF_upper = 1 - self.SF_lower
        self.CHF = -np.log(self.SF)
        self.CHF_lower = -np.log(self.SF_upper)
        self.CHF_upper = -np.log(self.SF_lower)  # this will be inf when SF=0

        # assemble the pandas dataframe for the output
        DATA = {
            "Failure times": d,
            "Censoring code (censored=0)": censor_code,
            "Items remaining": remaining_array,
            "Rank Adjustment Estimate": self.RA,
            "Lower CI bound": RA_lower,
            "Upper CI bound": RA_upper,
        }
        self.results = pd.DataFrame(
            DATA,
            columns=[
                "Failure times",
                "Censoring code (censored=0)",
                "Items remaining",
                "Rank Adjustment Estimate",
                "Lower CI bound",
                "Upper CI bound",
            ],
        )

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)
        self.__CI_rounded = CI_rounded
        self.__xmax = max(d)

    def plot(self, plot_type="SF", plot_CI=True, **kwargs):
        """Plot the Rank-Adjustment  estimate for the specified plot type.

        Parameters
        ----------
            plot_type (str): The type of plot to generate. Valid options are "SF" (Survival Function),
                "CDF" (Cumulative Density Function), or "CHF" (Cumulative Hazard Function). Default is SF
            plot_CI (bool, optional): Whether to plot the confidence bounds. Defaults to True.
            **kwargs: Plotting keywords that are passed directly to matplotlib for the
                plot (e.g. color, label, linestyle)

        Raises
        ------
            ValueError: If the `plot_type` is not one of the valid options.

        Returns
        -------
            None

        """
        if plot_type not in ["CDF", "SF", "CHF", "cdf", "sf", "chf"]:
            msg = "plot_type must be CDF, SF, or CHF. Default is SF."
            raise ValueError(msg)
        xlim_upper = plt.xlim(auto=None)[1]
        if plot_type in ["SF", "sf"]:
            p = plt.plot(self.xvals, self.SF, **kwargs)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str("Rank-Adjustment SF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds")
                plt.fill_between(
                    self.xvals,
                    self.SF_lower,
                    self.SF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Rank Adjustment estimate of Survival Function"
            plt.xlabel("Failure units")
            plt.ylabel("Reliability")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim([0, 1.1])
        elif plot_type in ["CDF", "cdf"]:
            p = plt.plot(self.xvals, self.CDF, **kwargs)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str(
                    "Rank Adjustment CDF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds",
                )
                plt.fill_between(
                    self.xvals,
                    self.CDF_lower,
                    self.CDF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Rank Adjustment estimate of Cumulative Density Function"
            plt.xlabel("Failure units")
            plt.ylabel("Fraction Failing")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim([0, 1.1])
        elif plot_type in ["CHF", "chf"]:
            ylims = plt.ylim(
                auto=None,
            )  # get the existing ylims so other plots are considered when setting the limits
            p = plt.plot(self.xvals, self.CHF, **kwargs)
            CHF_upper = np.nan_to_num(self.CHF_upper, posinf=1e10)
            if plot_CI is True:  # plots the confidence bounds
                title_text = str(
                    "Rank Adjustment CHF estimate\n with " + str(self.__CI_rounded) + "% confidence bounds",
                )
                plt.fill_between(
                    self.xvals,
                    self.CHF_lower,
                    CHF_upper,
                    color=p[0].get_color(),
                    alpha=0.3,
                    linewidth=0,
                )
            else:
                title_text = "Rank Adjustment estimate of Cumulative Hazard Function"
            plt.xlabel("Failure units")
            plt.ylabel("Cumulative Hazard")
            plt.title(title_text)
            plt.xlim([0, max(self.__xmax, xlim_upper)])
            plt.ylim(
                [0, max(ylims[1], self.CHF[-2] * 1.2)],
            )  # set the limits for y. Need to do this because the upper CI bound is inf.
        else:
            msg = "plot_type must be CDF, SF, CHF"
            raise ValueError(msg)

    def print_results(self):
        """Prints the results from RankAdjustment analysis with confidence interval.

        This method prints the results of the KaplanMeier analysis, including the confidence interval,
        in a formatted manner.

        Args:
        ----
            None

        Returns:
        -------
            None

        """
        colorprint(
            str("Results from RankAdjustment (" + str(self.__CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print(self.results.to_string(index=False), "\n")


def times_and_censor_codes(failures, right_censored) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # turn the failures and right censored times into a two lists of times and censoring codes
    times: npt.NDArray[np.float64] = np.hstack([failures, right_censored])
    F: npt.NDArray[np.float64] = np.ones_like(failures)
    RC: npt.NDArray[np.float64] = np.zeros_like(right_censored)  # censored values are given the code of 0
    cens_code: npt.NDArray[np.float64] = np.hstack([F, RC])
    ind = np.argsort(times)
    d: npt.NDArray[np.float64] = times[ind]
    c: npt.NDArray[np.float64] = cens_code[ind]
    return d, c

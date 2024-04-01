"""Repairable systems

reliability_growth - Fits a reliability growth model to failure data using
    either the Duane model or the Crow-AMSAA model.
optimal_replacement_time - Calculates the cost model to determine how cost
    varies with replacement time. The cost model may be NHPP (as good as old)
    or HPP (as good as new).
ROCOF - rate of occurrence of failures. Uses the Laplace test to determine if
    there is a trend in the failure times.
MCF_nonparametric - Mean CUmulative Function Non-parametric. Used to determine
    if a repairable system (or collection of identical systems) is improving,
    constant, or worsening based on the rate of failures over time.
MCF_parametric - Mean Cumulative Function Parametric. Fits a parametric model to
    the data obtained from MCF_nonparametric
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss
from matplotlib.axes import SubplotBase
from matplotlib.ticker import ScalarFormatter
from scipy import integrate
from scipy.optimize import curve_fit

from reliability.Utils import colorprint, round_and_string

if TYPE_CHECKING:
    import numpy.typing as npt
class reliability_growth:
    """Fits a reliability growth model tos failure data using either the Duane
    model or the Crow-AMSAA model.

    Parameters
    ----------
    times : list, array
        The failure times relative to an initial start time. These are actual
        failure times measured from the start of the test NOT failure
        interarrival times.
    target_MTBF : float, int, optional
        The target MTBF for the reliability growth curve. Default is None.
    log_scale : bool, optional
        Sets the x and y scales to log scales. Only used if show_plot is True.
    model : str, optional
        The model to use. Must be 'Duane' or 'Crow-AMSAA'. Default is 'Duane'.

    Returns
    -------
    Lambda : float
        The Lambda parameter from the Crow-AMSAA model. Only returned if
        model='Crow-AMSAA'.
    Beta : float
        The Beta parameter from the Crow-AMSAA model. Only returned if
        model='Crow-AMSAA'.
    growth_rate : float
        The growth rate of the Crow-AMSAA model. Growth rate = 1 - Beta.
        Only returned if model='Crow-AMSAA'.
    A : float
        The A parameter from the Duane model. Only returned if model='Duane'.
    Alpha : float
        The Alpha parameter from the Duane model. Only returned if
        model='Duane'.
    DMTBF_C : float
        The Demonstrated cumulative MTBF. The is the cumulative MTBF at the
        final failure time.
    DMTBF_I : float
        The Demonstrated instantaneous MTBF. The is the instantaneous MTBF at
        the final failure time.
    DFI_C : float
        The demonstrated cumulative failure intensity. This is 1/DMTBF_C.
    DFI_I : float
        The demonstrated instantaneous failure intensity. This is 1/DMTBF_I.
    time_to_target : float, str
        The time to reach target_MTBF. If target_MTBF is None then
        time_to_target will be a str asking for the target_MTBF to be specified.
        This uses the model for cumulative MTBF.

    Notes
    -----
    For more information see the `documentation <https://reliability.readthedocs.io/en/latest/Reliability%20growth.html>`_.

    """

    def __init__(
        self,
        times: npt.NDArray[np.float64] | list[int] | None =None,
        target_MTBF=None,
        model="Duane",
    ) -> None:
        if type(times) in [list, np.ndarray]:
            times = np.sort(np.asarray(times))
        else:
            raise ValueError("times must be an array or list of failure times")

        if min(times) <= 0:
            raise ValueError("failure times cannot be negative. times must be an array or list of failure times")
        if not isinstance(model, str):
            raise ValueError('model must be either "Duane" or "Crow-AMSAA".')
        if model.upper() in ["DUANE", "D"]:
            model = "Duane"
        elif model.upper() in [
            "CROW AMSAA",
            "CROW-AMSAA",
            "CROWAMSAA",
            "CROW",
            "AMSAA",
            "CA",
            "C",
        ]:
            model = "Crow-AMSAA"
        else:
            raise ValueError('method must be either "Duane" or "Crow-AMSAA".')

        self.__model = model
        self.__target_MTBF = target_MTBF

        n: int = len(times)
        max_time: np.int32 = max(times)
        failure_numbers: npt.NDArray[np.int32] = np.array(range(1, n + 1))
        MTBF_c: npt.NDArray[np.float64] = times / failure_numbers

        if model == "Crow-AMSAA":
            self.Beta = n / (n * np.log(max_time) - np.log(times).sum())
            self.Lambda = n / (max_time**self.Beta)
            self.growth_rate = 1 - self.Beta
            self.DMTBF_I = 1 / (
                self.Lambda * self.Beta * max_time ** (self.Beta - 1)
            )  # Demonstrated MTBF (instantaneous). Reported by reliasoft
            self.DFI_I = 1 / self.DMTBF_I  # Demonstrated failure intensity (instantaneous). Reported by reliasoft
            self.DMTBF_C = (1 / self.Lambda) * max_time ** (
                1 - self.Beta
            )  # Demonstrated failure intensity (cumulative)
            self.DFI_C = 1 / self.DMTBF_C  # Demonstrated MTBF (cumulative)
        else:  # Duane
            x: npt.NDArray[np.float64] = np.log(times)
            y: npt.NDArray[np.float64] = np.log(MTBF_c)
            # fit a straight line to the data to get the model parameters
            z: npt.NDArray[np.float64] = np.polyfit(x, y, 1)
            self.Alpha: np.float64 = z[0]
            b: np.float64 = np.exp(z[1])
            self.DMTBF_C: np.float64 = b * (max_time**self.Alpha)  # Demonstrated MTBF (cumulative)
            self.DFI_C: np.float64 = 1 / self.DMTBF_C  # Demonstrated failure intensity (cumulative)
            self.DFI_I: np.float64 = (
                1 - self.Alpha
            ) * self.DFI_C  # Demonstrated failure intensity (instantaneous). Reported by reliasoft
            self.DMTBF_I: np.float64 = 1 / self.DFI_I  # Demonstrated MTBF (instantaneous). Reported by reliasoft
            self.A: np.float64 = 1 / b
            self.__b = b

        if target_MTBF is not None:
            if model == "Crow-AMSAA":
                t_target: np.float64 = (1 / (self.Lambda * target_MTBF)) ** (1 / (self.Beta - 1))
            else:  # Duane
                t_target: np.float64 = (target_MTBF / b) ** (1 / self.Alpha)
            self.time_to_target: np.float64 = t_target
        else:
            t_target = np.float64(0)
            self.time_to_target: np.float64 = np.float64(0)
            print("Specify target_MTBF to obtain the time_to_target")
        self.__times = times
        self.__MTBF_c = MTBF_c
        self.__t_target = t_target
        self.__max_time = max_time

    def print_results(self) -> None:
        """
        Prints the results of the reliability growth model parameters and demonstrated metrics.

        If the model is "Crow-AMSAA", it prints the Crow-AMSAA reliability growth model parameters:
        - Beta
        - Lambda
        - Growth rate

        If the model is not "Crow-AMSAA" (assumed to be "Duane"), it prints the Duane reliability growth model parameters:
        - Alpha
        - A

        It also prints the following demonstrated metrics:
        - Demonstrated MTBF (cumulative)
        - Demonstrated MTBF (instantaneous)
        - Demonstrated failure intensity (cumulative)
        - Demonstrated failure intensity (instantaneous)

        If a target MTBF is specified, it also prints the time to reach the target MTBF.

        Returns:
        None
        """
        if self.__model == "Crow-AMSAA":
            colorprint(
                "Crow-AMSAA reliability growth model parameters:",
                bold=True,
                underline=True,
            )
            print("Beta:", round_and_string(self.Beta))
            print("Lambda:", round_and_string(self.Lambda))
            print("Growth rate:", round_and_string(self.growth_rate))
        else:  # Duane
            colorprint(
                "Duane reliability growth model parameters:",
                bold=True,
                underline=True,
            )
            print("Alpha:", round_and_string(self.Alpha))
            print("A:", round_and_string(self.A))
        print("Demonstrated MTBF (cumulative):", round_and_string(self.DMTBF_C))
        print("Demonstrated MTBF (instantaneous):", round_and_string(self.DMTBF_I))
        print("Demonstrated failure intensity (cumulative):", round_and_string(self.DFI_C))
        print("Demonstrated failure intensity (instantaneous):", round_and_string(self.DFI_I))

        if self.__target_MTBF is not None:
            print("Time to reach target MTBF:", round_and_string(self.time_to_target))
        print("")  # blank line

    def plot(self, log_scale=False, **kwargs) -> plt.Axes:
        """
        Plot the reliability growth curve.

        Parameters:
        - log_scale (bool): If True, the x and y axes will be displayed in logarithmic scale.
        - **kwargs: Additional keyword arguments to customize the plot.

        Returns:
        - matplotlib.axes.Axes: The current axes instance.

        """
        if log_scale is True:
            xmax = 10 ** np.ceil(np.log10(max(self.__max_time, self.__t_target)))
            x_array = np.geomspace(0.00001, xmax * 100, 1000)
        else:
            xmax = max(self.__max_time, self.__t_target) * 2
            x_array = np.linspace(0, xmax, 1000)

        if self.__model == "Crow-AMSAA":
            MTBF = 1 / (self.Lambda * x_array ** (self.Beta - 1))
        else:  # Duane
            MTBF = self.__b * x_array**self.Alpha

        # kwargs handling
        c = kwargs.pop("color") if "color" in kwargs else "steelblue"
        marker = kwargs.pop("marker") if "marker" in kwargs else "o"
        if "label" in kwargs:
            label = kwargs.pop("label")
        elif self.__model == "Crow-AMSAA":
            label = "Crow-AMSAA reliability growth curve"
        else:
            label = "Duane reliability growth curve"

        plt.plot(x_array, MTBF, color=c, label=label, **kwargs)
        plt.scatter(self.__times, self.__MTBF_c, color="k", marker=marker)

        if self.__target_MTBF is not None:
            # this section checks if "Target MTBF" is already in the legend
            # and if so it doesn't add it again. This is done since plotting
            # Duane on top of Crow-AMSAA would create duplicates in the
            # legend
            leg = plt.gca().get_legend()
            if leg is not None:
                target_plotted = False
                for item in leg.texts:
                    if item._text == "Target MTBF":
                        target_plotted = True
                target_label = None if target_plotted is True else "Target MTBF"
            else:
                target_label = "Target MTBF"
            # plot the red line tracing the target MTBF
            plt.plot(
                np.array([0, self.__t_target, self.__t_target]),
                np.array([self.__target_MTBF, self.__target_MTBF, 0]),
                color="red",
                linewidth=1,
                label=target_label,
            )
        plt.title("MTBF vs Time")
        plt.xlabel("Time")
        plt.ylabel("Cumulative MTBF")
        plt.legend()

        if log_scale is True:
            ymin = 10 ** np.floor(np.log10(min(self.__MTBF_c)))
            if self.__target_MTBF is not None:
                xmin = 10 ** np.floor(np.log10(min(min(self.__times), self.__target_MTBF)))
                ymax = 10 ** np.ceil(np.log10(max(max(self.__MTBF_c), self.__target_MTBF) * 1.2))
            else:
                xmin = 10 ** np.floor(np.log10(min(self.__times)))
                ymax = 10 ** np.ceil(np.log10(max(self.__MTBF_c) * 1.2))
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.xscale("log")
            plt.yscale("log")
            plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        else:
            plt.xlim(0, xmax)
            if self.__target_MTBF is not None:
                plt.ylim(0, max(max(self.__MTBF_c), self.__target_MTBF) * 1.2)
            else:
                plt.ylim(0, max(self.__MTBF_c) * 1.2)
        plt.tight_layout()
        return plt.gca()


class optimal_replacement_time:
    """Calculates the cost model to determine how cost varies with replacement time.
    The cost model may be HPP (good as new replacement) or NHPP (as good as old
    replacement). Default is HPP.

    Parameters
    ----------
    Cost_PM : int, float
        The cost of preventative maintenance (must be smaller than Cost_CM)
    Cost_CM : int, float
        The cost of corrective maintenance (must be larger than Cost_PM)
    weibull_alpha : int, float
        The scale parameter of the underlying Weibull distribution.
    weibull_beta : int, float
        The shape parameter of the underlying Weibull distribution. Should be
        greater than 1 otherwise conducting PM is not economical.
    q : int, optional
        The restoration factor. Must be 0 or 1. Use q=1 for Power Law NHPP
        (as good as old) or q=0 for HPP (as good as new). Default is q=0 (as
        good as new).

    Returns
    -------
    ORT : float
        The optimal replacement time
    min_cost : float
        The minimum cost per unit time

    """

    def __init__(
        self,
        cost_PM: float,
        cost_CM: float,
        weibull_alpha: float,
        weibull_beta: float,
        q: int = 0,
    ):
        if cost_PM > cost_CM:
            raise ValueError(
                "Cost_PM must be less than Cost_CM otherwise preventative maintenance should not be conducted.",
            )
        if weibull_beta < 1:
            colorprint(
                "WARNING: weibull_beta is < 1 so the hazard rate is decreasing, therefore preventative maintenance should not be conducted.",
                text_color="red",
            )

        if q == 1:  # as good as old
            alpha_multiple = 4
            t = np.linspace(1, weibull_alpha * alpha_multiple, 100000)
            CPUT = ((cost_PM * (t / weibull_alpha) ** weibull_beta) + cost_CM) / t
            ORT = weibull_alpha * ((cost_CM / (cost_PM * (weibull_beta - 1))) ** (1 / weibull_beta))
            min_cost = ((cost_PM * (ORT / weibull_alpha) ** weibull_beta) + cost_CM) / ORT
        elif q == 0:  # as good as new
            alpha_multiple = 3
            t = np.linspace(1, weibull_alpha * alpha_multiple, 10000)

            # survival function and its integral
            def calc_SF(x):
                return np.exp(-((x / weibull_alpha) ** weibull_beta))

            def integrate_SF(x):
                return integrate.quad(calc_SF, 0, x)[0]

            # vectorize them
            vcalc_SF = np.vectorize(calc_SF)
            vintegrate_SF = np.vectorize(integrate_SF)

            # calculate the SF and intergral at each time
            sf = vcalc_SF(t)
            integral = vintegrate_SF(t)

            self.__sf = sf
            self.__integral = integral

            CPUT = (cost_PM * sf + cost_CM * (1 - sf)) / integral
            idx = np.argmin(CPUT)
            min_cost = CPUT[idx]  # minimum cost per unit time
            ORT = t[idx]  # optimal replacement time
        else:
            raise ValueError(
                'q must be 0 or 1. Default is 0. Use 0 for "as good as new" and use 1 for "as good as old".',
            )
        self.ORT = ORT
        self.min_cost = min_cost
        self.__min_cost_rounded = round_and_string(min_cost, decimals=2)
        self.__ORT_rounded = round_and_string(ORT, decimals=2)
        self.__q = q
        self.__t = t
        self.__CPUT = CPUT
        self.__weibull_alpha = weibull_alpha
        self.__weibull_beta = weibull_beta
        self.__alpha_multiple = alpha_multiple
        self.__cost_CM = cost_CM
        self.__cost_PM = cost_PM

    def print_results(self) -> None:
            """
            Prints the results from the optimal_replacement_time calculation.

            The method prints the cost model assumption and the minimum cost per unit time
            along with the optimal replacement time.

            Parameters:
                None

            Returns:
                None
            """
            colorprint("Results from optimal_replacement_time:", bold=True, underline=True)
            if self.__q == 0:
                print("Cost model assuming as good as new replacement (q=0):")
            else:
                print("Cost model assuming as good as old replacement (q=1):")
            print(
                "The minimum cost per unit time is",
                self.__min_cost_rounded,
                "\nThe optimal replacement time is",
                self.__ORT_rounded,
            )

    def show_time_plot(self, subplot=None, **kwargs):
        """
        Display a time plot of the repairable system.

        Args:
            subplot (matplotlib.axes.Axes, optional): The subplot to use for the plot. If not provided, a new figure will be created.
            **kwargs: Additional keyword arguments to customize the plot.

        Returns:
            matplotlib.axes.Axes: The current axes instance.

        Raises:
            None

        Example usage:
            system = RepairableSystem()
            system.show_time_plot(subplot=ax, color='red', linestyle='--')

        """
        c = kwargs.pop("color") if "color" in kwargs else "steelblue"
        if subplot is not None and issubclass(type(subplot), SubplotBase):
            plt.sca(ax=subplot)  # use the axes passed
        else:
            plt.figure()  # if no axes is passed, make a new figure
        plt.plot(self.__t, self.__CPUT, color=c, **kwargs)
        plt.plot(self.ORT, self.min_cost, "o", color=c)
        text_str = str(
            "\nMinimum cost per unit time is "
            + str(self.__min_cost_rounded)
            + "\nOptimal replacement time is "
            + str(self.__ORT_rounded),
        )
        plt.text(self.ORT, self.min_cost, text_str, va="top")
        plt.xlabel("Replacement time")
        plt.ylabel("Cost per unit time")
        plt.title("Optimal replacement time estimation")
        plt.ylim([0, self.min_cost * 2])
        plt.xlim([0, self.__weibull_alpha * self.__alpha_multiple])
        return plt.gca()

    def show_ratio_plot(self, subplot=None):
        """
        Displays a plot of the optimal replacement interval across a range of CM costs.

        Args:
            subplot (matplotlib.axes.Axes, optional): The subplot to use for the plot. If not provided, a new figure will be created.

        Returns:
            matplotlib.axes.Axes: The current axes instance.

        Raises:
            None
        """
        if subplot is not None and issubclass(type(subplot), SubplotBase):
            plt.sca(ax=subplot)  # use the axes passed
        else:
            plt.figure()  # if no axes is passed, make a new figure
        xupper = np.round(self.__cost_CM / self.__cost_PM, 0) * 2
        CC_CP = np.linspace(1, xupper, 200)  # cost CM / cost PM
        CC = CC_CP * self.__cost_PM
        ORT_array = []  # optimal replacement time

        # get the ORT from the minimum CPUT for each CC

        def calc_ORT(x):
            if self.__q == 1:
                return self.__weibull_alpha * (x / (self.__cost_PM * (self.__weibull_beta - 1))) ** (1 / self.__weibull_beta)

            else:  # q = 0
                return self.__t[np.argmin((self.__cost_PM * self.__sf + x * (1 - self.__sf)) / self.__integral)]

        vcalc_ORT = np.vectorize(calc_ORT)
        ORT_array = vcalc_ORT(CC)

        plt.plot(CC_CP, ORT_array)
        plt.xlim(1, xupper)
        plt.ylim(0, self.ORT * 2)
        plt.scatter(self.__cost_CM / self.__cost_PM, self.ORT)
        # vertical alignment based on plot increasing or decreasing
        if ORT_array[50] > ORT_array[40]:
            va = "top"
            mult = 0.95
        else:
            va = "bottom"
            mult = 1.05
        plt.text(
            s=str(
                "$cost_{CM} = $"
                + str(self.__cost_CM)
                + "\n$cost_{PM} = $"
                + str(self.__cost_PM)
                + "\nInterval = "
                + round_and_string(self.ORT, 2),
            ),
            x=self.__cost_CM / self.__cost_PM * 1.05,
            y=self.ORT * mult,
            ha="left",
            va=va,
        )
        plt.xlabel(r"Cost ratio $\left(\frac{CM}{PM}\right)$")
        plt.ylabel("Replacement Interval")
        plt.title("Optimal replacement interval\nacross a range of CM costs")
        return plt.gca()


class ROCOF:
    """Uses the failure times or failure interarrival times to determine if there
    is a trend in those times. The test for statistical significance is the
    Laplace test which compares the Laplace test statistic (U) with the z value
    (z_crit) from the standard normal distribution. If there is a statistically
    significant trend, the parameters of the model (Lambda_hat and Beta_hat) are
    calculated. By default the results are printed and a plot of the times and
    MTBF is plotted.

    Parameters
    ----------
    times_between_failures : array, list, optional
        The failure interarrival times. See the Notes below.
    failure_times : array, list, optional
        The actual failure times. See the Notes below.
    test_end : int, float, optional
        Use this to specify the end of the test if the test did not end at the
        time of the last failure. Default = None which will result in the last
        failure being treated as the end of the test.
    CI : float
        The confidence interval for the Laplace test. Must be between 0 and 1.
        Default is 0.95 for 95% CI.

    Returns
    -------
    U : float
        The Laplace test statistic
    z_crit : tuple
        (lower,upper) bound on z value. This is based on the CI.
    trend : str
        'improving','worsening','constant'. This is based on the comparison of U
        with z_crit
    Beta_hat : float, str
        The Beta parameter for the NHPP Power Law model. Only calculated if the
        trend is not constant, else a string is returned.
    Lambda_hat : float, str
        The Lambda parameter for the NHPP Power Law model. Only calculated if
        the trend is not constant.
    ROCOF : float, str
        The Rate of OCcurrence Of Failures. Only calculated if the trend is
        constant. If trend is not constant then ROCOF changes over time in
        accordance with Beta_hat and Lambda_hat. In this case a string will be
        returned.

    Notes
    -----
    You can specify either times_between_failures OR failure_times but not both.
    Both options are provided for convenience so the conversion between the two
    is done internally. failure_times should be the same as
    np.cumsum(times_between_failures).

    The repair time is assumed to be negligible. If the repair times are not
    negligibly small then you will need to manually adjust your input to factor
    in the repair times.

    If show_plot is True, the ROCOF plot will be produced. Use plt.show() to
    show the plot.

    """

    def __init__(
        self,
        times_between_failures=None,
        failure_times=None,
        CI=0.95,
        test_end=None,
    ):
        if times_between_failures is not None and failure_times is not None:
            raise ValueError(
                "You have specified both times_between_failures and failure times. You can specify one but not both. Use times_between_failures for failure interarrival times, and failure_times for the actual failure times. failure_times should be the same as np.cumsum(times_between_failures)",
            )
        if times_between_failures is not None:
            if any(t <= 0 for t in times_between_failures):
                raise ValueError("times_between_failures cannot be less than zero")
            if isinstance(times_between_failures, list):
                ti = times_between_failures
            elif type(times_between_failures) == np.ndarray:
                ti = list(times_between_failures)
            else:
                raise ValueError("times_between_failures must be a list or array")
        if failure_times is not None:
            if any(t <= 0 for t in failure_times):
                raise ValueError("failure_times cannot be less than zero")
            if isinstance(failure_times, list):
                failure_times = np.sort(np.array(failure_times))
            elif type(failure_times) == np.ndarray:
                failure_times = np.sort(failure_times)
            else:
                raise ValueError("failure_times must be a list or array")
            failure_times[1:] -= failure_times[:-1].copy()  # this is the opposite of np.cumsum
            ti = list(failure_times)
        if test_end is not None and type(test_end) not in [float, int]:
            raise ValueError(
                "test_end should be a float or int. Use test_end to specify the end time of a test which was not failure terminated.",
            )
        if CI <= 0 or CI >= 1:
            raise ValueError("CI must be between 0 and 1. Default is 0.95 for 95% confidence interval.")
        if test_end is None:
            tn = sum(ti)
            n = len(ti) - 1
        else:
            tn = test_end
            n = len(ti)
            if tn < sum(ti):
                raise ValueError("test_end cannot be less than the final test time")

        tc = np.cumsum(ti[0:n])
        sum_tc = sum(tc)
        z_crit = ss.norm.ppf((1 - CI) / 2)  # z statistic based on CI
        U = (sum_tc / n - tn / 2) / (tn * (1 / (12 * n)) ** 0.5)
        self.U = U
        self.z_crit = (z_crit, -z_crit)
        results_str = str(
            "Laplace test results: U = "
            + str(round(U, 3))
            + ", z_crit = ("
            + str(round(z_crit, 2))
            + ",+"
            + str(round(-z_crit, 2))
            + ")",
        )

        x = np.arange(1, len(ti) + 1)
        if z_crit > U:
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn**B)
            self.trend = "improving"
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = "ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t."
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            x_to_plot = x if test_end is not None else x[:-1]
        elif -z_crit < U:
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn**B)
            self.trend = "worsening"
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = "ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t."
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            x_to_plot = x if test_end is not None else x[:-1]
        else:
            rocof = (n + 1) / sum(ti)
            self.trend = "constant"
            self.ROCOF = rocof
            self.Beta_hat = "not calculated when trend is constant"
            self.Lambda_hat = "not calculated when trend is constant"
            x_to_plot = x
            MTBF = np.ones_like(x_to_plot) / rocof

        self.__MTBF = MTBF
        self.__x = x
        self.__ti = ti
        self.__x_to_plot = x_to_plot
        self.__results_str = results_str
        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)
        self.__CI_rounded = CI_rounded

    def print_results(self) -> None:
        """
        Print the results from the ROCOF analysis.

        This method prints the results of the ROCOF analysis, including the confidence level,
        the type of trend (improving, worsening, or constant), and the parameters of the ROCOF assuming NHPP or HPP.

        Returns:
            None
        """
        colorprint("Results from ROCOF analysis:", bold=True, underline=True)
        print(self.__results_str)
        if self.z_crit[0] > self.U:
            if isinstance(self.Beta_hat, (float, int)) and isinstance(self.Lambda_hat, (float, int)):
                print(str("At " + str(self.__CI_rounded) + "% confidence level the ROCOF is IMPROVING. Assume NHPP."))
                print(
                    "ROCOF assuming NHPP has parameters: Beta_hat =",
                    round_and_string(float(self.Beta_hat), decimals=3),
                    ", Lambda_hat =",
                    round_and_string(float(self.Lambda_hat), decimals=4),
                )
            else:
                print("Invalid type for Beta_hat or Lambda_hat.")
        elif self.z_crit[1] < self.U:
            if isinstance(self.Beta_hat, (float, int)) and isinstance(self.Lambda_hat, (float, int)):
                print(str("At " + str(self.__CI_rounded) + "% confidence level the ROCOF is IMPROVING. Assume NHPP."))
                print(str("At " + str(self.__CI_rounded) + "% confidence level the ROCOF is WORSENING. Assume NHPP."))
                print(
                    "ROCOF assuming NHPP has parameters: Beta_hat =",
                    round_and_string(self.Beta_hat, decimals=3),
                    ", Lambda_hat =",
                    round_and_string(self.Lambda_hat, decimals=4),
                )
            else:
                print("Invalid type for Beta_hat or Lambda_hat.")
        else:
            print(str("At " + str(self.__CI_rounded) + "% confidence level the ROCOF is CONSTANT. Assume HPP."))
            if isinstance(self.ROCOF, (float, int)):
                rocof_value = round_and_string(self.ROCOF, decimals=4)
            else:
                rocof_value = "ROCOF is not provided when trend is not constant."
            print(
                "ROCOF assuming HPP is",
                rocof_value,
                "failures per unit time.",
            )

    def plot(self, **kwargs) -> plt.Axes:
        """
        Plot the failure interarrival times and mean time between failures (MTBF) of a repairable system.

        Parameters:
        - linestyle (str, optional): The line style for the MTBF plot. Default is '--'.
        - label (str, optional): The label for the failure interarrival times plot. Default is 'Failure interarrival times'.
        - **kwargs: Additional keyword arguments to be passed to the scatter plot function.

        Returns:
        - matplotlib.axes.Axes: The current axes instance.

        """
        ls = kwargs.pop("linestyle") if "linestyle" in kwargs else "--"
        label_1 = kwargs.pop("label") if "label" in kwargs else "Failure interarrival times"
        plt.plot(self.__x_to_plot, self.__MTBF, linestyle=ls, label="MTBF")
        plt.scatter(self.__x , self.__ti, label=label_1, **kwargs)
        plt.ylabel("Times between failures")
        plt.xlabel("Failure number")
        title_str = str(
            "Failure interarrival times vs failure number\nAt "
            + str(self.__CI_rounded)
            + "% confidence level the ROCOF is "
            + self.trend.upper(),
        )
        plt.title(title_str)
        plt.legend()
        return plt.gca()


class MCF_nonparametric:
    """The Mean Cumulative Function (MCF) is a cumulative history function that
    shows the cumulative number of recurrences of an event, such as repairs over
    time. In the context of repairs over time, the value of the MCF can be
    thought of as the average number of repairs that each system will have
    undergone after a certain time. It is only applicable to repairable systems
    and assumes that each event (repair) is identical, but it does not assume
    that each system's MCF is identical (which is an assumption of the
    parametric MCF). The non-parametric estimate of the MCF provides both the
    estimate of the MCF and the confidence bounds at a particular time.

    The shape of the MCF is a key indicator that shows whether the systems are
    improving, worsening, or staying the same over time. If the MCF is concave
    down (appearing to level out) then the system is improving. A straight line
    (constant increase) indicates it is staying the same. Concave up (getting
    steeper) shows the system is worsening as repairs are required more
    frequently as time progresses.

    Parameters
    ----------
    data : list
        The repair times for each system. Format this as a list of lists. eg.
        data=[[4,7,9],[3,8,12]] would be the data for 2 systems. The largest
        time for each system is assumed to be the retirement time and is treated
        as a right censored value. If the system was retired immediately after
        the last repair then you must include a repeated value at the end as
        this will be used to indicate a right censored value. eg. A system that
        had repairs at 4, 7, and 9 then was retired after the last repair would
        be entered as data = [4,7,9,9] since the last value is treated as a
        right censored value. If you only have data from 1 system you may enter
        the data in a single list as data = [3,7,12] and it will be nested
        within another list automatically.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default = 0.95 for 95% CI
        (one sided).

    Returns
    -------
    results : dataframe
        This is a dataframe of the results that are printed. It includes the
        blank lines for censored values.
    time : array
        This is the time column from results. Blank lines for censored values
        are removed.
    MCF : array
        This is the MCF column from results. Blank lines for censored values are
        removed.
    variance : array
        This is the Variance column from results. Blank lines for censored
        values are removed.
    lower : array
        This is the MCF_lower column from results. Blank lines for censored
        values are removed.
    upper : array
        This is the MCF_upper column from results. Blank lines for censored
        values are removed

    Notes
    -----
    This example is taken from Reliasoft's example (available at
    http://reliawiki.org/index.php/Recurrent_Event_Data_Analysis). The failure
    times and retirement times (retirement time is indicated by +) of 5 systems
    are:

    +------------+--------------+
    | System     | Times        |
    +------------+--------------+
    | 1          | 5,10,15,17+  |
    +------------+--------------+
    | 2          | 6,13,17,19+  |
    +------------+--------------+
    | 3          | 12,20,25,26+ |
    +------------+--------------+
    | 4          | 13,15,24+    |
    +------------+--------------+
    | 5          | 16,22,25,28+ |
    +------------+--------------+

    .. code:: python

        from reliability.Repairable_systems import MCF_nonparametric
        times = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
        MCF_nonparametric(data=times)

    """

    def __init__(self, data, CI=0.95):
        # check input is a list
        if isinstance(data, list):
            pass
        elif type(data) == np.ndarray:
            data = list(data)
        else:
            raise ValueError("data must be a list or numpy array")

        # check each item is a list and fix up any ndarrays to be lists.
        test_for_single_system = []
        for i, item in enumerate(data):
            if isinstance(item, list):
                test_for_single_system.append(False)
            elif type(item) == np.ndarray:
                data[i] = list(item)
                test_for_single_system.append(False)
            elif isinstance(item, (int, float)):
                test_for_single_system.append(True)
            else:
                raise ValueError("Each item in the data must be a list or numpy array. eg. data = [[1,3,5],[3,6,8]]")
        # Wraps the data in another list if all elements were numbers.
        if all(test_for_single_system):  # checks if all are True
            data = [data]
        elif not any(test_for_single_system):  # checks if all are False
            pass
        else:
            raise ValueError(
                "Mixed data types found in the data. Each item in the data must be a list or numpy array. eg. data = [[1,3,5],[3,6,8]].",
            )

        end_times = []
        repair_times = []
        for system in data:
            system.sort()  # sorts the values in ascending order
            for i, t in enumerate(system):
                if i < len(system) - 1:
                    repair_times.append(t)
                else:
                    end_times.append(t)

        if CI < 0 or CI > 1:
            raise ValueError("CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals (two sided).")

        if max(end_times) < max(repair_times):
            raise ValueError("The final end time must not be less than the final repair time.")
        last_time = max(end_times)
        C_array = ["C"] * len(end_times)
        F_array = ["F"] * len(repair_times)

        Z = -ss.norm.ppf(1 - CI)  # confidence interval converted to Z-value

        # sort the inputs and extract the sorted values for later use
        times = np.hstack([repair_times, end_times])
        states = np.hstack([F_array, C_array])
        data = {"times": times, "states": states}
        df = pd.DataFrame(data, columns=["times", "states"])
        df_sorted = df.sort_values(
            by=["times", "states"],
            ascending=[True, False],
        )  # sorts the df by times and then by states, ensuring that states are F then C where the same time occurs. This ensures a failure is counted then the item is retired.
        times_sorted = df_sorted.times.values
        states_sorted = df_sorted.states.values

        # MCF calculations
        MCF_array = []
        Var_array = []
        MCF_lower_array = []
        MCF_upper_array = []
        r = len(end_times)
        r_inv = 1 / r
        C_seq = 0  # sequential number of censored values
        for i in range(len(times)):
            if i == 0:
                if states_sorted[i] == "F":  # first event is a failure
                    MCF_array.append(r_inv)
                    Var_array.append((r_inv**2) * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2))
                    MCF_lower_array.append(MCF_array[i] / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i]))
                    MCF_upper_array.append(MCF_array[i] * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i]))
                else:  # first event is censored
                    MCF_array.append("")
                    Var_array.append("")
                    MCF_lower_array.append("")
                    MCF_upper_array.append("")
                    r -= 1
                    if (
                        times_sorted[i] not in end_times
                    ):  # check if this system only has one event. If not then increment the number censored count for this system
                        C_seq += 1
            elif states_sorted[i] == "F":  # failure event
                i_adj = i - C_seq
                r_inv = 1 / r
                if (
                    MCF_array[i_adj - 1] == ""
                ):  # this is the case where the first system only has one event that was censored and there is no data on the first line
                    MCF_array.append(r_inv)
                    Var_array.append((r_inv**2) * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2))
                    MCF_lower_array.append(MCF_array[i] / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i]))
                    MCF_upper_array.append(MCF_array[i] * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i]))
                else:  # this the normal case where there was previous data
                    MCF_array.append(r_inv + MCF_array[i_adj - 1])
                    Var_array.append(
                        (r_inv**2) * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2) + Var_array[i_adj - 1],
                    )
                    MCF_lower_array.append(MCF_array[i] / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i]))
                    MCF_upper_array.append(MCF_array[i] * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i]))
                C_seq = 0
            else:  # censored event
                r -= 1
                C_seq += 1
                MCF_array.append("")
                Var_array.append("")
                MCF_lower_array.append("")
                MCF_upper_array.append("")
                if r > 0:
                    r_inv = 1 / r

        # format output as dataframe
        data = {
            "state": states_sorted,
            "time": times_sorted,
            "MCF_lower": MCF_lower_array,
            "MCF": MCF_array,
            "MCF_upper": MCF_upper_array,
            "variance": Var_array,
        }
        printable_results = pd.DataFrame(data, columns=["state", "time", "MCF_lower", "MCF", "MCF_upper", "variance"])

        indices_to_drop = printable_results[printable_results["state"] == "C"].index
        plotting_results = printable_results.drop(indices_to_drop, inplace=False)
        RESULTS_time = plotting_results.time.values
        RESULTS_MCF = plotting_results.MCF.values
        RESULTS_variance = plotting_results.variance.values
        RESULTS_lower = plotting_results.MCF_lower.values
        RESULTS_upper = plotting_results.MCF_upper.values

        self.results = printable_results
        self.time = RESULTS_time
        self.MCF = RESULTS_MCF
        self.lower = RESULTS_lower
        self.upper = RESULTS_upper
        self.variance = RESULTS_variance
        self.last_time = last_time

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)
        self.CI_rounded = CI_rounded

    def print_results(self) -> None:
            """
            Prints the Mean Cumulative Function results with confidence interval.

            This method sets display options for pandas dataframe to prevent wrapping and truncation,
            and then prints the results with a header indicating the confidence interval.

            Args:
                None

            Returns:
                None
            """
            pd.set_option("display.width", 200)  # prevents wrapping after default 80 characters
            pd.set_option("display.max_columns", 9)  # shows the dataframe without ... truncation
            colorprint(
                str("Mean Cumulative Function results (" + str(self.CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print(self.results.to_string(index=False), "\n")

    def plot(self, plot_CI=True, **kwargs) -> plt.Axes:
        """
        Plot the non-parametric estimate of the Mean Cumulative Function (MCF) for repairable systems.

        Args:
            plot_CI (bool, optional): Whether to plot the confidence interval bounds. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the `plt.plot` function.

        Returns:
            matplotlib.axes.Axes: The current axes instance.

        Raises:
            None

        Example:
            To plot the MCF with confidence interval bounds:
            >>> repairable_system.plot(plot_CI=True, color='red')

        """
        x_MCF = [0, self.time[0]]
        y_MCF = [0, 0]
        y_upper = [0, 0]
        y_lower = [0, 0]
        x_MCF.append(self.time[0])
        y_MCF.append(self.MCF[0])
        y_upper.append(self.upper[0])
        y_lower.append(self.lower [0])
        for i, _ in enumerate(self.time):
            if i > 0:
                x_MCF.append(self.time[i])
                y_MCF.append(self.MCF[i - 1])
                y_upper.append(self.upper[i - 1])
                y_lower.append(self.lower [i - 1])
                x_MCF.append(self.time[i])
                y_MCF.append(self.MCF[i])
                y_upper.append(self.upper[i])
                y_lower.append(self.lower [i])
        x_MCF.append(self.last_time)  # add the last horizontal line
        y_MCF.append(self.MCF[-1])
        y_upper.append(self.upper[-1])
        y_lower.append(self.lower [-1])
        title_str = "Non-parametric estimate of the Mean Cumulative Function"

        col = kwargs.pop("color") if "color" in kwargs else "steelblue"
        if plot_CI is True:
            plt.fill_between(x_MCF, y_lower, y_upper, color=col, alpha=0.3, linewidth=0)
            title_str = str(title_str + "\nwith " + str(self.CI_rounded) + "% one-sided confidence interval bounds")
        plt.plot(x_MCF, y_MCF, color=col, **kwargs)
        plt.xlabel("Time")
        plt.ylabel("Mean cumulative number of failures")
        plt.title(title_str)
        plt.xlim(0, self.last_time)
        plt.ylim(0, max(self.upper) * 1.05)
        return plt.gca()


class MCF_parametric:
    """The Mean Cumulative Function (MCF) is a cumulative history function that
    shows the cumulative number of recurrences of an event, such as repairs over
    time. In the context of repairs over time, the value of the MCF can be
    thought of as the average number of repairs that each system will have
    undergone after a certain time. It is only applicable to repairable systems
    and assumes that each event (repair) is identical. In the case of the fitted
    paramertic MCF, it is assumed that each system's MCF is identical.

    The shape (beta parameter) of the MCF is a key indicator that shows whether
    the systems are improving (beta<1), worsening (beta>1), or staying the same
    (beta=1) over time. If the MCF is concave down (appearing to level out) then
    the system is improving. A straight line (constant increase) indicates it is
    staying the same. Concave up (getting steeper) shows the system is worsening
    as repairs are required more frequently as time progresses.

    Parameters
    ----------
    data : list
        The repair times for each system. Format this as a list of lists. eg.
        data=[[4,7,9],[3,8,12]] would be the data for 2 systems. The largest
        time for each system is assumed to be the retirement time and is treated
        as a right censored value. If the system was retired immediately after
        the last repair then you must include a repeated value at the end as
        this will be used to indicate a right censored value. eg. A system that
        had repairs at 4, 7, and 9 then was retired after the last repair would
        be entered as data = [4,7,9,9] since the last value is treated as a
        right censored value. If you only have data from 1 system you may enter
        the data in a single list as data = [3,7,12] and it will be nested
        within another list automatically.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default = 0.95 for 95% CI
        (one sided).

    Returns
    -------
    times : array
        This is the times (x values) from the scatter plot. This value is
        calculated using MCF_nonparametric.
    MCF : array
        This is the MCF (y values) from the scatter plot. This value is
        calculated using MCF_nonparametric.
    alpha : float
        The calculated alpha parameter from MCF = (t/alpha)^beta
    beta : float
        The calculated beta parameter from MCF = (t/alpha)^beta
    alpha_SE : float
        The standard error in the alpha parameter
    beta_SE : float
        The standard error in the beta parameter
    cov_alpha_beta : float
        The covariance between the parameters
    alpha_upper : float
        The upper CI estimate of the parameter
    alpha_lower : float
        The lower CI estimate of the parameter
    beta_upper : float
        The upper CI estimate of the parameter
    beta_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)

    Notes
    -----
    This example is taken from Reliasoft's example (available at
    http://reliawiki.org/index.php/Recurrent_Event_Data_Analysis). The failure
    times and retirement times (retirement time is indicated by +) of 5 systems
    are:

    +------------+--------------+
    | System     | Times        |
    +------------+--------------+
    | 1          | 5,10,15,17+  |
    +------------+--------------+
    | 2          | 6,13,17,19+  |
    +------------+--------------+
    | 3          | 12,20,25,26+ |
    +------------+--------------+
    | 4          | 13,15,24+    |
    +------------+--------------+
    | 5          | 16,22,25,28+ |
    +------------+--------------+

    .. code:: python

        from reliability.Repairable_systems import MCF_parametric
        times = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
        MCF_parametric(data=times)

    """

    def __init__(self, data, CI=0.95):
        if CI <= 0 or CI >= 1:
            raise ValueError("CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval.")

        MCF_NP = MCF_nonparametric(
            data=data,
        )  # all the MCF calculations to get the plot points are done in MCF_nonparametric
        self.times = MCF_NP.time
        self.MCF = MCF_NP.MCF

        # initial guess using least squares regression of linearised function
        # we must convert this back to list due to an issue within numpy dealing with the log of floats
        ln_x = np.log(list(self.times))
        ln_y = np.log(list(self.MCF))
        guess_fit = np.polyfit(ln_x, ln_y, deg=1)
        beta_guess = guess_fit[0]
        alpha_guess = np.exp(-guess_fit[1] / beta_guess)
        guess = [
            alpha_guess,
            beta_guess,
        ]  # guess for curve_fit. This guess is good but curve fit makes it much better.

        # actual fitting using curve_fit with initial guess from least squares
        def __MCF_eqn(t, a, b):  # objective function for curve_fit
            return (t / a) ** b

        fit = curve_fit(__MCF_eqn, self.times, self.MCF, p0=guess)
        alpha = fit[0][0]
        beta = fit[0][1]
        var_alpha = fit[1][0][0]  # curve_fit returns the variance and covariance from the optimizer
        var_beta = fit[1][1][1]
        cov_alpha_beta = fit[1][0][1]

        Z = -ss.norm.ppf((1 - CI) / 2)
        self.alpha = alpha
        self.alpha_SE = var_alpha**0.5
        self.beta = beta
        self.beta_SE = var_beta**0.5
        self.cov_alpha_beta = cov_alpha_beta
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))
        self.CI = CI
        self.Z = Z
        Data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
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

    def print_results(self) -> None:
        """
        Print the results of the Mean Cumulative Function Parametric Model.

        This method calculates and prints the results of the Mean Cumulative Function Parametric Model,
        including the Confidence Interval (CI), the MCF equation, and the system repair rate.

        Returns:
            None
        """
        CI_rounded = self.CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.CI * 100)
        colorprint(
            str("Mean Cumulative Function Parametric Model (" + str(CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print("MCF = (t/α)^β")
        print(self.results.to_string(index=False), "\n")
        if self.beta_upper <= 1:
            print("Since Beta is less than 1, the system repair rate is IMPROVING over time.")
        elif self.beta_lower < 1 and self.beta_upper > 1:
            print("Since Beta is approximately 1, the system repair rate is remaining CONSTANT over time.")
        else:
            print("Since Beta is greater than 1, the system repair rate is WORSENING over time.")

    def plot(self, plot_CI=True, **kwargs) -> plt.Axes:
        """
        Plot the parametric estimate of the Mean Cumulative Function (MCF) for repairable systems.

        Args:
            plot_CI (bool, optional): Whether to plot the confidence interval. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            plt.Axes: The matplotlib Axes object containing the plot.

        """
        if "color" in kwargs:
            color = kwargs.pop("color")
            marker_color = "k"
        else:
            color = "steelblue"
            marker_color = "k"

        marker = kwargs.pop("marker") if "marker" in kwargs else "."

        label = kwargs.pop("label") if "label" in kwargs else "$\\hat{MCF} = (\\frac{t}{\\alpha})^\\beta$"

        x_line = np.linspace(0.001, max(self.times) * 10, 1000)
        y_line = (x_line / self.alpha) ** self.beta
        plt.plot(x_line, y_line, color=color, label=label, **kwargs)

        if plot_CI is True:
            p1 = -(self.beta / self.alpha) * (x_line / self.alpha) ** self.beta
            p2 = ((x_line / self.alpha) ** self.beta) * np.log(x_line / self.alpha)
            var = self.var_alpha * p1**2 + self.var_beta * p2**2 + 2 * p1 * p2 * self.cov_alpha_beta
            SD = var**0.5
            y_line_lower = y_line * np.exp((-self.Z * SD) / y_line)
            y_line_upper = y_line * np.exp((self.Z * SD) / y_line)
            plt.fill_between(
                x_line,
                y_line_lower,
                y_line_upper,
                color=color,
                alpha=0.3,
                linewidth=0,
            )

        plt.scatter(np.array(self.times), np.array(self.MCF), marker=marker, color=marker_color, **kwargs)
        plt.ylabel("Mean cumulative number of failures")
        plt.xlabel("Time")
        title_str = str(
            "Parametric estimate of the Mean Cumulative Function\n"
            + r"$MCF = (\frac{t}{\alpha})^\beta$ with α="
            + str(round(self.alpha, 4))
            + ", β="
            + str(round(self.beta, 4)),
        )
        plt.xlim(0, max(self.times) * 1.2)
        plt.ylim(0, max(self.MCF) * 1.4)
        plt.title(title_str)
        return plt.gca()

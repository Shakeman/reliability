from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as ss
from scipy import integrate

from reliability.Utils import (
    colorprint,
    distribution_confidence_intervals,
    distributions_input_checking,
    extract_CI,
    get_axes_limits,
    restore_axes_limits,
    round_and_string,
    unpack_single_arrays,
    zeroise_below_gamma,
)

dec = 4  # number of decimals to use when rounding descriptive statistics and parameter titles
np.seterr(divide="ignore", invalid="ignore")  # ignore the divide by zero warnings


class Loglogistic_Distribution:
    """Loglogistic probability distribution. Creates a probability distribution
    object.

    Parameters
    ----------
    alpha : float, int
        Scale parameter. Must be > 0
    beta : float, int
        Shape parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Loglogistic'
    name2 : 'str
        'Loglogistic_2P' or 'Loglogistic_3P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Loglogistic Distribution (α=5,β=2)'
    param_title : str
        'α=5,β=2'
    parameters : list
        [alpha,beta,gamma]
    alpha : float
    beta : float
    gamma : float
    mean : float
    variance : float
    standard_deviation : float
    skewness : float
    kurtosis : float
    excess_kurtosis : float
    median : float
    mode : float
    b5 : float
    b95 : float

    Notes
    -----
    kwargs are used internally to generate the confidence intervals

    """

    def __init__(self, alpha=None, beta=None, gamma: float | np.float64 = 0.0, **kwargs):
        self.name = "Loglogistic"
        if alpha is None or beta is None:
            raise ValueError(
                "Parameters alpha and beta must be specified. Eg. Loglogistic_Distribution(alpha=5,beta=2)",
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.parameters = np.array([self.alpha, self.beta, self.gamma])

        if self.beta > 1:
            self.mean = float(ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="m"))
        else:
            self.mean = r"no mean when $\beta \leq 1$"
        if self.beta > 2:
            self.variance = float(ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="v"))
            self.standard_deviation = self.variance**0.5
        else:
            self.variance = r"no variance when $\beta \leq 2$"
            self.standard_deviation = r"no stdev when $\beta \leq 2$"
        if self.beta > 3:
            self.skewness = float(ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="s"))
        else:
            self.skewness = r"no skewness when $\beta \leq 3$"
        if self.beta > 4:
            self.excess_kurtosis = float(ss.fisk.stats(self.beta, scale=self.alpha, loc=self.gamma, moments="k"))
            self.kurtosis = self.excess_kurtosis + 3
        else:
            self.excess_kurtosis = r"no kurtosis when $\beta \leq 4$"
            self.kurtosis = r"no kurtosis when $\beta \leq 4$"

        self.median = ss.fisk.median(self.beta, scale=self.alpha, loc=self.gamma)
        if self.beta >= 1:
            self.mode = self.alpha * ((self.beta - 1) / (self.beta + 1)) ** (1 / self.beta) + self.gamma
        else:
            self.mode = self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "α="
                + round_and_string(self.alpha, dec)
                + ",β="
                + round_and_string(self.beta, dec)
                + ",γ="
                + round_and_string(self.gamma, dec),
            )
            self.param_title_long = str(
                "Loglogistic Distribution (α="
                + round_and_string(self.alpha, dec)
                + ",β="
                + round_and_string(self.beta, dec)
                + ",γ="
                + round_and_string(self.gamma, dec)
                + ")",
            )
            self.name2 = "Loglogistic_3P"
        else:
            self.param_title = str("α=" + round_and_string(self.alpha, dec) + ",β=" + round_and_string(self.beta, dec))
            self.param_title_long = str(
                "Loglogistic Distribution (α="
                + round_and_string(self.alpha, dec)
                + ",β="
                + round_and_string(self.beta, dec)
                + ")",
            )
            self.name2 = "Loglogistic_2P"
        self.b5 = ss.fisk.ppf(0.05, self.beta, scale=self.alpha, loc=self.gamma)
        self.b95 = ss.fisk.ppf(0.95, self.beta, scale=self.alpha, loc=self.gamma)

        # extracts values for confidence interval plotting
        if "alpha_SE" in kwargs:
            self.alpha_SE = kwargs.pop("alpha_SE")
        else:
            self.alpha_SE = None
        if "beta_SE" in kwargs:
            self.beta_SE = kwargs.pop("beta_SE")
        else:
            self.beta_SE = None
        if "Cov_alpha_beta" in kwargs:
            self.Cov_alpha_beta = kwargs.pop("Cov_alpha_beta")
        else:
            self.Cov_alpha_beta = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        if "CI_type" in kwargs:
            self.CI_type = kwargs.pop("CI_type")
        else:
            self.CI_type = "time"
        for item in kwargs:
            colorprint(
                str(
                    "WARNING:"
                    + item
                    + " is not recognised as an appropriate entry in kwargs. Appropriate entries are alpha_SE, beta_SE, Cov_alpha_beta, CI, and CI_type.",
                ),
                text_color="red",
            )
        self._pdf0 = ss.fisk.pdf(
            0,
            self.beta,
            scale=self.alpha,
            loc=0,
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.fisk.pdf(0, self.beta, scale=self.alpha, loc=0) / ss.fisk.sf(
            0,
            self.beta,
            scale=self.alpha,
            loc=0,
        )  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

    def plot(self, xvals=None, xmin=None, xmax=None):
        """Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure

        Parameters
        ----------
        xvals : list, array, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting

        Returns
        -------
        None

        Notes
        -----
        The plot will be shown. No need to use plt.show().
        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters. No plotting keywords are
        accepted.

        """
        X, xvals, xmin, xmax = distributions_input_checking(self, "ALL", xvals, xmin, xmax)

        pdf = ss.fisk.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        cdf = ss.fisk.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        sf = ss.fisk.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        hf = ((self.beta / self.alpha) * ((X - self.gamma) / self.alpha) ** (self.beta - 1)) / (
            1 + ((X - self.gamma) / self.alpha) ** self.beta
        )
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        chf = np.log(1 + ((X - self.gamma) / self.alpha) ** self.beta)
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)

        plt.figure(figsize=(9, 7))
        text_title = str("Loglogistic Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="PDF",
            X=X,
            Y=pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(X, cdf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="CDF",
            X=X,
            Y=cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(X, sf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="SF",
            X=X,
            Y=sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        plt.plot(X, hf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="HF",
            X=X,
            Y=hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(X, chf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="CHF",
            X=X,
            Y=chf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Hazard\nFunction")

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_median = str("Median = " + round_and_string(self.median, dec))
        text_b5 = str("$5^{th}$ quantile = " + round_and_string(self.b5, dec))
        text_b95 = str("$95^{th}$ quantile = " + round_and_string(self.b95, dec))
        text_mode = str("Mode = " + round_and_string(self.mode, dec))

        if isinstance(self.mean, str):
            text_mean = str("Mean = " + str(self.mean))  # required when mean is str
        else:
            text_mean = str("Mean = " + round_and_string(self.mean, dec))

        if isinstance(self.standard_deviation, str):
            text_std = str(
                "Standard deviation = " + str(self.standard_deviation),
            )  # required when standard deviation is str
        else:
            text_std = str("Standard deviation = " + round_and_string(self.standard_deviation, dec))

        if isinstance(self.variance, str):
            text_var = str("Variance = " + str(self.variance))  # required when variance is str
        else:
            text_var = str("Variance = " + round_and_string(self.variance, dec))

        if isinstance(self.skewness, str):
            text_skew = str("Skewness = " + str(self.skewness))  # required when skewness is str
        else:
            text_skew = str("Skewness = " + round_and_string(self.skewness, dec))

        if isinstance(self.excess_kurtosis, str):
            text_ex_kurt = str("Excess kurtosis = " + str(self.excess_kurtosis))  # required when excess kurtosis is str
        else:
            text_ex_kurt = str("Excess kurtosis = " + round_and_string(self.excess_kurtosis, dec))

        plt.text(0, 9, text_mean)
        plt.text(0, 8, text_median)
        plt.text(0, 7, text_mode)
        plt.text(0, 6, text_b5)
        plt.text(0, 5, text_b95)
        plt.text(0, 4, text_std)
        plt.text(0, 3, text_var)
        plt.text(0, 2, text_skew)
        plt.text(0, 1, text_ex_kurt)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, top=0.84)
        plt.show()

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array, float
            The y-values of the plot

        Notes
        -----
        The plot will be shown if show_plot is True (which it is by default).

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        X, xvals, xmin, xmax, show_plot = distributions_input_checking(
            self,
            "PDF",
            xvals,
            xmin,
            xmax,
            show_plot,
        )  # lgtm [py/mismatched-multiple-assignment]

        pdf = ss.fisk.pdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        pdf = unpack_single_arrays(pdf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Loglogistic Distribution\n" + " Probability Density Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=X,
                Y=pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

        return pdf

    def CDF(
        self,
        xvals: npt.NDArray[np.float64] | None = None,
        xmin: np.float64 | None = None,
        xmax: np.float64 | None = None,
        show_plot: bool = True,
        plot_CI: bool = True,
        CI_type: Literal["time", "reliability", "none"] | None = None,
        CI: np.float64 | None = None,
        CI_y: np.float64 | None = None,
        CI_x: np.float64 | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64, npt.NDArray[np.float64]]:
        """Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        show_plot : bool, optional
            True or False. Default = True
        plot_CI : bool, optional
            True or False. Default = True. Only used if the distribution object
            was created by Fitters.
        CI_type : str, optional
            Must be either "time", "reliability", or "none". Default is "time". Only used
            if the distribution object was created by Fitters.
        CI : float, optional
            The confidence interval between 0 and 1. Only used if the
            distribution object was created by Fitters.
        CI_y : list, array, optional
            The confidence interval y-values to trace. Only used if the
            distribution object was created by Fitters and CI_type='time'.
        CI_x : list, array, optional
            The confidence interval x-values to trace. Only used if the
            distribution object was created by Fitters and
            CI_type='reliability'.
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array, float
            The y-values of the plot. Only returned if CI_x and CI_y are not
            specified.
        lower_estimate, point_estimate, upper_estimate : tuple
            A tuple of arrays or floats of the confidence interval estimates
            based on CI_x or CI_y. Only returned if CI_x or CI_y is specified
            and the confidence intervals are available. If CI_x is specified,
            the point estimate is the y-values from the distribution at CI_x. If
            CI_y is specified, the point estimate is the x-values from the
            distribution at CI_y.

        Notes
        -----
        The plot will be shown if show_plot is True (which it is by default).

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        (
            X,
            xvals,
            xmin,
            xmax,
            show_plot,
            plot_CI,
            CI_type,
            CI,
            CI_y,
            CI_x,
        ) = distributions_input_checking(self, "CDF", xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x)

        cdf = ss.fisk.cdf(X, self.beta, scale=self.alpha, loc=self.gamma)
        cdf = unpack_single_arrays(cdf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Loglogistic Distribution\n" + " Cumulative Distribution Function " + "\n" + self.param_title,
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=X,
                Y=cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.loglogistic_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

        lower_CI, upper_CI = extract_CI(dist=self, func="CDF", CI_type=CI_type, CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_type == "time":
                return lower_CI, self.quantile(CI_y), upper_CI
            elif CI_type == "reliability":
                cdf_point = ss.fisk.cdf(CI_x, self.beta, scale=self.alpha, loc=self.gamma)
                return lower_CI, unpack_single_arrays(cdf_point), upper_CI
        return cdf

    def SF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_CI=True,
        CI_type=None,
        CI=None,
        CI_y=None,
        CI_x=None,
        **kwargs,
    ):
        """Plots the SF (survival function)

        Parameters
        ----------
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        show_plot : bool, optional
            True or False. Default = True
        plot_CI : bool, optional
            True or False. Default = True. Only used if the distribution object
            was created by Fitters.
        CI_type : str, optional
            Must be either "time", "reliability", or "none". Default is "time". Only used
            if the distribution object was created by Fitters.
        CI : float, optional
            The confidence interval between 0 and 1. Only used if the
            distribution object was created by Fitters.
        CI_y : list, array, optional
            The confidence interval y-values to trace. Only used if the
            distribution object was created by Fitters and CI_type='time'.
        CI_x : list, array, optional
            The confidence interval x-values to trace. Only used if the
            distribution object was created by Fitters and
            CI_type='reliability'.
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array, float
            The y-values of the plot. Only returned if CI_x and CI_y are not
            specified.
        lower_estimate, point_estimate, upper_estimate : tuple
            A tuple of arrays or floats of the confidence interval estimates
            based on CI_x or CI_y. Only returned if CI_x or CI_y is specified
            and the confidence intervals are available. If CI_x is specified,
            the point estimate is the y-values from the distribution at CI_x. If
            CI_y is specified, the point estimate is the x-values from the
            distribution at CI_y.

        Notes
        -----
        The plot will be shown if show_plot is True (which it is by default).

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        (
            X,
            xvals,
            xmin,
            xmax,
            show_plot,
            plot_CI,
            CI_type,
            CI,
            CI_y,
            CI_x,
        ) = distributions_input_checking(self, "SF", xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x)

        sf = ss.fisk.sf(X, self.beta, scale=self.alpha, loc=self.gamma)
        sf = unpack_single_arrays(sf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Loglogistic Distribution\n" + " Survival Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=X,
                Y=sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.loglogistic_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

        lower_CI, upper_CI = extract_CI(dist=self, func="SF", CI_type=CI_type, CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_type == "time":
                return lower_CI, self.inverse_SF(CI_y), upper_CI
            elif CI_type == "reliability":
                sf_point = ss.fisk.sf(CI_x, self.beta, scale=self.alpha, loc=self.gamma)
                return lower_CI, unpack_single_arrays(sf_point), upper_CI
        else:
            return sf

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array, float
            The y-values of the plot

        Notes
        -----
        The plot will be shown if show_plot is True (which it is by default).

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        X, xvals, xmin, xmax, show_plot = distributions_input_checking(
            self,
            "HF",
            xvals,
            xmin,
            xmax,
            show_plot,
        )  # lgtm [py/mismatched-multiple-assignment]

        hf = ((self.beta / self.alpha) * ((X - self.gamma) / self.alpha) ** (self.beta - 1)) / (
            1 + ((X - self.gamma) / self.alpha) ** self.beta
        )
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        hf = unpack_single_arrays(hf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Loglogistic Distribution\n" + " Hazard Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=X,
                Y=hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

        return hf

    def CHF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_CI=True,
        CI_type=None,
        CI=None,
        CI_y=None,
        CI_x=None,
        **kwargs,
    ):
        """Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        xvals : array, list, optional
            x-values for plotting
        xmin : int, float, optional
            minimum x-value for plotting
        xmax : int, float, optional
            maximum x-value for plotting
        show_plot : bool, optional
            True or False. Default = True
        plot_CI : bool, optional
            True or False. Default = True. Only used if the distribution object
            was created by Fitters.
        CI_type : str, optional
            Must be either "time", "reliability", or "none". Default is "time". Only used
            if the distribution object was created by Fitters.
        CI : float, optional
            The confidence interval between 0 and 1. Only used if the
            distribution object was created by Fitters.
        CI_y : list, array, optional
            The confidence interval y-values to trace. Only used if the
            distribution object was created by Fitters and CI_type='time'.
        CI_x : list, array, optional
            The confidence interval x-values to trace. Only used if the
            distribution object was created by Fitters and
            CI_type='reliability'.
        kwargs
            Plotting keywords that are passed directly to matplotlib
            (e.g. color, linestyle)

        Returns
        -------
        yvals : array, float
            The y-values of the plot. Only returned if CI_x and CI_y are not
            specified.
        lower_estimate, point_estimate, upper_estimate : tuple
            A tuple of arrays or floats of the confidence interval estimates
            based on CI_x or CI_y. Only returned if CI_x or CI_y is specified
            and the confidence intervals are available. If CI_x is specified,
            the point estimate is the y-values from the distribution at CI_x. If
            CI_y is specified, the point estimate is the x-values from the
            distribution at CI_y.

        Notes
        -----
        The plot will be shown if show_plot is True (which it is by default).

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        (
            X,
            xvals,
            xmin,
            xmax,
            show_plot,
            plot_CI,
            CI_type,
            CI,
            CI_y,
            CI_x,
        ) = distributions_input_checking(self, "CHF", xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x)

        chf = np.log(1 + ((X - self.gamma) / self.alpha) ** self.beta)
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)
        chf = unpack_single_arrays(chf)
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str("Loglogistic Distribution\n" + " Cumulative Hazard Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=X,
                Y=chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

            distribution_confidence_intervals.loglogistic_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

        lower_CI, upper_CI = extract_CI(dist=self, func="CHF", CI_type=CI_type, CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_type == "time":
                return lower_CI, self.inverse_SF(np.exp(-CI_y)), upper_CI
            elif CI_type == "reliability":
                chf_point = zeroise_below_gamma(
                    X=CI_x,
                    Y=np.log(1 + ((CI_x - self.gamma) / self.alpha) ** self.beta),
                    gamma=self.gamma,
                )
                return lower_CI, unpack_single_arrays(chf_point), upper_CI
        else:
            return chf

    def quantile(self, q):
        """Quantile calculator

        Parameters
        ----------
        q : float, list, array
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q

        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type float, list, array")
        ppf = ss.fisk.ppf(q, self.beta, scale=self.alpha, loc=self.gamma)
        return unpack_single_arrays(ppf)

    def inverse_SF(self, q):
        """Inverse survival function calculator

        Parameters
        ----------
        q : float, list, array
            Quantile to be calculated. Must be between 0 and 1.

        Returns
        -------
        x : float
            The inverse of the SF at q.

        """
        if type(q) in [int, float, np.float64]:
            if q < 0 or q > 1:
                raise ValueError("Quantile must be between 0 and 1")
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError("Quantile must be between 0 and 1")
        else:
            raise ValueError("Quantile must be of type float, list, array")
        isf = ss.fisk.isf(q, self.beta, scale=self.alpha, loc=self.gamma)
        return unpack_single_arrays(isf)

    def mean_residual_life(self, t):
        """Mean Residual Life calculator

        Parameters
        ----------
        t : int, float
            Time (x-value) at which mean residual life is to be evaluated

        Returns
        -------
        MRL : float
            The mean residual life

        """

        def R(x):
            return ss.fisk.sf(x, self.beta, scale=self.alpha, loc=self.gamma)

        integral_R, error = integrate.quad(R, t, np.inf)
        MRL = integral_R / R(t)
        return MRL

    def stats(self):
        """Descriptive statistics of the probability distribution.
        These are the same as the statistics shown using .plot() but printed to
        the console.

        Parameters
        ----------
        None

        Return
        ------
        None

        """
        if self.gamma == 0:
            colorprint(
                str(
                    "Descriptive statistics for Loglogistic distribution with alpha = "
                    + str(self.alpha)
                    + " and beta = "
                    + str(self.beta),
                ),
                bold=True,
                underline=True,
            )
        else:
            colorprint(
                str(
                    "Descriptive statistics for Loglogistic distribution with alpha = "
                    + str(self.alpha)
                    + ", beta = "
                    + str(self.beta)
                    + ", and gamma = "
                    + str(self.gamma),
                ),
                bold=True,
                underline=True,
            )
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, seed=None):
        """Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        samples : array
            The random samples

        Notes
        -----
        This is the same as rvs in scipy.stats

        """
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)
        RVS = ss.fisk.rvs(self.beta, scale=self.alpha, loc=self.gamma, size=number_of_samples)
        return RVS

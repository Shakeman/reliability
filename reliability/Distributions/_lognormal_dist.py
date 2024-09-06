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
)

dec = 4  # number of decimals to use when rounding descriptive statistics and parameter titles
np.seterr(divide="ignore", invalid="ignore")  # ignore the divide by zero warnings


class Lognormal_Distribution:
    """Lognormal probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    mu : float, int
        Location parameter
    sigma : float, int
        Scale parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Lognormal'
    name2 : 'str
        'Lognormal_2P' or 'Lognormal_3P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Lognormal Distribution (μ=5,σ=2)'
    param_title : str
        'μ=5,σ=2'
    parameters : list
        [mu,sigma,gamma]
    mu : float
    sigma : float
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

    def __init__(self, mu=None, sigma=None, gamma: float | np.float64 = 0.0, **kwargs):
        self.name = "Lognormal"
        if mu is None or sigma is None:
            raise ValueError("Parameters mu and sigma must be specified. Eg. Lognormal_Distribution(mu=5,sigma=2)")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.parameters: npt.NDArray[np.float64] = np.array([self.mu, self.sigma, self.gamma])
        mean, var, skew, kurt = ss.lognorm.stats(self.sigma, self.gamma, np.exp(self.mu), moments="mvsk")
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var**0.5
        self.skewness = float(skew)
        self.kurtosis: np.float64 = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.lognorm.median(self.sigma, self.gamma, np.exp(self.mu))
        self.mode: np.float64 = np.exp(self.mu - self.sigma**2) + self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "μ="
                + round_and_string(self.mu, dec)
                + ",σ="
                + round_and_string(self.sigma, dec)
                + ",γ="
                + round_and_string(self.gamma, dec),
            )
            self.param_title_long = str(
                "Lognormal Distribution (μ="
                + round_and_string(self.mu, dec)
                + ",σ="
                + round_and_string(self.sigma, dec)
                + ",γ="
                + round_and_string(self.gamma, dec)
                + ")",
            )
            self.name2 = "Lognormal_3P"
        else:
            self.param_title = str("μ=" + round_and_string(self.mu, dec) + ",σ=" + round_and_string(self.sigma, dec))
            self.param_title_long = str(
                "Lognormal Distribution (μ="
                + round_and_string(self.mu, dec)
                + ",σ="
                + round_and_string(self.sigma, dec)
                + ")",
            )
            self.name2 = "Lognormal_2P"
        self.b5 = ss.lognorm.ppf(
            0.05,
            self.sigma,
            self.gamma,
            np.exp(self.mu),
        )  # note that scipy uses mu in a log way compared to most other software, so we must take the exp of the input
        self.b95 = ss.lognorm.ppf(0.95, self.sigma, self.gamma, np.exp(self.mu))

        # extracts values for confidence interval plotting
        if "mu_SE" in kwargs:
            self.mu_SE = kwargs.pop("mu_SE")
        else:
            self.mu_SE = None
        if "sigma_SE" in kwargs:
            self.sigma_SE = kwargs.pop("sigma_SE")
        else:
            self.sigma_SE = None
        if "Cov_mu_sigma" in kwargs:
            self.Cov_mu_sigma = kwargs.pop("Cov_mu_sigma")
        else:
            self.Cov_mu_sigma = None
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
                    "WARNING: "
                    + item
                    + "is not recognised as an appropriate entry in kwargs. Appropriate entries are mu_SE, sigma_SE, Cov_mu_sigma, CI, and CI_type.",
                ),
                text_color="red",
            )

        self._pdf0 = ss.lognorm.pdf(
            0,
            self.sigma,
            0,
            np.exp(self.mu),
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.lognorm.pdf(0, self.sigma, 0, np.exp(self.mu)) / ss.lognorm.sf(
            0,
            self.sigma,
            0,
            np.exp(self.mu),
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
        input_check = distributions_input_checking(self, "ALL", xvals, xmin, xmax)
        X, xvals, xmin, xmax = input_check.X, input_check.xvals, input_check.xmin, input_check.xmax
        pdf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu))
        cdf = ss.lognorm.cdf(X, self.sigma, self.gamma, np.exp(self.mu))
        sf = ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu))
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str("Lognormal Distribution" + "\n" + self.param_title)
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        plt.title("Probability Density\nFunction")
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

        plt.subplot(232)
        plt.plot(X, cdf)
        plt.title("Cumulative Distribution\nFunction")
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

        plt.subplot(233)
        plt.plot(X, sf)
        plt.title("Survival Function")
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

        plt.subplot(234)
        plt.plot(X, hf)
        plt.title("Hazard Function")
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

        plt.subplot(235)
        plt.plot(X, chf)
        plt.title("Cumulative Hazard\nFunction")
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

        # descriptive statistics section
        plt.subplot(236)
        plt.axis("off")
        plt.ylim([0, 10])
        plt.xlim([0, 10])
        text_mean = str("Mean = " + round_and_string(self.mean, dec))
        text_median = str("Median = " + round_and_string(self.median, dec))
        text_mode = str("Mode = " + round_and_string(self.mode, dec))
        text_b5 = str("$5^{th}$ quantile = " + round_and_string(self.b5, dec))
        text_b95 = str("$95^{th}$ quantile = " + round_and_string(self.b95, dec))
        text_std = str("Standard deviation = " + round_and_string(self.standard_deviation, dec))
        text_var = str("Variance = " + round_and_string(self.variance, dec))
        text_skew = str("Skewness = " + round_and_string(self.skewness, dec))
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
        input_check = distributions_input_checking(
            self,
            "PDF",
            xvals,
            xmin,
            xmax,
            show_plot,
        )
        X, xvals, xmin, xmax, show_plot = (
            input_check.X,
            input_check.xvals,
            input_check.xmin,
            input_check.xmax,
            input_check.show_plot,
        )
        pdf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu))
        pdf = unpack_single_arrays(pdf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Lognormal Distribution\n" + " Probability Density Function " + "\n" + self.param_title)
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
        input_check = distributions_input_checking(
            self, "CDF", xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x
        )
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
        ) = (
            input_check.X,
            input_check.xvals,
            input_check.xmin,
            input_check.xmax,
            input_check.show_plot,
            input_check.plot_CI,
            input_check.CI_type,
            input_check.CI,
            input_check.CI_y,
            input_check.CI_x,
        )
        cdf = ss.lognorm.cdf(X, self.sigma, self.gamma, np.exp(self.mu))
        cdf = unpack_single_arrays(cdf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Lognormal Distribution\n" + " Cumulative Distribution Function " + "\n" + self.param_title,
            )
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.lognormal_CI(
                self,
                func="CDF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

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

        lower_CI, upper_CI = extract_CI(dist=self, func="CDF", CI_type=CI_type, CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_type == "time":
                return lower_CI, self.quantile(CI_y), upper_CI
            elif CI_type == "reliability":
                cdf_point = ss.lognorm.cdf(CI_x, self.sigma, self.gamma, np.exp(self.mu))
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
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64, npt.NDArray[np.float64]]:
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
        input_check = distributions_input_checking(
            self, "SF", xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x
        )
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
        ) = (
            input_check.X,
            input_check.xvals,
            input_check.xmin,
            input_check.xmax,
            input_check.show_plot,
            input_check.plot_CI,
            input_check.CI_type,
            input_check.CI,
            input_check.CI_y,
            input_check.CI_x,
        )
        sf = ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu))
        sf = unpack_single_arrays(sf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Lognormal Distribution\n" + " Survival Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.lognormal_CI(
                self,
                func="SF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

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

        lower_CI, upper_CI = extract_CI(dist=self, func="SF", CI_type=CI_type, CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_type == "time":
                return lower_CI, self.inverse_SF(CI_y), upper_CI
            elif CI_type == "reliability":
                sf_point = ss.lognorm.sf(CI_x, self.sigma, self.gamma, np.exp(self.mu))
                return lower_CI, unpack_single_arrays(sf_point), upper_CI
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
        input_check = distributions_input_checking(
            self,
            "HF",
            xvals,
            xmin,
            xmax,
            show_plot,
        )
        X, xvals, xmin, xmax, show_plot = (
            input_check.X,
            input_check.xvals,
            input_check.xmin,
            input_check.xmax,
            input_check.show_plot,
        )
        hf = ss.lognorm.pdf(X, self.sigma, self.gamma, np.exp(self.mu)) / ss.lognorm.sf(
            X,
            self.sigma,
            self.gamma,
            np.exp(self.mu),
        )
        hf = unpack_single_arrays(hf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Lognormal Distribution\n" + " Hazard Function " + "\n" + self.param_title)
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
        CI_type: None | Literal["time", "reliability"] = None,
        CI=None,
        CI_y=None,
        CI_x=None,
        **kwargs,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64, npt.NDArray[np.float64]]:
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
        input_check = distributions_input_checking(
            self, "CHF", xvals, xmin, xmax, show_plot, plot_CI, CI_type, CI, CI_y, CI_x
        )
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
        ) = (
            input_check.X,
            input_check.xvals,
            input_check.xmin,
            input_check.xmax,
            input_check.show_plot,
            input_check.plot_CI,
            input_check.CI_type,
            input_check.CI,
            input_check.CI_y,
            input_check.CI_x,
        )
        chf = -np.log(ss.lognorm.sf(X, self.sigma, self.gamma, np.exp(self.mu)))
        chf = unpack_single_arrays(chf)
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str("Lognormal Distribution\n" + " Cumulative Hazard Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.lognormal_CI(
                self,
                func="CHF",
                CI_type=CI_type,
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

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

        lower_CI, upper_CI = extract_CI(dist=self, func="CHF", CI_type=CI_type, CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_type == "time":
                return lower_CI, self.inverse_SF(np.exp(-CI_y)), upper_CI
            elif CI_type == "reliability":
                chf_point = -np.log(ss.lognorm.sf(CI_x, self.sigma, self.gamma, np.exp(self.mu)))
                return lower_CI, unpack_single_arrays(chf_point), upper_CI
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
        ppf = ss.lognorm.ppf(q, self.sigma, self.gamma, np.exp(self.mu))
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
        isf = ss.lognorm.isf(q, self.sigma, self.gamma, np.exp(self.mu))
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
            return ss.lognorm.sf(x, self.sigma, self.gamma, np.exp(self.mu))

        integral_R, error = integrate.quad(R, t, np.inf)
        MRL: float = integral_R / R(t)
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
                    "Descriptive statistics for Lognormal distribution with mu = "
                    + str(self.mu)
                    + " and sigma = "
                    + str(self.sigma),
                ),
                bold=True,
                underline=True,
            )
        else:
            colorprint(
                str(
                    "Descriptive statistics for Lognormal distribution with mu = "
                    + str(self.mu)
                    + ", sigma = "
                    + str(self.sigma)
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
            rng = np.random.default_rng(seed)
        RVS: npt.NDArray[np.float64] = ss.lognorm.rvs(
            self.sigma, self.gamma, np.exp(self.mu), size=number_of_samples, random_state=rng
        )
        return RVS

from dataclasses import dataclass
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


@dataclass
class Normal_Distribution:
    """Normal probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    mu : float, int
        Location parameter
    sigma : float, int
        Scale parameter. Must be > 0

    Returns
    -------
    name : str
        'Normal'
    name2 : 'str
        'Normal_2P'
    param_title_long : str
        'Normal Distribution (μ=5,σ=2)'
    param_title : str
        'μ=5,σ=2'
    parameters : list
        [mu,sigma]
    mu : float
    sigma : float
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

    mu: float | np.float64
    sigma: float | np.float64
    CI: float | None = None
    CI_type: str = "time"
    mean_standard_error: float | np.float64 | None = None
    sigma_standard_error: float | np.float64 | None = None
    Cov_mu_sigma: float | np.float64 | None = None

    def __post_init__(self):
        self.mean: float = self.mu
        self.variance: float = self.sigma**2
        self.standard_deviation: float = self.sigma
        self.median: float = self.mu
        self.mode: float = self.mu
        self.param_title = str("μ=" + round_and_string(self.mu, dec) + ",σ=" + round_and_string(self.sigma, dec))
        self.param_title_long = str(
            "Normal Distribution (μ="
            + round_and_string(self.mu, dec)
            + ",σ="
            + round_and_string(self.sigma, dec)
            + ")",
        )
        self.name: str = "Normal"
        self.name2: str = "Normal_2P"
        self.skewness: int = 0
        self.kurtosis: int = 3
        self.excess_kurtosis: int = 0
        self._pdf0 = 0  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = 0  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self.b5: float = ss.norm.ppf(0.05, loc=self.mu, scale=self.sigma)
        self.b95: float = ss.norm.ppf(0.95, loc=self.mu, scale=self.sigma)
        self.parameters: list[float] = [self.mu, self.sigma]
        if self.CI is not None:
            self.Z: float | None = -ss.norm.ppf((1 - self.CI) / 2)
        else:
            self.Z = None
        if self.CI_type is None:
            self.CI_type: str = "time"

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
        pdf = ss.norm.pdf(X, self.mu, self.sigma)
        cdf = ss.norm.cdf(X, self.mu, self.sigma)
        sf = ss.norm.sf(X, self.mu, self.sigma)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str("Normal Distribution" + "\n" + self.param_title)
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

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs) -> npt.NDArray[np.float64]:
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
        pdf: npt.NDArray[np.float64] = ss.norm.pdf(X, self.mu, self.sigma)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Normal Distribution\n" + " Probability Density Function " + "\n" + self.param_title)
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
        xmin: float | None = None,
        xmax: float | None = None,
        show_plot: bool | None = True,
        plot_CI: bool | None = True,
        CI_type: Literal["time", "reliability"] | None = None,
        CI: float | None = None,
        CI_y: float | None = None,
        CI_x: float | None = None,
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
        )  # type: ignore
        cdf = ss.norm.cdf(X, self.mu, self.sigma)
        cdf = unpack_single_arrays(cdf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str("Normal Distribution\n" + " Cumulative Distribution Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.normal_CI(
                self,
                func="CDF",
                CI_type=CI_type,  # type: ignore
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
                cdf_point = ss.norm.cdf(CI_x, self.mu, self.sigma)
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
        sf = ss.norm.sf(X, self.mu, self.sigma)
        sf = unpack_single_arrays(sf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Normal Distribution\n" + " Survival Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.normal_CI(
                self,
                func="SF",
                CI_type=CI_type,  # type: ignore
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
                sf_point = ss.norm.sf(CI_x, self.mu, self.sigma)
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
        hf = ss.norm.pdf(X, self.mu, self.sigma) / ss.norm.sf(X, self.mu, self.sigma)
        hf = unpack_single_arrays(hf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Normal Distribution\n" + " Hazard Function " + "\n" + self.param_title)
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
        chf: npt.NDArray[np.float64] = -np.log(ss.norm.sf(X, self.mu, self.sigma))
        chf = unpack_single_arrays(chf)
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str("Normal Distribution\n" + " Cumulative Hazard Function " + "\n" + self.param_title)
            plt.title(text_title)
            plt.subplots_adjust(top=0.81)

            distribution_confidence_intervals.normal_CI(
                self,
                func="CHF",
                CI_type=CI_type,  # type: ignore
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
                chf_point = -np.log(ss.norm.sf(CI_x, self.mu, self.sigma))
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
        ppf = ss.norm.ppf(q, loc=self.mu, scale=self.sigma)
        return ppf

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
        isf = ss.norm.isf(q, loc=self.mu, scale=self.sigma)
        return isf

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
            return ss.norm.sf(x, loc=self.mu, scale=self.sigma)

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
        colorprint(
            str(
                "Descriptive statistics for Normal distribution with mu = "
                + str(self.mu)
                + " and sigma = "
                + str(self.sigma),
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

    def random_samples(self, number_of_samples: int, seed: int | None = None) -> npt.NDArray[np.float64]:
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
        RVS: npt.NDArray[np.float64] = ss.norm.rvs(loc=self.mu, scale=self.sigma, size=number_of_samples)
        return RVS

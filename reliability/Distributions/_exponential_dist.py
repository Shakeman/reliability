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

class Exponential_Distribution:
    """Exponential probability distribution. Creates a probability distribution
    object.

    Parameters
    ----------
    Lambda : float, int
        Scale parameter. Must be > 0
    gamma : float, int, optional
        threshold (offset) parameter. Must be >= 0. Default = 0

    Returns
    -------
    name : str
        'Exponential'
    name2 : 'str
        'Exponential_1P' or 'Exponential_2P' depending on the value of the gamma
        parameter
    param_title_long : str
        'Exponential Distribution (λ=5)'
    param_title : str
        'λ=5'
    parameters : list
        [Lambda,gamma]
    Lambda : float
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

    def __init__(self, Lambda=None, gamma=0, **kwargs):
        self.name = "Exponential"
        if Lambda is None:
            raise ValueError("Parameter Lambda must be specified. Eg. Exponential_Distribution(Lambda=3)")
        self.Lambda: float = float(Lambda)
        self.gamma: float = float(gamma)
        self.parameters: npt.NDArray[np.float64] = np.array([self.Lambda, self.gamma])
        mean, var, skew, kurt = ss.expon.stats(scale=1 / self.Lambda, loc=self.gamma, moments="mvsk")
        self.mean: float = float(mean)
        self.variance: float = float(var)
        self.standard_deviation: np.float64 = var**0.5
        self.skewness = float(skew)
        self.kurtosis: float = kurt + 3
        self.excess_kurtosis: float = float(kurt)
        self.median: np.float64 = ss.expon.median(scale=1 / self.Lambda, loc=self.gamma)
        self.mode: float = self.gamma
        if self.gamma != 0:
            self.param_title = str(
                "λ=" + round_and_string(self.Lambda, dec) + ",γ=" + round_and_string(self.gamma, dec),
            )
            self.param_title_long = str(
                "Exponential Distribution (λ="
                + round_and_string(self.Lambda, dec)
                + ",γ="
                + round_and_string(gamma, dec)
                + ")",
            )
            self.name2 = "Exponential_2P"
        else:
            self.param_title: str = str("λ=" + round_and_string(self.Lambda, dec))
            self.param_title_long: str = str("Exponential Distribution (λ=" + round_and_string(self.Lambda, dec) + ")")
            self.name2: str = "Exponential_1P"
        self.b5: np.float64 = ss.expon.ppf(0.05, scale=1 / self.Lambda, loc=self.gamma)
        self.b95: np.float64 = ss.expon.ppf(0.95, scale=1 / self.Lambda, loc=self.gamma)

        # extracts values for confidence interval plotting
        if "Lambda_SE" in kwargs:
            self.Lambda_SE = kwargs.pop("Lambda_SE")
        else:
            self.Lambda_SE = None
        if "CI" in kwargs:
            CI = kwargs.pop("CI")
            self.Z = -ss.norm.ppf((1 - CI) / 2)
        else:
            self.Z = None
        for item in kwargs:
            colorprint(
                str(
                    "WARNING: "
                    + item
                    + " is not recognised as an appropriate entry in kwargs. Appropriate entries are Lambda_SE and CI.",
                ),
                text_color="red",
            )
        self._pdf0 = ss.expon.pdf(
            0,
            scale=1 / self.Lambda,
            loc=0,
        )  # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array.
        self._hf0 = self.Lambda  # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array

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

        pdf = ss.expon.pdf(X, scale=1 / self.Lambda, loc=self.gamma)
        cdf = ss.expon.cdf(X, scale=1 / self.Lambda, loc=self.gamma)
        sf = ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma)
        hf = np.ones_like(X) * self.Lambda
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        chf = (X - self.gamma) * self.Lambda
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)

        plt.figure(figsize=(9, 7))
        text_title = str("Exponential Distribution" + "\n" + self.param_title)
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

        pdf = ss.expon.pdf(X, scale=1 / self.Lambda, loc=self.gamma)
        pdf = unpack_single_arrays(pdf)

        if show_plot is True:
            limits = get_axes_limits()

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Exponential Distribution\n" + " Probability Density Function " + "\n" + self.param_title)
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
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_CI=True,
        CI=None,
        CI_y=None,
        CI_x=None,
        **kwargs,
    ):
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
        if "CI_type" in kwargs:
            kwargs.pop("CI_type")
            colorprint(
                "WARNING: CI_type is not required for the Exponential Distribution since bounds on time and bounds on reliability are identical.",
                text_color="red",
            )
        (
            X,
            xvals,
            xmin,
            xmax,
            show_plot,
            plot_CI,
            _,
            CI,
            CI_y,
            CI_x,
        ) = distributions_input_checking(self, "CDF", xvals, xmin, xmax, show_plot, plot_CI, None, CI, CI_y, CI_x)

        cdf = ss.expon.cdf(X, scale=1 / self.Lambda, loc=self.gamma)
        cdf = unpack_single_arrays(cdf)

        if show_plot is True:
            limits = get_axes_limits()

            p = plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(
                "Exponential Distribution\n" + " Cumulative Distribution Function " + "\n" + self.param_title,
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

            distribution_confidence_intervals.exponential_CI(
                self,
                func="CDF",
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

        lower_CI, upper_CI = extract_CI(dist=self, func="CDF", CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_y is not None:
                return lower_CI, self.quantile(CI_y), upper_CI
            elif CI_x is not None:
                cdf_point = ss.expon.cdf(CI_x, scale=1 / self.Lambda, loc=self.gamma)
                return lower_CI, unpack_single_arrays(cdf_point), upper_CI
        else:
            return cdf

    def SF(
        self,
        xvals=None,
        xmin=None,
        xmax=None,
        show_plot=True,
        plot_CI=True,
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
        if "CI_type" in kwargs:
            kwargs.pop("CI_type")
            colorprint(
                "WARNING: CI_type is not required for the Exponential Distribution since bounds on time and bounds on reliability are identical.",
                text_color="red",
            )
        (
            X,
            xvals,
            xmin,
            xmax,
            show_plot,
            plot_CI,
            _,
            CI,
            CI_y,
            CI_x,
        ) = distributions_input_checking(self, "SF", xvals, xmin, xmax, show_plot, plot_CI, None, CI, CI_y, CI_x)

        sf = ss.expon.sf(X, scale=1 / self.Lambda, loc=self.gamma)
        sf = unpack_single_arrays(sf)

        if show_plot is True:
            limits = get_axes_limits()

            p = plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Exponential Distribution\n" + " Survival Function " + "\n" + self.param_title)
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

            distribution_confidence_intervals.exponential_CI(
                self,
                func="SF",
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

        lower_CI, upper_CI = extract_CI(dist=self, func="SF", CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_y is not None:
                return lower_CI, self.inverse_SF(CI_y), upper_CI
            elif CI_x is not None:
                sf_point = ss.expon.sf(CI_x, scale=1 / self.Lambda, loc=self.gamma)
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

        hf = np.ones_like(X) * self.Lambda
        hf = zeroise_below_gamma(X=X, Y=hf, gamma=self.gamma)
        hf = unpack_single_arrays(hf)

        if show_plot is True:
            limits = get_axes_limits()

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Exponential Distribution\n" + " Hazard Function " + "\n" + self.param_title)
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
        if "CI_type" in kwargs:
            kwargs.pop("CI_type")
            colorprint(
                "WARNING: CI_type is not required for the Exponential Distribution since bounds on time and bounds on reliability are identical.",
                text_color="red",
            )
        (
            X,
            xvals,
            xmin,
            xmax,
            show_plot,
            plot_CI,
            _,
            CI,
            CI_y,
            CI_x,
        ) = distributions_input_checking(self, "CHF", xvals, xmin, xmax, show_plot, plot_CI, None, CI, CI_y, CI_x)

        chf = (X - self.gamma) * self.Lambda
        chf = zeroise_below_gamma(X=X, Y=chf, gamma=self.gamma)
        chf = unpack_single_arrays(chf)

        if show_plot is True:
            limits = get_axes_limits()

            p = plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str("Exponential Distribution\n" + " Cumulative Hazard Function " + "\n" + self.param_title)
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

            distribution_confidence_intervals.exponential_CI(
                self,
                func="CHF",
                plot_CI=plot_CI,
                CI=CI,
                text_title=text_title,
                color=p[0].get_color(),
            )

        lower_CI, upper_CI = extract_CI(dist=self, func="CHF", CI=CI, CI_y=CI_y, CI_x=CI_x)
        if lower_CI is not None and upper_CI is not None:
            if CI_y is not None:
                return lower_CI, self.inverse_SF(np.exp(-CI_y)), upper_CI
            elif CI_x is not None:
                chf_point = zeroise_below_gamma(X=CI_x, Y=(CI_x - self.gamma) * self.Lambda, gamma=self.gamma)
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
        ppf = ss.expon.ppf(q, scale=1 / self.Lambda, loc=self.gamma)
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
        isf = ss.expon.isf(q, scale=1 / self.Lambda, loc=self.gamma)
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
            return ss.expon.sf(x, scale=1 / self.Lambda, loc=self.gamma)

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
                str("Descriptive statistics for Exponential distribution with lambda = " + str(self.Lambda)),
                bold=True,
                underline=True,
            )
        else:
            colorprint(
                str(
                    "Descriptive statistics for Exponential distribution with lambda = "
                    + str(self.Lambda)
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
        RVS = ss.expon.rvs(scale=1 / self.Lambda, loc=self.gamma, size=number_of_samples)
        return RVS

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as ss
from scipy import integrate

from reliability.Utils import (
    colorprint,
    distributions_input_checking,
    get_axes_limits,
    restore_axes_limits,
    round_and_string,
    unpack_single_arrays,
)

dec = 4  # number of decimals to use when rounding descriptive statistics and parameter titles
np.seterr(divide="ignore", invalid="ignore")  # ignore the divide by zero warnings


class Beta_Distribution:
    """Beta probability distribution. Creates a probability distribution object.

    Parameters
    ----------
    alpha : float, int
        Shape parameter 1. Must be > 0
    beta : float, int
        Shape parameter 2. Must be > 0

    Returns
    -------
    name : str
        'Beta'
    name2 : 'str
        'Beta_2P'
    param_title_long : str
        'Beta Distribution (α=5,β=2)'
    param_title : str
        'α=5,β=2'
    parameters : list
        [alpha,beta]
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
    kwargs are not accepted

    """

    def __init__(self, alpha=None, beta=None):
        self.name = "Beta"
        self.name2 = "Beta_2P"
        if alpha is None or beta is None:
            raise ValueError("Parameters alpha and beta must be specified. Eg. Beta_Distribution(alpha=5,beta=2)")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.parameters: npt.NDArray[np.float64] = np.array([self.alpha, self.beta])
        mean, var, skew, kurt = ss.beta.stats(self.alpha, self.beta, 0, 1, moments="mvsk")
        self.mean = float(mean)
        self.variance = float(var)
        self.standard_deviation = var**0.5
        self.skewness = float(skew)
        self.kurtosis: np.float64 = kurt + 3
        self.excess_kurtosis = float(kurt)
        self.median = ss.beta.median(self.alpha, self.beta, 0, 1)
        if self.alpha > 1 and self.beta > 1:
            self.mode = (self.alpha - 1) / (self.beta + self.alpha - 2)
        else:
            self.mode = r"No mode exists unless $\alpha$ > 1 and $\beta$ > 1"
        self.param_title = str("α=" + round_and_string(self.alpha, dec) + ",β=" + round_and_string(self.beta, dec))
        self.param_title_long = str(
            "Beta Distribution (α="
            + round_and_string(self.alpha, dec)
            + ",β="
            + round_and_string(self.beta, dec)
            + ")",
        )
        self.b5 = ss.beta.ppf(0.05, self.alpha, self.beta, 0, 1)
        self.b95 = ss.beta.ppf(0.95, self.alpha, self.beta, 0, 1)

        # the pdf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._pdf0 = ss.beta.pdf(0, self.alpha, self.beta, 0, 1)
        # the hf at 0. Used by Utils.restore_axes_limits and Utils.generate_X_array
        self._hf0 = ss.beta.pdf(0, self.alpha, self.beta, 0, 1) / ss.beta.sf(0, self.alpha, self.beta, 0, 1)
        self.Z = None  # this is necessary because distributions_input_checking looks for this value

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
        pdf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1)
        cdf = ss.beta.cdf(X, self.alpha, self.beta, 0, 1)
        sf = ss.beta.sf(X, self.alpha, self.beta, 0, 1)
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        text_title = str("Beta Distribution" + "\n" + self.param_title)
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
        if isinstance(self.mode, str):
            text_mode = str("Mode = " + str(self.mode))  # required when mode is str
        else:
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
        pdf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1)
        pdf = unpack_single_arrays(pdf)

        if show_plot is True:
            limits = get_axes_limits()

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Beta Distribution\n" + " Probability Density Function " + "\n" + self.param_title)
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

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """Plots the CDF (cumulative distribution function)

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
        input_check = distributions_input_checking(self, "CDF", xvals, xmin, xmax, show_plot)
        X, xvals, xmin, xmax, show_plot = (
            input_check.X,
            input_check.xvals,
            input_check.xmin,
            input_check.xmax,
            input_check.show_plot,
        )
        cdf = ss.beta.cdf(X, self.alpha, self.beta, 0, 1)
        cdf = unpack_single_arrays(cdf)

        if show_plot is True:
            limits = get_axes_limits()

            plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str("Beta Distribution\n" + " Cumulative Distribution Function " + "\n" + self.param_title)
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

        return cdf

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """Plots the SF (survival function)

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
            "SF",
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
        sf = ss.beta.sf(X, self.alpha, self.beta, 0, 1)
        sf = unpack_single_arrays(sf)

        if show_plot is True:
            limits = get_axes_limits()

            plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Beta Distribution\n" + " Survival Function " + "\n" + self.param_title)
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
        hf = ss.beta.pdf(X, self.alpha, self.beta, 0, 1) / ss.beta.sf(X, self.alpha, self.beta, 0, 1)
        hf = unpack_single_arrays(hf)

        if show_plot is True:
            limits = get_axes_limits()

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Beta Distribution\n" + " Hazard Function " + "\n" + self.param_title)
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

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, **kwargs):
        """Plots the CHF (cumulative hazard function)

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
            "CHF",
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
        chf = -np.log(ss.beta.sf(X, self.alpha, self.beta, 0, 1))
        chf = unpack_single_arrays(chf)
        self._chf = chf  # required by the CI plotting part
        self._X = X

        if show_plot is True:
            limits = get_axes_limits()

            plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str("Beta Distribution\n" + " Cumulative Hazard Function " + "\n" + self.param_title)
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
        ppf = ss.beta.ppf(q, self.alpha, self.beta, 0, 1)
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
        isf = ss.beta.isf(q, self.alpha, self.beta, 0, 1)
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
            return ss.beta.sf(x, self.alpha, self.beta, 0, 1)

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
                "Descriptive statistics for Beta distribution with alpha = "
                + str(self.alpha)
                + " and beta = "
                + str(self.beta),
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
        RVS = ss.beta.rvs(self.alpha, self.beta, 0, 1, size=number_of_samples, random_state=rng)

        # this section is for resampling so that we always get numbers below 1.
        # For a Beta Distribution, 1 should be impossible, but scipy.stats will
        # still return 1's for Beta Distributions skewed towards 1 such as when
        # alpha = 0.01, beta = 0.01. This causes the fitters to fail since it
        # cannot handle 0 or 1 for a Beta Distribution.
        checked = False
        while checked is False:
            RVS = RVS[RVS < 1]  # remove 1's
            if len(RVS) < number_of_samples:  # resample if required
                resamples = ss.beta.rvs(self.alpha, self.beta, 0, 1, size=number_of_samples - len(RVS))
                RVS = np.append(RVS, resamples)  # append the new samples
            else:
                checked = True

        return RVS

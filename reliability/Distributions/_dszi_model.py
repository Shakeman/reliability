
import matplotlib.pyplot as plt
import numpy as np

from reliability.Distributions._beta_dist import Beta_Distribution
from reliability.Distributions._exponential_dist import Exponential_Distribution
from reliability.Distributions._gamma_dist import Gamma_Distribution
from reliability.Distributions._gumbel_dist import Gumbel_Distribution
from reliability.Distributions._loglogistic_dist import Loglogistic_Distribution
from reliability.Distributions._lognormal_dist import Lognormal_Distribution
from reliability.Distributions._normal_dist import Normal_Distribution
from reliability.Distributions._weibull_dist import Weibull_Distribution
from reliability.Utils import (
    colorprint,
    generate_X_array,
    get_axes_limits,
    restore_axes_limits,
    round_and_string,
    unpack_single_arrays,
)

dec = 4  # number of decimals to use when rounding descriptive statistics and parameter titles
np.seterr(divide="ignore", invalid="ignore")  # ignore the divide by zero warnings



class DSZI_Model:
    """Defective Subpopulation Zero Inflated Model. This model should be used when
    there are failures at t=0 ("dead on arrival") creating a zero inflated (ZI)
    distribution and/or many right censored failures creating a defective
    subpopulation (DS) model. The parameters DS and ZI represent the maximum and
    minimum of the CDF respectively. Their default values are 1 and 0 which is
    equivalent to a non-DS and non-ZI model. Leaving one as the default and
    specifying the other can be used to create a DS or ZI model, while
    specifying both parameters creates a DSZI model.

    The output API is similar to the other probability distributions (Weibull,
    Normal, etc.) as shown below.

    Parameters
    ----------
    distribution: object
        A probability distribution object representing the base distribution to
        which the DS and ZI transformations are made.
    DS : float, optional
        The defective subpopulation fraction. Must be between 0 and 1. Must be
        greater than ZI. This is the maximum of the CDF. Default is 1 which is
        equivalent to a non-DS CDF (ie. everything fails eventually).
    ZI : float, optional
        The zero inflated fraction. Must be between 0 and 1. Must be less than
        DS. This is the minimum of the CDF. Default is 0 which is
        equivalent to a non-ZI CDF (ie. no failures at t=0).

    Returns
    -------
    DS : float
    ZI : float
    name : str
        'DSZI'
    name2 : str
        'Defective Subpopulation Zero Inflated Weibull'. Exact string depends on
        the values of DS and ZI, and the type of base distribution.
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
    DS and ZI are optional but at least one of them must be specified. Leaving
    them both unspecified is equivalent to the base distribution specified in
    the "distribution" parameter.

    """

    def __init__(self, distribution, DS=None, ZI=None):
        if DS is None and ZI is None:
            raise ValueError(
                "DS and ZI cannot both be unspecified. Please specify one or both of these parameters to create a DS, ZI, or DSZI model.",
            )
        if DS is None:
            DS = float(1)
        if ZI is None:
            ZI = float(0)
        if DS == 1 and ZI == 0:
            colorprint(
                "WARNING: A DSZI model with DS = 1 and ZI = 0 is equivalent to the base distribution. There is no need to use a DSZI distribution if DS = 1 and ZI = 0.",
                text_color="red",
            )
        if ZI > DS:
            raise ValueError(
                "DS can not be greater than ZI. DS is the maximum of the CDF. ZI is the minimum of the CDF.",
            )
        if ZI >= 1 or ZI < 0:
            raise ValueError("ZI must be >= 0 and < 1. ZI is the minimum of the CDF.")
        if DS > 1 or DS <= 0:
            raise ValueError("DS must be > 0 and <= 1. DS is the maximum of the CDF")

        if type(distribution) not in [
            Weibull_Distribution,
            Normal_Distribution,
            Lognormal_Distribution,
            Exponential_Distribution,
            Beta_Distribution,
            Gamma_Distribution,
            Loglogistic_Distribution,
            Gumbel_Distribution,
        ]:
            raise ValueError(
                "distribution must be an array or list of probability distributions. Each distribution must be created using the reliability.Distributions module.",
            )

        self.__base_distribution = distribution
        self.DS = DS
        self.ZI = ZI
        self.mean = self.__base_distribution.mean
        self.variance = self.__base_distribution.variance
        self.standard_deviation = self.__base_distribution.standard_deviation
        self.skewness = self.__base_distribution.skewness
        self.kurtosis = self.__base_distribution.kurtosis
        self.excess_kurtosis = self.__base_distribution.excess_kurtosis
        self.median = self.__base_distribution.median
        self.mode = self.__base_distribution.mode
        self.name = "DSZI"
        if distribution.parameters[-1] == 0:
            params_in_dist = len(distribution.parameters) - 1
        else:
            params_in_dist = len(distribution.parameters)
        if ZI == 0:
            self.__model_title = "DS Model"
            self.name2 = "Defective Subpopulation " + self.__base_distribution.name
            self.__number_of_params = params_in_dist + 1
        elif DS == 1:
            self.__model_title = "ZI Model"
            self.name2 = "Zero Inflated " + self.__base_distribution.name
            self.__number_of_params = params_in_dist + 1
        else:  # DSZI
            self.__model_title = "DSZI Model"
            self.name2 = "Defective Subpopulation Zero Inflated " + self.__base_distribution.name
            self.__number_of_params = params_in_dist + 2

        xmax = self.__base_distribution.quantile(1 - 1e-10)
        xmin = self.__base_distribution.quantile(1e-10)
        X = np.linspace(xmin, xmax, 1000000)
        pdf0 = self.__base_distribution.PDF(xvals=X, show_plot=False)
        pdf = pdf0 * (self.DS - self.ZI)  # the DSZI formula for the PDF
        cdf0 = self.__base_distribution.CDF(xvals=X, show_plot=False)
        cdf = cdf0 * (self.DS - self.ZI) + self.ZI  # the DSZI formula for the CDF
        self.__pdf_init = pdf
        self.__cdf_init = cdf
        self.__xvals_init = X
        self.b5 = X[np.argmin(abs(cdf - 0.05))]
        self.b95 = X[np.argmin(abs(cdf - 0.95))]

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
        X = generate_X_array(dist=self.__base_distribution, xvals=xvals, xmin=xmin, xmax=xmax)  # obtain the X array

        pdf0 = self.__base_distribution.PDF(xvals=X, show_plot=False)
        pdf = pdf0 * (self.DS - self.ZI)  # the DSZI formula for the PDF
        cdf0 = self.__base_distribution.CDF(xvals=X, show_plot=False)
        cdf = cdf0 * (self.DS - self.ZI) + self.ZI  # the DSZI formula for the CDF
        sf = 1 - cdf
        hf = pdf / sf
        chf = -np.log(sf)

        plt.figure(figsize=(9, 7))
        plt.suptitle(self.__model_title, fontsize=15)

        plt.subplot(231)
        plt.plot(X, pdf)
        restore_axes_limits(
            [(0, 1), (0, 1), False],
            dist=self.__base_distribution,
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
            [(0, 1), (0, 1), False],
            dist=self.__base_distribution,
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
            [(0, 1), (0, 1), False],
            dist=self.__base_distribution,
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
            [(0, 1), (0, 1), False],
            dist=self.__base_distribution,
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
            [(0, 1), (0, 1), False],
            dist=self.__base_distribution,
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
        # obtain the X array
        if xmin is None and xmax is None and type(xvals) not in [list, np.ndarray, type(None)]:
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self.__base_distribution, xvals=xvals, xmin=xmin, xmax=xmax)

        pdf0 = self.__base_distribution.PDF(xvals=X, show_plot=False)
        pdf = pdf0 * (self.DS - self.ZI)  # the DSZI formula for the PDF
        pdf = unpack_single_arrays(pdf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, pdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str(self.__model_title + "\n" + "Probability Density Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self.__base_distribution,
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
        # obtain the X array
        if xmin is None and xmax is None and type(xvals) not in [list, np.ndarray, type(None)]:
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self.__base_distribution, xvals=xvals, xmin=xmin, xmax=xmax)

        cdf0 = self.__base_distribution.CDF(xvals=X, show_plot=False)
        cdf = cdf0 * (self.DS - self.ZI) + self.ZI  # the DSZI formula for the CDF
        cdf = unpack_single_arrays(cdf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, cdf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str(self.__model_title + "\n" + "Cumulative Distribution Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self.__base_distribution,
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
        # obtain the X array
        if xmin is None and xmax is None and type(xvals) not in [list, np.ndarray, type(None)]:
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self.__base_distribution, xvals=xvals, xmin=xmin, xmax=xmax)

        cdf0 = self.__base_distribution.CDF(xvals=X, show_plot=False)
        cdf = cdf0 * (self.DS - self.ZI) + self.ZI  # the DSZI formula for the CDF
        sf = 1 - cdf
        sf = unpack_single_arrays(sf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, sf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str(self.__model_title + "\n" + "Survival Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self.__base_distribution,
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
        # obtain the X array
        if xmin is None and xmax is None and type(xvals) not in [list, np.ndarray, type(None)]:
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self.__base_distribution, xvals=xvals, xmin=xmin, xmax=xmax)

        pdf0 = self.__base_distribution.PDF(xvals=X, show_plot=False)
        pdf = pdf0 * (self.DS - self.ZI)  # the DSZI formula for the PDF
        cdf0 = self.__base_distribution.CDF(xvals=X, show_plot=False)
        cdf = cdf0 * (self.DS - self.ZI) + self.ZI  # the DSZI formula for the CDF
        hf = pdf / (1 - cdf)  # pdf/sf
        hf = unpack_single_arrays(hf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, hf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str(self.__model_title + "\n" + "Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self.__base_distribution,
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
        # obtain the X array
        if xmin is None and xmax is None and type(xvals) not in [list, np.ndarray, type(None)]:
            X = xvals
            show_plot = False
        else:
            X = generate_X_array(dist=self.__base_distribution, xvals=xvals, xmin=xmin, xmax=xmax)

        cdf0 = self.__base_distribution.CDF(xvals=X, show_plot=False)
        cdf = cdf0 * (self.DS - self.ZI) + self.ZI  # the DSZI formula for the CDF
        chf = -np.log(1 - cdf)  # -ln(sf)
        chf = unpack_single_arrays(chf)

        if show_plot is True:
            limits = get_axes_limits()  # get the previous axes limits

            plt.plot(X, chf, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative hazard")
            text_title = str(self.__model_title + "\n" + "Cumulative Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.85)

            restore_axes_limits(
                limits,
                dist=self.__base_distribution,
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
            Quantile to be calculated. Must be between ZI and DS.
            If q < ZI or q > DS then a ValueError will be raised.

        Returns
        -------
        x : float
            The inverse of the CDF at q. This is the probability that a random
            variable from the distribution is < q

        """
        if type(q) in [int, float, np.float64]:
            if q < self.ZI or q > self.DS:
                raise ValueError(
                    "Quantile must be between ZI and DS. ZI = " + str(self.ZI) + ", DS = " + str(self.DS) + ".",
                )
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                raise ValueError(
                    "Quantile must be between ZI and DS. ZI = " + str(self.ZI) + ", DS = " + str(self.DS) + ".",
                )
        else:
            raise ValueError("Quantile must be of type float, list, array")
        ppf = self.__xvals_init[np.argmin(abs(self.__cdf_init - q))]
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
        isf = self.__xvals_init[np.argmin(abs((1 - self.__cdf_init) - q))]
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
            The mean residual life.

        Notes
        -----
        If DS < 1 the MRL will return np.inf

        """
        MRL = np.inf if self.DS < 1 else self.__base_distribution.mean_residual_life(t=t)
        # infinite life if the CDF never reaches 1
        # the MRL of the scaled distribution is the same as that of the base distribution
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
        colorprint("Descriptive statistics for DSZI Model", bold=True, underline=True)
        print("Mean = ", self.mean)
        print("Median =", self.median)
        print("Mode =", self.mode)
        print("5th quantile =", self.b5)
        print("95th quantile =", self.b95)
        print("Standard deviation =", self.standard_deviation)
        print("Variance =", self.variance)
        print("Skewness =", self.skewness)
        print("Excess kurtosis =", self.excess_kurtosis)

    def random_samples(self, number_of_samples, right_censored_time=None, seed=None):
        """Draws random samples from the probability distribution

        Parameters
        ----------
        number_of_samples : int
            The number of samples to be drawn. Must be greater than 0.
        right_censored_time : float
            The time to use as the right censored value. Only required if DS is
            not 1. The right_censored_time can be thought of as the end of the
            observation period.
        seed : int, optional
            The random seed passed to numpy. Default = None

        Returns
        -------
        failures, right_censored : array, array
            failures is an array of the random samples of the failure times.
            right_censored is an array of the right censored random samples.
            failures will contain zeros is ZI > 0.
            right_censored will be empty if DS = 1, otherwise all of the
            All the right_censored samples will be equal to the parameter
            right_censored_time.

        Notes
        -----
        If any of the failure times exceed the right_censored_time, these
        failures will be converted to right_censored_time and moved into the
        right_censored array. A warning will be printed if this occurs. To
        prevent this from occurring, select a higher right_censored_time.

        """
        if right_censored_time is None:
            if self.DS < 1:
                raise ValueError("right_censored_time must be provided if DS is not 1")
            else:
                right_censored_time = 1  # dummy value which is multiplied by an empty array.
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("number_of_samples must be an integer greater than 0")
        if seed is not None:
            np.random.seed(seed)

        samples0 = np.random.choice(
            [0, 1, 2],
            size=number_of_samples,
            p=[self.ZI, self.DS - self.ZI, 1 - self.DS],
        )  # 0 is number of ZI, 1 is number of failures, 2 is number of right censored
        frequency = np.histogram(samples0, bins=[0, 1, 2, 3])[0]
        num_ZI = frequency[0]
        num_failures = frequency[1]
        num_DS = frequency[2]
        zeros = np.zeros(num_ZI)
        failures = self.__base_distribution.random_samples(int(num_failures), seed=seed)
        right_censored = np.ones(num_DS) * right_censored_time
        failures = np.hstack([zeros, failures])  # combine the zeros and failure times into "failures"

        # this moves failures into right_censored if they exceed right_censored_time. Only applied if there are right censored times (DS < 1)
        if self.DS < 1 and max(failures) > right_censored_time:
            colorprint(
                "WARNING: some of the failure times exceeded the right_censored_time. These failures have been converted to the right_censored_time and moved into the right_censored array. This may bias the random samples so you should set a higher right_censored_time if possible.",
                text_color="red",
            )
            num_failure_beyond_right_censored_time = len(np.where(failures > right_censored_time)[0])
            failures = np.delete(failures, np.where(failures > right_censored_time))
            right_censored = np.append(
                right_censored,
                np.ones(num_failure_beyond_right_censored_time) * right_censored_time,
            )

        return failures, right_censored

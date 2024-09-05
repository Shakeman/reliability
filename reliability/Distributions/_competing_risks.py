import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

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
    get_axes_limits,
    restore_axes_limits,
    round_and_string,
    unpack_single_arrays,
)

dec = 4  # number of decimals to use when rounding descriptive statistics and parameter titles
np.seterr(divide="ignore", invalid="ignore")  # ignore the divide by zero warnings


class Competing_Risks_Model:
    """The competing risks model is used to model the effect of multiple risks
    (expressed as probability distributions) that act on a system over time.
    The model is obtained using the product of the survival functions:

    :math:`SF_{total} = SF_1 × SF_2 × SF_3 × ... × SF_n`

    The output API is similar to the other probability distributions (Weibull,
    Normal, etc.) as shown below.

    Parameters
    ----------
    distributions : list, array
        a list or array of probability distribution objects used to construct
        the model

    Returns
    -------
    name : str
        'Competing risks'
    name2 : str
        'Competing risks using 3 distributions'. The exact name depends on the
        number of distributions used
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
    An equivalent form of this model is to sum the hazard or cumulative hazard
    functions which will give the same result. In this way, we see the CDF, HF,
    and CHF of the overall model being equal to or higher than any of the
    constituent distributions. Similarly, the SF of the overall model will
    always be equal to or lower than any of the constituent distributions.
    The PDF occurs earlier in time since the earlier risks cause the population
    to fail sooner leaving less to fail due to the later risks.

    This model should be used when a data set has been divided by failure mode
    and each failure mode has been modelled separately. The competing risks
    model can then be used to recombine the constituent distributions into a
    single model. Unlike the mixture model, there are no proportions as the
    risks are competing to cause failure rather than being mixed.

    As this process is multiplicative for the survival function, and may accept
    many distributions of different types, the mathematical formulation quickly
    gets complex. For this reason, the algorithm combines the models numerically
    rather than empirically so there are no simple formulas for many of the
    descriptive statistics (mean, median, etc.). Also, the accuracy of the model
    is dependent on xvals. If the xvals array is small (<100 values) then the
    answer will be 'blocky' and inaccurate. The variable xvals is only accepted
    for PDF, CDF, SF, HF, CHF. The other methods (like random samples) use the
    default xvals for maximum accuracy. The default number of values generated
    when xvals is not given is 1000. Consider this carefully when specifying
    xvals in order to avoid inaccuracies in the results.

    """

    def __init__(self, distributions):
        if type(distributions) not in [list, np.ndarray]:
            raise ValueError("distributions must be a list or array of distribution objects.")
        contains_normal_or_gumbel = False
        for dist in distributions:
            if type(dist) not in [
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
                    "distributions must be an array or list of probability distributions. Each distribution must be created using the reliability.Distributions module.",
                )
            if type(dist) in [Normal_Distribution, Gumbel_Distribution]:
                contains_normal_or_gumbel = (
                    True  # check if we can have negative xvals (allowable if only normal and gumbel are in the mixture)
                )
        self.__contains_normal_or_gumbel = contains_normal_or_gumbel
        self.distributions = distributions  # this just passes the distributions to the __combiner which is used by the other functions along with the xvals. No combining can occur without xvals.
        self.name = "Competing risks"
        self.num_dists = len(distributions)
        self.name2 = str("Competing risks using " + str(self.num_dists) + " distributions")

        # This is essentially just the same as the __combiner method but more automated with a high amount of detail for the X array to minimize errors
        xmax = -1e100
        xmin = 1e100
        xmax999 = -1e100
        xmin001 = 1e100
        xmax_inf = -1e100
        number_of_params = 0
        for dist in distributions:
            params_in_dist = len(dist.parameters) - 1 if dist.parameters[-1] == 0 else len(dist.parameters)
            number_of_params += params_in_dist
            xmax = max(xmax, dist.quantile(1 - 1e-10))
            xmin = min(xmin, dist.quantile(1e-10))
            xmax999 = max(xmax999, dist.quantile(0.999))
            xmin001 = min(xmin001, dist.quantile(0.001))
            xmax_inf = max(xmax_inf, dist.quantile(1 - 1e-10))  # effective infinity used by MRL
        self.__number_of_params = number_of_params
        self.__xmax999 = xmax999
        self.__xmin001 = xmin001
        self.__xmax_inf = xmax_inf

        X = np.linspace(xmin, xmax, 1000000)
        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative_zeros = np.zeros_like(X_negative)
        Y_negative_ones = np.ones_like(X_negative)

        sf = np.ones_like(X)
        hf = np.zeros_like(X)
        # combine the distributions using the product of the survival functions: SF_total = SF_1 x SF_2 x SF_3 x ....x SF_n
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                sf *= distributions[i].SF(X, show_plot=False)
                hf += distributions[i].HF(X, show_plot=False)
            else:
                sf *= np.hstack([Y_negative_ones, distributions[i].SF(X_positive, show_plot=False)])
                hf += np.hstack([Y_negative_zeros, distributions[i].HF(X_positive, show_plot=False)])
        pdf = hf * sf
        np.nan_to_num(
            pdf,
            copy=False,
            nan=0.0,
            posinf=None,
            neginf=None,
        )  # because the hf is nan (which is expected due to being pdf/sf=0/0)

        self.__xvals_init = X  # used by random_samples
        self.__pdf_init = pdf  # used by random_samples
        self.__sf_init = sf  # used by quantile and inverse_SF
        self.mean = integrate.simpson(pdf * X, x=X)
        self.standard_deviation = (integrate.simpson(pdf * (X - self.mean) ** 2, x=X)) ** 0.5
        self.variance = self.standard_deviation**2
        self.skewness = integrate.simpson(pdf * ((X - self.mean) / self.standard_deviation) ** 3, x=X)
        self.kurtosis = integrate.simpson(pdf * ((X - self.mean) / self.standard_deviation) ** 4, x=X)
        self.mode = X[np.argmax(pdf)]
        self.median = X[np.argmin(abs(sf - 0.5))]
        self.excess_kurtosis = self.kurtosis - 3
        self.b5 = X[np.argmin(abs((1 - sf) - 0.05))]
        self.b95 = X[np.argmin(abs((1 - sf) - 0.95))]

    def __combiner(self, xvals=None, xmin=None, xmax=None):
        """This is a hidden function used to combine the distributions numerically.
        It is necessary to do this outside of the __init__ method as it needs to be called by each function (PDF, CDF...) so that xvals is used consistently.
        This approach keeps the API the same as the other probability distributions.
        Users should never need to access this function directly.
        """
        distributions = self.distributions

        # obtain the X values
        if xvals is not None:
            X = xvals
        else:
            if xmin is None:
                if self.__xmin001 > 0 and self.__xmin001 - (self.__xmax999 - self.__xmin001) * 0.3 < 0:
                    xmin = 0  # if its positive but close to zero then just make it zero
                else:
                    xmin = self.__xmin001
            if xmax is None:
                xmax = self.__xmax999
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            X = np.linspace(
                xmin,
                xmax,
                1000,
            )  # this is a big array because everything is numerical rather than empirical. Small array sizes will lead to blocky (inaccurate) results.

        # convert to numpy array if given list. raise error for other types. check for values below 0.
        if type(X) in [int, float]:
            X = np.array([X])
        elif type(X) in [np.ndarray, list]:
            X = np.asarray(X)
        else:
            raise ValueError("unexpected type in xvals. Must be  list, or array")

        if min(X) < 0 and self.__contains_normal_or_gumbel is False:
            raise ValueError(
                "xvals was found to contain values below 0. This is only allowed if some of the mixture components are Normal or Gumbel distributions.",
            )

        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative_zeros = np.zeros_like(X_negative).astype(np.float64)
        Y_negative_ones = np.ones_like(X_negative).astype(np.float64)

        sf = np.ones_like(X).astype(np.float64)
        hf = np.zeros_like(X).astype(np.float64)
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                sf *= distributions[i].SF(X, show_plot=False)
                hf += distributions[i].HF(X, show_plot=False)
            else:
                sf *= np.hstack([Y_negative_ones, distributions[i].SF(X_positive, show_plot=False)])
                hf += np.hstack([Y_negative_zeros, distributions[i].HF(X_positive, show_plot=False)])
        pdf = sf * hf
        np.nan_to_num(
            pdf,
            copy=False,
            nan=0.0,
            posinf=None,
            neginf=None,
        )  # because the hf may contain nan (which is expected due to being pdf/sf=0/0 for high xvals)

        # these are all hidden to the user but can be accessed by the other functions in this module
        self.__xvals = X
        self.__pdf = pdf
        self.__cdf = 1 - sf
        self.__sf = sf
        self.__hf = hf
        self.__chf = -np.log(sf)
        self._pdf0 = pdf[0]
        self._hf0 = hf[0]

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
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)
        plt.figure(figsize=(9, 7))
        text_title = "Competing Risks Model"
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        self.__pdf[self.__pdf > 1e100] = 1e100
        plt.plot(self.__xvals, self.__pdf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="PDF",
            X=self.__xvals,
            Y=self.__pdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Probability Density\nFunction")

        plt.subplot(232)
        plt.plot(self.__xvals, self.__cdf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="CDF",
            X=self.__xvals,
            Y=self.__cdf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Cumulative Distribution\nFunction")

        plt.subplot(233)
        plt.plot(self.__xvals, self.__sf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="SF",
            X=self.__xvals,
            Y=self.__sf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Survival Function")

        plt.subplot(234)
        self.__hf[self.__hf > 1e100] = 1e100
        plt.plot(self.__xvals, self.__hf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="HF",
            X=self.__xvals,
            Y=self.__hf,
            xvals=xvals,
            xmin=xmin,
            xmax=xmax,
        )
        plt.title("Hazard Function")

        plt.subplot(235)
        plt.plot(self.__xvals, self.__chf)
        restore_axes_limits(
            ((0, 1), (0, 1), False),
            dist=self,
            func="CHF",
            X=self.__xvals,
            Y=self.__chf,
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

    def PDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, plot_components=False, **kwargs):
        """Plots the PDF (probability density function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
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
        The plot will be shown if show_plot is True (which it is by default) and
        len(xvals) >= 2.

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if len(self.__xvals) < 2:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.PDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.PDF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Competing risks model"
            limits = get_axes_limits()
            self.__pdf[self.__pdf > 1e100] = 1e100
            plt.plot(self.__xvals, self.__pdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Competing Risks Model\n" + " Probability Density Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="PDF",
                X=self.__xvals,
                Y=self.__pdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

        return unpack_single_arrays(self.__pdf)

    def CDF(self, xvals=None, xmin=None, xmax=None, show_plot=True, plot_components=False, **kwargs):
        """Plots the CDF (cumulative distribution function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
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
        The plot will be shown if show_plot is True (which it is by default) and
        len(xvals) >= 2.

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if len(self.__xvals) < 2:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CDF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__cdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str("Competing Risks Model\n" + " Cumulative Distribution Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="CDF",
                X=self.__xvals,
                Y=self.__cdf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

        return unpack_single_arrays(self.__cdf)

    def SF(self, xvals=None, xmin=None, xmax=None, show_plot=True, plot_components=False, **kwargs):
        """Plots the SF (survival function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
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
        The plot will be shown if show_plot is True (which it is by default) and
        len(xvals) >= 2.

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if len(self.__xvals) < 2:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.SF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.SF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__sf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Competing Risks Model\n" + " Survival Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="SF",
                X=self.__xvals,
                Y=self.__sf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

        return unpack_single_arrays(self.__sf)

    def HF(self, xvals=None, xmin=None, xmax=None, show_plot=True, plot_components=False, **kwargs):
        """Plots the HF (hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
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
        The plot will be shown if show_plot is True (which it is by default) and
        len(xvals) >= 2.

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if len(self.__xvals) < 2:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.HF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.HF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Competing risks model"
            limits = get_axes_limits()
            self.__hf[self.__hf > 1e100] = 1e100
            plt.plot(self.__xvals, self.__hf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Competing Risks Model\n" + " Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="HF",
                X=self.__xvals,
                Y=self.__hf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

        return unpack_single_arrays(self.__hf)

    def CHF(self, xvals=None, xmin=None, xmax=None, show_plot=True, plot_components=False, **kwargs):
        """Plots the CHF (cumulative hazard function)

        Parameters
        ----------
        show_plot : bool, optional
            True or False. Default = True
        plot_components : bool
            Option to plot the components of the model. True or False. Default = False.
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
        The plot will be shown if show_plot is True (which it is by default) and
        len(xvals) >= 2.

        If xvals is specified, it will be used. If xvals is not specified but
        xmin and/or xmax are specified then an array with 200 elements will be
        created using these limits. If nothing is specified then the range will
        be based on the distribution's parameters.

        """
        Competing_Risks_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        if len(self.__xvals) < 2:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CHF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CHF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Competing risks model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__chf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative Hazard")
            text_title = str("Competing Risks Model\n" + " Cumulative Hazard Function")
            plt.title(text_title)
            plt.subplots_adjust(top=0.87)

            restore_axes_limits(
                limits,
                dist=self,
                func="CHF",
                X=self.__xvals,
                Y=self.__chf,
                xvals=xvals,
                xmin=xmin,
                xmax=xmax,
            )

        return unpack_single_arrays(self.__chf)

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
        ppf = self.__xvals_init[np.argmin(abs((1 - self.__sf_init) - q))]
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
        isf = self.__xvals_init[np.argmin(abs(self.__sf_init - q))]
        return unpack_single_arrays(isf)

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
            "Descriptive statistics for Competing Risks Model",
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

        def __subcombiner(X):
            """This function does what __combiner does but more efficiently and
            also accepts single values
            """
            if isinstance(X, np.ndarray):
                sf = np.ones_like(X)
                X_positive = X[X >= 0]
                X_negative = X[X < 0]
                Y_negative = np.ones_like(X_negative)
                for i in range(len(self.distributions)):
                    if type(self.distributions[i]) in [
                        Normal_Distribution,
                        Gumbel_Distribution,
                    ]:
                        sf *= self.distributions[i].SF(X, show_plot=False)
                    else:
                        sf *= np.hstack(
                            [
                                Y_negative,
                                self.distributions[i].SF(X_positive, show_plot=False),
                            ],
                        )
            else:
                sf = 1
                for i in range(len(self.distributions)):
                    if (
                        type(self.distributions[i])
                        in [
                            Normal_Distribution,
                            Gumbel_Distribution,
                        ]
                        or X > 0
                    ):
                        sf *= self.distributions[i].SF(X, show_plot=False)
            return sf

        t_full = np.linspace(t, self.__xmax_inf, 1000000)
        sf_full = __subcombiner(t_full)
        sf_single = __subcombiner(t)
        MRL = integrate.simpson(sf_full, x=t_full) / sf_single
        return MRL

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
        rng = np.random.default_rng(seed)
        return rng.choice(
            a=self.__xvals_init,
            size=number_of_samples,
            p=self.__pdf_init / np.sum(self.__pdf_init),
        )

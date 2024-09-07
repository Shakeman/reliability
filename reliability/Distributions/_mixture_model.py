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


class Mixture_Model:
    """The mixture model is used to create a distribution that contains parts from
    multiple distributions. This allows for a more complex model to be
    constructed as the sum of other distributions, each multiplied by a
    proportion (where the proportions sum to 1). The model is obtained using the
    sum of the cumulative distribution functions.

    :math:`CDF_{total} = (CDF_1 x p_1) + (CDF_2 x p_2) + (CDF_3 x p_3) + ... + (CDF_n x p_n)`

    The output API is similar to the other probability distributions (Weibull,
    Normal, etc.) as shown below.

    Parameters
    ----------
    distributions : list, array
        List or array of probability distribution objects used to construct the
        model.
    proportions : list, array
        List or array of floats specifying how much of each distribution to
        add to the mixture. The sum of proportions must always be 1.

    Returns
    -------
    name : str
        'Mixture'
    name2 : str
        'Mixture using 3 distributions'. The exact name depends on the number of
        distributions used.
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
    An equivalent form of this model is to sum the PDF. SF is obtained as 1-CDF.
    Note that you cannot simply sum the HF or CHF as this method would be
    equivalent to the competing risks model. In this way, we see the mixture
    model will always lie somewhere between the constituent models.

    This model should be used when a data set cannot be modelled by a single
    distribution, as evidenced by the shape of the PDF, CDF or probability plot
    (points do not form a straight line). Unlike the competing risks model, this
    model requires the proportions to be supplied.

    As this process is additive for the survival function, and may accept many
    distributions of different types, the mathematical formulation quickly gets
    complex. For this reason, the algorithm combines the models numerically
    ather than empirically so there are no simple formulas for many of the
    descriptive statistics (mean, median, etc.). Also, the accuracy of the model
    is dependent on xvals. If the xvals array is small (<100 values) then the
    answer will be 'blocky' and inaccurate. The variable xvals is only accepted
    for PDF, CDF, SF, HF, CHF. The other methods (like random samples) use the
    default xvals for maximum accuracy. The default number of values generated
    when xvals is not given is 1000. Consider this carefully when specifying
    xvals in order to avoid inaccuracies in the results.

    """

    def __init__(self, distributions, proportions=None) -> None:
        if type(distributions) not in [list, np.ndarray]:
            msg = "distributions must be a list or array of distribution objects."
            raise ValueError(msg)
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
                msg = "distributions must be an array or list of probability distributions. Each distribution must be created using the reliability.Distributions module."
                raise ValueError(
                    msg,
                )
            if type(dist) in [Normal_Distribution, Gumbel_Distribution]:
                contains_normal_or_gumbel = (
                    True  # check if we can have negative xvals (allowable if only normal and gumbel are in the mixture)
                )
        self.__contains_normal_or_gumbel = contains_normal_or_gumbel

        if proportions is not None:
            if np.sum(proportions) != 1:
                msg = "the sum of the proportions must be 1"
                raise ValueError(msg)
            if len(proportions) != len(distributions):
                msg = "the length of the proportions array must match the length of the distributions array"
                raise ValueError(msg)
        else:
            proportions = np.ones_like(distributions) / len(
                distributions,
            )  # if proportions are not specified they are assumed to all be the same proportion

        self.proportions = proportions  # this just passes the proportions to the __combiner which is used by the other functions along with the xvals. No combining can occur without xvals.
        self.distributions = distributions  # this just passes the distributions to the __combiner which is used by the other functions along with the xvals. No combining can occur without xvals.
        self.name = "Mixture"
        self.num_dists = len(distributions)
        self.name2 = str("Mixture using " + str(self.num_dists) + " distributions")

        # This is essentially just the same as the __combiner method but more automated with a high amount of detail for the X array to minimize errors
        xmax = -1e100
        xmin = 1e100
        xmax999 = -1e100
        xmin001 = 1e100
        xmax_inf = -1e100
        number_of_params = 0
        for dist in distributions:
            params_in_dist = len(dist.parameters) - 1 if dist.parameters[-1] == 0 else len(dist.parameters)
            number_of_params += params_in_dist + 1  # plus 1 for the proportion
            xmax = max(xmax, dist.quantile(1 - 1e-10))
            xmin = min(xmin, dist.quantile(1e-10))
            xmax999 = max(xmax999, dist.quantile(0.999))
            xmin001 = min(xmin001, dist.quantile(0.001))
            xmax_inf = max(xmax_inf, dist.quantile(1 - 1e-10))  # effective infinity used by MRL
        self.__number_of_params = number_of_params - 1  # minus 1 because last proportion is not needed as they sum to 1
        self.__xmax999 = xmax999
        self.__xmin001 = xmin001
        self.__xmax_inf = xmax_inf

        X = np.linspace(xmin, xmax, 1000000)
        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative = np.zeros_like(X_negative)

        cdf = np.zeros_like(X)
        pdf = np.zeros_like(X)
        # combine the distributions using the sum of the cumulative distribution functions: CDF_total = (CDF_1 x p_1) + (CDF_2 x p2) x (CDF_3 x p3) + .... + (CDF_n x pn)
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                cdf += distributions[i].CDF(X, show_plot=False) * proportions[i]
                pdf += distributions[i].PDF(X, show_plot=False) * proportions[i]
            else:
                cdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].CDF(X_positive, show_plot=False) * proportions[i],
                    ],
                )
                pdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].PDF(X_positive, show_plot=False) * proportions[i],
                    ],
                )
        self.__pdf_init = pdf
        self.__cdf_init = cdf
        self.__xvals_init = X
        self.mean = integrate.simpson(pdf * X, x=X)
        self.standard_deviation = (integrate.simpson(pdf * (X - self.mean) ** 2, x=X)) ** 0.5
        self.variance = self.standard_deviation**2
        self.skewness = integrate.simpson(pdf * ((X - self.mean) / self.standard_deviation) ** 3, x=X)
        self.kurtosis = integrate.simpson(pdf * ((X - self.mean) / self.standard_deviation) ** 4, x=X)
        self.mode = X[np.argmax(pdf)]
        self.median = X[np.argmin(abs((1 - cdf) - 0.5))]
        self.excess_kurtosis = self.kurtosis - 3
        self.b5 = X[np.argmin(abs(cdf - 0.05))]
        self.b95 = X[np.argmin(abs(cdf - 0.95))]

    def __combiner(self, xvals=None, xmin=None, xmax=None):
        """This is a hidden function used to combine the distributions numerically.
        It is necessary to do this outside of the __init__ method as it needs to be called by each function (PDF, CDF...) so that xvals is used consistently.
        This approach keeps the API the same as the other probability distributions.
        Users should never need to access this function directly.
        """
        distributions = self.distributions
        proportions = self.proportions

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
            msg = "unexpected type in xvals. Must be  list, or array"
            raise ValueError(msg)
        if min(X) < 0 and self.__contains_normal_or_gumbel is False:
            msg = "xvals was found to contain values below 0. This is only allowed if some of the mixture components are Normal or Gumbel distributions."
            raise ValueError(
                msg,
            )

        X_positive = X[X >= 0]
        X_negative = X[X < 0]
        Y_negative = np.zeros_like(X_negative).astype(np.float64)

        cdf = np.zeros_like(X).astype(np.float64)
        pdf = np.zeros_like(X).astype(np.float64)
        # combine the distributions using the sum of the cumulative distribution functions: CDF_total = (CDF_1 x p_1) + (CDF_2 x p2) x (CDF_3 x p3) + .... + (CDF_n x pn)
        for i in range(len(distributions)):
            if type(distributions[i]) in [Normal_Distribution, Gumbel_Distribution]:
                cdf += distributions[i].CDF(X, show_plot=False) * proportions[i]
                pdf += distributions[i].PDF(X, show_plot=False) * proportions[i]
            else:
                cdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].CDF(X_positive, show_plot=False) * proportions[i],
                    ],
                )
                pdf += np.hstack(
                    [
                        Y_negative,
                        distributions[i].PDF(X_positive, show_plot=False) * proportions[i],
                    ],
                )

        # these are all hidden to the user but can be accessed by the other functions in this module
        hf = pdf / (1 - cdf)
        self.__xvals = X
        self.__pdf = pdf
        self.__cdf = cdf
        self.__sf = 1 - cdf
        self.__hf = hf
        self.__chf = -np.log(1 - cdf)
        self._pdf0 = pdf[0]
        self._hf0 = hf[0]

    def plot(self, xvals=None, xmin=None, xmax=None):
        """Plots all functions (PDF, CDF, SF, HF, CHF) and descriptive statistics
        in a single figure.

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
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        plt.figure(figsize=(9, 7))
        text_title = "Mixture Model"
        plt.suptitle(text_title, fontsize=15)

        plt.subplot(231)
        PDF_CEILING = 1e100
        self.__pdf[self.__pdf > PDF_CEILING] = PDF_CEILING
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
        MAX_HF_VALUE = 1e100
        self.__hf[self.__hf > MAX_HF_VALUE] = MAX_HF_VALUE
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
        """Plots the PDF (probability density function).

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
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        MIN_XVALS_LENGTH = 2
        if len(self.__xvals) < MIN_XVALS_LENGTH:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.PDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.PDF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Mixture model"

            limits = get_axes_limits()
            MAX_PDF_VALUE = 1e100
            self.__pdf[self.__pdf > MAX_PDF_VALUE] = MAX_PDF_VALUE
            plt.plot(self.__xvals, self.__pdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Probability density")
            text_title = str("Mixture Model\n" + " Probability Density Function")
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
        """Plots the CDF (cumulative distribution function).

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
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        MIN_XVALS_LENGTH = 2
        if len(self.__xvals) < MIN_XVALS_LENGTH:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CDF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CDF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Mixture model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__cdf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction failing")
            text_title = str("Mixture Model\n" + " Cumulative Distribution Function")
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
        """Plots the SF (survival function).

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
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        MIN_XVALS_LENGTH = 2
        if len(self.__xvals) < MIN_XVALS_LENGTH:
            show_plot = False

        if show_plot is True:
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.SF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.SF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Mixture model"
            limits = get_axes_limits()
            plt.plot(self.__xvals, self.__sf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Fraction surviving")
            text_title = str("Mixture Model\n" + " Survival Function")
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
        """Plots the HF (hazard function).

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
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)

        MIN_XVALS_LENGTH = 2
        if len(self.__xvals) < MIN_XVALS_LENGTH:
            show_plot = False

        if show_plot is True:
            limits = get_axes_limits()
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.HF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.HF(xvals=self.__xvals, label=dist.param_title_long)
            textlabel = kwargs.pop("label") if "label" in kwargs else "Mixture model"
            MAX_HF_VALUE = 1e100
            self.__hf[self.__hf > MAX_HF_VALUE] = MAX_HF_VALUE
            plt.plot(self.__xvals, self.__hf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Hazard")
            text_title = str("Mixture Model\n" + " Hazard Function")
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
        """Plots the CHF (cumulative hazard function).

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
        Mixture_Model.__combiner(self, xvals=xvals, xmin=xmin, xmax=xmax)
        MAX_XVALS_LENGTH = 2
        if len(self.__xvals) < MAX_XVALS_LENGTH:
            show_plot = False

        if show_plot is True:
            limits = get_axes_limits()
            if plot_components is True:  # this will plot the distributions that make up the components of the model
                X_positive = self.__xvals[self.__xvals >= 0]
                for dist in self.distributions:
                    if type(dist) not in [Normal_Distribution, Gumbel_Distribution]:
                        dist.CHF(xvals=X_positive, label=dist.param_title_long)
                    else:
                        dist.CHF(xvals=self.__xvals, label=dist.param_title_long)
                        print("here")
            textlabel = kwargs.pop("label") if "label" in kwargs else "Mixture model"
            plt.plot(self.__xvals, self.__chf, label=textlabel, **kwargs)
            plt.xlabel("x values")
            plt.ylabel("Cumulative Hazard")
            text_title = str("Mixture Model\n" + " Cumulative Hazard Function")
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
        """Quantile calculator.

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
                msg = "Quantile must be between 0 and 1"
                raise ValueError(msg)
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                msg = "Quantile must be between 0 and 1"
                raise ValueError(msg)
        else:
            msg = "Quantile must be of type float, list, array"
            raise ValueError(msg)
        ppf = self.__xvals_init[np.argmin(abs(self.__cdf_init - q))]
        return unpack_single_arrays(ppf)

    def inverse_SF(self, q):
        """Inverse survival function calculator.

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
                msg = "Quantile must be between 0 and 1"
                raise ValueError(msg)
        elif type(q) in [list, np.ndarray]:
            if min(q) < 0 or max(q) > 1:
                msg = "Quantile must be between 0 and 1"
                raise ValueError(msg)
        else:
            msg = "Quantile must be of type float, list, array"
            raise ValueError(msg)
        isf = self.__xvals_init[np.argmin(abs((1 - self.__cdf_init) - q))]
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
        colorprint("Descriptive statistics for Mixture Model", bold=True, underline=True)
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
        """Mean Residual Life calculator.

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
            also accepts single values.
            """
            if isinstance(X, np.ndarray):
                cdf = np.zeros_like(X)
                X_positive = X[X >= 0]
                X_negative = X[X < 0]
                Y_negative = np.zeros_like(X_negative)
                for i in range(len(self.distributions)):
                    if type(self.distributions[i]) in [
                        Normal_Distribution,
                        Gumbel_Distribution,
                    ]:
                        cdf += self.distributions[i].CDF(X, show_plot=False) * self.proportions[i]
                    else:
                        cdf += np.hstack(
                            [
                                Y_negative,
                                self.distributions[i].CDF(X_positive, show_plot=False) * self.proportions[i],
                            ],
                        )
            else:
                cdf = 0
                for i in range(len(self.distributions)):
                    if (
                        type(self.distributions[i])
                        in [
                            Normal_Distribution,
                            Gumbel_Distribution,
                        ]
                        or X > 0
                    ):
                        cdf += self.distributions[i].CDF(X, show_plot=False) * self.proportions[i]
            return 1 - cdf

        t_full = np.linspace(t, self.__xmax_inf, 1000000)
        sf_full = __subcombiner(t_full)
        sf_single = __subcombiner(t)
        MRL: float = integrate.simpson(sf_full, x=t_full) / sf_single
        return MRL

    def random_samples(self, number_of_samples, seed=None):
        """Draws random samples from the probability distribution.

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
            msg = "number_of_samples must be an integer greater than 0"
            raise ValueError(msg)
        generator = np.random.default_rng(seed=seed)
        return generator.choice(
            self.__xvals_init,
            size=number_of_samples,
            p=self.__pdf_init / np.sum(self.__pdf_init),
        )

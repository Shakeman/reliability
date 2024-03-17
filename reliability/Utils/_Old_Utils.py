"""Utils (utilities)

This is a collection of utilities that are used throughout the python
reliability library. Functions have been placed here as to declutter the
dropdown lists of your IDE and to provide a common resource across multiple
modules. It is not expected that users will be using any utils directly.

Included functions are:

- ALT_MLE_optimization - performs optimization for the ALT_Fitters
- ALT_fitters_input_checking - performs input checking for the ALT_Fitters
- ALT_least_squares - least squares estimation for ALT_Fitters
- ALT_prob_plot - probability plotting for ALT_Fitters
- LS_optimization - least squares optimization for Fitters
- MLE_optimization - maximum likelihood estimation optimization for Fitters
- anderson_darling - calculated the anderson darling (AD) goodness of fit statistic
- axes_transforms - Custom scale functions used in Probability_plotting
- clean_CI_arrays - cleans the CI arrays of nan and illegal values
- colorprint - prints to the console in color, bold, italic, and underline
- distribution_confidence_intervals - calculates and plots the confidence intervals for the distributions
- fill_no_autoscale - creates a shaded region without adding it to the global list of objects to consider when autoscale is calculated
- fitters_input_checking - error checking and default values for all the fitters
- generate_X_array - generates the X values for all distributions
- get_axes_limits - gets the current axes limits
- least_squares - provides parameter estimates for distributions using the method of least squares. Used extensively by Fitters.
- life_stress_plot - generates the life stress plot for ALT_Fitters
- line_no_autoscale - creates a line without adding it to the global list of objects to consider when autoscale is calculated
- linear_regression - given x and y data it will return slope and intercept of line of best fit. Includes options to specify slope or intercept.
- make_fitted_dist_params_for_ALT_probplots - creates a class structure for the ALT probability plots to give to Probability_plotting
- no_reverse - corrects for reversals in confidence intervals
- probability_plot_xylims - sets the x and y limits on probability plots
- probability_plot_xyticks - sets the x and y ticks on probability plots
- removeNaNs - removes nan
- restore_axes_limits - restores the axes limits based on values from get_axes_limits()
- round_and_string - applies different rounding rules and converts to string
- show_figure_from_object - Re-shows a figure from an axes or figure handle even after the figure has been closed.
- transform_spaced - Creates linearly spaced array (in transform space) based on a specified transform. This is like np.logspace but it can make an array that is weibull spaced, normal spaced, etc.
- validate_CI_params - checks that the confidence intervals have all the right parameters to be generated
- write_df_to_xlsx - converts a dataframe to an xlsx file
- xy_transform - provides conversions between spatial (-inf,inf) and axes coordinates (0,1).
- zeroise_below_gamma - sets all y values to zero when x < gamma. Used when the HF and CHF equations are specified
"""

import warnings

import autograd.numpy as anp
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from autograd import jacobian as jac  # type: ignore
from autograd_gamma import gammaincc as agammaincc
from autograd_gamma import gammainccinv as agammainccinv
from matplotlib import colors
from matplotlib.axes import _axes
from scipy.optimize import OptimizeWarning  # type: ignore

from reliability.Utils._ancillary_utils import colorprint, round_and_string
from reliability.Utils._array_utils import (
    clean_CI_arrays,
    no_reverse,
    transform_spaced,
)
from reliability.Utils._input_checking_utils import (
    validate_CI_params,
)
from reliability.Utils._plot_utils import fill_no_autoscale, line_no_autoscale, probability_plot_xyticks

warnings.filterwarnings(
    action="ignore",
    category=OptimizeWarning,
)  # ignores the optimize warning that curve_fit sometimes outputs when there are 3 data points to fit a 3P curve
warnings.filterwarnings(
    action="ignore",
    category=RuntimeWarning,
)  # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required

class axes_transforms:
    """Custom scale functions used in Probability_plotting
    Each of these functions is either a forward or inverse transform.

    There are no parameters for this class, only a collection of subfunctions
    which can be called individually to perform the transforms.
    """

    @staticmethod
    def weibull_forward(F: np.float64):
        return np.log(-np.log(1 - F))

    @staticmethod
    def weibull_inverse(R: np.float64):
        return 1 - np.exp(-np.exp(R))

    @staticmethod
    def loglogistic_forward(F: np.float64):
        return np.log(1 / (1 - F) - 1)

    @staticmethod
    def loglogistic_inverse(R: np.float64):
        return 1 - 1 / (np.exp(R) + 1)

    @staticmethod
    def exponential_forward(F: np.float64):
        return ss.expon.ppf(F)

    @staticmethod
    def exponential_inverse(R: np.float64):
        return ss.expon.cdf(R)

    @staticmethod
    def normal_forward(F: np.float64):
        return ss.norm.ppf(F)

    @staticmethod
    def normal_inverse(R: np.float64):
        return ss.norm.cdf(R)

    @staticmethod
    def gumbel_forward(F: np.float64):
        return ss.gumbel_l.ppf(F)

    @staticmethod
    def gumbel_inverse(R: np.float64):
        return ss.gumbel_l.cdf(R)

    @staticmethod
    def gamma_forward(F: np.float64, beta: np.float64):
        return ss.gamma.ppf(F, a=beta)

    @staticmethod
    def gamma_inverse(R: np.float64, beta: np.float64):
        return ss.gamma.cdf(R, a=beta)

    @staticmethod
    def beta_forward(F: np.float64, alpha: np.float64, beta: np.float64):
        return ss.beta.ppf(F, a=alpha, b=beta)

    @staticmethod
    def beta_inverse(R: np.float64, alpha: np.float64, beta: np.float64):
        return ss.beta.cdf(R, a=alpha, b=beta)





def probability_plot_xylims(x, y, dist, spacing=0.1, gamma_beta=None, beta_alpha=None, beta_beta=None):
    """This function finds what the x and y limits of probability plots should be
    and sets these limits. This is similar to autoscaling, but the rules here
    are different to the matplotlib defaults.
    It is used extensively by the functions within the probability_plotting
    module to achieve the plotting style used within this library.

    Parameters
    ----------
    x : list, array
        The x-values from the plot
    y : list, array
        The y-values from the plot
    dist : str
        Must be either "weibull", "lognormal", "loglogistic", "normal", "gamma",
        "exponential", "beta", or "gumbel".
    spacing : float
        The spacing between the points and the edge of the plot. Default is 0.1
        for 10% spacing.
    gamma_beta : float, int, optional
        The beta parameter from the gamma distribution. Only required if dist =
        "gamma".
    beta_alpha : float, int, optional
        The alpha parameter from the beta distribution. Only required if dist =
        "beta".
    beta_beta : float, int, optional
        The beta parameter from the beta distribution. Only required if dist =
        "beta".

    Returns
    -------
    None
        There are no outputs from this function. It will set the xlim() and
        ylim() of the probability plot automatically.

    """
    # remove inf
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    # x limits
    if dist in ["weibull", "lognormal", "loglogistic"]:
        min_x_log = np.log10(min_x)
        max_x_log = np.log10(max_x)
        dx_log = max_x_log - min_x_log
        xlim_lower = 10 ** (min_x_log - dx_log * spacing)
        xlim_upper = 10 ** (max_x_log + dx_log * spacing)
        if xlim_lower == xlim_upper:
            xlim_lower = 10 ** (np.log10(xlim_lower) - 10 * spacing)
            xlim_upper = 10 ** (np.log10(xlim_upper) + 10 * spacing)
    elif dist in ["normal", "gamma", "exponential", "beta", "gumbel"]:
        dx = max_x - min_x
        xlim_lower = min_x - dx * spacing
        xlim_upper = max_x + dx * spacing
        if xlim_lower == xlim_upper:
            xlim_lower = 0
            xlim_upper = xlim_upper * 2
    else:
        raise ValueError("dist is unrecognised")
    if xlim_lower < 0 and dist not in ["normal", "gumbel"]:
        xlim_lower = 0
    # set xlims
    plt.xlim(xlim_lower, xlim_upper)

    # y limits
    if dist == "weibull":
        min_y_tfm = axes_transforms.weibull_forward(min_y)
        max_y_tfm = axes_transforms.weibull_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.weibull_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.weibull_inverse(max_y_tfm + dy_tfm * spacing)
    if dist == "exponential":
        min_y_tfm = axes_transforms.exponential_forward(min_y)
        max_y_tfm = axes_transforms.exponential_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.exponential_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.exponential_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "gamma":
        min_y_tfm = axes_transforms.gamma_forward(min_y, gamma_beta)
        max_y_tfm = axes_transforms.gamma_forward(max_y, gamma_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.gamma_inverse(min_y_tfm - dy_tfm * spacing, gamma_beta)
        ylim_upper = axes_transforms.gamma_inverse(max_y_tfm + dy_tfm * spacing, gamma_beta)
    elif dist in ["normal", "lognormal"]:
        min_y_tfm = axes_transforms.normal_forward(min_y)
        max_y_tfm = axes_transforms.normal_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.normal_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.normal_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "gumbel":
        min_y_tfm = axes_transforms.gumbel_forward(min_y)
        max_y_tfm = axes_transforms.gumbel_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.gumbel_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.gumbel_inverse(max_y_tfm + dy_tfm * spacing)
    elif dist == "beta":
        min_y_tfm = axes_transforms.beta_forward(min_y, beta_alpha, beta_beta)
        max_y_tfm = axes_transforms.beta_forward(max_y, beta_alpha, beta_beta)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.beta_inverse(min_y_tfm - dy_tfm * spacing, beta_alpha, beta_beta)
        ylim_upper = axes_transforms.beta_inverse(max_y_tfm + dy_tfm * spacing, beta_alpha, beta_beta)
    elif dist == "loglogistic":
        min_y_tfm = axes_transforms.loglogistic_forward(min_y)
        max_y_tfm = axes_transforms.loglogistic_forward(max_y)
        dy_tfm = max_y_tfm - min_y_tfm
        ylim_lower = axes_transforms.loglogistic_inverse(min_y_tfm - dy_tfm * spacing)
        ylim_upper = axes_transforms.loglogistic_inverse(max_y_tfm + dy_tfm * spacing)
    if ylim_upper == ylim_lower:
        dx = min(1 - ylim_upper, ylim_upper - 1)
        ylim_upper = ylim_upper - spacing * dx
        ylim_lower = ylim_lower + spacing * dx

    # correction for the case where ylims are is 0 or 1
    if ylim_lower == 0:
        ylim_lower = min_y if min_y > 0 else 1e-05
    if ylim_upper == 1:
        ylim_upper = max_y if max_y < 1 else 0.99999
    # set ylims
    plt.ylim(ylim_lower, ylim_upper)



def anderson_darling(fitted_cdf, empirical_cdf):
    """Calculates the Anderson-Darling goodness of fit statistic.
    These formulas are based on the method used in MINITAB which gives an
    adjusted form of the original AD statistic described on Wikipedia.

    Parameters
    ----------
    fitted_cdf : list, array
        The fitted CDF values at the data points
    empirical_cdf  : list, array
        The empirical (rank adjustment) CDF values at the data points

    Returns
    -------
    AD : float
        The anderson darling (adjusted) test statistic.

    """
    if type(fitted_cdf) != np.ndarray:
        fitted_cdf = [fitted_cdf]  # required when there is only 1 failure
    Z = np.sort(np.asarray(fitted_cdf))
    Zi = np.hstack([Z, 1 - 1e-12])
    Zi_1 = (np.hstack([0, Zi]))[0:-1]  # Z_i-1
    FnZi = np.sort(np.asarray(empirical_cdf))
    FnZi_1 = np.hstack([0, FnZi])  # Fn(Z_i-1)
    lnZi = np.log(Zi)
    lnZi_1 = np.hstack([0, lnZi])[0:-1]

    A = -Zi - np.log(1 - Zi) + Zi_1 + np.log(1 - Zi_1)
    B = 2 * np.log(1 - Zi) * FnZi_1 - 2 * np.log(1 - Zi_1) * FnZi_1
    C = lnZi * FnZi_1**2 - np.log(1 - Zi) * FnZi_1**2 - lnZi_1 * FnZi_1**2 + np.log(1 - Zi_1) * FnZi_1**2
    n = len(fitted_cdf)
    AD = n * ((A + B + C).sum())
    return AD


class distribution_confidence_intervals:
    """This class contains several subfunctions that provide all the confidence
    intervals for CDF, SF, CHF for each distribution for which it is
    implemented.

    The class has no parameters or returns as it is used primarily to create the
    confidence interval object which is used by the subfunctions.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    @staticmethod
    def exponential_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """Generates the confidence intervals for CDF, SF, and CHF of the
        Exponential distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF"
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None.
        x : array, list, optional
            The x-values to be calculated. Default is None.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if q is not None.
        t_upper :array
            The upper bounds on time. Only returned if q is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.Lambda_SE and self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For the Exponential distribution, the bounds on time and reliability are
        the same.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200

        # this section plots the confidence interval
        if self.Lambda_SE is not None and self.Z is not None and (plot_CI is True or q is not None or x is not None):
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError("q must be a list or array of quantiles. Default is None")
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError("x must be a list or array of x-values. Default is None")
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)

                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)  # removes decimals if the only decimal is 0
                # Adds the CI and CI_type to the title
                text_title = str(text_title + "\n" + str(CI_100) + "% confidence bounds")
                # add a line to the plot title to include the confidence bounds information
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            Lambda_upper = self.Lambda * (np.exp(Z * (self.Lambda_SE / self.Lambda)))
            Lambda_lower = self.Lambda * (np.exp(-Z * (self.Lambda_SE / self.Lambda)))

            if x is not None:
                t = x - self.gamma
            else:
                t0 = self.quantile(0.00001) - self.gamma
                if t0 <= 0:
                    t0 = 0.0001
                t = np.geomspace(
                    t0,
                    self.quantile(0.99999) - self.gamma,
                    points,
                )

            # calculate the CIs using the formula for SF
            Y_lower = np.exp(-Lambda_lower * t)
            Y_upper = np.exp(-Lambda_upper * t)

            # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
            t, t, Y_lower, Y_upper = clean_CI_arrays(
                xlower=t,
                xupper=t,
                ylower=Y_lower,
                yupper=Y_upper,
                plot_type=func,
                q=q,
                x=x,
            )
            # artificially correct for any reversals
            if (x is None or q is None) and len(Y_lower) > 2 and len(Y_upper) > 2:
                Y_lower = no_reverse(Y_lower, CI_type=None, plot_type=func)
                Y_upper = no_reverse(Y_upper, CI_type=None, plot_type=func)

            if func == "CDF":
                yy_upper = 1 - Y_upper
                yy_lower = 1 - Y_lower
            elif func == "SF":
                yy_upper = Y_upper
                yy_lower = Y_lower
            elif func == "CHF":
                yy_upper = -np.log(Y_upper)  # same as -np.log(SF)
                yy_lower = -np.log(Y_lower)

            if plot_CI is True:
                fill_no_autoscale(
                    xlower=t + self.gamma,
                    xupper=t + self.gamma,
                    ylower=yy_lower,
                    yupper=yy_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                )
                line_no_autoscale(
                    x=t + self.gamma,
                    y=yy_lower,
                    color=color,
                    linewidth=0,
                )  # these are invisible but need to be added to the plot for crosshairs() to find them
                line_no_autoscale(
                    x=t + self.gamma,
                    y=yy_upper,
                    color=color,
                    linewidth=0,
                )  # still need to specify color otherwise the invisible CI lines will consume default colors
                # plt.scatter(t + self.gamma, yy_lower,color='blue',marker='.')
                # plt.scatter(t + self.gamma, yy_upper, color='red', marker='.')

            if q is not None:
                t_lower = -np.log(q) / Lambda_upper + self.gamma
                t_upper = -np.log(q) / Lambda_lower + self.gamma
                return t_lower, t_upper
            elif x is not None:
                return Y_lower, Y_upper

    @staticmethod
    def weibull_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """Generates the confidence intervals for CDF, SF, and CHF of the
        Weibull distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF"
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if
            CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if
            CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z) is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError("q must be a list or array of quantiles. Default is None")
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError("x must be a list or array of x-values. Default is None")
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type)
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, alpha, beta):  # u = ln(-ln(R))
                return beta * (anp.log(t) - anp.log(alpha))  # weibull SF linearized

            def v(R, alpha, beta):  # v = ln(t)
                return (1 / beta) * anp.log(-anp.log(R)) + anp.log(alpha)  # weibull SF rearranged for t

            du_da = jac(u, 1)  # derivative wrt alpha (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt beta (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt alpha (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt beta (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + du_db(v, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2 * du_da(v, self.alpha, self.beta) * du_db(v, self.alpha, self.beta) * self.Cov_alpha_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + dv_db(u, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2 * dv_da(u, self.alpha, self.beta) * dv_db(u, self.alpha, self.beta) * self.Cov_alpha_beta
                )

            # Confidence bounds on time (in terms of reliability)
            if CI_type == "time":
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                elif q is not None:
                    Y = q
                else:
                    Y = transform_spaced("weibull", y_lower=1e-8, y_upper=1 - 1e-8, num=points)

                # v is ln(t)
                v_lower = v(Y, self.alpha, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.alpha, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma  # transform back from ln(t)
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )

                    line_no_autoscale(
                        x=t_lower,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            # Confidence bounds on Reliability (in terms of time)
            elif CI_type == "reliability":
                if x is not None:
                    t = x - self.gamma
                else:
                    t0 = self.quantile(0.00001) - self.gamma
                    if t0 <= 0:
                        t0 = 0.0001
                    t = np.geomspace(
                        t0,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability ln(-ln(R))
                u_lower = (
                    u(t, self.alpha, self.beta) + Z * var_u(self, t) ** 0.5
                )  # note that gamma is incorporated into u but not in var_u. This is the same as just shifting a Weibull_2P across
                u_upper = u(t, self.alpha, self.beta) - Z * var_u(self, t) ** 0.5

                Y_lower = np.exp(-np.exp(u_lower))  # transform back from ln(-ln(R))
                Y_upper = np.exp(-np.exp(u_upper))

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_lower,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_upper,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t + self.gamma, yy_upper, color='red')
                    # plt.scatter(t + self.gamma, yy_lower, color='blue')

                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def gamma_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """Generates the confidence intervals for CDF, SF, and CHF of the
        Gamma distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.beta_SE, self.Cov_mu_beta, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.

        if (
            validate_CI_params(self.mu_SE, self.beta_SE, self.Cov_mu_beta, self.Z) is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError("q must be a list or array of quantiles. Default is None")
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError("x must be a list or array of x-values. Default is None")
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type)
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, beta):  # u = R
                return agammaincc(beta, t / anp.exp(mu))

            def v(R, mu, beta):  # v = ln(t)
                return anp.log(agammainccinv(beta, R)) + mu

            du_dm = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt beta (bounds on reliability)
            dv_dm = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt beta (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_dm(v, self.mu, self.beta) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.beta) ** 2 * self.beta_SE**2
                    + 2 * du_dm(v, self.mu, self.beta) * du_db(v, self.mu, self.beta) * self.Cov_mu_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_dm(u, self.mu, self.beta) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.beta) ** 2 * self.beta_SE**2
                    + 2 * dv_dm(u, self.mu, self.beta) * dv_db(u, self.mu, self.beta) * self.Cov_mu_beta
                )

            # Confidence bounds on time (in terms of reliability)
            if CI_type == "time":
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                elif q is not None:
                    Y = q
                elif self.beta > 3:
                    Y = transform_spaced(
                        "gamma",
                        y_lower=1e-8,
                        y_upper=1 - 1e-8,
                        beta=self.beta,
                        num=points,
                    )
                else:
                    Y = np.linspace(1e-8, 1 - 1e-8, points)

                # v is ln(t)
                v_lower = v(Y, self.mu, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.mu, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            # Confidence bounds on Reliability (in terms of time)
            elif CI_type == "reliability":
                if x is not None:
                    t = x - self.gamma
                else:
                    t0 = 0.0001 if self.gamma == 0 else self.quantile(1e-07)
                    t = np.linspace(
                        t0 - self.gamma,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability
                # note that gamma is incorporated into u but not in var_u. This is the same as just shifting a Gamma_2P across
                R = u(t, self.mu, self.beta)
                varR = var_u(self, t)
                R_lower = R / (R + (1 - R) * np.exp((Z * varR**0.5) / (R * (1 - R))))
                R_upper = R / (R + (1 - R) * np.exp((-Z * varR**0.5) / (R * (1 - R))))

                # transform back from u = R
                Y_lower = R_lower
                Y_upper = R_upper

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )

                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_lower,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_upper,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t + self.gamma, yy_upper, color='red')
                    # plt.scatter(t + self.gamma, yy_lower, color='blue')
                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def normal_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """Generates the confidence intervals for CDF, SF, and CHF of the
        Normal distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z) is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError("q must be a list or array of quantiles. Default is None")
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError("x must be a list or array of x-values. Default is None")
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type)
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, sigma):  # u = phiinv(R)
                return (mu - t) / sigma  # normal SF linearlized

            def v(R, mu, sigma):  # v = t
                return mu - sigma * ss.norm.ppf(R)

            # for consistency with other distributions, the derivatives are da for d_sigma and db for d_mu. Just think of a is first parameter and b is second parameter.
            du_da = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt sigma (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt sigma (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2 * du_da(v, self.mu, self.sigma) * du_db(v, self.mu, self.sigma) * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2 * dv_da(u, self.mu, self.sigma) * dv_db(u, self.mu, self.sigma) * self.Cov_mu_sigma
                )

            # Confidence bounds on time (in terms of reliability)
            if CI_type == "time":
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    Y = q if q is not None else transform_spaced("normal", y_lower=1e-08, y_upper=1 - 1e-08, num=points)

                # v is t
                t_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                t_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            # Confidence bounds on Reliability (in terms of time)
            elif CI_type == "reliability":
                t = x if x is not None else np.linspace(self.quantile(1e-05), self.quantile(0.99999), points)

                # u is reliability u = phiinv(R)
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = ss.norm.cdf(u_lower)  # transform back from u = phiinv(R)
                Y_upper = ss.norm.cdf(u_upper)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t,
                        xupper=t,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t,
                        y=yy_lower,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t,
                        y=yy_upper,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t, yy_upper, color='red')
                    # plt.scatter(t, yy_lower, color='blue')

                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def lognormal_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """Generates the confidence intervals for CDF, SF, and CHF of the
        Lognormal distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z) is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError("q must be a list or array of quantiles. Default is None")
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError("x must be a list or array of x-values. Default is None")

            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type)
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, sigma):  # u = phiinv(R)
                return (mu - np.log(t)) / sigma  # lognormal SF linearlized

            def v(R, mu, sigma):  # v = ln(t)
                return mu - sigma * ss.norm.ppf(R)

            # for consistency with other distributions, the derivatives are da for d_sigma and db for d_mu. Just think of a is first parameter and b is second parameter.
            du_da = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt sigma (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt sigma (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2 * du_da(v, self.mu, self.sigma) * du_db(v, self.mu, self.sigma) * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2 * dv_da(u, self.mu, self.sigma) * dv_db(u, self.mu, self.sigma) * self.Cov_mu_sigma
                )

            if CI_type == "time":
                # Confidence bounds on time (in terms of reliability)
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    Y = q if q is not None else transform_spaced("normal", y_lower=1e-08, y_upper=1 - 1e-08, num=points)

                # v is ln(t)
                v_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            elif CI_type == "reliability":
                # Confidence bounds on Reliability (in terms of time)
                if x is not None:
                    t = x - self.gamma
                else:
                    t0 = self.quantile(0.00001) - self.gamma
                    if t0 <= 0:
                        t0 = 0.0001
                    t = np.geomspace(
                        t0,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability u = phiinv(R)
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = ss.norm.cdf(u_lower)  # transform back from u = phiinv(R)
                Y_upper = ss.norm.cdf(u_upper)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_lower,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_upper,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t+ self.gamma, yy_upper, color='red')
                    # plt.scatter(t+ self.gamma, yy_lower, color='blue')

                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def loglogistic_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """Generates the confidence intervals for CDF, SF, and CHF of the
        Loglogistic distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.alpha_SE, self.beta_SE, self.Cov_alpha_beta, self.Z) is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError("q must be a list or array of quantiles. Default is None")
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError("x must be a list or array of x-values. Default is None")
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type)
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, alpha, beta):  # u = ln(1/R - 1)
                return beta * (anp.log(t) - anp.log(alpha))  # loglogistic SF linearized

            def v(R, alpha, beta):  # v = ln(t)
                return (1 / beta) * anp.log(1 / R - 1) + anp.log(alpha)  # loglogistic SF rearranged for t

            du_da = jac(u, 1)  # derivative wrt alpha (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt beta (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt alpha (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt beta (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + du_db(v, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2 * du_da(v, self.alpha, self.beta) * du_db(v, self.alpha, self.beta) * self.Cov_alpha_beta
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.alpha, self.beta) ** 2 * self.alpha_SE**2
                    + dv_db(u, self.alpha, self.beta) ** 2 * self.beta_SE**2
                    + 2 * dv_da(u, self.alpha, self.beta) * dv_db(u, self.alpha, self.beta) * self.Cov_alpha_beta
                )

            if CI_type == "time":  # Confidence bounds on time (in terms of reliability)
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                elif q is not None:
                    Y = q
                else:
                    Y = transform_spaced("loglogistic", y_lower=1e-8, y_upper=1 - 1e-8, num=points)

                # v is ln(t)
                v_lower = v(Y, self.alpha, self.beta) - Z * (var_v(self, Y) ** 0.5)
                v_upper = v(Y, self.alpha, self.beta) + Z * (var_v(self, Y) ** 0.5)

                t_lower = np.exp(v_lower) + self.gamma  # transform back from ln(t)
                t_upper = np.exp(v_upper) + self.gamma

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            elif CI_type == "reliability":  # Confidence bounds on Reliability (in terms of time)
                if x is not None:
                    t = x - self.gamma
                else:
                    t0 = self.quantile(0.00001) - self.gamma
                    if t0 <= 0:
                        t0 = 0.0001
                    t = np.geomspace(
                        t0,
                        self.quantile(0.99999) - self.gamma,
                        points,
                    )

                # u is reliability ln(1/R - 1)
                u_lower = (
                    u(t, self.alpha, self.beta) + Z * var_u(self, t) ** 0.5
                )  # note that gamma is incorporated into u but not in var_u. This is the same as just shifting a Weibull_2P across
                u_upper = u(t, self.alpha, self.beta) - Z * var_u(self, t) ** 0.5

                Y_lower = 1 / (np.exp(u_lower) + 1)  # transform back from ln(1/R - 1)
                Y_upper = 1 / (np.exp(u_upper) + 1)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t + self.gamma,
                        xupper=t + self.gamma,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_lower,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t + self.gamma,
                        y=yy_upper,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t + self.gamma, yy_upper, color='red')
                    # plt.scatter(t + self.gamma, yy_lower, color='blue')
                if x is not None:
                    return Y_lower, Y_upper

    @staticmethod
    def gumbel_CI(
        self,
        func="CDF",
        plot_CI=None,
        CI_type=None,
        CI=None,
        text_title="",
        color=None,
        q=None,
        x=None,
    ):
        """Generates the confidence intervals for CDF, SF, and CHF of the
        Gumbel distribution.

        Parameters
        ----------
        self : object
            The distribution object
        func : str
            Must be either "CDF", "SF" or "CHF". Default is "CDF".
        plot_CI : bool, None
            The confidence intervals will only be plotted if plot_CI is True.
        CI_type : str
            Must be either "time" or "reliability"
        CI : float
            The confidence interval. Must be between 0 and 1
        text_title : str
            The existing CDF/SF/CHF text title to which the confidence interval
            string will be added.
        color : str
            The color to be used to fill the confidence intervals.
        q : array, list, optional
            The quantiles to be calculated. Default is None. Only used if CI_type='time'.
        x : array, list, optional
            The x-values to be calculated. Default is None. Only used if CI_type='reliability'.

        Returns
        -------
        t_lower : array
            The lower bounds on time. Only returned if CI_type is "time" and q
            is not None.
        t_upper :array
            The upper bounds on time. Only returned if CI_type is "time" and q
            is not None.
        R_lower : array
            The lower bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.
        R_upper :array
            The upper bounds on reliability. Only returned if CI_type is
            "reliability" and x is not None.

        Notes
        -----
        self must contain particular values for this function to work. These
        include self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z.

        As a Utils function, there is very limited error checking done, as this
        function is not intended for users to access directly.

        For an explaination of how the confidence inervals are calculated,
        please see the `documentation <https://reliability.readthedocs.io/en/latest/How%20are%20the%20confidence%20intervals%20calculated.html>`_.

        """
        points = 200  # the number of data points in each confidence interval (upper and lower) line

        # this determines if the user has specified for the CI bounds to be shown or hidden.
        if (
            validate_CI_params(self.mu_SE, self.sigma_SE, self.Cov_mu_sigma, self.Z) is True
            and (plot_CI is True or q is not None or x is not None)
            and CI_type is not None
        ):
            if CI_type in ["time", "t", "T", "TIME", "Time"]:
                CI_type = "time"
            elif CI_type in [
                "reliability",
                "r",
                "R",
                "RELIABILITY",
                "rel",
                "REL",
                "Reliability",
            ]:
                CI_type = "reliability"
            if func not in ["CDF", "SF", "CHF"]:
                raise ValueError("func must be either CDF, SF, or CHF")
            if type(q) not in [list, np.ndarray, type(None)]:
                raise ValueError("q must be a list or array of quantiles. Default is None")
            if type(x) not in [list, np.ndarray, type(None)]:
                raise ValueError("x must be a list or array of x-values. Default is None")
            if q is not None:
                q = np.asarray(q)
            if x is not None:
                x = np.asarray(x)

            Z = -ss.norm.ppf((1 - CI) / 2)  # converts CI to Z

            if plot_CI is True:
                # formats the confidence interval value ==> 0.95 becomes 95
                CI_100 = round(CI * 100, 4)
                # removes decimals if the only decimal is 0
                if CI_100 % 1 == 0:
                    CI_100 = int(CI_100)
                # Adds the CI and CI_type to the title
                text_title = str(text_title + "\n" + str(CI_100) + "% confidence bounds on " + CI_type)
                plt.title(text_title)
                plt.subplots_adjust(top=0.81)

            def u(t, mu, sigma):  # u = ln(-ln(R))
                return (t - mu) / sigma  # gumbel SF linearlized

            def v(R, mu, sigma):  # v = t
                return mu + sigma * anp.log(-anp.log(R))  # Gumbel SF rearranged for t

            # for consistency with other distributions, the derivatives are da for d_sigma and db for d_mu. Just think of a is first parameter and b is second parameter.
            du_da = jac(u, 1)  # derivative wrt mu (bounds on reliability)
            du_db = jac(u, 2)  # derivative wrt sigma (bounds on reliability)
            dv_da = jac(v, 1)  # derivative wrt mu (bounds on time)
            dv_db = jac(v, 2)  # derivative wrt sigma (bounds on time)

            def var_u(self, v):  # v is time
                return (
                    du_da(v, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + du_db(v, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2 * du_da(v, self.mu, self.sigma) * du_db(v, self.mu, self.sigma) * self.Cov_mu_sigma
                )

            def var_v(self, u):  # u is reliability
                return (
                    dv_da(u, self.mu, self.sigma) ** 2 * self.mu_SE**2
                    + dv_db(u, self.mu, self.sigma) ** 2 * self.sigma_SE**2
                    + 2 * dv_da(u, self.mu, self.sigma) * dv_db(u, self.mu, self.sigma) * self.Cov_mu_sigma
                )

            if CI_type == "time":  # Confidence bounds on time (in terms of reliability)
                # Y is reliability (R)
                if func == "CHF":
                    chf_array = np.geomspace(1e-8, self._chf[-1] * 1.5, points)
                    Y = np.exp(-chf_array)
                else:  # CDF and SF
                    Y = q if q is not None else transform_spaced("gumbel", y_lower=1e-08, y_upper=1 - 1e-08, num=points)

                # v is t
                t_lower = v(Y, self.mu, self.sigma) - Z * (var_v(self, Y) ** 0.5)
                t_upper = v(Y, self.mu, self.sigma) + Z * (var_v(self, Y) ** 0.5)

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t_lower, t_upper, Y, Y = clean_CI_arrays(
                    xlower=t_lower,
                    xupper=t_upper,
                    ylower=Y,
                    yupper=Y,
                    plot_type=func,
                    q=q,
                )
                # artificially correct for any reversals
                if q is None and len(t_lower) > 2 and len(t_upper) > 2:
                    t_lower = no_reverse(t_lower, CI_type=CI_type, plot_type=func)
                    t_upper = no_reverse(t_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy = 1 - Y
                elif func == "SF":
                    yy = Y
                elif func == "CHF":
                    yy = -np.log(Y)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t_lower,
                        xupper=t_upper,
                        ylower=yy,
                        yupper=yy,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t_lower,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t_upper,
                        y=yy,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t_lower, yy, linewidth=1, color='blue')
                    # plt.scatter(t_upper, yy, linewidth=1, color='red')

                if q is not None:
                    return t_lower, t_upper

            elif CI_type == "reliability":  # Confidence bounds on Reliability (in terms of time)
                t = x if x is not None else np.linspace(self.quantile(1e-05), self.quantile(0.99999), points)

                # u is reliability u = ln(-ln(R))
                u_lower = u(t, self.mu, self.sigma) + Z * var_u(self, t) ** 0.5
                u_upper = u(t, self.mu, self.sigma) - Z * var_u(self, t) ** 0.5

                Y_lower = np.exp(-np.exp(u_lower))  # transform back from ln(-ln(R))
                Y_upper = np.exp(-np.exp(u_upper))

                # clean the arrays of illegal values (<=0, nans, >=1 (if CDF or SF))
                t, t, Y_lower, Y_upper = clean_CI_arrays(
                    xlower=t,
                    xupper=t,
                    ylower=Y_lower,
                    yupper=Y_upper,
                    plot_type=func,
                    x=x,
                )
                # artificially correct for any reversals
                if x is None and len(Y_lower) > 2 and len(Y_upper) > 2:
                    Y_lower = no_reverse(Y_lower, CI_type=CI_type, plot_type=func)
                    Y_upper = no_reverse(Y_upper, CI_type=CI_type, plot_type=func)

                if func == "CDF":
                    yy_lower = 1 - Y_lower
                    yy_upper = 1 - Y_upper
                elif func == "SF":
                    yy_lower = Y_lower
                    yy_upper = Y_upper
                elif func == "CHF":
                    yy_lower = -np.log(Y_lower)
                    yy_upper = -np.log(Y_upper)

                if plot_CI is True:
                    fill_no_autoscale(
                        xlower=t,
                        xupper=t,
                        ylower=yy_lower,
                        yupper=yy_upper,
                        color=color,
                        alpha=0.3,
                        linewidth=0,
                    )
                    line_no_autoscale(
                        x=t,
                        y=yy_lower,
                        color=color,
                        linewidth=0,
                    )  # these are invisible but need to be added to the plot for crosshairs() to find them
                    line_no_autoscale(
                        x=t,
                        y=yy_upper,
                        color=color,
                        linewidth=0,
                    )  # still need to specify color otherwise the invisible CI lines will consume default colors
                    # plt.scatter(t, yy_upper, color='red')
                    # plt.scatter(t, yy_lower, color='blue')
                if x is not None:
                    return Y_lower, Y_upper


class make_fitted_dist_params_for_ALT_probplots:
    """This function creates a class structure for the ALT probability plots to
    give to Probability_plotting.

    Parameters
    ----------
    dist : str
        The distribution. Must be either "Weibull", "Lognormal", "Normal", or
        "Exponential".
    params : list, array
        The parameters of the model. Must be 2 elements for Weibull, Lognormal,
        and Normal, and must be 1 element for Exponential.

    Returns
    -------
    alpha : float
        Only returned for Weibull
    beta : float
        Only returned for Weibull
    gamma : int
        This will always be 0. Only returned for Weibull, Lognormal, and
        Exponential.
    alpha_SE : None
        Only returned for Weibull
    beta_SE : None
        Only returned for Weibull
    Cov_alpha_beta : None
        Only returned for Weibull
    mu : float
        Only returned for Normal and Lognormal
    sigma : float
        Only returned for Normal and Lognormal
    Cov_mu_sigma : None
        Only returned for Normal and Lognormal
    Lambda : float
        Only returned for Exponential
    Lambda_SE : None
        Only returned for Exponential

    Notes
    -----
    This function only exists to convert a list or array of parameters into a
    class with the correct parameters for the probability plots to use.

    """

    def __init__(self, dist, params):
        if dist == "Weibull":
            self.alpha = params[0]
            self.beta = params[1]
            self.gamma = 0
            self.alpha_SE = None
            self.beta_SE = None
            self.Cov_alpha_beta = None
        elif dist == "Lognormal":
            self.mu = np.log(params[0])
            self.sigma = params[1]
            self.gamma = 0
            self.mu_SE = None
            self.sigma_SE = None
            self.Cov_mu_sigma = None
        elif dist == "Normal":
            self.mu = params[0]
            self.sigma = params[1]
            self.mu_SE = None
            self.sigma_SE = None
            self.Cov_mu_sigma = None
        elif dist == "Exponential":
            self.Lambda = 1 / params[0]
            self.Lambda_SE = None
            self.gamma = 0
        else:
            raise ValueError("dist must be one of Weibull, Normal, Lognormal, Exponential")


def ALT_prob_plot(
    dist,
    model,
    stresses_for_groups,
    failure_groups,
    right_censored_groups,
    life_func,
    shape,
    scale_for_change_df,
    shape_for_change_df,
    use_level_stress=None,
    ax=True,
):
    """Generates an ALT probability plot using the inputs provided.

    Parameters
    ----------
    dist : str
        Must be either "Weibull", "Exponential", "Lognormal", or "Normal"
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power".
    stresses_for_groups : list
        The stresses for the failure groups
    failure_groups : list
        The failure groups. This is a list of lists.
    right_censored_groups
        The failure groups. This is a list of lists.
    life_func : function
        The life function for the ALT life model.
    shape : float, int
        The shape parameter of the model.
    scale_for_change_df : array, list
        The list of scale parameters for the lines.
    shape_for_change_df
        The list of shape parameters for the lines.
    use_level_stress : float, int, array, list, None
        The use level stress. This must be an array or list for dual stress
        models. Default is None.
    ax : axis, bool, optional
        The axis handle to use. Default is True which will create a new plot.
        If False then no plot will be generated.

    Returns
    -------
    current_axis : axis
        The axis handle of the plot. If ax is specified in the inputs then this
        will be the same handle.

    """
    if ax is True or type(ax) == _axes.Axes:
        if type(ax) == _axes.Axes:
            plt.sca(ax=ax)  # use the axes passed
        else:
            plt.figure()  # if no axes is passed, make a new figure

        from reliability.Probability_plotting import plotting_positions

        if dist == "Weibull":
            from reliability.Distributions import Weibull_Distribution as Distribution
            from reliability.Probability_plotting import (
                Weibull_probability_plot as probplot,
            )
        elif dist == "Lognormal":
            from reliability.Distributions import Lognormal_Distribution as Distribution
            from reliability.Probability_plotting import (
                Lognormal_probability_plot as probplot,
            )
        elif dist == "Normal":
            from reliability.Distributions import Normal_Distribution as Distribution
            from reliability.Probability_plotting import (
                Normal_probability_plot as probplot,
            )
        elif dist == "Exponential":
            from reliability.Distributions import (
                Exponential_Distribution as Distribution,
            )
            from reliability.Probability_plotting import (
                Exponential_probability_plot_Weibull_Scale as probplot,
            )
        else:
            raise ValueError("dist must be either Weibull, Lognormal, Normal, Exponential")

        if model in ["Dual_Exponential", "Power_Exponential", "Dual_Power"]:
            dual_stress = True
        elif model in ["Exponential", "Eyring", "Power"]:
            dual_stress = False
        else:
            raise ValueError(
                "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power",
            )

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # gets the default color cycle
        x_array = []
        y_array = []
        if dual_stress is True:
            for i, stress in enumerate(stresses_for_groups):
                f = failure_groups[i]
                rc = None if right_censored_groups is None else right_censored_groups[i]
                # get the plotting positions so they can be given to probability_plot_xylims for autoscaling
                x, y = plotting_positions(failures=f, right_censored=rc)
                x_array.extend(x)
                y_array.extend(y)
                # generate the probability plot and the line from the life-stress model
                fitted_dist_params = make_fitted_dist_params_for_ALT_probplots(
                    dist=dist,
                    params=[life_func(S1=stress[0], S2=stress[1]), shape],
                )
                probplot(
                    failures=f,
                    right_censored=rc,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_cycle[i],
                    label=str(round_and_string(stress[0]) + ", " + round_and_string(stress[1])),
                )
                # plot the original fitted line
                if dist == "Exponential":
                    if scale_for_change_df[i] != "":
                        Distribution(1 / scale_for_change_df[i]).CDF(linestyle="--", alpha=0.5, color=color_cycle[i])
                elif scale_for_change_df[i] != "":
                    Distribution(scale_for_change_df[i], shape_for_change_df[i]).CDF(
                        linestyle="--",
                        alpha=0.5,
                        color=color_cycle[i],
                    )

            if use_level_stress is not None:
                if dist in ["Weibull", "Normal"]:
                    distribution_at_use_stress = Distribution(
                        life_func(S1=use_level_stress[0], S2=use_level_stress[1]),
                        shape,
                    )
                elif dist == "Lognormal":
                    distribution_at_use_stress = Distribution(
                        np.log(life_func(S1=use_level_stress[0], S2=use_level_stress[1])),
                        shape,
                    )
                elif dist == "Exponential":
                    distribution_at_use_stress = Distribution(
                        1 / life_func(S1=use_level_stress[0], S2=use_level_stress[1]),
                    )
                distribution_at_use_stress.CDF(
                    color=color_cycle[i + 1],
                    label=str(
                        round_and_string(use_level_stress[0])
                        + ", "
                        + round_and_string(use_level_stress[1])
                        + " (use stress)",
                    ),
                )
                x_array.extend(
                    [
                        distribution_at_use_stress.quantile(min(y_array)),
                        distribution_at_use_stress.quantile(max(y_array)),
                    ],
                )  # this ensures the plot limits include the use stress distribution

            plt.legend(title="     Stress 1, Stress 2")

        else:
            for i, stress in enumerate(stresses_for_groups):
                f = failure_groups[i]
                rc = None if right_censored_groups is None else right_censored_groups[i]
                # get the plotting positions so they can be given to probability_plot_xylims for autoscaling
                x, y = plotting_positions(failures=f, right_censored=rc)
                x_array.extend(x)
                y_array.extend(y)
                # generate the probability plot and the line from the life-stress model
                fitted_dist_params = make_fitted_dist_params_for_ALT_probplots(
                    dist=dist,
                    params=[life_func(S1=stress), shape],
                )
                probplot(
                    failures=f,
                    right_censored=rc,
                    __fitted_dist_params=fitted_dist_params,
                    color=color_cycle[i],
                    label=round_and_string(stress),
                )
                # plot the original fitted line
                if dist == "Exponential":
                    if scale_for_change_df[i] != "":
                        Distribution(1 / scale_for_change_df[i]).CDF(linestyle="--", alpha=0.5, color=color_cycle[i])
                elif scale_for_change_df[i] != "":
                    Distribution(scale_for_change_df[i], shape_for_change_df[i]).CDF(
                        linestyle="--",
                        alpha=0.5,
                        color=color_cycle[i],
                    )

            if use_level_stress is not None:
                if dist in ["Weibull", "Normal"]:
                    distribution_at_use_stress = Distribution(life_func(S1=use_level_stress), shape)
                elif dist == "Lognormal":
                    distribution_at_use_stress = Distribution(np.log(life_func(S1=use_level_stress)), shape)
                elif dist == "Exponential":
                    distribution_at_use_stress = Distribution(1 / life_func(S1=use_level_stress))
                distribution_at_use_stress.CDF(
                    color=color_cycle[i + 1],
                    label=str(round_and_string(use_level_stress) + " (use stress)"),
                )
                x_array.extend(
                    [
                        distribution_at_use_stress.quantile(min(y_array)),
                        distribution_at_use_stress.quantile(max(y_array)),
                    ],
                )  # this ensures the plot limits include the use stress distribution

            plt.legend(title="Stress")

        probplot_type = dist.lower()
        if dist == "Exponential":
            probplot_type = "weibull"

        probability_plot_xylims(x=x_array, y=y_array, dist=probplot_type, spacing=0.1)
        probability_plot_xyticks()
        plt.title("Probability plot\n" + dist + "_" + model + " Model")
        plt.tight_layout()
        return plt.gca()


def life_stress_plot(
    model,
    dist,
    life_func,
    failure_groups,
    stresses_for_groups,
    use_level_stress=None,
    ax=True,
):
    """Generates a life stress plot using the inputs provided. The life stress plot
    is an output from each of the ALT_fitters.

    Parameters
    ----------
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power".
    dist : str
        Must be either "Weibull", "Exponential", "Lognormal", or "Normal"
    life_func : function
        The life function for the ALT life model.
    failure_groups : list
        The failure groups. This is a list of lists.
    stresses_for_groups : list
        The stresses for the failure groups
    use_level_stress : float, int, array, list, None
        The use level stress. This must be an array or list for dual stress
        models. Default is None.
    ax : axes, bool, optional
        The axes handle to use. Default is True which will create a new plot.
        If False then no plot will be generated.
        This is also used to flip the x and y axes to make a stress-life plot. Use ax='swap' to make the life-stress
        plot change its axes to be a stress-life plot (similar to SN diagram).

    Returns
    -------
    current_axes : axes
        The axes handle of the plot. If ax is specified in the inputs then this
        will be the same handle.

    """
    if ax in [
        "swap",
        "swapped",
        "flip",
        "flipped",
        "SWAP",
        "SWAPPED",
        "FLIP",
        "FLIPPED",
        "Swap",
        "Swapped",
        "Flip",
        "Flipped",
    ]:
        ax = True
        swap_xy = True
    else:
        swap_xy = False

    if ax is True or type(ax) == _axes.Axes:
        if model in ["Dual_Exponential", "Power_Exponential", "Dual_Power"]:
            dual_stress = True
        elif model in ["Exponential", "Eyring", "Power"]:
            dual_stress = False
        else:
            raise ValueError(
                "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power",
            )

        if type(ax) == _axes.Axes:
            if dual_stress is False:
                if hasattr(ax, "get_zlim") is False:
                    plt.sca(ax=ax)  # use the axes passed if 2d
                else:
                    colorprint(
                        "WARNING: The axes passed to the life_stress_plot has been ignored as it contains 3d projection. Only specify 3d projection in life stress plots for dual stress models.",
                        text_color="red",
                    )
                    plt.figure(figsize=(9, 9))
            elif hasattr(ax, "get_zlim") is True:
                plt.sca(ax=ax)  # use the axes passed if 3d
            else:
                colorprint(
                    "WARNING: The axes passed to the life_stress_plot has been ignored as it does not have 3d projection. This is a requirement of life stress plots for all dual stress models.",
                    text_color="red",
                )
                fig = plt.figure(figsize=(9, 9))
                ax = fig.add_subplot(111, projection="3d")
        else:
            fig = plt.figure(figsize=(9, 9))  # if no axes is passed, make a new figure
            if dual_stress is True:
                ax = fig.add_subplot(111, projection="3d")

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # gets the default color cycle

        if dist == "Weibull":
            line_label = r"$\alpha$"
        elif dist == "Lognormal":
            line_label = r"$ln(\sigma)$"
        elif dist == "Normal":
            line_label = r"$\sigma$"
        elif dist == "Exponential":
            line_label = r"$1/\lambda$"
        else:
            raise ValueError("dist must be either Weibull, Lognormal, Normal, Exponential")

        if dual_stress is True:
            # collect all the stresses so we can find their min and max
            stress_1_array0 = []
            stress_2_array0 = []
            for stress in stresses_for_groups:
                stress_1_array0.append(stress[0])
                stress_2_array0.append(stress[1])
            if use_level_stress is not None:
                stress_1_array0.append(use_level_stress[0])
                stress_2_array0.append(use_level_stress[1])
            min_stress_1 = min(stress_1_array0)
            max_stress_1 = max(stress_1_array0)
            min_stress_2 = min(stress_2_array0)
            max_stress_2 = max(stress_2_array0)
            # find the upper and lower limits so we can generate the grid of points for the surface
            stress_1_delta_log = np.log(max_stress_1) - np.log(min_stress_1)
            stress_2_delta_log = np.log(max_stress_2) - np.log(min_stress_2)
            stress_1_array_lower = np.exp(np.log(min_stress_1) - stress_1_delta_log * 0.2)
            stress_2_array_lower = np.exp(np.log(min_stress_2) - stress_2_delta_log * 0.2)
            stress_1_array_upper = np.exp(np.log(max_stress_1) + stress_1_delta_log * 0.2)
            stress_2_array_upper = np.exp(np.log(max_stress_2) + stress_2_delta_log * 0.2)
            stress_1_array = np.linspace(stress_1_array_lower, stress_1_array_upper, 50)
            stress_2_array = np.linspace(stress_2_array_lower, stress_2_array_upper, 50)
            X, Y = np.meshgrid(stress_1_array, stress_2_array)
            Z = life_func(S1=X, S2=Y)
            # plot the surface showing stress_1 and stress_2 vs life
            normalized_colors = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
            ax.plot_surface(
                X,
                Y,
                Z,
                cmap="jet_r",
                norm=normalized_colors,
                linewidth=1,
                antialiased=False,
                alpha=0.5,
                zorder=0,
            )
            for i, stress in enumerate(stresses_for_groups):
                # plot the failures as a scatter plot
                ax.scatter(
                    stress[0],
                    stress[1],
                    failure_groups[i],
                    color=color_cycle[i],
                    s=30,
                    label=str(
                        "Failures at stress of " + round_and_string(stress[0]) + ", " + round_and_string(stress[1]),
                    ),
                    zorder=1,
                )
            if use_level_stress is not None:
                # plot the use level stress
                ax.scatter(
                    use_level_stress[0],
                    use_level_stress[1],
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1]),
                    color=color_cycle[i + 1],
                    s=30,
                    label=str(
                        "Use stress of "
                        + round_and_string(use_level_stress[0])
                        + ", "
                        + round_and_string(use_level_stress[1]),
                    ),
                    marker="^",
                    zorder=2,
                )
            ax.set_zlabel("Life")
            ax.set_zlim(bottom=0)
            ax.computed_zorder = False  # computed_zorder ensures the specified zorder is respected so that the scatter plot sits on top of the surface
            plt.xlabel("Stress 1")
            plt.ylabel("Stress 2")
            plt.xlim(min(stress_1_array), max(stress_1_array))
            plt.ylim(min(stress_2_array), max(stress_2_array))
            plt.legend(loc="upper right")
            plt.title("Life-stress plot\n" + dist + "_" + model + " model")

        else:  # single stress model
            if use_level_stress is not None:
                min_stress = min(min(stresses_for_groups), use_level_stress)
            else:
                min_stress = min(stresses_for_groups)
            max_stress = max(stresses_for_groups)
            stress_delta_log = np.log(max_stress) - np.log(min_stress)
            # lower and upper lim
            stress_array_lower = np.exp(np.log(min_stress) - stress_delta_log * 0.2)
            stress_array_upper = np.exp(np.log(max_stress) + stress_delta_log * 0.2)
            # array for the life-stress line
            stress_array = np.linspace(0, stress_array_upper * 10, 1000)
            life_array = life_func(S1=stress_array)
            if swap_xy is True:
                plt.plot(
                    life_array,
                    stress_array,
                    label=str("Characteristic life (" + line_label + ")"),
                    color="k",
                )
                plt.xlabel("Life")
                plt.ylabel("Stress")
            else:
                plt.plot(
                    stress_array,
                    life_array,
                    label=str("Characteristic life (" + line_label + ")"),
                    color="k",
                )
                plt.ylabel("Life")
                plt.xlabel("Stress")
            for i, stress in enumerate(stresses_for_groups):
                failure_points = failure_groups[i]
                stress_points = np.ones_like(failure_points) * stress
                if swap_xy is True:
                    plt.scatter(
                        failure_points,
                        stress_points,
                        color=color_cycle[i],
                        alpha=0.7,
                        label=str("Failures at stress of " + round_and_string(stress)),
                    )
                else:
                    plt.scatter(
                        stress_points,
                        failure_points,
                        color=color_cycle[i],
                        alpha=0.7,
                        label=str("Failures at stress of " + round_and_string(stress)),
                    )
            if use_level_stress is not None:
                alpha_at_use_stress = life_func(S1=use_level_stress)
                if swap_xy is True:
                    plt.plot(
                        [-1e20, alpha_at_use_stress, alpha_at_use_stress],
                        [use_level_stress, use_level_stress, plt.xlim()[0]],
                        label=str("Use stress of " + round_and_string(use_level_stress)),
                        color=color_cycle[i + 1],
                    )
                else:
                    plt.plot(
                        [use_level_stress, use_level_stress, plt.xlim()[0]],
                        [-1e20, alpha_at_use_stress, alpha_at_use_stress],
                        label=str("Use stress of " + round_and_string(use_level_stress)),
                        color=color_cycle[i + 1],
                    )
            # this is a list comprehension to flatten the list of lists. np.ravel won't work here
            flattened_failure_groups = [item for sublist in failure_groups for item in sublist]
            if swap_xy is True:
                plt.xlim(
                    0,
                    1.2 * max(life_func(S1=stress_array_lower), max(flattened_failure_groups)),
                )
                plt.ylim(stress_array_lower, stress_array_upper)
                plt.title("Stress-life plot\n" + dist + "-" + model + " model")
            else:
                plt.ylim(
                    0,
                    1.2 * max(life_func(S1=stress_array_lower), max(flattened_failure_groups)),
                )
                plt.xlim(stress_array_lower, stress_array_upper)
                plt.title("Life-stress plot\n" + dist + "-" + model + " model")
            plt.legend(loc="upper right")
            plt.tight_layout()
        return plt.gca()



def extract_CI(dist, func="CDF", CI_type="time", CI=0.95, CI_y=None, CI_x=None):
    """Extracts the confidence bounds at CI_x or CI_y.

    Parameters
    ----------
    dist : object
        Distribution object from reliability.Distributions
    func : str
        Must be either 'CDF', 'SF', 'CHF'
    CI_type : str
        Must be either 'time' or 'reliability'
    CI : float
        The confidence interval. Must be between 0 and 1.
    CI_y : list, array
        The y-values from which to extract the confidence interval (x-values)
        for bounds on time.
    CI_x : list, array
        The x-values from which to extract the confidence interval (y-values)
        for bounds on reliability.

    Returns
    -------
    lower : array
        An array of the lower confidence bounds at CI_x or CI_y
    upper : array
        An array of the upper confidence bounds at CI_x or CI_y

    Notes
    -----
    If CI_type="time" then CI_y must be specified in order to extract the
    confidence bounds on time.

    If CI_type="reliability" then CI_x must be specified in order to extract the
    confidence bounds on reliability.

    """
    if dist.name == "Exponential":
        if CI_y is not None and CI_x is not None:
            raise ValueError("Both CI_x and CI_y have been provided. Please provide only one.")
        if CI_y is not None:
            if func == "SF":
                q = np.asarray(CI_y)
            elif func == "CDF":
                q = 1 - np.asarray(CI_y)
            elif func == "CHF":
                q = np.exp(-np.asarray(CI_y))
            else:
                raise ValueError("func must be CDF, SF, or CHF")
            SF_time = distribution_confidence_intervals.exponential_CI(self=dist, CI=CI, q=q)
            lower, upper = SF_time[0], SF_time[1]
        elif CI_x is not None:
            SF_rel = distribution_confidence_intervals.exponential_CI(self=dist, CI=CI, x=CI_x)
            if func == "SF":
                lower, upper = SF_rel[1], SF_rel[0]
            elif func == "CDF":
                lower, upper = 1 - SF_rel[0], 1 - SF_rel[1]
            elif func == "CHF":
                lower, upper = -np.log(SF_rel[0]), -np.log(SF_rel[1])
            else:
                raise ValueError("func must be CDF, SF, or CHF")
        else:
            lower, upper = None, None
    else:
        if CI_y is not None and CI_x is not None:
            raise ValueError("Both CI_x and CI_y have been provided. Please provide only one.")
        if CI_x is not None and CI_y is None and CI_type == "time":
            colorprint(
                'WARNING: If CI_type="time" then CI_y must be specified in order to extract the confidence bounds on time.',
                text_color="red",
            )
            lower, upper = None, None
        elif CI_y is not None and CI_x is None and CI_type == "reliability":
            colorprint(
                'WARNING: If CI_type="reliability" then CI_x must be specified in order to extract the confidence bounds on reliability.',
                text_color="red",
            )
            lower, upper = None, None
        elif (CI_y is not None and CI_type == "time") or (CI_x is not None and CI_type == "reliability"):
            if CI_type == "time":
                if func == "SF":
                    q = np.asarray(CI_y)
                elif func == "CDF":
                    q = 1 - np.asarray(CI_y)
                elif func == "CHF":
                    q = np.exp(-np.asarray(CI_y))
                else:
                    raise ValueError("func must be CDF, SF, or CHF")
            if dist.name == "Weibull":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.weibull_CI(self=dist, CI_type="time", CI=CI, q=q)
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.weibull_CI(
                        self=dist,
                        CI_type="reliability",
                        CI=CI,
                        x=CI_x,
                    )
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            elif dist.name == "Normal":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.normal_CI(self=dist, CI_type="time", CI=CI, q=q)
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.normal_CI(
                        self=dist,
                        CI_type="reliability",
                        CI=CI,
                        x=CI_x,
                    )
                    if func == "SF":
                        lower, upper = SF_rel[1], SF_rel[0]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[0], 1 - SF_rel[1]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[0]), -np.log(SF_rel[1])
            elif dist.name == "Lognormal":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.lognormal_CI(self=dist, CI_type="time", CI=CI, q=q)
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.lognormal_CI(
                        self=dist,
                        CI_type="reliability",
                        CI=CI,
                        x=CI_x,
                    )
                    if func == "SF":
                        lower, upper = SF_rel[1], SF_rel[0]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[0], 1 - SF_rel[1]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[0]), -np.log(SF_rel[1])
            elif dist.name == "Gamma":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.gamma_CI(self=dist, CI_type="time", CI=CI, q=q)
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.gamma_CI(self=dist, CI_type="reliability", CI=CI, x=CI_x)
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            elif dist.name == "Gumbel":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.gumbel_CI(self=dist, CI_type="time", CI=CI, q=q)
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.gumbel_CI(
                        self=dist,
                        CI_type="reliability",
                        CI=CI,
                        x=CI_x,
                    )
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            elif dist.name == "Loglogistic":
                if CI_type == "time":
                    SF_time = distribution_confidence_intervals.loglogistic_CI(self=dist, CI_type="time", CI=CI, q=q)
                    lower, upper = SF_time[0], SF_time[1]
                elif CI_type == "reliability":
                    SF_rel = distribution_confidence_intervals.loglogistic_CI(
                        self=dist,
                        CI_type="reliability",
                        CI=CI,
                        x=CI_x,
                    )
                    if func == "SF":
                        lower, upper = SF_rel[0], SF_rel[1]
                    elif func == "CDF":
                        lower, upper = 1 - SF_rel[1], 1 - SF_rel[0]
                    elif func == "CHF":
                        lower, upper = -np.log(SF_rel[1]), -np.log(SF_rel[0])
            else:
                raise ValueError("Unknown distribution")
        else:
            lower, upper = None, None
    if type(lower) is not type(None) and len(lower) == 1:  # unpack arrays of length 1
        lower, upper = lower[0], upper[0]
    return lower, upper

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from matplotlib.axes import _axes

from reliability.Utils._ancillary_utils import round_and_string
from reliability.Utils._plot_utils import probability_plot_xyticks


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
    ax: _axes.Axes | bool | None = True,
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
                    _fitted_dist_params=fitted_dist_params,
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
                    _fitted_dist_params=fitted_dist_params,
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
    if xlim_lower <= 0 and plt.gca().get_xscale() == "log":
        xlim_lower = 1e-10
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

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

from reliability.Utils._ancillary_utils import colorprint, round_and_string, write_df_to_xlsx
from reliability.Utils._array_utils import (
    anderson_darling,
    clean_CI_arrays,
    generate_X_array,
    no_reverse,
    removeNaNs,
    transform_spaced,
    unpack_single_arrays,
    xy_downsample,
    zeroise_below_gamma,
)
from reliability.Utils._dist_utils import distribution_confidence_intervals, extract_CI, life_stress_plot
from reliability.Utils._input_checking_utils import (
    ALT_fitters_input_checking,
    distributions_input_checking,
    fitters_input_checking,
    validate_CI_params,
)
from reliability.Utils._optimization_utils import ALT_MLE_optimization, LS_optimization, MLE_optimization
from reliability.Utils._plot_utils import (
    fill_no_autoscale,
    get_axes_limits,
    line_no_autoscale,
    linear_regression,
    probability_plot_xyticks,
    reshow_figure,
    restore_axes_limits,
    xy_transform,
)
from reliability.Utils._probability_plot_utils import (
    ALT_prob_plot,
    axes_transforms,
    make_fitted_dist_params_for_ALT_probplots,
    probability_plot_xylims,
)
from reliability.Utils._statstic_utils import ALT_least_squares, least_squares

__all__ = [
    "round_and_string",
    "transform_spaced",
    "axes_transforms",
    "get_axes_limits",
    "restore_axes_limits",
    "generate_X_array",
    "no_reverse",
    "zeroise_below_gamma",
    "xy_transform",
    "probability_plot_xylims",
    "probability_plot_xyticks",
    "anderson_darling",
    "colorprint",
    "fitters_input_checking",
    "ALT_fitters_input_checking",
    "validate_CI_params",
    "clean_CI_arrays",
    "fill_no_autoscale",
    "line_no_autoscale",
    "distribution_confidence_intervals",
    "linear_regression",
    "least_squares",
    "ALT_least_squares",
    "LS_optimization",
    "MLE_optimization",
    "ALT_MLE_optimization",
    "write_df_to_xlsx",
    "removeNaNs",
    "make_fitted_dist_params_for_ALT_probplots",
    "ALT_prob_plot",
    "life_stress_plot",
    "xy_downsample",
    "distributions_input_checking",
    "extract_CI",
    "unpack_single_arrays",
    "reshow_figure",
    ]

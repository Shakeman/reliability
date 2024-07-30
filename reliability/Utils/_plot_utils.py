import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import ticker
from matplotlib.axes import _axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError

from reliability.Utils._ancillary_utils import round_and_string


def reshow_figure(handle: _axes.Axes | Figure | None):
    """Shows a figure from an axes handle or figure handle.
    This is useful if the handle is saved to a variable but the figure has been
    closed.
    Note that the Navigation Toolbar (for pan, zoom, and save) is still
    connected to the old figure. There is no known work around for this issue.

    Parameters
    ----------
    handle : object
        The axes handle (type(_axes.Axes)) or figure handle (type(Figure))

    Returns
    -------
    None
        The figure is automatically shown using plt.show().

    """
    if type(handle) is not Figure and not isinstance(handle, _axes.Axes):
        # check that the handle is either an axes or a figure
        raise ValueError("handle must be an axes handle or a figure handle")
    if isinstance(handle, _axes.Axes):
        # if the handle is an axes then extract the Figure
        handle = handle.figure  # type: ignore

    # rebuild the figure
    if handle is not None:
        figsize: npt.NDArray[np.float64] = handle.get_size_inches()
        fig_new = plt.figure()
        new_manager: plt.FigureManagerBase = fig_new.canvas.manager
        new_manager.canvas.figure = handle
        handle.set_canvas(new_manager.canvas)
        handle.set_size_inches(figsize)  # type: ignore
        plt.show()
    else:
        raise ValueError("handle is None. It must be a valid axes or figure handle")


def fill_no_autoscale(xlower, xupper, ylower, yupper, **kwargs):
    """Creates a filled region (polygon) without adding it to the global list of
    autoscale objects. Use this function when you want to plot something but not
    have it considered when autoscale sets the range.

    Parameters
    ----------
    xlower : list, array
        The lower x array for the polygon.
    xupper : list, array
        The upper x array for the polygon.
    ylower : list, array
        The lower y array for the polygon.
    ylower : list, array
        The upper y array for the polygon.
    kwargs
        keyword arguments passed to the matplotlib PolyCollection

    Returns
    -------
    None
        The filled polygon will be added to the plot.

    """
    # generate the polygon
    xstack = np.hstack([xlower, xupper[::-1]])
    ystack = np.hstack([ylower, yupper[::-1]])

    # this corrects illegal yvalues in the probability plot
    if plt.gca().get_yscale() == "function":
        ystack[np.where(ystack >= 1)] = 0.9999999
        ystack[np.where(ystack <= 0)] = 0.0000001

    polygon = np.column_stack([xstack, ystack])
    # this is equivalent to fill as it makes a polygon
    col = PolyCollection([polygon], **kwargs)
    plt.gca().add_collection(col, autolim=False)


def line_no_autoscale(x, y, **kwargs):
    """Creates a line without adding it to the global list of autoscale objects.
    Use this when you want to plot something but not have it considered when
    autoscale sets the range.

    Parameters
    ----------
    x : array, list
        The x values for the line
    y : array, list
        The y values for the line
    kwargs
        keyword arguments passed to the matplotlib LineCollection

    Returns
    -------
    None
        The line will be added to the plot.

    """
    # this is equivalent to plot as it makes a line
    line = np.column_stack([x, y])
    col = LineCollection([line], **kwargs)
    plt.gca().add_collection(col, autolim=False)


def probability_plot_xyticks(yticks=None):
    """This function sets the x and y ticks for probability plots.

    X ticks are selected using either MaxNLocator or LogLocator. X ticks are
    formatted using a custom formatter.

    Y ticks are specified with FixedLocator due to their irregular spacing.
    Minor y ticks use MaxNLocator. Y ticks are formatted using a custom Percent
    Formatter that handles decimals better than the default.

    This function is used by all the probability plots.

    Within this function are several sub functions that are called internally.

    Parameters
    ----------
    yticks : list, array
        The yticks to use. If unspecified, the default yticks are [0.0001, 0.001,
        0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.999999].

    Returns
    -------
    None
        This function will set the ticks but it does not return anything.

    """

    def get_tick_locations(major_or_minor, in_lims=True, axis="x"):
        """Returns the major or minor tick locations for the current axis.

        Parameters
        ----------
        major_or_minor : str
            Specifies which ticks to get. Must be either "major" or
            "minor".
        in_lims : bool, optional
            If in_lims=True then it will only return the ticks that are within
            the current xlim() or ylim(). Default is True.
        axis : str, optional
            Specifies which axis to get the ticks for. Must be "x" or "y".
            Default is "x".

        Returns
        -------
        locations : array
            The tick locations

        """
        if axis == "x":
            AXIS = ax.xaxis
            L = xlower
            U = xupper
        elif axis == "y":
            AXIS = ax.yaxis
        #          L = ylower
        #           U = yupper
        else:
            raise ValueError("axis must be x or y. Default is x")

        if major_or_minor == "major":
            all_locations = AXIS.get_major_locator().tick_values(L, U)
        elif major_or_minor == "minor":
            all_locations = AXIS.get_minor_locator().tick_values(L, U)
        else:
            raise ValueError('major_or_minor must be "major" or "minor"')
        if in_lims is True:
            locations = []
            locations = [item for item in all_locations if item >= L and item <= U]
        else:
            locations = all_locations
        return locations

    def customFormatter(value, _):
        """Provides custom string formatting that is used for the xticks

        Parameters
        ----------
        value : int, float
            The value to be formatted

        Returns
        -------
        label : str
            The formatted string

        """
        if value == 0:
            label = "0"
        elif (
            abs(value) >= 10000 or abs(value) <= 0.0001
        ):  # small numbers and big numbers are formatted with scientific notation
            if value < 0:
                sign = "-"
                value *= -1
            else:
                sign = ""
            exponent = int(np.floor(np.log10(value)))
            multiplier = value / (10**exponent)
            if multiplier % 1 < 0.0000001:
                multiplier = int(multiplier)
            if multiplier == 1:
                label = str((r"$%s%s^{%d}$") % (sign, 10, exponent))
            else:
                label = str((r"$%s%g\times%s^{%d}$") % (sign, multiplier, 10, exponent))
        else:  # numbers between 0.0001 and 10000 are formatted without scientific notation
            label = str(f"{value:g}")
        return label

    def customPercentFormatter(value, _):
        """Provides custom percent string formatting that is used for the yticks
        This function is slightly different than matplotlib's PercentFormatter
        as it does not force a particular number of decimals. ie. in this
        function 99.00 becomes 99 while 99.99 still displays as such.

        Parameters
        ----------
        value : int, float
            The value to be formatted

        Returns
        -------
        label : str
            The formatted string

        """
        value100 = value * 100
        value100dec = round(
            value100 % 1,
            8,
        )  # this breaks down after 8 decimal places due to python's auto rounding. Not likely to be an issue as we're rarely dealing with this many decimals
        if value100dec == 0:
            value100dec = int(value100dec)
        value100whole = int(value100 - value100dec)
        combined = value100dec + value100whole
        label = str(str(combined) + "%")
        return label

    def get_edge_distances():
        """Finds the sum of the distance (in axes coords (0 to 1)) of the distances
        from the edge ticks to the edges

        Parameters
        ----------
        None

        Returns
        -------
        distances : float
            The edge distances

        """
        xtick_locations = get_tick_locations("major", axis="x")
        left_tick_distance = xy_transform(xtick_locations[0], direction="forward", axis="x") - xy_transform(
            xlower,
            direction="forward",
            axis="x",
        )
        right_tick_distance = xy_transform(xupper, direction="forward", axis="x") - xy_transform(
            xtick_locations[-1],
            direction="forward",
            axis="x",
        )
        return left_tick_distance + right_tick_distance

    ################# xticks
    MaxNLocator = ticker.MaxNLocator(nbins=10, min_n_ticks=2, steps=[1, 2, 5, 10])
    LogLocator = ticker.LogLocator()
    ax = plt.gca()
    xlower, xupper = plt.xlim()
    if xlower <= 0:  # can't use log scale if 0 (gamma) or negative numbers (normal and gumbel)
        loc_x = MaxNLocator
    elif xupper < 0.1:  # very small positive values
        loc_x = ticker.LogLocator()
    elif (
        xupper < 1000 or np.log10(xupper) - np.log10(xlower) < 1.5
    ):  # not too big and not too small OR it may be big but not too spread out
        loc_x = MaxNLocator
    else:  # it is really big (>1000) and spread out
        loc_x = ticker.LogLocator()
    if xupper < xlower:
        raise ValueError("xupper must be greater than xlower")
    ax.xaxis.set_major_locator(loc_x)  # apply the tick locator
    # do not apply a minor locator. It is never as good as the default

    if (
        get_edge_distances() > 0.5
    ):  # 0.5 means 50% of the axis is without ticks on either side. Above this is considered unacceptable. This has a weakness where there's only 1 tick it will return 0. Changing 0 to 1 can make things too crowded
        # find which locator is better
        ax.xaxis.set_major_locator(MaxNLocator)
        edges_maxNLocator = get_edge_distances()
        ax.xaxis.set_major_locator(LogLocator)
        edges_LogLocator = get_edge_distances()
        if edges_LogLocator < edges_maxNLocator:
            ax.xaxis.set_major_locator(LogLocator)  # apply a new locator
        else:
            ax.xaxis.set_major_locator(MaxNLocator)  # apply a new locator
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(customFormatter),
    )  # the custom formatter is always applied to the major ticks

    num_major_x_ticks_shown = len(get_tick_locations("major", axis="x"))
    num_minor_x_xticks_shown = len(get_tick_locations("minor", axis="x"))
    max_minor_ticks = 15 if max(abs(xlower), abs(xupper)) < 1000 and min(abs(xlower), abs(xupper)) > 0.001 else 10
    if num_major_x_ticks_shown < 2 and num_minor_x_xticks_shown <= max_minor_ticks:
        ax.xaxis.set_minor_formatter(
            ticker.FuncFormatter(customFormatter),
        )  # if there are less than 2 major ticks within the plotting limits then the minor ticks should be labeled. Only do this if there aren't too many minor ticks

    ################# yticks
    if yticks is None:
        yticks = [
            0.0001,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
            0.9999,
            0.999999,
        ]
    loc_y = ticker.FixedLocator(yticks)
    loc_y_minor = ticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10])
    ax.yaxis.set_major_locator(loc_y)  # sets the tick spacing
    ax.yaxis.set_minor_locator(loc_y_minor)  # sets the tick spacing
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(customPercentFormatter))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(customPercentFormatter))
    ax.format_coord = (
        lambda x, y: f"x={x:g}, y={y:.1%}"
    )  # sets the formatting of the axes coordinates in the bottom right of the figure. Without this the FuncFormatter raw strings make it into the axes coords and don't look good.


def xy_transform(value, direction="forward", axis="x") -> np.float64 | npt.NDArray[np.float64] | list[float]:
    """This function converts between data values and axes coordinates (based on
    xlim() or ylim()).

    If direction is forward the returned value will always be between 0 and 1
    provided the value is on the plot.

    If direction is reverse the input should be between 0 and 1 and the returned
    value will be the data value based on the current plot lims

    Parameters
    ----------
    value: int, float, list, array
        The value/s to be transformed
    direction : str, optional
        Must be "forward" or "inverse". Default is "forward"
    axis : str, optional
        Must be "x" or "y". Default is "x".

    Returns
    -------
    transformed_values : float, array
        The transformed values. This will be a float if the input was int or
        float, or an array if the input was list or array.

    """
    if direction not in ["reverse", "inverse", "inv", "rev", "forward", "fwd"]:
        raise ValueError('direction must be "forward" or "reverse"')
    if axis not in ["X", "x", "Y", "y"]:
        raise ValueError("axis must be x or y. Default is x")

    ax = plt.gca()
    if direction in ["reverse", "inverse", "inv", "rev"]:
        if type(value) in [int, float, np.float64]:
            if axis == "x":
                # x transform
                transformed_values: np.float64 | npt.NDArray[np.float64] | list[float] = (
                    ax.transData.inverted().transform((ax.transAxes.transform((value, 0.5))[0], 0.5))[0]
                )
            else:
                # y transform
                transformed_values = ax.transData.inverted().transform((1, ax.transAxes.transform((1, value))[1]))[1]
        elif type(value) in [list, np.ndarray]:
            transformed_values = []
            for item in value:
                if axis == "x":
                    transformed_values.append(
                        ax.transData.inverted().transform((ax.transAxes.transform((item, 0.5))[0], 0.5))[0],
                    )  # x transform
                else:
                    transformed_values.append(
                        ax.transData.inverted().transform((1, ax.transAxes.transform((1, item))[1]))[1],
                    )  # y transform
        else:
            raise ValueError("type of value is not recognized")
    elif type(value) in [int, float, np.float64]:
        if axis == "x":
            transformed_values = ax.transAxes.inverted().transform(ax.transData.transform((value, 0.5)))[
                0
            ]  # x transform
        else:
            transformed_values = ax.transAxes.inverted().transform(ax.transData.transform((1, value)))[1]  # y transform
    elif type(value) in [list, np.ndarray]:
        transformed_values = []
        for item in value:
            if axis == "x":
                transformed_values.append(
                    ax.transAxes.inverted().transform(ax.transData.transform((item, 0.5)))[0],
                )  # x transform
            else:
                transformed_values.append(
                    ax.transAxes.inverted().transform(ax.transData.transform((1, value)))[1],
                )  # y transform
    else:
        raise ValueError("type of value is not recognized")
    return transformed_values


def restore_axes_limits(
    limits: tuple[tuple[float, float], tuple[float, float], bool],
    dist,
    func,
    X,
    Y,
    xvals=None,
    xmin=None,
    xmax=None,
):
    """This function works in a pair with get_axes_limits. Using the values
    producted by get_axes_limits which are [xlims, ylims, use_prev_lims], this
    function will determine how to change the axes limits to meet the style
    requirements of the library.

    Parameters
    ----------
    limits : list
        A list of [xlims, ylims, use_prev_lims] created by get_axes_limits
    dist : object
        The distribution object which the axes limits are influenced by.
    X : array, list
        The x-values of the plot
    Y : array, list
        The y-values of the plot
    xvals : array, list, optional
        The plot xvals if specified. May be None if not specified.
    xmin : int, float, optional
        The plot xmin if specified. May be None if not specified.
    xmax : int, float, optional
        The plot xmax if specified. May be None if not specified.

    Returns
    -------
    None
        This function will scale the plot but it does not return anything

    Notes
    -----
    No scaling will be done if the axes are not linear due to errors that result
    from log and function scaled axes when a limit of 0 is used. This means that
    this function is not able to be applied to the probability plots are they
    have non-linear scaled axes.

    """
    xlims = limits[0]
    ylims = limits[1]
    use_prev_lims = limits[2]

    ################## XLIMS ########################
    # obtain the xlims as if we did not consider prev limits
    if xvals is None:
        if xmin is None:
            if dist.name in [
                "Weibull",
                "Gamma",
                "Loglogistic",
                "Exponential",
                "Lognormal",
            ]:
                if dist.gamma == 0:
                    xlim_lower = 0
                else:
                    diff = dist.quantile(0.999) - dist.quantile(0.001)
                    xlim_lower = max(0, dist.quantile(0.001) - diff * 0.1)
            elif dist.name in ["Normal", "Gumbel"]:
                xlim_lower = dist.quantile(0.001)
            elif dist.name == "Beta":
                xlim_lower = 0
            elif dist.name in ["Mixture", "Competing risks"]:
                # DSZI not required here as limits are same as base distribution
                xlim_lower = min(X)
            else:
                raise ValueError("Unrecognised distribution name")
        else:
            xlim_lower = xmin

        xlim_upper = (1 if dist.name == "Beta" else dist.quantile(0.999)) if xmax is None else xmax

        if xlim_lower > xlim_upper:
            xlim_lower, xlim_upper = (
                xlim_upper,
                xlim_lower,
            )  # switch them if xmin and xmax were given in the wrong order
    else:  # if the xlims have been specified then these are the limits to be used
        xlim_lower = min(xvals)
        xlim_upper = max(xvals)

    # determine what to set the xlims based on whether to use_prev_lims
    if use_prev_lims is True:
        xlim_lower = min(xlim_lower, xlims[0])
        xlim_upper = max(xlim_upper, xlims[1])

    if plt.gca().get_xscale() == "linear" and len(X) > 1:
        plt.xlim(xlim_lower, xlim_upper, auto=None)

    ################## YLIMS ########################

    top_spacing = 1.1  # the amount of space between the max value and the upper axis limit. 1.1 means the axis lies 10% above the max value
    if func in ["pdf", "PDF"]:
        if not np.isfinite(dist._pdf0) and not np.isfinite(Y[-1]):  # asymptote on the left and right
            ylim_upper = min(Y) * 5
        elif not np.isfinite(Y[-1]) or dist._pdf0 == np.inf or dist._pdf0 > 10:  # asymptote on the right or on the left
            ylim_upper = max(Y)
        else:  # an increasing pdf. Not asymptote
            ylim_upper = max(Y) * top_spacing
    elif func in ["cdf", "CDF", "SF", "sf"]:
        ylim_upper = top_spacing
    elif func in ["hf", "HF"]:
        if not np.isfinite(dist._hf0) and not np.isfinite(Y[-1]):  # asymptote on the left and right
            ylim_upper = min(Y) * 5
        elif not np.isfinite(Y[-1]) or dist._hf0 == np.inf or dist._hf0 > 10:  # asymptote of the right or on the left
            ylim_upper = max(Y)
        elif max(Y) > Y[-1]:  # a peaked hf
            ylim_upper = max(Y) * top_spacing
        else:  # an increasing hf. Not an asymptote
            idx = len(X) - 1 if (len(np.where(plt.xlim()[1] <= X)[0]) == 0) else np.where(plt.xlim()[1] <= X)[0][0]
            # this is for the mixture model and CR model
            ylim_upper = Y[idx] * top_spacing
    elif func in ["chf", "CHF"]:
        idx = (len(X) - 1) if (len(np.where(xlim_upper <= X)[0]) == 0) else np.where(xlim_upper <= X)[0][0]
        # this is for the mixture model and CR model
        # else is the index of the chf where it is equal to b95
        ylim_upper = Y[idx] * top_spacing if np.isfinite(Y[idx]) else Y[idx - 1] * top_spacing
    else:
        raise ValueError("func is invalid")
    ylim_lower = 0

    # determine what to set the ylims based on whether to use_prev_lims
    if use_prev_lims is False:
        ylim_LOWER = ylim_lower
        ylim_UPPER = ylim_upper
    else:  # need to consider previous axes limits
        ylim_LOWER = min(ylim_lower, ylims[0])
        ylim_UPPER = max(ylim_upper, ylims[1])

    if plt.gca().get_yscale() == "linear" and len(Y) > 1:
        if ylim_LOWER != ylim_UPPER and np.isfinite(ylim_UPPER):
            plt.ylim(ylim_LOWER, ylim_UPPER, auto=None)
        else:
            plt.ylim(bottom=ylim_LOWER, auto=None)


def get_axes_limits() -> tuple[tuple[float, float], tuple[float, float], bool]:
    """This function works in a pair with restore_axes_limits
    This function gets the previous xlim and ylim and also checks whether there
    was a previous plot (based on whether the default 0,1 axes had been
    changed).

    It returns a list of items that are used by restore_axes_limits after the
    plot has been performed.

    Parameters
    ----------
    None
        The plot properties are extracted automatically for analysis

    Returns
    -------
    output : list
        A list of [xlims, ylims, use_prev_lims]. These values are used by
        restore_axes_limits to determine how the axes limits need to be
        changed after plotting.

    """
    xlims: tuple[float, float] = plt.xlim(auto=None)  # type: ignore # get previous xlim
    ylims: tuple[float, float] = plt.ylim(auto=None)  # type: ignore # get previous ylim
    use_prev_lims = not (
        xlims == (0, 1) and ylims == (0, 1)
    )  # this checks if there was a previous plot. If the lims were 0,1 and 0,1 then there probably wasn't
    out: tuple[tuple[float, float], tuple[float, float], bool] = (xlims, ylims, use_prev_lims)
    return out


def linear_regression(
    x,
    y,
    slope=None,
    x_intercept=None,
    y_intercept=None,
    RRX_or_RRY="RRX",
    show_plot=False,
    **kwargs,
):
    """This function provides the linear algebra solution to find line of best fit
    passing through points (x,y). Options to specify slope or intercept enable
    these parameters to be forced.

    Rank regression can be on X (RRX) or Y (RRY). Default is RRX.
    Note that slope depends on RRX_or_RRY. If you use RRY then slope is dy/dx
    but if you use RRX then slope is dx/dy.

    Parameters
    ----------
    x : array, list
        The x values
    y : array, list
        The y values
    slope : float, int, optional
        Used to force the slope. Default is None.
    x_intercept : float, int, optional
        Used to force the x-intercept. Default is None. Only used for RRY.
    y_intercept : float, int, optional
        Used to force the y-intercept. Default is None. Only used for RRX.
    RRX_or_RRY : str, optional
        Must be "RRY" or "RRX". Default is "RRY".
    show_plot : bool, optional
        If True, a plot of the line and points will be generated. Use plt.show()
        to show it.
    kwargs
        Keyword arguments for the plot that are passed to matplotlib for the
        line.

    Returns
    -------
    slope : float
        The slope of the line.
    intercept : float
        The intercept (x or y depending on RRX_or_RRY) of the line.

    Notes
    -----
    The equation of a line used here is Y = slope * X + intercept. This is the
    RRY form. For RRX it can be rearranged to X = (Y - intercept)/slope

    For more information on linear regression, see the `documentation <https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html>`_.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y are different lengths")
    if RRX_or_RRY not in ["RRX", "RRY"]:
        raise ValueError('RRX_or_RRY must be either "RRX" or "RRY". Default is "RRY".')
    if x_intercept is not None and RRX_or_RRY == "RRY":
        raise ValueError("RRY must use y_intercept not x_intercept")
    if y_intercept is not None and RRX_or_RRY == "RRX":
        raise ValueError("RRX must use x_intercept not y_intercept")
    if slope is not None and (x_intercept is not None or y_intercept is not None):
        raise ValueError("You can not specify both slope and intercept")

    if RRX_or_RRY == "RRY":
        if y_intercept is not None:  # only the slope must be found
            min_pts = 1
            xx = np.array([x]).T
            yy = (y - y_intercept).T
        elif slope is not None:  # only the intercept must be found
            min_pts = 1
            xx = np.array([np.ones_like(x)]).T
            yy = (y - slope * x).T
        else:  # both slope and intercept must be found
            min_pts = 2
            xx = np.array([x, np.ones_like(x)]).T
            yy = y.T
    elif x_intercept is not None:  # only the slope must be found
        min_pts = 1
        yy = np.array([y]).T
        xx = (x - x_intercept).T
    elif slope is not None:  # only the intercept must be found
        min_pts = 1
        yy = np.array([np.ones_like(y)]).T
        xx = (x - slope * y).T
    else:  # both slope and intercept must be found
        min_pts = 2
        yy = np.array([y, np.ones_like(y)]).T
        xx = x.T

    if len(x) < min_pts:
        if slope is not None:
            err_str = "A minimum of 1 point is required to fit the line when the slope is specified."
        elif x_intercept is not None and y_intercept is not None:
            err_str = "A minimum of 1 point is required to fit the line when the intercept is specified."
        else:
            err_str = "A minimum of 2 points are required to fit the line when slope or intercept are not specified."
        raise ValueError(err_str)

    if RRX_or_RRY == "RRY":
        try:
            solution = np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)  # linear regression formula for RRY
        except LinAlgError:
            raise RuntimeError(
                "An error has occurred when attempting to find the initial guess using least squares estimation.\nThis error is caused by a non-invertable matrix.\nThis can occur when there are only two very similar failure times like 10 and 10.000001.\nThere is no solution to this error, other than to use failure times that are more unique.",
            ) from None
        if y_intercept is not None:
            m = solution[0]
            c = y_intercept
        elif slope is not None:
            m = slope
            c = solution[0]
        else:
            m = solution[0]
            c = solution[1]
    else:  # RRX
        try:
            solution = np.linalg.inv(yy.T.dot(yy)).dot(yy.T).dot(xx)  # linear regression formula for RRX
        except LinAlgError:
            raise RuntimeError(
                "An error has occurred when attempting to find the initial guess using least squares estimation.\nThis error is caused by a non-invertable matrix.\nThis can occur when there are only two very similar failure times like 10 and 10.000001.\nThere is no solution to this error, other than to use failure times that are more unique.",
            ) from None
        if x_intercept is not None:
            m_x = solution[0]
            m = 1 / m_x
            c = -x_intercept / m_x
        elif slope is not None:
            m = 1 / slope
            c_x = solution[0]
            c = -c_x / slope
        else:
            m_x = solution[0]
            c_x = solution[1]
            m = 1 / m_x
            c = -c_x / m_x

    if show_plot is True:
        plt.scatter(x, y, marker=".", color="k")
        delta_x = max(x) - min(x)
        delta_y = max(y) - min(y)
        xvals = np.linspace(min(x) - delta_x, max(x) + delta_x, 10)
        yvals = m * xvals + c
        if "label" in kwargs:
            label = kwargs.pop("label")
        else:
            label = str("y=" + round_and_string(m, 2) + ".x + " + round_and_string(c, 2))
        plt.plot(xvals, yvals, label=label, **kwargs)
        plt.xlim(min(x) - delta_x * 0.2, max(x) + delta_x * 0.2)
        plt.ylim(min(y) - delta_y * 0.2, max(y) + delta_y * 0.2)
    return m, c


import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError

from reliability.Utils._ancillary_utils import round_and_string


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

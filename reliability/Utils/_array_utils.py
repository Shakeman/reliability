
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.stats as ss

from reliability.Utils._ancillary_utils import colorprint


def anderson_darling(fitted_cdf, empirical_cdf) -> float:
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
    n: int = len(fitted_cdf)
    AD: float = n * ((A + B + C).sum())
    return AD

def unpack_single_arrays(array):
    """Unpacks arrays with a single element to return just that element

    Parameters
    ----------
    array : float, int, list, array
        The value for unpacking

    Returns
    -------
    output : float, list, int, array
        If the input was a single length numpy array then the output will be the
        item from the array. If the input was anything else then the output will
        match the input

    """
    out = (array[0] if len(array) == 1 else array) if type(array) == np.ndarray else array
    return out


def generate_X_array(dist, xvals=None, xmin=None, xmax=None):
    """Generates the array of X values for each of the PDf, CDF, SF, HF, CHF
    functions within reliability.Distributions

    This is done with a variety of cases in order to ensure that for regions of
    high gradient (particularly asymptotes to inf) the points are more
    concentrated. This ensures that the line always looks as smooth as possible
    using only 200 data points.

    Parameters
    ----------
    dist : object
        The distribution object
    xvals : array, list, optional
        The xvals for the plot if specified
    xmin : array, list, optional
        The xmin for the plot if specified
    xmax : array, list, optional
        The xmax for the plot if specified

    Returns
    -------
    X : array
        The X array that was generated.

    """
    # obtain the xvals array
    points = 200  # the number of points to use when generating the X array
    points_right = 25  # the number of points given to the area above QU. The total points is still equal to 'points' so the area below QU receives 'points - points_right'
    QL: np.float64 = dist.quantile(0.0001)  # quantile lower
    QU: np.float64  = dist.quantile(0.99)  # quantile upper
    if xvals is not None:
        X = xvals
        if type(X) in [float, int, np.float64]:
            if X < 0 and dist.name not in ["Normal", "Gumbel"]:
                raise ValueError("the value given for xvals is less than 0")
            if X > 1 and dist.name == "Beta":
                raise ValueError(
                    "the value given for xvals is greater than 1. The beta distribution is bounded between 0 and 1.",
                )
            X = np.array([X])
        elif isinstance(X, list):
            X = np.array(X)
        elif type(X) is np.ndarray:
            pass
        else:
            raise ValueError("unexpected type in xvals. Must be int, float, list, or array")
        if type(X) is np.ndarray and min(X) < 0 and dist.name not in ["Normal", "Gumbel"]:
            raise ValueError("xvals was found to contain values below 0")
        if type(X) is np.ndarray and max(X) > 1 and dist.name == "Beta":
            raise ValueError(
                "xvals was found to contain values above 1. The beta distribution is bounded between 0 and 1.",
            )
    elif dist.name in ["Weibull", "Lognormal", "Loglogistic", "Exponential", "Gamma"]:
        if xmin is None:
            xmin = 0
        if xmin < 0:
            raise ValueError(
                "xmin must be greater than or equal to 0 for all distributions except the Normal and Gumbel distributions",
            )
        if xmax is None:
            xmax = dist.quantile(0.9999)
        if xmin > xmax:
            xmin, xmax = (
                xmax,
                xmin,
            )  # switch them if they are given in the wrong order
        if (xmin < QL and xmax < QL) or (xmin >= QL and xmax <= QU) or (xmin > QU and xmax > QU):
            X = np.linspace(xmin, xmax, points)
        elif xmin < QL and xmax > QL and xmax < QU:
            if dist.gamma == 0:
                if dist._pdf0 == 0:
                    X = np.hstack([xmin, np.linspace(QL, xmax, points - 1)])
                else:  # pdf is asymptotic to inf at x=0
                    X = np.hstack([xmin, np.geomspace(QL, xmax, points - 1)])
            elif dist._pdf0 == 0:
                X = np.hstack([xmin, dist.gamma - 1e-8, np.linspace(QL, xmax, points - 2)])
            else:  # pdf is asymptotic to inf at x=0
                detail = np.geomspace(QL - dist.gamma, xmax - dist.gamma, points - 2) + dist.gamma
                X = np.hstack([xmin, dist.gamma - 1e-8, detail])
        elif xmin > QL and xmin < QU and xmax > QU:
            if dist._pdf0 == 0:
                X = np.hstack(
                    [
                        np.linspace(xmin, QU, points - points_right),
                        np.linspace(QU, xmax, points_right),
                    ],
                )
            else:  # pdf is asymptotic to inf at x=0
                try:
                    detail = (
                        np.geomspace(
                            xmin - dist.gamma,
                            QU - dist.gamma,
                            points - points_right,
                        )
                        + dist.gamma
                    )
                    right = np.geomspace(QU - dist.gamma, xmax - dist.gamma, points_right) + dist.gamma
                except ValueError:  # occurs for very low shape params causing QL-gamma to be zero
                    detail = np.linspace(xmin, QU, points - points_right)
                    right = np.linspace(QU, xmax, points_right)
                X = np.hstack([detail, right])
        elif dist.gamma == 0:
            if dist._pdf0 == 0:
                X = np.hstack(
                    [
                        xmin,
                        np.linspace(QL, QU, points - (points_right + 1)),
                        np.geomspace(QU, xmax, points_right),
                    ],
                )
            else:  # pdf is asymptotic to inf at x=0
                try:
                    X: npt.NDArray[np.float64] = np.hstack(
                        [
                            xmin,
                            np.geomspace(QL, QU, points - (points_right + 1)),
                            np.geomspace(QU, xmax, points_right),
                        ],
                    )
                except ValueError:  # occurs for very low shape params causing QL to be zero
                    X = np.hstack(
                        [
                            xmin,
                            np.linspace(QL, QU, points - (points_right + 1)),
                            np.geomspace(QU, xmax, points_right),
                        ],
                    )
        elif dist._pdf0 == 0:
            X = np.hstack(
                [
                    xmin,
                    dist.gamma - 1e-8,
                    np.linspace(QL, QU, points - (points_right + 2)),
                    np.geomspace(QU - dist.gamma, xmax - dist.gamma, points_right) + dist.gamma,
                ],
            )
        else:  # pdf is asymptotic to inf at x=0
            try:
                detail = (
                    np.geomspace(
                        QL - dist.gamma,
                        QU - dist.gamma,
                        points - (points_right + 2),
                    )
                    + dist.gamma
                )
                right = np.geomspace(QU - dist.gamma, xmax - dist.gamma, points_right) + dist.gamma
            except ValueError:  # occurs for very low shape params causing QL-gamma to be zero
                detail = np.linspace(QL, QU, points - (points_right + 2))
                right = np.linspace(QU, xmax, points_right)
            X = np.hstack([xmin, dist.gamma - 1e-8, detail, right])
    elif dist.name in ["Normal", "Gumbel"]:
        if xmin is None:
            xmin = dist.quantile(0.0001)
        if xmax is None:
            xmax = dist.quantile(0.9999)
        if xmin > xmax:
            xmin, xmax = (
                xmax,
                xmin,
            )  # switch them if they are given in the wrong order
        if xmin <= 0 or xmin > dist.quantile(0.0001):
            X = np.linspace(xmin, xmax, points)
        else:
            X = np.hstack(
                [0, np.linspace(xmin, xmax, points - 1)],
            )  # this ensures that the distribution is at least plotted from 0 if its xmin is above 0
    elif dist.name == "Beta":
        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = 1
        if xmax > 1:
            raise ValueError("xmax must be less than or equal to 1 for the beta distribution")
        if xmin > xmax:
            xmin, xmax = (
                xmax,
                xmin,
            )  # switch them if they are given in the wrong order
        X = np.linspace(xmin, xmax, points)
    else:
        raise ValueError("Unrecognised distribution name")
    return X


def xy_downsample(x, y, downsample_factor=None, default_max_values=1000):
    """This function downsamples the x and y arrays. This exists to make plotting
    much faster, particularly when matplotlib becomes very slow for tens of
    thousands of datapoints.

    Parameters
    ----------
    x : array, list
        The x values
    y : array, list
        The y values
    downsample_factor : int, optional
        How must downsampling to do. See Notes for more detail.
    default_max_values : int, optional
        The maximum number of values to be returned if downsample_factor is
        None. See Notes for more detail.

    Returns
    -------
    x : array
        The downsampled x values
    y : array
        The downsampled y values

    Notes
    -----
    Downsampling is done using the downsample_factor. If the down_sample factor
    is 2 then every second value will be returned, if 3 then every third value
    will be returned. The first and last items will always be included in the
    downsampled dataset.

    If downsample_factor is not specified, downsampling will only occur if there
    are more than default_max_values. The downsample factor will aim for the
    number of values to be returned to be between default_max_values/2 and
    default_max_values. By default this is between 500 and 1000.

    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    len_x = len(x)
    if downsample_factor is False:
        return x, y
    if downsample_factor in [None, True] and len_x < default_max_values:
        return x, y
    else:
        if downsample_factor in [None, True]:
            downsample_factor = np.floor(len_x / (0.5 * default_max_values))
        elif not isinstance(downsample_factor, int):
            raise ValueError("downsample_factor must be an integer")
        if len_x / downsample_factor < 2:
            return x, y
        else:
            indices = np.arange(start=0, stop=len_x, step=int(np.floor(downsample_factor)), dtype=int)

            if len_x - 1 not in indices:
                indices[-1] = len_x - 1
            x_downsample = []
            y_downsample = []
            for idx in indices:
                x_downsample.append(x_sorted[idx])
                y_downsample.append(y_sorted[idx])
            return x_downsample, y_downsample

def removeNaNs(X):
    """Removes NaNs from a list or array.

    Parameters
    ----------
    X : array, list
        The array or list to be processed.

    Returns
    -------
    output : list, array
        A list or array of the same type as the input with the NaNs removed.

    Notes
    -----
    This is better than simply using "x = x[numpy.logical_not(numpy.isnan(x))]"
    as numpy crashes for str and bool.

    """
    if type(X) == np.ndarray:
        X = list(X)
        arr_out = True
    else:
        arr_out = False
    out = []
    for i in X:
        if type(i) in [str, bool, np.str_]:
            if i != "nan":
                out.append(i)
        elif np.logical_not(np.isnan(i)):  # this only works for numbers
            out.append(i)
    if arr_out is True:
        out = np.asarray(out)
    return out

def clean_CI_arrays(xlower, xupper, ylower, yupper, plot_type="CDF", x=None, q=None):
    """This function cleans the CI arrays of nans and numbers <= 0 and also removes
    numbers >= 1 if plot_type is CDF or SF.

    Parameters
    ----------
    xlower : list, array
        The lower x array for the confidence interval
    xupper : list, array
        The upper x array for the confidence interval
    ylower : list, array
        The lower y array for the confidence interval
    yupper : list, array
        The upper y array for the confidence interval
    plot_type : str, optional
        Must be "CDF", "SF", "CHF". Default is "CDF"
    x : array, optional
        The x array for CI extraction
    q : array, optional
        The q array for CI extraction

    Returns
    -------
    xlower : array
        The "cleaned" lower x array for the confidence interval
    xupper : array
        The "cleaned" upper x array for the confidence interval
    ylower : array
        The "cleaned" lower y array for the confidence interval
    ylower : array
        The "cleaned" upper y array for the confidence interval

    Notes
    -----
    The returned arrays will all be the same length

    The cleaning is done by deleting values. If the cleaned arrays are < 2 items
    in length then an error will be triggered.

    """
    # format the input as arrays
    xlower = np.asarray(xlower)
    xupper = np.asarray(xupper)
    ylower = np.asarray(ylower)
    yupper = np.asarray(yupper)

    # create empty arrays to fill with cleaned values
    xlower_out = np.array([])
    xupper_out = np.array([])
    ylower_out = np.array([])
    yupper_out = np.array([])

    xlower_out2 = np.array([])
    xupper_out2 = np.array([])
    ylower_out2 = np.array([])
    yupper_out2 = np.array([])

    xlower_out3 = np.array([])
    xupper_out3 = np.array([])
    ylower_out3 = np.array([])
    yupper_out3 = np.array([])

    # remove nans in all arrays
    for i in np.arange(len(xlower)):
        if np.isfinite(xlower[i]) and np.isfinite(xupper[i]) and np.isfinite(ylower[i]) and np.isfinite(yupper[i]):
            xlower_out = np.append(xlower_out, xlower[i])
            xupper_out = np.append(xupper_out, xupper[i])
            ylower_out = np.append(ylower_out, ylower[i])
            yupper_out = np.append(yupper_out, yupper[i])

    # remove values >= 1 for CDF and SF
    if plot_type.upper() in ["CDF", "SF"]:
        for i in np.arange(len(xlower_out)):
            if ylower_out[i] < 1 and yupper_out[i] < 1:
                xlower_out2 = np.append(xlower_out2, xlower_out[i])
                xupper_out2 = np.append(xupper_out2, xupper_out[i])
                ylower_out2 = np.append(ylower_out2, ylower_out[i])
                yupper_out2 = np.append(yupper_out2, yupper_out[i])
    else:  # do nothing
        xlower_out2 = xlower_out
        xupper_out2 = xupper_out
        ylower_out2 = ylower_out
        yupper_out2 = yupper_out

    # remove values <=0 for all cases
    tol = 1e-50  # tolerance for equivalent to zero. Accounts for precision error
    for i in np.arange(len(xlower_out2)):
        if ylower_out2[i] > tol and yupper_out2[i] > tol:
            xlower_out3 = np.append(xlower_out3, xlower_out2[i])
            xupper_out3 = np.append(xupper_out3, xupper_out2[i])
            ylower_out3 = np.append(ylower_out3, ylower_out2[i])
            yupper_out3 = np.append(yupper_out3, yupper_out2[i])

    # checks whether CI_x or CI_y was specified and resulted in values being deleted due to being illegal values. Raises a more detailed error for the user.
    if len(xlower_out3) != len(xlower) and x is not None:
        raise ValueError(
            "The confidence intervals for CI_x cannot be returned because they are NaN. This may occur when the SF=0. Try specifying CI_x values closer to the mean of the distribution.",
        )
    if len(ylower_out3) != len(ylower) and q is not None:
        raise ValueError(
            "The confidence intervals for CI_y cannot be returned because they are NaN. This may occur when the CI_y is near 0 or 1. Try specifying CI_y values closer to 0.5.",
        )

    # final error check for lengths matching and there still being at least 2 elements remaining
    if (
        len(xlower_out3) != len(xupper_out3)
        or len(xlower_out3) != len(yupper_out3)
        or len(xlower_out3) != len(ylower_out3)
        or len(xlower_out3) < 1
    ):
        colorprint(
            "ERROR in clean_CI_arrays: Confidence intervals could not be plotted due to the presence of too many NaNs in the arrays.",
            text_color="red",
        )

    return xlower_out3, xupper_out3, ylower_out3, yupper_out3

def zeroise_below_gamma(X, Y, gamma):
    """This will make all Y values 0 for the corresponding X values being below
    gamma (the threshold parameter for Weibull, Exponential, Gamma, Loglogistic,
    and Lognormal).

    Used by HF and CHF which need to be zeroized if the gamma shifted form of
    the equation is used.

    Parameters
    ----------
    X : array, list
        The x values of the distribution. These areused to determine whis Y
        values to zeroize.
    Y : array, list
        The y-values of the distribution
    gamma : float, int
        The gamma parameter. This is the point at which Y values corresponding
        to X values below gamma will be zeroized.

    Returns
    -------
    Y : array
        The zeroized Y array

    """
    if gamma > 0:
        if len(np.where(gamma < X)[0]) == 0:
            Y[0::] = 0  # zeroize everything if there is no X values above gamma
        else:
            Y[0 : (np.where(gamma < X)[0][0])] = 0  # zeroize below X=gamma
    return Y

def no_reverse(x, CI_type, plot_type):
    """This is used to convert an array that decreases and then increases into an
    array that decreases then is constant at its minimum.

    The always decreasing rule will apply unless CI_type = 'time' and
    plot_type = 'CHF'

    This function is used to provide a correction to the confidence intervals
    which mathematically are correct but practically should never decrease.

    Parameters
    ----------
    x : array, list
        The array or list to which the no_reverse correction is applied
    CI_type : str
        Must be either 'time' or 'reliability'
    plot_type : str
        Must be either 'CDF', 'SF', or 'CHF'

    Returns
    -------
    x : array
        A corrected form of the input x that obeys the always decreasing rule
        (or the always increasing rule in the case of CI_type = 'time' and
        plot_type = 'CHF').

    """
    if type(x) not in [np.ndarray, list]:
        raise ValueError("x must be a list or array")
    if len(x) < 2:
        raise ValueError("x must be a list or array with length greater than 1")
    decreasing = not (CI_type == "time" and plot_type == "CHF")

    x = np.copy(np.asarray(x))
    if all(np.isfinite(x)):
        # it will not work if there are any nans
        if decreasing is True:
            idxmin = np.where(x == min(x))[0][0]
            if idxmin < len(x) - 1:
                x[idxmin::] = min(x)
        elif decreasing is False:
            idxmax = np.where(x == max(x))[0][0]
            if idxmax < len(x) - 1:
                x[idxmax::] = max(x)
        else:
            return ValueError("The parameter 'decreasing' must be True or False")
    return x


def transform_spaced(
    transform: str,
    y_lower: float = 1e-8,
    y_upper: float = 1 - 1e-8,
    num: int = 1000,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
):
    """Creates linearly spaced array based on a specified transform.

    This is similar to np.linspace or np.logspace but is designed for weibull
    space, exponential space, normal space, gamma space, loglogistic space,
    gumbel space and beta space.

    It is useful if the points generated are going to be plotted on axes that
    are scaled using the same transform and need to look equally spaced in the
    transform space.

    Parameters
    ----------
    transform : str
        The transform name. Must be either weibull, exponential, normal, gamma,
        gumbel, loglogistic, or beta.
    y_upper : float, optional
        The lower bound (must be within the bounds 0 to 1). Default is 1e-8
    y_lower : float, optional
        The upper bound (must be within the bounds 0 to 1). Default is 1-1e-8
    num : int, optional
        The number of values in the array. Default is 1000.
    alpha : int, float, optional
        The alpha value of the beta distribution. Only used if the transform is
        beta
    beta : int, float, optional
        The beta value of the beta or gamma distribution. Only used if the
        transform is beta or gamma

    Returns
    -------
    transformed_array : array
        transform spaced array. This appears linearly spaced when plotted in
        transform space.

    Notes
    -----
    Note that lognormal is the same as normal, since the x-axis is what is
    transformed in lognormal, not the y-axis.

    """
    np.seterr("ignore")  # this is required due to an error in scipy.stats
    if y_lower > y_upper:
        y_lower, y_upper = y_upper, y_lower
    if y_lower <= 0 or y_upper >= 1:
        raise ValueError("y_lower and y_upper must be within the range 0 to 1")
    if num <= 2:
        raise ValueError("num must be greater than 2")
    if transform in ["normal", "Normal", "norm", "Norm"]:

        def fwd(x: float):
            return ss.norm.ppf(x)

        def inv(x: npt.NDArray[np.float64]):
            return ss.norm.cdf(x)

    elif transform in ["gumbel", "Gumbel", "gbl", "gum", "Gum", "Gbl"]:

        def fwd(x: float):
            return ss.gumbel_l.ppf(x)

        def inv(x: npt.NDArray[np.float64]):
            return ss.gumbel_l.cdf(x)

    elif transform in ["weibull", "Weibull", "weib", "Weib", "wbl"]:

        def fwd(x: float):
            return np.log(-np.log(1 - x))

        def inv(x: npt.NDArray[np.float64]): #type: ignore
            return 1 - np.exp(-np.exp(x))

    elif transform in ["loglogistic", "Loglogistic", "LL", "ll", "loglog"]:

        def fwd(x: float):
            return np.log(1 / x - 1)

        def inv(x: npt.NDArray[np.float64]): #type: ignore
            return 1 / (np.exp(x) + 1)

    elif transform in ["exponential", "Exponential", "expon", "Expon", "exp", "Exp"]:

        def fwd(x: float):
            return ss.expon.ppf(x)

        def inv(x: npt.NDArray[np.float64]):
            return ss.expon.cdf(x)

    elif transform in ["gamma", "Gamma", "gam", "Gam"]:
        if beta is None:
            raise ValueError("beta must be specified to use the gamma transform")
        else:

            def fwd(x: float):
                return ss.gamma.ppf(x, a=beta)

            def inv(x: npt.NDArray[np.float64]):
                return ss.gamma.cdf(x, a=beta)

    elif transform in ["beta", "Beta"]:
        if alpha is None or beta is None:
            raise ValueError("alpha and beta must be specified to use the beta transform")
        else:

            def fwd(x: float):
                return ss.beta.ppf(x, a=beta, b=alpha)

            def inv(x: npt.NDArray[np.float64]):
                return ss.beta.cdf(x, a=beta, b=alpha)

    elif transform in [
        "lognormal",
        "Lognormal",
        "LN",
        "ln",
        "lognorm",
        "Lognorm",
    ]:  # the transform is the same, it's just the xscale that is ln for lognormal
        raise ValueError("the Lognormal transform is the same as the normal transform. Specify normal and try again")
    else:
        raise ValueError("transform must be either exponential, normal, weibull, loglogistic, gamma, or beta")

    # find the value of the bounds in tranform space
    upper = fwd(y_upper)
    lower = fwd(y_lower)
    # generate the array in transform space
    arr: npt.NDArray[np.float64] = np.linspace(lower, upper, num, dtype=np.float64)
    # convert the array back from transform space
    transform_array = inv(arr)
    return transform_array


import numpy as np
import numpy.typing as npt
import scipy.stats as ss
from numpy.linalg import LinAlgError
from scipy.optimize import curve_fit, minimize  # type: ignore
from scipy.special import betainc, erf, gammainc  # type: ignore

from reliability.Utils._ancillary_utils import colorprint
from reliability.Utils._plot_utils import linear_regression


def least_squares(dist: str, failures: npt.NDArray[np.float64], right_censored: npt.NDArray[np.float64], method: str ="RRX", force_shape=None):
    """Uses least squares or non-linear least squares estimation to fit the
    parameters of the distribution to the plotting positions.

    Plotting positions are based on failures and right_censored so while least
    squares estimation does not consider the right_censored data in the same way
    as MLE, the plotting positions do. This means that right censored data are
    not ignored by least squares estimation, but the way the values are treated
    differs between least squares and MLE.

    The output of this method may be used as the initial guess for the MLE
    method.

    Parameters
    ----------
    dist : str
        The description of the distribution
    failures : array, list
        The failure data
    right_censored : array, list
        The right censored data. If there is no data then this should be an
        empty list.
    method : str, optional
        Must be "RRX" or "RRY". Default is RRX.
    force_shape : float, int, optional
        Used to force the shape (beta or sigma) parameter. Default is None which
        will not force the slope.

    Returns
    -------
    model_parameters : list
        The model's parameters in a list. eg. for "Weibull_2P" it will return
        [alpha,beta]. For "Weibull_3P" it will return [alpha,beta,gamma].

    Notes
    -----
    For more information on least squares estimation, see the `documentation <https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html>`_.

    For cases where the CDF is not linearizable (eg. Weibull_3P), this function
    uses non-linear least squares (NLLS) which uses scipy's curve_fit to find
    the parameters. This may sometimes fail as curve_fit is an optimization
    routine that needs an initial guess provided by this function.

    """
    if min(failures) <= 0 and dist not in ["Normal_2P", "Gumbel_2P"]:
        raise ValueError(
            "failures contains zeros or negative values which are only suitable when dist is Normal_2P or Gumbel_2P",
        )
    if max(failures) >= 1 and dist == "Beta_2P":
        raise ValueError(
            "failures contains values greater than or equal to one which is not allowed when dist is Beta_2P",
        )
    if force_shape is not None and dist not in [
        "Weibull_2P",
        "Normal_2P",
        "Lognormal_2P",
    ]:
        raise ValueError("force_shape can only be applied to Weibull_2P, Normal_2P, and Lognormal_2P")
    if method not in ["RRX", "RRY"]:
        raise ValueError('method must be either "RRX" or "RRY". Default is RRX.')

    from reliability.Probability_plotting import (
        plotting_positions,
    )

    x, y = plotting_positions(failures=failures, right_censored=right_censored)
    x: npt.NDArray[np.float64] = np.array(x)
    y: npt.NDArray[np.float64] = np.array(y)
    gamma0: np.float64 | float = (
        min(np.hstack([failures, right_censored])) - 0.001
    )  # initial guess for gamma when it is required for the 3P fitters
    if gamma0 < 0:
        gamma0 = 0.0

    if dist == "Weibull_2P":
        guess: list[np.float64]  = Weibull_2P_guess(x, y, method, force_shape)

    elif dist == "Weibull_3P":
        guess = Weibull_3P_guess(x, y, method, gamma0, failures, force_shape)

    elif dist == "Exponential_1P":
        guess = Exponential_1P_guess(x, y, method)

    elif dist == "Exponential_2P":
        guess = Exponential_2P_guess(x, y, method, gamma0, failures)

    elif dist == "Normal_2P":
        guess = Normal_2P_guess(x, y, method, force_shape)

    elif dist == "Gumbel_2P":
        guess = Gumbel_2P_guess(x, y, method)

    elif dist == "Lognormal_2P":
        guess = Lognormal_2P_guess(x, y, method, force_shape)

    elif dist == "Lognormal_3P":
        guess = Lognormal_3P_guess(x, y, method, force_shape, gamma0, failures)

    elif dist == "Loglogistic_2P":
        guess = Loglogistic_2P_guess(x, y, method)

    elif dist == "Loglogistic_3P":
        guess = Loglogistic_3P_guess(x, y, method, force_shape, gamma0, failures)

    elif dist == "Gamma_2P":
        guess = Gamma_2P_guess(x, y, method, failures)

    elif dist == "Gamma_3P":
        guess = Gamma_3P_guess(x, y, method, gamma0, failures)

    elif dist == "Beta_2P":
        guess = Beta_2P_guess(x, y, method, failures)
    else:
        raise ValueError('Unknown dist. Use the correct name. eg. "Weibull_2P"')
    return guess

def Weibull_2P_guess(x, y, method, force_shape) -> list[np.float64]:
    """
    Calculates the initial guess for the parameters of a 2-parameter Weibull distribution.

    Args:
        x (array-like): The x-values of the data points.
        y (array-like): The y-values of the data points.
        method (str): The method used for regression. Can be "RRX" or "RRY".
        force_shape (float or None): The shape parameter to be forced. If None, no shape parameter is forced.

    Returns:
        list[np.float64]: The initial guess for the parameters [alpha, beta] of the Weibull distribution.

    """
    xlin: npt.NDArray[np.float64] = np.log(x)
    ylin: npt.NDArray[np.float64] = np.log(-np.log(1 - y))
    if force_shape is not None and method == "RRX":
        force_shape = 1 / force_shape  # only needs to be inverted for RRX not RRY
    slope, intercept = linear_regression(xlin, ylin, slope=force_shape, RRX_or_RRY=method)
    LS_beta: np.float64 = slope
    LS_alpha: np.float64 = np.exp(-intercept / LS_beta)
    guess = [LS_alpha, LS_beta]
    return guess

def Weibull_3P_guess(x, y, method, gamma0, failures, force_shape) -> list[np.float64]:
    # Weibull_2P estimate to create the guess for Weibull_3P
    guess_x =  x - gamma0
    guess_2P = Weibull_2P_guess(x=guess_x, y=y, method=method, force_shape=force_shape)
    LS_alpha = guess_2P[0]
    LS_beta = guess_2P[1]

    # NLLS for Weibull_3P
    def __weibull_3P_CDF(t, alpha, beta, gamma):
        return 1 - np.exp(-(((t - gamma) / alpha) ** beta))

    try:
        curve_fit_bounds = (
            [0, 0, 0],
            [1e20, 1000, gamma0],
        )  # ([alpha_lower,beta_lower,gamma_lower],[alpha_upper,beta_upper,gamma_upper])
        popt, _ = curve_fit(
            __weibull_3P_CDF,
            x,
            y,
            p0=[LS_alpha, LS_beta, gamma0],
            bounds=curve_fit_bounds,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta,gamma]
        NLLS_alpha = popt[0]
        NLLS_beta = popt[1]
        NLLS_gamma = popt[2]
        guess = [NLLS_alpha, NLLS_beta, NLLS_gamma]
    except (ValueError, LinAlgError, RuntimeError):
        colorprint(
            "WARNING: Non-linear least squares for Weibull_3P failed. The result returned is an estimate that is likely to be incorrect.",
            text_color="red",
        )
        guess = [LS_alpha, LS_beta, gamma0]
    return guess

def Exponential_1P_guess(x, y, method) -> list[np.float64]:
    if method == "RRY":
        x_intercept = None
        y_intercept = 0
    elif method == "RRX":
        y_intercept = None
        x_intercept = 0

    ylin = -np.log(1 - y)
    slope, _ = linear_regression(
        x,
        ylin,
        x_intercept=x_intercept,
        y_intercept=y_intercept,
        RRX_or_RRY=method,
    )  # equivalent to y = m.x
    LS_Lambda = slope
    guess = [LS_Lambda]
    return guess

def Exponential_2P_guess(x, y, method, gamma0, failures) -> list[np.float64]:
    # Exponential_1P estimate to create the guess for Exponential_2P
    # while it is mathematically possible to use ordinary least squares (y=mx+c) for this, the LS method does not allow bounds on gamma. This can result in gamma > min(data) which should be impossible and will cause other errors.
    xlin = x - gamma0
    ylin = -np.log(1 - y)
    slope, _ = linear_regression(xlin, ylin, x_intercept=0, RRX_or_RRY="RRX")
    LS_Lambda = slope

    # NLLS for Exponential_2P
    def __exponential_2P_CDF(t, Lambda, gamma):
        return 1 - np.exp(-Lambda * (t - gamma))

    try:
        curve_fit_bounds = (
            [0, 0],
            [1e20, gamma0],
        )  # ([Lambda_lower,gamma_lower],[Lambda_upper,gamma_upper])
        popt, _ = curve_fit(
            __exponential_2P_CDF,
            x,
            y,
            p0=[LS_Lambda, gamma0],
            bounds=curve_fit_bounds,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )
        NLLS_Lambda = popt[0]
        NLLS_gamma = popt[1]
        guess = [NLLS_Lambda, NLLS_gamma]
    except (ValueError, LinAlgError, RuntimeError):
        colorprint(
            "WARNING: Non-linear least squares for Exponential_2P failed. The result returned is an estimate that is likely to be incorrect.",
            text_color="red",
        )
        guess = [LS_Lambda, gamma0]
    return guess

def Normal_2P_guess(x, y, method, force_shape) -> list[np.float64]:
    ylin = ss.norm.ppf(y)
    if force_shape is not None and method == "RRY":
        force_shape = 1 / force_shape  # only needs to be inverted for RRY not RRX
    slope, intercept = linear_regression(x, ylin, slope=force_shape, RRX_or_RRY=method)
    LS_sigma = 1 / slope
    LS_mu = -intercept * LS_sigma
    guess = [LS_mu, LS_sigma]
    return guess

def Gumbel_2P_guess(x, y, method) -> list[np.float64]:
    ylin = np.log(-np.log(1 - y))
    slope, intercept = linear_regression(x, ylin, RRX_or_RRY=method)
    LS_sigma = 1 / slope
    LS_mu = -intercept * LS_sigma
    guess = [LS_mu, LS_sigma]
    return guess

def Lognormal_2P_guess(x, y, method, force_shape) -> list[np.float64]:
    xlin = np.log(x)
    ylin = ss.norm.ppf(y)
    if force_shape is not None and method == "RRY":
        force_shape = 1 / force_shape  # only needs to be inverted for RRY not RRX
    slope, intercept = linear_regression(xlin, ylin, slope=force_shape, RRX_or_RRY=method)
    LS_sigma = 1 / slope
    LS_mu = -intercept * LS_sigma
    guess = [LS_mu, LS_sigma]
    return guess

def Lognormal_3P_guess(x, y, method, force_shape, gamma0, failures) -> list[np.float64]:
    # uses least squares to fit a normal distribution to the log of the data and minimizes the correlation coefficient (1 - R^2)
    def __gamma_optimizer(gamma_guess, x, y):
        xlin = np.log(x - gamma_guess)
        ylin = ss.norm.ppf(y)
        _, _, r, _, _ = ss.linregress(xlin, ylin)
        return 1 - (r**2)

    # NLLS for Normal_2P which is used by Lognormal_3P by taking the log of the data. This is more accurate than doing it with Lognormal_3P.
    def __normal_2P_CDF(t, mu, sigma):
        return (1 + erf(((t - mu) / sigma) / 2**0.5)) / 2

    res = minimize(
        __gamma_optimizer,
        gamma0,
        args=(x, y),
        method="TNC",
        bounds=[([0, gamma0])],
    )  # this obtains gamma
    gamma = res.x[0]

    try:
        curve_fit_bounds = (
            [-1e20, 0],
            [1e20, 1000],
        )  # ([mu_lower,sigma_lower],[mu_upper,sigma_upper])
        popt, _ = curve_fit(
            __normal_2P_CDF,
            np.log(x - gamma),
            y,
            p0=[np.mean(np.log(x - gamma)), np.std(np.log(x - gamma))],
            bounds=curve_fit_bounds,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )  # This is the non-linear least squares method. p0 is the initial guess for [mu,sigma].
        NLLS_mu = popt[0]
        NLLS_sigma = popt[1]
        guess = [NLLS_mu, NLLS_sigma, gamma]
    except (ValueError, LinAlgError, RuntimeError):
        colorprint(
            "WARNING: Non-linear least squares for Lognormal_3P failed. The result returned is an estimate that is likely to be incorrect.",
            text_color="red",
        )
        guess = [np.mean(np.log(x - gamma)), np.std(np.log(x - gamma)), gamma]
    return guess

def Loglogistic_2P_guess(x, y, method) -> list[np.float64]:
    xlin = np.log(x)
    ylin = np.log(1 / y - 1)
    slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
    LS_beta = -slope
    LS_alpha = np.exp(intercept / LS_beta)
    guess = [LS_alpha, LS_beta]
    return guess

def Loglogistic_3P_guess(x, y, method, force_shape, gamma0, failures) -> list[np.float64]:

    def __loglogistic_3P_CDF(t, alpha, beta, gamma):
        return 1 / (1 + ((t - gamma) / alpha) ** -beta)

    # Loglogistic_2P estimate to create the guess for Loglogistic_3P
    xlin = np.log(x - gamma0)
    ylin = np.log(1 / y - 1)
    slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
    LS_beta = -slope
    LS_alpha = np.exp(intercept / LS_beta)

    try:
        # Loglogistic_3P estimate
        curve_fit_bounds = (
            [0, 0, 0],
            [1e20, 1000, gamma0],
        )  # ([alpha_lower,beta_lower,gamma_lower],[alpha_upper,beta_upper,gamma_upper])
        popt, _ = curve_fit(
            __loglogistic_3P_CDF,
            x,
            y,
            p0=[LS_alpha, LS_beta, gamma0],
            bounds=curve_fit_bounds,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta,gamma].
        NLLS_alpha = popt[0]
        NLLS_beta = popt[1]
        NLLS_gamma = popt[2]
        guess = [NLLS_alpha, NLLS_beta, NLLS_gamma]
    except (ValueError, LinAlgError, RuntimeError):
        colorprint(
            "WARNING: Non-linear least squares for Loglogistic_3P failed. The result returned is an estimate that is likely to be incorrect.",
            text_color="red",
        )
        guess = [LS_alpha, LS_beta, gamma0]
    return guess

def Gamma_2P_guess(x, y, method, failures) -> list[np.float64]:

    def __gamma_2P_CDF(t, alpha, beta):
        return gammainc(beta, t / alpha)

    # Weibull_2P estimate which is converted to a Gamma_2P initial guess
    xlin = np.log(x)
    ylin = np.log(-np.log(1 - y))
    slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
    LS_beta = slope
    LS_alpha = np.exp(-intercept / LS_beta)

    # conversion of weibull parameters to gamma parameters. These values were found empirically and the relationship is only an approximate model
    beta_guess = abs(0.6932 * LS_beta**2 - 0.0908 * LS_beta + 0.2804)
    alpha_guess = abs(LS_alpha / (-0.00095 * beta_guess**2 + 1.1119 * beta_guess))

    def __perform_curve_fit():  # separated out for repeated use
        curve_fit_bounds = (
            [0, 0],
            [1e20, 1000],
        )  # ([alpha_lower,beta_lower],[alpha_upper,beta_upper])
        popt, _ = curve_fit(
            __gamma_2P_CDF,
            x,
            y,
            p0=[alpha_guess, beta_guess],
            bounds=curve_fit_bounds,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta]
        return [popt[0], popt[1]]

    try:
        # Gamma_2P estimate
        guess = __perform_curve_fit()
    except (ValueError, LinAlgError, RuntimeError):
        try:
            guess = __perform_curve_fit()
            # We repeat the same attempt at a curve_fit because of a very strange event.
            # When Fit_Gamma_2P is run twice in a row, the second attempt fails if there was a probability plot generated for the first attempt.
            # This was unable to debugged since the curve_fit has identical inputs each run and the curve_fit should not interact with the probability plot in any way.
            # One possible cause of this error may relate to memory usage though this is not confirmed.
            # By simply repeating the attempted curve_fit one more time, it often will work perfectly on the second try.
            # If it fails the second try then we report the failure and return the initial guess.
        except (ValueError, LinAlgError, RuntimeError):
            colorprint(
                "WARNING: Non-linear least squares for Gamma_2P failed. The result returned is an estimate that is likely to be incorrect.",
                text_color="red",
            )
            guess = [alpha_guess, beta_guess]
    return guess

def Gamma_3P_guess(x, y, method, gamma0, failures) -> list[np.float64]:

    def __gamma_2P_CDF(t, alpha, beta):
        return gammainc(beta, t / alpha)

    def __gamma_3P_CDF(t, alpha, beta, gamma):
        return gammainc(beta, (t - gamma) / alpha)

    # Weibull_2P estimate which is converted to a Gamma_2P initial guess
    xlin = np.log(x - gamma0 * 0.98)
    ylin = np.log(-np.log(1 - y))
    slope, intercept = linear_regression(xlin, ylin, RRX_or_RRY=method)
    LS_beta = slope
    LS_alpha = np.exp(-intercept / LS_beta)

    # conversion of weibull parameters to gamma parameters. These values were found empirically and the relationship is only an approximate model
    beta_guess = abs(0.6932 * LS_beta**2 - 0.0908 * LS_beta + 0.2804)
    alpha_guess = abs(LS_alpha / (-0.00095 * beta_guess**2 + 1.1119 * beta_guess))

    def __perform_curve_fit_gamma_2P():  # separated out for repeated use
        curve_fit_bounds = (
            [0, 0],
            [1e20, 1000],
        )  # ([alpha_lower,beta_lower],[alpha_upper,beta_upper])
        popt, _ = curve_fit(
            __gamma_2P_CDF,
            x - gamma0 * 0.98,
            y,
            p0=[alpha_guess, beta_guess],
            bounds=curve_fit_bounds,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta]
        return [popt[0], popt[1]]

    def __perform_curve_fit_gamma_3P():  # separated out for repeated use
        curve_fit_bounds_3P = (
            [0, 0, 0],
            [1e20, 1000, gamma0],
        )  # ([alpha_lower,beta_lower,gamma_lower],[alpha_upper,beta_upper,gamma_upper])
        popt, _ = curve_fit(
            __gamma_3P_CDF,
            x,
            y,
            p0=[NLLS_alpha_2P, NLLS_beta_2P, gamma0 * 0.98],
            bounds=curve_fit_bounds_3P,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta,gamma]
        return [popt[0], popt[1], popt[2]]

    try:
        # Gamma_2P estimate to create the guess for Gamma_3P
        guess_2P = __perform_curve_fit_gamma_2P()
        NLLS_alpha_2P, NLLS_beta_2P = guess_2P[0], guess_2P[1]
        try:
            # Gamma_3P estimate
            guess = __perform_curve_fit_gamma_3P()
        except (ValueError, LinAlgError, RuntimeError):
            try:
                # try gamma_3P a second time
                guess = __perform_curve_fit_gamma_3P()
            except (ValueError, LinAlgError, RuntimeError):
                colorprint(
                    "WARNING: Non-linear least squares for Gamma_3P failed during Gamma_3P optimization. The result returned is an estimate that is likely to be incorrect.",
                    text_color="red",
                )
                guess = [NLLS_alpha_2P, NLLS_beta_2P, gamma0 * 0.98]
    except (ValueError, LinAlgError, RuntimeError):
        # We repeat the same attempt at a curve_fit because of a very strange event.
        # When Fit_Gamma_3P is run twice in a row, the second attempt fails if there was a probability plot generated for the first attempt.
        # This was unable to debugged since the curve_fit has identical inputs each run and the curve_fit should not interact with the probability plot in any way.
        # One possible cause of this error may relate to memory usage though this is not confirmed.
        # By simply repeating the attempted curve_fit one more time, it often will work perfectly on the second try.
        # If it fails the second try then we report the failure and return the initial guess.
        try:
            guess_2P = __perform_curve_fit_gamma_2P()
            NLLS_alpha_2P, NLLS_beta_2P = guess_2P[0], guess_2P[1]
            try:
                # Gamma_3P estimate
                guess = __perform_curve_fit_gamma_3P()
            except (ValueError, LinAlgError, RuntimeError):
                try:
                    # try gamma_3P for a second time
                    guess = __perform_curve_fit_gamma_3P()
                except (ValueError, LinAlgError, RuntimeError):
                    colorprint(
                        "WARNING: Non-linear least squares for Gamma_3P failed during Gamma_3P optimization. The result returned is an estimate that is likely to be incorrect.",
                        text_color="red",
                    )
                    guess = [NLLS_alpha_2P, NLLS_beta_2P, gamma0 * 0.98]
        except (ValueError, LinAlgError, RuntimeError):
            colorprint(
                "WARNING: Non-linear least squares for Gamma_3P failed during Gamma_2P optimization. The result returned is an estimate that is likely to be incorrect.",
                text_color="red",
            )
            guess = [alpha_guess, beta_guess, gamma0 * 0.98]
    return guess

def Beta_2P_guess(x, y, method, failures) -> list[np.float64 | float]:

    def __beta_2P_CDF(t, alpha, beta):
        return betainc(alpha, beta, t)

    try:
        curve_fit_bounds = (
            [0, 0],
            [100, 100],
        )  # ([alpha_lower,beta_lower],[alpha_upper,beta_upper])
        popt, _ = curve_fit(
            __beta_2P_CDF,
            x,
            y,
            p0=[2, 1],
            bounds=curve_fit_bounds,
            jac="3-point",
            method="trf",
            max_nfev=300 * len(failures),
        )  # This is the non-linear least squares method. p0 is the initial guess for [alpha,beta]
        NLLS_alpha = popt[0]
        NLLS_beta = popt[1]
        guess = [NLLS_alpha, NLLS_beta]
    except (ValueError, LinAlgError, RuntimeError):
        colorprint(
            "WARNING: Non-linear least squares for Beta_2P failed. The result returned is an estimate that is likely to be incorrect.",
            text_color="red",
        )
        guess = [2.0, 1.0]
    return guess

def ALT_least_squares(model, failures, stress_1_array, stress_2_array=None):
    """Uses least squares estimation to fit the parameters of the ALT stress-life
    model to the time to failure data.

    Unlike least_squares for probability distributions, this function does not
    use the plotting positions because it is working on the life-stress model
    L(S) and not the life distribution F(t), so it uses the failure data directly.

    This function therefore only fits the parameters of the model to the failure
    data and does not take into account the right censored data. Right censored
    data is only used when fitting the life-stress distribution (eg. "Weibull
    Eyring") which is done using MLE.

    The output of this function is used as the initial guess for the MLE
    method for the life-stress distribution.

    Parameters
    ----------
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power"
    failures : array, list
        The failure data
    stress_1_array : list, array
        The stresses corresponding to the failure data.
    stress_2_array : list, array, optional
        The second stresses corresponding to the failure data. Used only for
        dual-stress models. Default is None.

    Returns
    -------
    model_parameters : list
        The model's parameters in a list. This depends on the model.
        Exponential - [a,b], Eyring - [a,c], Power - [a,n],
        Dual_Exponential - [a,b,c], Power_Exponential - [a,c,n],
        Dual_Power - [c,m,n]

    Notes
    -----
    For more information on least squares estimation, see the `documentation <https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html>`_.
    For more information on fitting ALT models, see the `documentation <https://reliability.readthedocs.io/en/latest/Equations%20of%20ALT%20models.html>`_.

    For models with more than two parameters, linear algebra is equally valid,
    but in these cases it is not finding the line of best fit, it is instead
    finding the plane of best fit.

    """

    def non_invertable_handler(xx, yy, model) -> list[np.float64]:
        """This subfunction performs the linear algebra to find the solution.
        It also handles the occasional case of a non-invertable matrix.
        This function is separated out for repeated use.

        Parameters
        ----------
        xx : list, array
            The x data
        yy : list, array
            The y data
        model : str
            The model. Used only for printing the correct error string.

        Returns
        -------
        model_parameters : list
            The model parameters in a list. These are the linearized form and
            need to be converted back to find the actual model parameters.

        """
        try:
            # linear regression formula for RRY
            out = np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)
        except LinAlgError:
            try:
                # if the matrix is perfectly symmetrical, it becomes non-invertable
                # in this case we introduce a small amount of noise so it can be inverted
                # this noise doesn't affect the result much and it doesn't really matter since it is just a guess being passed to MLE for optimization
                noise = np.random.normal(loc=0, scale=0.01, size=(3, 3))
                out = np.linalg.inv(xx.T.dot(xx) + noise).dot(xx.T).dot(yy)
            except LinAlgError:
                colorprint(
                    "WARNING: Least squares estimates failed for " + model + " model.",
                    text_color="red",
                )
                out = [1, 2, 3]  # return a dummy solution for MLE to deal with
        return out

    L: npt.NDArray[np.float64] = np.asarray(failures)
    S1: npt.NDArray[np.float64] = np.asarray(stress_1_array)
    S2: npt.NDArray[np.float64] | None = np.asarray(stress_2_array) if stress_2_array is not None else None
    if model == "Exponential":
        m, c = linear_regression(x=1 / S1, y=np.log(L), RRX_or_RRY="RRY")
        output = [m, np.exp(c)]  # a,b
    elif model == "Eyring":
        m, c = linear_regression(x=1 / S1, y=np.log(L) + np.log(S1), RRX_or_RRY="RRY")
        output = [m, -c]  # a,c
    elif model == "Power":
        m, c = linear_regression(x=np.log(S1), y=np.log(L), RRX_or_RRY="RRY")
        output = [np.exp(c), m]  # a,n
    elif model == "Dual_Exponential":
        X = 1 / S1
        Y = 1 / S2
        Z = np.log(L)
        yy = Z.T
        xx = np.array([np.ones_like(X), X, Y]).T
        # linear regression formula for RRY
        solution = non_invertable_handler(xx, yy, model)
        output = [solution[1], solution[2], np.exp(solution[0])]  # a,b,c
    elif model == "Power_Exponential":
        X = 1 / S1
        Y = np.log(S2)
        Z = np.log(L)
        yy = Z.T
        xx = np.array([np.ones_like(X), X, Y]).T
        # linear regression formula for RRY
        solution = non_invertable_handler(xx, yy, model)
        output = [solution[1], np.exp(solution[0]), solution[2]]  # a,c,n
    elif model == "Dual_Power":
        X = np.log(S1)
        Y = np.log(S2)
        Z = np.log(L)
        yy = Z.T
        xx = np.array([np.ones_like(X), X, Y]).T
        solution = non_invertable_handler(xx, yy, model)
        output = [np.exp(solution[0]), solution[1], solution[2]]  # c,m,n
    else:
        raise ValueError(
            "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power.",
        )
    return output

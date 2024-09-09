from __future__ import annotations

from typing import Literal

import numpy as np
from autograd import value_and_grad  # type: ignore
from scipy.optimize import minimize  # type: ignore

from reliability.Utils._ancillary_utils import colorprint
from reliability.Utils._statstic_utils import least_squares


class LS_optimization:
    """This function is a control function for least squares regression and it is
    used by each of the Fitters. There is no actual "optimization" done here,
    with the exception of checking which method (RRX or RRY) gives the better
    solution.

    Parameters
    ----------
    func_name : str
        The function to be fitted. Eg. "Weibull_2P".
    LL_func : function
        The log-likelihood function from the fitter
    failures : list, array
        The failure data
    right_censored : list, array
        The right censored data. If there is no right censored data then this
        should be an empty array.
    method : str, optional
        Must be either "RRX", "RRY", "LS", or "NLLS". Default is "LS".
    force_shape : float, int, optional
        The shape parameter to be forced. Default is None which results in no
        forcing of the shape parameter.
    LL_func_force : function
        The log-likelihood function for when the shape parameter is forced. Only
        required if force_shape is not None.

    Returns
    -------
    guess : list
        The guess of the models parameters. The length of this list depends on
        the number of parameters in the model. The guess is obtained using
        Utils.least_squares
    method : str
        The method used. This will be either "RRX", "RRY" or "NLLS".

    Notes
    -----
    If method="LS" then both "RRX" and "RRY" will be tried and the best one will
    be returned.

    """

    def __init__(
        self,
        func_name,
        LL_func,
        failures,
        right_censored,
        method: Literal["RRX", "RRY", "LS", "NLLS"] = "LS",
        force_shape=None,
        LL_func_force=None,
    ) -> None:
        if method not in ["RRX", "RRY", "LS", "NLLS"]:
            msg = "method must be either RRX, RRY, LS, or NLLS. Default is LS"
            raise ValueError(msg)
        if func_name in [
            "Weibull_3P",
            "Gamma_2P",
            "Gamma_3P",
            "Beta_2P",
            "Lognormal_3P",
            "Loglogistic_3P",
            "Exponential_2P",
        ]:
            guess: tuple[np.float64, ...] = least_squares(
                dist=func_name, failures=failures, right_censored=right_censored
            )
            LS_method = "NLLS"
        elif method in ["RRX", "RRY"]:
            guess = least_squares(
                dist=func_name,
                failures=failures,
                right_censored=right_censored,
                method=method,  # type: ignore
                force_shape=force_shape,
            )
            LS_method = method
        else:  # LS
            # RRX
            guess_RRX: tuple[np.float64, ...] = least_squares(
                dist=func_name,
                failures=failures,
                right_censored=right_censored,
                method="RRX",
                force_shape=force_shape,
            )
            if force_shape is not None and LL_func_force is not None:
                loglik_RRX = -LL_func_force(guess_RRX, failures, right_censored, force_shape)
            else:
                loglik_RRX = -LL_func(guess_RRX, failures, right_censored)
            # RRY
            guess_RRY: tuple[np.float64, ...] = least_squares(
                dist=func_name,
                failures=failures,
                right_censored=right_censored,
                method="RRY",
                force_shape=force_shape,
            )
            if force_shape is not None and LL_func_force is not None:
                loglik_RRY = -LL_func_force(guess_RRY, failures, right_censored, force_shape)
            else:
                loglik_RRY = -LL_func(guess_RRY, failures, right_censored)
            # take the best one
            if abs(loglik_RRX) < abs(loglik_RRY):  # RRX is best
                LS_method = "RRX"
                guess = guess_RRX
            else:  # RRY is best
                LS_method = "RRY"
                guess = guess_RRY
        self.guess: tuple[np.float64, ...] = guess
        self.method: Literal["NLLS", "RRX", "RRY", "LS"] = LS_method


class MLE_optimization:
    """This function performs Maximum Likelihood Estimation (MLE) to find the
    optimal parameters of the probability distribution. This functions is used
    by each of the fitters.

    Parameters
    ----------
    func_name : str
        The function to be fitted. Eg. "Weibull_2P".
    LL_func : function
        The log-likelihood function from the fitter
    initial_guess : list, array
        The initial guess of the model parameters that is used by the optimizer.
    failures : list, array
        The failure data
    right_censored : list, array
        The right censored data. If there is no right censored data then this
        should be an empty array.
    optimizer : str, None
        This must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best",
        "all" or None. Fot detail on how these optimizers are used, please see
        the `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    force_shape : float, int, optional
        The shape parameter to be forced. Default is None which results in no
        forcing of the shape parameter.
    LL_func_force : function
        The log-likelihood function for when the shape parameter is forced. Only
        required if force_shape is not None.

    Returns
    -------
    scale : float
        Only returned for Weibull_2P, Weibull_3P, Lognormal_2P, Lognormal_3P,
        Gamma_2P, Gamma_3P, Loglogistic_2P, Loglogistic_3P, Exponential_1P,
        Exponential_2P, Normal_2P, Beta_2P, and Gumbel_2P
    shape : float
        Only returned for Weibull_2P, Weibull_3P, Lognormal_2P, Lognormal_3P,
        Gamma_2P, Gamma_3P, Loglogistic_2P, Loglogistic_3P, Normal_2P, Beta_2P,
        and Gumbel_2P
    alpha : float
        Only returned for Weibull_DS, Weibull_ZI and Weibull_DSZI
    beta : float
        Only returned for Weibull_DS, Weibull_ZI and Weibull_DSZI
    gamma : float
        Only returned for Weibull_3P, Exponential_2P, Gamma_3P, Lognormal_3P,
        and Loglogistic_3P.
    DS : float
        Only returned for Weibull_DS and Weibull_DSZI
    ZI : float
        Only returned for Weibull_ZI and Weibull_DSZI
    alpha_1 : float
        Only returned for Weibull_mixture and Weibull_CR
    beta_1 : float
        Only returned for Weibull_mixture and Weibull_CR
    alpha_2 : float
        Only returned for Weibull_mixture and Weibull_CR
    beta_2 : float
        Only returned for Weibull_mixture and Weibull_CR
    proportion_1 : float
        Only returned for Weibull_mixture
    proportion_2 : float
        Only returned for Weibull_mixture
    success : bool
        Whether at least one optimizer succeeded. If False then the least
        squares result will be returned in place of the MLE result.
    optimizer : str, None
        The optimizer used. If MLE failed then None is returned as the
        optimizer.

    Notes
    -----
    Not all of the above returns are always returned. It depends on which model
    is being used.

    If the MLE method fails then the initial guess (from least squares) will be
    returned with a printed warning.

    """

    def __init__(
        self,
        func_name,
        LL_func,
        initial_guess,
        failures,
        right_censored,
        optimizer: str | None,
        force_shape=None,
        LL_func_force=None,
    ) -> None:
        # this sub-function does the actual optimization. It is called each time a new optimizer is tried
        def loglik_optimizer(
            LL_func,
            guess,
            failures,
            right_censored,
            bounds,
            optimizer: str | None,
            force_shape,
            LL_func_force,
            func_name,
        ):
            """This sub-function does the actual optimization. It is called each
            time a new optimizer is tried.

            Parameters
            ----------
            LL_func : function
                The log-likelihood function from the fitter
            guess : list, array
                The initial guess of the model parameters that is used by the optimizer.
            failures : list, array
                The failure data
            right_censored : list, array
                The right censored data. If there is no right censored data then this
                should be an empty array.
            bounds : list
                The bounds on the solution
            optimizer : str, None
                This must be either "TNC", "L-BFGS-B", "nelder-mead", or
                "powell".
            force_shape : float, int, optional
                The shape parameter to be forced. Default is None which results in no
                forcing of the shape parameter.
            LL_func_force : function
                The log-likelihood function for when the shape parameter is forced. Only
                required if force_shape is not None.
            func_name : str
                The function name. eg. "Weibull_2P"

            Returns
            -------
            success : bool
                Whether the optimizer was successful
            log_likelihood : float
                The log-likelihood of the solution
            model_parameters : array
                The model parameters of the solution

            Notes
            -----
            The returns are provided in a tuple of success, log_likelihood,
            model_parameters.

            """
            delta_LL = 1
            LL_array = [1000000]
            runs = 0

            ZI = func_name in ["Weibull_ZI", "Weibull_DSZI"]

            if ZI is True:  # Zero Inflated distribution (applies to ZI and DSZI)
                args = (failures[failures == 0], failures[failures > 0], right_censored)
            else:
                args = (failures, right_censored)

            if force_shape is None:
                EPSILON = 0.001
                MAX_RUNS = 5
                while delta_LL > EPSILON and runs < MAX_RUNS:
                    # exits after LL convergence or 5 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(LL_func),
                        guess,
                        args=args,
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    guess = result.x  # update the guess each iteration
                    if ZI is True:
                        LL2 = 2 * LL_func(
                            guess,
                            failures[failures == 0],
                            failures[failures > 0],
                            right_censored,
                        )
                    else:
                        LL2 = 2 * result.fun
                    LL_array.append(np.abs(LL2))
                    delta_LL: int = abs(LL_array[-1] - LL_array[-2])
            else:  # this will only be run for Weibull_2P, Normal_2P, and Lognormal_2P so the guess is structured with this in mind
                bounds = [bounds[0]]
                guess = [guess[0]]
                EPSILON = 0.001
                MAX_RUNS = 5
                while delta_LL > EPSILON and runs < MAX_RUNS:  # exits after LL convergence or 5 iterations
                    runs += 1
                    result = minimize(
                        value_and_grad(LL_func_force),
                        guess,
                        args=(failures, right_censored, force_shape),
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    guess = result.x
                    LL2 = 2 * result.fun
                    LL_array.append(np.abs(LL2))
                    delta_LL = abs(LL_array[-1] - LL_array[-2])
                    guess = result.x  # update the guess each iteration
            return result.success, LL_array[-1], result.x

        # generate the bounds on the solution
        gamma0: int = max(0, min(np.hstack([failures, right_censored])) - 0.0001)
        if func_name in ["Weibull_2P", "Gamma_2P", "Beta_2P", "Loglogistic_2P"]:
            bounds = [(0, None), (0, None)]
        elif func_name in ["Weibull_3P", "Gamma_3P", "Loglogistic_3P"]:
            bounds = [(0, None), (0, None), (0, gamma0)]
        elif func_name in ["Normal_2P", "Gumbel_2P", "Lognormal_2P"]:
            bounds = [(None, None), (0, None)]
        elif func_name == "Lognormal_3P":
            bounds = [(None, None), (0, None), (0, gamma0)]
        elif func_name == "Exponential_1P":
            bounds = [(0, None)]
        elif func_name == "Exponential_2P":
            bounds = [(0, None), (0, gamma0)]
        elif func_name == "Weibull_mixture":
            bounds = [
                (0.0001, None),
                (0.0001, None),
                (0.0001, None),
                (0.0001, None),
                (0.0001, 0.9999),
            ]
        elif func_name == "Weibull_CR":
            bounds = [(0.0001, None), (0.0001, None), (0.0001, None), (0.0001, None)]
        elif func_name == "Weibull_DSZI":
            bounds = [(0.0001, None), (0.0001, None), (0.00001, 1), (0, 0.99999)]
        elif func_name == "Weibull_DS":
            bounds = [(0.0001, None), (0.0001, None), (0.00001, 1)]
        elif func_name == "Weibull_ZI":
            bounds = [(0.0001, None), (0.0001, None), (0, 0.99999)]
        else:
            msg = 'func_name is not recognised. Use the correct name e.g. "Weibull_2P"'
            raise ValueError(msg)

        # determine which optimizers to use
        stop_after_success = False
        if optimizer is None:  # default is to try in this order but stop after one succeeds
            optimizers_to_try = ["L-BFGS-B", "TNC", "nelder-mead", "powell"]
            stop_after_success = True
        elif optimizer in [
            "best",
            "BEST",
            "all",
            "ALL",
        ]:  # try all of the bounded optimizers
            optimizers_to_try = ["TNC", "L-BFGS-B", "nelder-mead", "powell"]
        elif optimizer.upper() == "TNC":
            optimizers_to_try = ["TNC"]
        elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
            optimizers_to_try = ["L-BFGS-B"]
        elif optimizer.upper() == "POWELL":
            optimizers_to_try = ["powell"]
        elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD"]:
            optimizers_to_try = ["nelder-mead"]
        else:
            raise ValueError(
                str(
                    str(optimizer)
                    + ' is not a valid optimizer. Please specify either "TNC", "L-BFGS-B", "nelder-mead", "powell" or "best".',
                ),
            )

        # use each of the optimizers specified
        ALL_successes = []
        ALL_loglik = []
        ALL_results = []
        ALL_opt_names = []
        optimizers_tried_str = "Optimizers tried:"
        for opt in optimizers_to_try:
            optim_results = loglik_optimizer(
                LL_func,
                initial_guess,
                failures,
                right_censored,
                bounds,
                opt,
                force_shape,
                LL_func_force,
                func_name,
            )
            ALL_successes.append(optim_results[0])
            ALL_loglik.append(optim_results[1])
            ALL_results.append(optim_results[2])
            ALL_opt_names.append(opt)
            optimizers_tried_str = optimizers_tried_str + " " + opt + ","
            if optim_results[0] is True and stop_after_success is True:
                break  # stops after it finds one that works
        optimizers_tried_str = optimizers_tried_str[0:-1]  # remove the last comma
        # extract the results
        if True not in ALL_successes:
            # everything failed, need to return the initial guess
            self.success = False
            self.optimizer = None
            if func_name == "Weibull_mixture":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_mixture. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha_1 = initial_guess[0]
                self.beta_1 = initial_guess[1]
                self.alpha_2 = initial_guess[2]
                self.beta_2 = initial_guess[3]
                self.proportion_1 = initial_guess[4]
                self.proportion_2 = 1 - initial_guess[4]
            elif func_name == "Weibull_CR":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_CR. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha_1 = initial_guess[0]
                self.beta_1 = initial_guess[1]
                self.alpha_2 = initial_guess[2]
                self.beta_2 = initial_guess[3]
            elif func_name == "Weibull_DSZI":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_DSZI. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha = initial_guess[0]
                self.beta = initial_guess[1]
                self.DS = initial_guess[2]
                self.ZI = initial_guess[3]
            elif func_name == "Weibull_DS":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_DS. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha = initial_guess[0]
                self.beta = initial_guess[1]
                self.DS = initial_guess[2]
            elif func_name == "Weibull_ZI":
                colorprint(
                    "WARNING: MLE estimates failed for Weibull_ZI. The initial estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                    text_color="red",
                )
                self.alpha = initial_guess[0]
                self.beta = initial_guess[1]
                self.ZI = initial_guess[2]
            else:
                colorprint(
                    str(
                        "WARNING: MLE estimates failed for "
                        + func_name
                        + ". The least squares estimates have been returned. These results may not be as accurate as MLE. "
                        + optimizers_tried_str,
                    ),
                    text_color="red",
                )
                if force_shape is None:
                    self.scale = initial_guess[0]  # alpha, mu, Lambda
                    if func_name not in ["Exponential_1P", "Exponential_2P"]:
                        self.shape = initial_guess[1]  # beta, sigma
                    elif func_name == "Exponential_2P":
                        self.gamma = initial_guess[1]  # gamma for Exponential_2P
                    if func_name in [
                        "Weibull_3P",
                        "Gamma_3P",
                        "Loglogistic_3P",
                        "Lognormal_3P",
                    ]:
                        # gamma for Weibull_3P, Gamma_3P, Loglogistic_3P, Lognormal_3P
                        self.gamma = initial_guess[2]
                # this will only be reached for Weibull_2P, Normal_2P and Lognormal_2P so the scale and shape extraction is fine for these
                else:
                    self.scale = initial_guess[0]
                    self.shape = force_shape
        else:
            # at least one optimizer succeeded. Need to drop the failed ones then get the best of the successes
            items = np.arange(0, len(ALL_successes))[::-1]

            for i in items:
                if ALL_successes[i] is not True:
                    ALL_successes.pop(i)
                    ALL_loglik.pop(i)
                    ALL_results.pop(i)
                    ALL_opt_names.pop(i)
            idx_best = ALL_loglik.index(min(ALL_loglik))
            params = ALL_results[idx_best]
            self.optimizer: str = ALL_opt_names[idx_best]
            self.success = True

            if func_name == "Weibull_mixture":
                self.alpha_1 = params[0]
                self.beta_1 = params[1]
                self.alpha_2 = params[2]
                self.beta_2 = params[3]
                self.proportion_1 = params[4]
                self.proportion_2 = 1 - params[4]
            elif func_name == "Weibull_CR":
                self.alpha_1 = params[0]
                self.beta_1 = params[1]
                self.alpha_2 = params[2]
                self.beta_2 = params[3]
            elif func_name == "Weibull_DSZI":
                self.alpha = params[0]
                self.beta = params[1]
                self.DS = params[2]
                self.ZI = params[3]
            elif func_name == "Weibull_DS":
                self.alpha = params[0]
                self.beta = params[1]
                self.DS = params[2]
            elif func_name == "Weibull_ZI":
                self.alpha = params[0]
                self.beta = params[1]
                self.ZI = params[2]
            elif force_shape is None:
                self.scale = params[0]  # alpha, mu, Lambda
                if func_name not in ["Exponential_1P", "Exponential_2P"]:
                    self.shape = params[1]  # beta, sigma
                elif func_name == "Exponential_2P":
                    self.gamma = params[1]  # gamma for Exponential_2P
                if func_name in [
                    "Weibull_3P",
                    "Gamma_3P",
                    "Loglogistic_3P",
                    "Lognormal_3P",
                ]:
                    self.gamma = params[2]  # gamma for Weibull_3P, Gamma_3P, Loglogistic_3P, Lognormal_3P
            else:  # this will only be reached for Weibull_2P, Normal_2P and Lognormal_2P so the scale and shape extraction is fine for these
                self.scale = params[0]
                self.shape = force_shape


class ALT_MLE_optimization:
    """This performs the MLE method to find the parameters.
    If the optimizer is None then all bounded optimizers will be tried and the
    best result (lowest log-likelihood) will be returned. If the optimizer is
    specified then it will be used. If it fails then the initial guess will be
    returned with a warning.

    Parameters
    ----------
    model : str
        Must be either "Exponential", "Eyring", "Power", "Dual_Exponential",
        "Power_Exponential", or "Dual_Power".
    dist : str
        Must be either "Weibull", "Exponential", "Lognormal", or "Normal".
    LL_func : function
        The log-likelihood function from the fitter
    initial_guess : list, array
        The initial guess of the model parameters that is used by the
        optimizer.
    optimizer : str, None
        This must be either "TNC", "L-BFGS-B", "nelder-mead", "powell", "best",
        "all" or None. Fot detail on how these optimizers are used, please see
        the `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.
    failures : list, array
        The failure data
    right_censored : list, array
        The right censored data. If there is no right censored data then this
        should be an empty array.
    failure_stress_1 : array, list
        The failure stresses.
    failure_stress_2 : array, list
        The failure second stresses. This is only used for daul stress
        models.
    right_censored_stress_1 : array, list
        The right censored stresses. If there is no right censored data
        then this should be an empty array.
    right_censored_stress_2 : array, list
        The right censored second stresses. If there is no right
        censored data then this should be an empty array. This is only
        used for daul stress models.

    Returns
    -------
    a : float
        Only returned for Exponential, Eyring, Power, Dual_exponential, and
        Power_Exponential
    b : float
        Only returned for Exponential and Dual_Exponential
    c : float
        Only returned for Eyring, Dual_Exponential, Power_Exponential and
        Dual_Power
    n : float
        Only returned for Power, Power_Exponential, and Dual_Power
    m : float
        Only returned for Dual_Power
    beta : float
        Only returned for Weibull models
    sigma : float
        Only returned for Normal and Lognormal models
    success : bool
        Whether at least one optimizer succeeded. If False then the least
        squares result will be returned in place of the MLE result.
    optimizer : str, None
        The optimizer used. If MLE failed then None is returned as the
        optimizer.

    Notes
    -----
    Not all of the above returns are always returned. It depends on which model
    is being used.

    If the MLE method fails then the initial guess (from least squares) will be
    returned with a printed warning.

    """

    def __init__(
        self,
        model,
        dist,
        LL_func,
        initial_guess,
        optimizer: str | None,
        failures,
        failure_stress_1,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
    ) -> None:
        def loglik_optimizer(
            initial_guess,
            dual_stress,
            LL_func,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
            bounds,
            optimizer,
        ):
            """This sub-function does the actual optimization. It is called each
            time a new optimizer is tried.

            Parameters
            ----------
            initial_guess : list, array
                The initial guess of the model parameters that is used by the
                optimizer.
            dual_stress : bool
                Whether this is a dual_stress model.
            LL_func : function
                The log-likelihood function from the fitter
            failures : list, array
                The failure data
            right_censored : list, array
                The right censored data. If there is no right censored data then this
                should be an empty array.
            failure_stress_1 : array, list
                The failure stresses.
            failure_stress_2 : array, list
                The failure second stresses. This is only used for daul stress
                models.
            right_censored_stress_1 : array, list
                The right censored stresses. If there is no right censored data
                then this should be an empty array.
            right_censored_stress_2 : array, list
                The right censored second stresses. If there is no right
                censored data then this should be an empty array. This is only
                used for daul stress models.
            bounds : list
                The bounds on the solution
            optimizer : str, None
                This must be either "TNC", "L-BFGS-B", "nelder-mead", or
                "powell".

            Returns
            -------
            success : bool
                Whether the optimizer was successful
            log_likelihood : float
                The log-likelihood of the solution
            model_parameters : array
                The model parameters of the solution

            Notes
            -----
            The returns are provided in a tuple of success, log_likelihood,
            model_parameters.

            """
            delta_LL = 1
            LL_array: list[int] = [1000000]
            runs = 0
            guess = initial_guess  # set the current guess as the initial guess and then update the current guess each iteration
            EPSILON = 0.001
            MAX_ITERATIONS = 5
            while delta_LL > EPSILON and runs < MAX_ITERATIONS:  # exits after BIC convergence or 5 iterations
                runs += 1
                # single stress model
                if dual_stress is False:
                    result = minimize(
                        value_and_grad(LL_func),
                        guess,
                        args=(
                            failures,
                            right_censored,
                            failure_stress_1,
                            right_censored_stress_1,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    LL2 = -LL_func(
                        result.x,
                        failures,
                        right_censored,
                        failure_stress_1,
                        right_censored_stress_1,
                    )
                else:
                    # dual stress model
                    result = minimize(
                        value_and_grad(LL_func),
                        guess,
                        args=(
                            failures,
                            right_censored,
                            failure_stress_1,
                            failure_stress_2,
                            right_censored_stress_1,
                            right_censored_stress_2,
                        ),
                        jac=True,
                        method=optimizer,
                        bounds=bounds,
                    )
                    LL2 = -LL_func(
                        result.x,
                        failures,
                        right_censored,
                        failure_stress_1,
                        failure_stress_2,
                        right_censored_stress_1,
                        right_censored_stress_2,
                    )
                LL_array.append(np.abs(LL2))
                delta_LL = abs(LL_array[-1] - LL_array[-2])
                guess = result.x  # update the guess each iteration
            return result.success, LL_array[-1], result.x

        if model == "Exponential":
            bounds = [(None, None), (0, None), (0, None)]  # a, b, shape
            dual_stress = False
        elif model == "Eyring":
            bounds = [(None, None), (None, None), (0, None)]  # a, c, shape
            dual_stress = False
        elif model == "Power":
            bounds = [(0, None), (None, None), (0, None)]  # a, n, shape
            dual_stress = False
        elif model == "Dual_Exponential":
            bounds = [
                (None, None),
                (None, None),
                (0, None),
                (0, None),
            ]  # a, b, c, shape
            dual_stress = True
        elif model == "Power_Exponential":
            bounds = [
                (None, None),
                (0, None),
                (None, None),
                (0, None),
            ]  # a, c, n, shape
            dual_stress = True
        elif model == "Dual_Power":
            bounds = [
                (0, None),
                (None, None),
                (None, None),
                (0, None),
            ]  # c, m, n, shape
            dual_stress = True
        else:
            msg = "model must be one of Exponential, Eyring, Power, Dual_Exponential, Power_Exponential, Dual_Power"
            raise ValueError(
                msg,
            )

        if dist not in ["Weibull", "Exponential", "Lognormal", "Normal"]:
            msg = "dist must be one of Weibull, Exponential, Lognormal, Normal."
            raise ValueError(msg)

        # remove the last bound as Exponential does not need a bound for shape
        if dist == "Exponential":
            bounds = bounds[0:-1]

        if right_censored is None:
            right_censored = []
            right_censored_stress_1 = []
            right_censored_stress_2 = []

        # determine which optimizers to use
        stop_after_success = False
        if optimizer is None:  # default is to try in this order but stop after one succeeds
            optimizers_to_try = ["L-BFGS-B", "TNC", "nelder-mead", "powell"]
            stop_after_success = True
        elif optimizer in [
            "best",
            "BEST",
            "all",
            "ALL",
        ]:  # try all of the bounded optimizers
            optimizers_to_try = ["L-BFGS-B", "TNC", "nelder-mead", "powell"]
        elif optimizer.upper() == "TNC":
            optimizers_to_try = ["TNC"]
        elif optimizer.upper() in ["L-BFGS-B", "LBFGSB"]:
            optimizers_to_try = ["L-BFGS-B"]
        elif optimizer.upper() == "POWELL":
            optimizers_to_try = ["powell"]
        elif optimizer.upper() in ["NELDER-MEAD", "NELDERMEAD"]:
            optimizers_to_try = ["nelder-mead"]
        else:
            raise ValueError(
                str(
                    str(optimizer)
                    + ' is not a valid optimizer. Please specify either "TNC", "L-BFGS-B", "nelder-mead", "powell" or "best".',
                ),
            )

        # use each of the optimizers specified
        ALL_successes = []
        ALL_loglik = []
        ALL_results = []
        ALL_opt_names = []
        optimizers_tried_str = "Optimizers tried:"
        for opt in optimizers_to_try:
            optim_results = loglik_optimizer(
                initial_guess,
                dual_stress,
                LL_func,
                failures,
                right_censored,
                failure_stress_1,
                failure_stress_2,
                right_censored_stress_1,
                right_censored_stress_2,
                bounds,
                opt,
            )
            ALL_successes.append(optim_results[0])
            ALL_loglik.append(optim_results[1])
            ALL_results.append(optim_results[2])
            ALL_opt_names.append(opt)
            optimizers_tried_str = optimizers_tried_str + " " + opt + ","
            if optim_results[0] is True and stop_after_success is True:
                break  # stops after it finds one that works
        optimizers_tried_str = optimizers_tried_str[0:-1]  # remove the last comma

        # extract the results
        if True not in ALL_successes:
            # everything failed, need to return the initial guess
            self.success = False
            self.optimizer = None
            colorprint(
                str(
                    "WARNING: MLE estimates failed for "
                    + dist
                    + "_"
                    + model
                    + ". The least squares estimates have been returned. These results may not be as accurate as MLE. "
                    + optimizers_tried_str,
                ),
                text_color="red",
            )

            if model == "Exponential":
                self.a = initial_guess[0]
                self.b = initial_guess[1]
            elif model == "Eyring":
                self.a = initial_guess[0]
                self.c = initial_guess[1]
            elif model == "Power":
                self.a = initial_guess[0]
                self.n = initial_guess[1]
            elif model == "Dual_Exponential":
                self.a = initial_guess[0]
                self.b = initial_guess[1]
                self.c = initial_guess[2]
            elif model == "Power_Exponential":
                self.a = initial_guess[0]
                self.c = initial_guess[1]
                self.n = initial_guess[2]
            elif model == "Dual_Power":
                self.c = initial_guess[0]
                self.m = initial_guess[1]
                self.n = initial_guess[2]

            if dual_stress is False:
                if dist == "Weibull":
                    self.beta = initial_guess[2]
                elif dist in ["Lognormal", "Normal"]:
                    self.sigma = initial_guess[2]
            elif dist == "Weibull":
                self.beta = initial_guess[3]
            elif dist in ["Lognormal", "Normal"]:
                self.sigma = initial_guess[3]
        else:
            # at least one optimizer succeeded. Need to drop the failed ones then get the best of the successes
            items = np.arange(0, len(ALL_successes))[::-1]
            for i in items:
                if ALL_successes[i] is not True:
                    ALL_successes.pop(i)
                    ALL_loglik.pop(i)
                    ALL_results.pop(i)
                    ALL_opt_names.pop(i)
            idx_best = ALL_loglik.index(min(ALL_loglik))
            params = ALL_results[idx_best]
            self.optimizer: str = ALL_opt_names[idx_best]
            self.success = True

            if model == "Exponential":
                self.a = params[0]
                self.b = params[1]
            elif model == "Eyring":
                self.a = params[0]
                self.c = params[1]
            elif model == "Power":
                self.a = params[0]
                self.n = params[1]
            elif model == "Dual_Exponential":
                self.a = params[0]
                self.b = params[1]
                self.c = params[2]
            elif model == "Power_Exponential":
                self.a = params[0]
                self.c = params[1]
                self.n = params[2]
            elif model == "Dual_Power":
                self.c = params[0]
                self.m = params[1]
                self.n = params[2]

            if dual_stress is False:
                if dist == "Weibull":
                    self.beta = params[2]
                elif dist in ["Lognormal", "Normal"]:
                    self.sigma = params[2]
            elif dist == "Weibull":
                self.beta = params[3]
            elif dist in ["Lognormal", "Normal"]:
                self.sigma = params[3]

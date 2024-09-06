from typing import Literal

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss
from autograd.differential_operators import hessian
from autograd.scipy.special import erf
from matplotlib.axes._axes import Axes
from numpy.linalg import LinAlgError

from reliability.Distributions import (
    Normal_Distribution,
)
from reliability.Fitters import Fit_Normal_2P
from reliability.Utils import (
    ALT_least_squares,
    ALT_MLE_optimization,
    ALT_prob_plot,
    alt_fitters_dual_stress_input_checking,
    alt_single_stress_fitters_input_checking,
    colorprint,
    life_stress_plot,
    round_and_string,
)

pd.set_option("display.width", 200)  # prevents wrapping after default 80 characters
pd.set_option("display.max_columns", 9)  # shows the dataframe without ... truncation
shape_change_threshold = 0.5


class Fit_Normal_Exponential:
    """This function will Fit the Normal-Exponential life-stress model to the data
    provided. Please see the online documentation for the equations of this
    model.

    This model is most appropriate to model a life-stress relationship with
    temperature. It is recommended that you ensure your temperature data are in
    Kelvin.

    If you are using this model for the Arrhenius equation, a = Ea/K_B. When
    results are printed Ea will be provided in eV.

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress : array, list
        The corresponding stresses (such as temperature) at which each failure
        occurred. This must match the length of failures as each failure is
        tied to a failure stress.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    right_censored_stress : array, list, optional
        The corresponding stresses (such as temperature) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a
        right_censored stress. Conditionally optional input. This must be
        provided if right_censored is provided.
    use_level_stress : int, float, optional
        The use level stress at which you want to know the mean life. Optional
        input.
    print_results : bool, optional
        True/False. Default is True. Prints the results to the console.
    show_probability_plot : bool, object, optional
        True/False. Default is True. Provides a probability plot of the fitted
        ALT model. If an axes object is passed it will be used.
    show_life_stress_plot : bool, str, object, optional
        If True the life-stress plot will be shown. To hide the life-stress
        plot use False. To swap the axes and show a stress-life plot use
        'swap'. If an axes handle is passed it will be used. Default is True.
    CI : float, optional
        Confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the
        `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.

    Returns
    -------
    a : float
        The fitted parameter from the Exponential model
    b : float
        The fitted parameter from the Exponential model
    sigma : float
        The fitted Normal_2P sigma parameter
    loglik2 : float
        Log Likelihood*-2 (as used in JMP Pro)
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    a_SE : float
        The standard error (sqrt(variance)) of the parameter
    b_SE : float
        The standard error (sqrt(variance)) of the parameter
    sigma_SE : float
        The standard error (sqrt(variance)) of the parameter
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    b_upper : float
        The upper CI estimate of the parameter
    b_lower : float
        The lower CI estimate of the parameter
    sigma_upper : float
        The upper CI estimate of the parameter
    sigma_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (mu and sigma) at
        each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    mu_at_use_stress : float
        The equivalent Normal mu parameter at the use level stress (only
        provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Normal distribution at the use level stress (only provided if
        use_level_stress is provided).
    probability_plot : object
        The figure object from the probability plot (only provided if
        show_probability_plot is True).
    life_stress_plot : object
        The figure object from the life-stress plot (only provided if
        show_life_stress_plot is True).

    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress: float | None = None,
        CI: float = 0.95,
        optimizer: str | None = None,
    ):
        inputs = alt_single_stress_fitters_input_checking(
            dist="Normal",
            life_stress_model="Exponential",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Exponential.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess: list[np.float64] = ALT_least_squares(
            model="Exponential", failures=failures, stress_1_array=failure_stress
        )
        if right_censored_groups is None:
            right_censored_groups = []
        # obtain the common shape parameter
        fits: list[Fit_Normal_2P | np.float64] = [
            Fit_Normal_2P(failures=f, right_censored=rc) if len(f) > 1 else np.float64(np.nan)
            for f, rc in zip(failure_groups, right_censored_groups)
        ]
        sigmas: list[np.float64] = [
            fit.sigma if len(f) > 1 else np.float64(np.nan) for f, fit in zip(failure_groups, fits)
        ]
        mus_for_change_df: list[np.float64] = [
            fit.mu if len(f) > 1 else np.float64(np.nan) for f, fit in zip(failure_groups, fits)
        ]

        common_sigma: float | Literal[1] = (
            float(np.nanmean(sigmas)) if len(sigmas) > 0 else 1
        )  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess: list[np.float64 | float] = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, b, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Exponential",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a: np.float64 = MLE_results.a
        self.b: np.float64 = MLE_results.b
        self.sigma: np.float64 = MLE_results.sigma
        self.success: bool = MLE_results.success
        self.optimizer: str = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z: np.float64 = -ss.norm.ppf((1 - CI) / 2)
        params: list[np.float64] = [self.a, self.b, self.sigma]
        hessian_matrix = hessian(LL_func)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.a_SE: np.float64 = abs(covariance_matrix[0][0]) ** 0.5
            self.b_SE: np.float64 = abs(covariance_matrix[1][1]) ** 0.5
            self.sigma_SE: np.float64 = abs(covariance_matrix[2][2]) ** 0.5
            # a can be positive or negative
            self.a_upper: np.float64 = self.a + (Z * self.a_SE)
            self.a_lower: np.float64 = self.a + (-Z * self.a_SE)
            # b is strictly positive
            self.b_upper: np.float64 = self.b * (np.exp(Z * (self.b_SE / self.b)))
            self.b_lower: np.float64 = self.b * (np.exp(-Z * (self.b_SE / self.b)))
            # sigma is strictly positive
            self.sigma_upper: np.float64 = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower: np.float64 = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Normal_Exponential model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0.0
            self.b_SE = 0.0
            self.sigma_SE = 0.0
            self.a_upper = self.a
            self.a_lower = self.a
            self.b_upper = self.b
            self.b_lower = self.b
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "sigma"],
            "Point Estimate": [self.a, self.b, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n: int = len(failures) + len(right_censored)
        k: int = len(guess)
        LL2: np.float64 = 2 * LL_func(params, failures, right_censored, failure_stress, right_censored_stress)
        self.loglik2: np.float64 = LL2
        self.loglik: np.float64 = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc: np.float64 = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC: np.float64 = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress: np.float64 = self.life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma)
            self.mean_life: np.float64 = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus: npt.NDArray[np.float64] = self.life_func(S1=stresses_for_groups)
        AF: npt.NDArray[np.float64] | None = (
            self.life_func(S1=use_level_stress) / new_mus if use_level_stress is not None else None
        )
        common_sigmas: npt.NDArray[np.float64] = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded: bool = False
        for i in range(len(stresses_for_groups)):
            if isinstance(sigmas[i], float) and np.isnan(sigmas[i]):
                sigma_differences.append(np.nan)
            else:
                sigma_diff = (common_sigmas[i] - sigmas[i]) / sigmas[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                sigma_differences.append(f"{sigma_diff * 100:+.2f}%")

        self.__mus_for_change_df: list[np.float64 | float] = mus_for_change_df
        self.__sigmas_for_change_df: list[np.float64 | float] = sigmas

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )
        self.__failures = failures
        self.__right_censored = right_censored
        self.__CI: float = CI
        self.__shape_change_exceeded: bool = shape_change_exceeded
        self.__use_level_stress: float | None = use_level_stress

    def print_results(self):
        n: int = len(self.__failures) + len(self.__right_censored)
        CI_rounded: float = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored = len(self.__right_censored) / n * 100
        if frac_censored % 1 < 1e-10:
            frac_censored = int(frac_censored)
        colorprint(
            str("Results from Fit_Normal_Exponential (" + str(CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print("Analysis method: Maximum Likelihood Estimation (MLE)")
        if self.optimizer is not None:
            print("Optimizer:", self.optimizer)
        print(
            "Failures / Right censored:",
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.change_of_parameters.to_string(index=False))
        if self.__shape_change_exceeded is True:
            print(
                str(
                    "The sigma parameter has been found to change significantly (>"
                    + str(int(shape_change_threshold * 100))
                    + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate.",
                ),
            )
        print("\n", self.goodness_of_fit.to_string(index=False), "\n")
        print(
            "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
            round(self.a * 8.617333262145 * 10**-5, 5),
            "eV\n",
        )

        if self.__use_level_stress is not None:
            print(
                str(
                    "At the use level stress of "
                    + round_and_string(self.__use_level_stress)
                    + ", the mean life is "
                    + str(round(self.mean_life, 5))
                    + "\n",
                ),
            )

    def probability_plot(self, ax: bool | Axes = True):
        return ALT_prob_plot(
            dist="Normal",
            model="Exponential",
            stresses_for_groups=self.__stresses_for_groups,
            failure_groups=self.__failure_groups,
            right_censored_groups=self.__right_censored_groups,
            life_func=self.life_func,
            shape=self.sigma,
            scale_for_change_df=self.__mus_for_change_df,
            shape_for_change_df=self.__sigmas_for_change_df,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_stress_plot(self, ax: bool | Axes = True):
        return life_stress_plot(
            dist="Normal",
            model="Exponential",
            life_func=self.life_func,
            failure_groups=self.__failure_groups,
            stresses_for_groups=self.__stresses_for_groups,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_func(self, S1):
        return self.b * np.exp(self.a / S1)

    @staticmethod
    def logf(t, T, a, b, sigma):  # Log PDF
        life = b * anp.exp(a / T)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(sigma * (2 * anp.pi) ** 0.5)

    @staticmethod
    def logR(t, T, a, b, sigma):  # Log SF
        life = b * anp.exp(a / T)
        return anp.log((1 + erf(((life - t) / sigma) / 2**0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Normal_Exponential.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Normal_Exponential.logR(t_rc, T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Eyring:
    """This function will Fit the Normal-Eyring life-stress model to the data
    provided. Please see the online documentation for the equations of this
    model.

    This model is most appropriate to model a life-stress relationship with
    temperature. It is recommended that you ensure your temperature data are in
    Kelvin.

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress : array, list
        The corresponding stresses (such as temperature) at which each failure
        occurred. This must match the length of failures as each failure is
        tied to a failure stress.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    right_censored_stress : array, list, optional
        The corresponding stresses (such as temperature) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a
        right_censored stress. Conditionally optional input. This must be
        provided if right_censored is provided.
    use_level_stress : int, float, optional
        The use level stress at which you want to know the mean life. Optional
        input.
    CI : float, optional
        Confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the
        `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.

    Returns
    -------
    a : float
        The fitted parameter from the Eyring model
    c : float
        The fitted parameter from the Eyring model
    sigma : float
        The fitted Normal_2P sigma parameter
    loglik2 : float
        Log Likelihood*-2 (as used in JMP Pro)
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    a_SE : float
        The standard error (sqrt(variance)) of the parameter
    c_SE : float
        The standard error (sqrt(variance)) of the parameter
    sigma_SE : float
        The standard error (sqrt(variance)) of the parameter
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    c_upper : float
        The upper CI estimate of the parameter
    c_lower : float
        The lower CI estimate of the parameter
    sigma_upper : float
        The upper CI estimate of the parameter
    sigma_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (mu and sigma) at
        each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    mu_at_use_stress : float
        The equivalent Normal mu parameter at the use level stress (only
        provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Normal distribution at the use level stress (only provided if
        use_level_stress is provided).
    probability_plot : object
        The figure object from the probability plot (only provided if
        show_probability_plot is True).
    life_stress_plot : object
        The figure object from the life-stress plot (only provided if
        show_life_stress_plot is True).

    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress: float | None = None,
        CI=0.95,
        optimizer=None,
    ):
        inputs = alt_single_stress_fitters_input_checking(
            dist="Normal",
            life_stress_model="Eyring",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Eyring.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(model="Eyring", failures=failures, stress_1_array=failure_stress)

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        common_sigma = float(np.average(sigmas)) if len(sigmas) > 0 else 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, c, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Eyring",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.sigma]
        hessian_matrix = hessian(LL_func)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
            # a can be positive or negative
            self.a_upper = self.a + (Z * self.a_SE)
            self.a_lower = self.a + (-Z * self.a_SE)
            # c can be positive or negative
            self.c_upper = self.c + (Z * self.c_SE)
            self.c_lower = self.c + (-Z * self.c_SE)
            # sigma is strictly positive
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Normal_Eyring model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.c_SE = 0
            self.sigma_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.c_upper = self.c
            self.c_lower = self.c
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "sigma"],
            "Point Estimate": [self.a, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(params, failures, right_censored, failure_stress, right_censored_stress)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = self.life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(self.life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(self.life_func(S1=use_level_stress) / self.life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (common_sigmas[i] - sigmas_for_change_df[i]) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(str("+" + str(round(sigma_diff * 100, 2)) + "%"))
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))
        self.__mus_for_change_df = mus_for_change_df
        self.__sigmas_for_change_df = sigmas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )
        self.__failures = failures
        self.__right_censored = right_censored
        self.__CI: float = CI
        self.__shape_change_exceeded = shape_change_exceeded
        self.__use_level_stress = use_level_stress

    def print_results(self) -> None:
        n: int = len(self.__failures) + len(self.__right_censored)
        CI_rounded: float = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored = len(self.__right_censored) / n * 100
        if frac_censored % 1 < 1e-10:
            frac_censored = int(frac_censored)
        colorprint(
            str("Results from Fit_Normal_Eyring (" + str(CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print("Analysis method: Maximum Likelihood Estimation (MLE)")
        if self.optimizer is not None:
            print("Optimizer:", self.optimizer)
        print(
            "Failures / Right censored:",
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.change_of_parameters.to_string(index=False))
        if self.__shape_change_exceeded is True:
            print(
                str(
                    "The sigma parameter has been found to change significantly (>"
                    + str(int(shape_change_threshold * 100))
                    + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate.",
                ),
            )
        print("\n", self.goodness_of_fit.to_string(index=False), "\n")

        if self.__use_level_stress is not None:
            print(
                str(
                    "At the use level stress of "
                    + round_and_string(self.__use_level_stress)
                    + ", the mean life is "
                    + str(round(self.mean_life, 5))
                    + "\n",
                ),
            )

    def probability_plot(self, ax: bool | Axes = True):
        return ALT_prob_plot(
            dist="Normal",
            model="Eyring",
            stresses_for_groups=self.__stresses_for_groups,
            failure_groups=self.__failure_groups,
            right_censored_groups=self.__right_censored_groups,
            life_func=self.life_func,
            shape=self.sigma,
            scale_for_change_df=self.__mus_for_change_df,
            shape_for_change_df=self.__sigmas_for_change_df,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_stress_plot(self, ax: bool | Axes = True):
        return life_stress_plot(
            dist="Normal",
            model="Eyring",
            life_func=self.life_func,
            failure_groups=self.__failure_groups,
            stresses_for_groups=self.__stresses_for_groups,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_func(self, S1):
        return 1 / S1 * np.exp(-(self.c - self.a / S1))

    @staticmethod
    def logf(t, T, a, c, sigma):  # Log PDF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(sigma * (2 * anp.pi) ** 0.5)

    @staticmethod
    def logR(t, T, a, c, sigma):  # Log SF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log((1 + erf(((life - t) / sigma) / 2**0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Normal_Eyring.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Normal_Eyring.logR(t_rc, T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Power:
    """This function will Fit the Normal-Power life-stress model to the data
    provided. Please see the online documentation for the equations of this
    model.

    This model is most appropriate to model a life-stress relationship with
    non-thermal stresses (typically in fatigue applications).

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress : array, list
        The corresponding stresses (such as load) at which each failure
        occurred. This must match the length of failures as each failure is
        tied to a failure stress.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    right_censored_stress : array, list, optional
        The corresponding stresses (such as load) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a
        right_censored stress. Conditionally optional input. This must be
        provided if right_censored is provided.
    use_level_stress : int, float, optional
        The use level stress at which you want to know the mean life. Optional
        input.
    print_results : bool, optional
        True/False. Default is True. Prints the results to the console.
    show_probability_plot : bool, object, optional
        True/False. Default is True. Provides a probability plot of the fitted
        ALT model. If an axes object is passed it will be used.
    show_life_stress_plot : bool, str, object, optional
        If True the life-stress plot will be shown. To hide the life-stress
        plot use False. To swap the axes and show a stress-life plot use
        'swap'. If an axes handle is passed it will be used. Default is True.
    CI : float, optional
        Confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the
        `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.

    Returns
    -------
    a : float
        The fitted parameter from the Power model
    n : float
        The fitted parameter from the Power model
    sigma : float
        The fitted Normal_2P sigma parameter
    loglik2 : float
        Log Likelihood*-2 (as used in JMP Pro)
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    a_SE : float
        The standard error (sqrt(variance)) of the parameter
    n_SE : float
        The standard error (sqrt(variance)) of the parameter
    sigma_SE : float
        The standard error (sqrt(variance)) of the parameter
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    n_upper : float
        The upper CI estimate of the parameter
    n_lower : float
        The lower CI estimate of the parameter
    sigma_upper : float
        The upper CI estimate of the parameter
    sigma_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (mu and sigma) at
        each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    mu_at_use_stress : float
        The equivalent Normal mu parameter at the use level stress (only
        provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Normal distribution at the use level stress (only provided if
        use_level_stress is provided).
    probability_plot : object
        The figure object from the probability plot (only provided if
        show_probability_plot is True).
    life_stress_plot : object
        The figure object from the life-stress plot (only provided if
        show_life_stress_plot is True).

    """

    def __init__(
        self,
        failures,
        failure_stress,
        right_censored=None,
        right_censored_stress=None,
        use_level_stress: float | None = None,
        CI=0.95,
        optimizer: str | None = None,
    ):
        inputs = alt_single_stress_fitters_input_checking(
            dist="Normal",
            life_stress_model="Power",
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress = inputs.failure_stress_1
        right_censored = inputs.right_censored
        right_censored_stress = inputs.right_censored_stress_1
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Power.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(model="Power", failures=failures, stress_1_array=failure_stress)

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        common_sigma = float(np.average(sigmas)) if len(sigmas) > 0 else 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            common_sigma,
        ]  # a, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Power",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.n, self.sigma]
        hessian_matrix = hessian(LL_func)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress)),
            np.array(tuple(right_censored_stress)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.n_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[2][2]) ** 0.5
            # a is strictly positive
            self.a_upper = self.a * (np.exp(Z * (self.a_SE / self.a)))
            self.a_lower = self.a * (np.exp(-Z * (self.a_SE / self.a)))
            # n can be positive or negative
            self.n_upper = self.n + (Z * self.n_SE)
            self.n_lower = self.n + (-Z * self.n_SE)
            # sigma is strictly positive
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Normal_Power model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.n_SE = 0
            self.sigma_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.n_upper = self.n
            self.n_lower = self.n
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        # results dataframe
        results_data = {
            "Parameter": ["a", "n", "sigma"],
            "Point Estimate": [self.a, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.n_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(params, failures, right_censored, failure_stress, right_censored_stress)
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = self.life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(self.life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(self.life_func(S1=use_level_stress) / self.life_func(S1=stress))
        common_sigmas = np.ones_like(stresses_for_groups) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (common_sigmas[i] - sigmas_for_change_df[i]) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(str("+" + str(round(sigma_diff * 100, 2)) + "%"))
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))
        self.__mus_for_change_df = mus_for_change_df
        self.__sigmas_for_change_df = sigmas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )
        self.__failures = failures
        self.__right_censored = right_censored
        self.__CI = CI
        self.__shape_change_exceeded = shape_change_exceeded
        self.__use_level_stress = use_level_stress

    def print_results(self):
        n = len(self.__failures) + len(self.__right_censored)
        CI_rounded = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored = len(self.__right_censored) / n * 100
        if frac_censored % 1 < 1e-10:
            frac_censored = int(frac_censored)
        colorprint(
            str("Results from Fit_Normal_Power (" + str(CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print("Analysis method: Maximum Likelihood Estimation (MLE)")
        if self.optimizer is not None:
            print("Optimizer:", self.optimizer)
        print(
            "Failures / Right censored:",
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.change_of_parameters.to_string(index=False))
        if self.__shape_change_exceeded is True:
            print(
                str(
                    "The sigma parameter has been found to change significantly (>"
                    + str(int(shape_change_threshold * 100))
                    + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate.",
                ),
            )
        print("\n", self.goodness_of_fit.to_string(index=False), "\n")

        if self.__use_level_stress is not None:
            print(
                str(
                    "At the use level stress of "
                    + round_and_string(self.__use_level_stress)
                    + ", the mean life is "
                    + str(round(self.mean_life, 5))
                    + "\n",
                ),
            )

    def probability_plot(self, ax=True):
        return ALT_prob_plot(
            dist="Normal",
            model="Power",
            stresses_for_groups=self.__stresses_for_groups,
            failure_groups=self.__failure_groups,
            right_censored_groups=self.__right_censored_groups,
            life_func=self.life_func,
            shape=self.sigma,
            scale_for_change_df=self.__mus_for_change_df,
            shape_for_change_df=self.__sigmas_for_change_df,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_stress_plot(self, ax=True):
        return life_stress_plot(
            dist="Normal",
            model="Power",
            life_func=self.life_func,
            failure_groups=self.__failure_groups,
            stresses_for_groups=self.__stresses_for_groups,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_func(self, S1):
        return self.a * S1**self.n

    @staticmethod
    def logf(t, T, a, n, sigma):  # Log PDF
        life = a * T**n
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(sigma * (2 * anp.pi) ** 0.5)

    @staticmethod
    def logR(t, T, a, n, sigma):  # Log SF
        life = a * T**n
        return anp.log((1 + erf(((life - t) / sigma) / 2**0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Normal_Power.logf(t_f, T_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Normal_Power.logR(t_rc, T_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Dual_Exponential:
    """This function will Fit the Normal_Dual_Exponential life-stress model to the
    data provided. Please see the online documentation for the equations of
    this model.

    This model is most appropriate to model a life-stress relationship with
    two thermal stresses (such as temperature-humidity). It is recommended that
    you ensure your temperature data are in Kelvin and humidity data range from
    0 to 1.

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress_1 : array, list
        The corresponding stress 1 (such as temperature) at which each failure
        occurred. This must match the length of failures as each failure is tied
        to a failure stress.
    failure_stress_2 : array, list
        The corresponding stress 2 (such as humidity) at which each failure
        occurred. This must match the length of failures as each failure is tied
        to a failure stress.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    right_censored_stress_1 : array, list, optional
        The corresponding stress 1 (such as temperature) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a right_censored
        stress. Conditionally optional input. This must be provided if
        right_censored is provided.
    right_censored_stress_2 : array, list, optional
        The corresponding stress 1 (such as humidity) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a right_censored
        stress. Conditionally optional input. This must be provided if
        right_censored is provided.
    use_level_stress : array, list optional
        A two element list or array of the use level stresses in the form
        [stress_1, stress_2] at which you want to know the mean life. Optional
        input.
    print_results : bool, optional
        True/False. Default is True. Prints the results to the console.
    show_probability_plot : bool, object, optional
        True/False. Default is True. Provides a probability plot of the fitted
        ALT model. If an axes object is passed it will be used.
    show_life_stress_plot : bool, str, object, optional
        If True the life-stress plot will be shown. To hide the life-stress
        plot use False. To swap the axes and show a stress-life plot use
        'swap'. If an axes handle is passed it will be used. Default is True.
    CI : float, optional
        Confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the
        `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.

    Returns
    -------
    a : float
        The fitted parameter from the Dual_Exponential model
    b : float
        The fitted parameter from the Dual_Exponential model
    c : float
        The fitted parameter from the Dual_Exponential model
    sigma : float
        The fitted Normal_2P sigma parameter
    loglik2 : float
        Log Likelihood*-2 (as used in JMP Pro)
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    a_SE : float
        The standard error (sqrt(variance)) of the parameter
    b_SE : float
        The standard error (sqrt(variance)) of the parameter
    c_SE : float
        The standard error (sqrt(variance)) of the parameter
    sigma_SE : float
        The standard error (sqrt(variance)) of the parameter
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    b_upper : float
        The upper CI estimate of the parameter
    b_lower : float
        The lower CI estimate of the parameter
    c_upper : float
        The upper CI estimate of the parameter
    c_lower : float
        The lower CI estimate of the parameter
    sigma_upper : float
        The upper CI estimate of the parameter
    sigma_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (mu and sigma) at
        each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    mu_at_use_stress : float
        The equivalent Normal mu parameter at the use level stress (only
        provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Normal distribution at the use level stress (only provided if
        use_level_stress is provided).
    probability_plot : object
        The figure object from the probability plot (only provided if
        show_probability_plot is True).
    life_stress_plot : object
        The figure object from the life-stress plot (only provided if
        show_life_stress_plot is True).

    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress: npt.NDArray[np.float64] | None = None,
        CI=0.95,
        optimizer=None,
    ):
        inputs = alt_fitters_dual_stress_input_checking(
            dist="Normal",
            life_stress_model="Dual_Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Dual_Exponential.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        common_sigma = float(np.average(sigmas)) if len(sigmas) > 0 else 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # a, b, c, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Dual_Exponential",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.c = MLE_results.c
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.c, self.sigma]
        hessian_matrix = hessian(LL_func)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.c_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
            # a can be positive or negative
            self.a_upper = self.a + (Z * self.a_SE)
            self.a_lower = self.a + (-Z * self.a_SE)
            # b can be positive or negative
            self.b_upper = self.b + (Z * self.b_SE)
            self.b_lower = self.b + (-Z * self.b_SE)
            # c is strictly positive
            self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
            self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
            # sigma is strictly positive
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Normal_Dual_Exponential model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.b_SE = 0
            self.c_SE = 0
            self.sigma_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.b_upper = self.b
            self.b_lower = self.b
            self.c_upper = self.c
            self.c_lower = self.c
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "c", "sigma"],
            "Point Estimate": [self.a, self.b, self.c, self.sigma],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        def life_func(S1, S2):
            return self.c * np.exp(self.a / S1 + self.b / S2)

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            self.distribution_at_use_stress = Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        AF = []
        stresses_for_groups_str = []
        for stress in stresses_for_groups:
            new_mus.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(str(round_and_string(stress[0]) + ", " + round_and_string(stress[1])))
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1]) / life_func(S1=stress[0], S2=stress[1]),
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (common_sigmas[i] - sigmas_for_change_df[i]) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(str("+" + str(round(sigma_diff * 100, 2)) + "%"))
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))
        self.__mus_for_change_df = mus_for_change_df
        self.__sigmas_for_change_df = sigmas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )
        self.__failures = failures
        self.__right_censored = right_censored
        self.__CI = CI
        self.__shape_change_exceeded = shape_change_exceeded
        self.__use_level_stress = use_level_stress

    def print_results(self) -> None:
        n: int = len(self.__failures) + len(self.__right_censored)
        CI_rounded: float = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored: float = len(self.__right_censored) / n * 100
        if frac_censored % 1 < 1e-10:
            frac_censored = int(frac_censored)
        colorprint(
            str("Results from Fit_Normal_Dual_Exponential (" + str(CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print("Analysis method: Maximum Likelihood Estimation (MLE)")
        if self.optimizer is not None:
            print("Optimizer:", self.optimizer)
        print(
            "Failures / Right censored:",
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.change_of_parameters.to_string(index=False))
        if self.__shape_change_exceeded is True:
            print(
                str(
                    "The sigma parameter has been found to change significantly (>"
                    + str(int(shape_change_threshold * 100))
                    + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate.",
                ),
            )
        print("\n", self.goodness_of_fit.to_string(index=False), "\n")

        if self.__use_level_stress is not None:
            print(
                str(
                    "At the use level stress of "
                    + round_and_string(self.__use_level_stress[0])
                    + ", "
                    + round_and_string(self.__use_level_stress[1])
                    + ", the mean life is "
                    + str(round(self.mean_life, 5))
                    + "\n",
                ),
            )

    def probability_plot(self, ax: bool | Axes = True) -> Axes | None:
        return ALT_prob_plot(
            dist="Normal",
            model="Dual_Exponential",
            stresses_for_groups=self.__stresses_for_groups,
            failure_groups=self.__failure_groups,
            right_censored_groups=self.__right_censored_groups,
            life_func=self.life_func,
            shape=self.sigma,
            scale_for_change_df=self.__mus_for_change_df,
            shape_for_change_df=self.__sigmas_for_change_df,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_stress_plot(self, ax: bool | Axes = True) -> Axes | None:
        return life_stress_plot(
            dist="Normal",
            model="Dual_Exponential",
            life_func=self.life_func,
            failure_groups=self.__failure_groups,
            stresses_for_groups=self.__stresses_for_groups,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_func(self, S1, S2):
        return self.c * np.exp(self.a / S1 + self.b / S2)

    @staticmethod
    def logf(t, S1, S2, a, b, c, sigma):  # Log PDF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(sigma * (2 * anp.pi) ** 0.5)

    @staticmethod
    def logR(t, S1, S2, a, b, c, sigma):  # Log SF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log((1 + erf(((life - t) / sigma) / 2**0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Normal_Dual_Exponential.logf(t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]).sum()
        # right censored times
        LL_rc = Fit_Normal_Dual_Exponential.logR(t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Power_Exponential:
    """This function will Fit the Normal_Power_Exponential life-stress model to
    the data provided. Please see the online documentation for the equations of
    this model.

    This model is most appropriate to model a life-stress relationship with
    thermal and non-thermal stresses. It is essential that you ensure your
    thermal stress is stress_1 (as this will be modeled by the Exponential) and
    your non-thermal stress is stress_2 (as this will be modeled by the Power).
    Also ensure that your temperature data are in Kelvin.

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress_1 : array, list
        The corresponding stress 1 (thermal-stress) at which each failure
        occurred. This must match the length of failures as each failure is tied
        to a failure stress.
    failure_stress_2 : array, list
        The corresponding stress 2 (non-thermal stress) at which each failure
        occurred. This must match the length of failures as each failure is tied
        to a failure stress.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    right_censored_stress_1 : array, list, optional
        The corresponding stress 1 (thermal stress) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a right_censored
        stress. Conditionally optional input. This must be provided if
        right_censored is provided.
    right_censored_stress_2 : array, list, optional
        The corresponding stress 1 (non-thermal stress) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a right_censored
        stress. Conditionally optional input. This must be provided if
        right_censored is provided.
    use_level_stress : array, list optional
        A two element list or array of the use level stresses in the form
        [stress_1, stress_2] at which you want to know the mean life. Optional
        input.
    print_results : bool, optional
        True/False. Default is True. Prints the results to the console.
    show_probability_plot : bool, object, optional
        True/False. Default is True. Provides a probability plot of the fitted
        ALT model. If an axes object is passed it will be used.
    show_life_stress_plot : bool, str, object, optional
        If True the life-stress plot will be shown. To hide the life-stress
        plot use False. To swap the axes and show a stress-life plot use
        'swap'. If an axes handle is passed it will be used. Default is True.
    CI : float, optional
        Confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the
        `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.

    Returns
    -------
    a : float
        The fitted parameter from the Power_Exponential model
    c : float
        The fitted parameter from the Power_Exponential model
    n : float
        The fitted parameter from the Power_Exponential model
    sigma : float
        The fitted Normal_2P sigma parameter
    loglik2 : float
        Log Likelihood*-2 (as used in JMP Pro)
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    a_SE : float
        The standard error (sqrt(variance)) of the parameter
    c_SE : float
        The standard error (sqrt(variance)) of the parameter
    n_SE : float
        The standard error (sqrt(variance)) of the parameter
    sigma_SE : float
        The standard error (sqrt(variance)) of the parameter
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    c_upper : float
        The upper CI estimate of the parameter
    c_lower : float
        The lower CI estimate of the parameter
    n_upper : float
        The upper CI estimate of the parameter
    n_lower : float
        The lower CI estimate of the parameter
    sigma_upper : float
        The upper CI estimate of the parameter
    sigma_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (mu and sigma) at
        each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    mu_at_use_stress : float
        The equivalent Normal mu parameter at the use level stress (only
        provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Normal distribution at the use level stress (only provided if
        use_level_stress is provided).
    probability_plot : object
        The figure object from the probability plot (only provided if
        show_probability_plot is True).
    life_stress_plot : object
        The figure object from the life-stress plot (only provided if
        show_life_stress_plot is True).

    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress: npt.NDArray[np.float64] | None = None,
        CI=0.95,
        optimizer=None,
    ):
        inputs = alt_fitters_dual_stress_input_checking(
            dist="Normal",
            life_stress_model="Power_Exponential",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Power_Exponential.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Power_Exponential",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")

        common_sigma = float(np.average(sigmas)) if len(sigmas) > 0 else 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # a, c, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Power_Exponential",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.a = MLE_results.a
        self.c = MLE_results.c
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.n, self.sigma]
        hessian_matrix = hessian(LL_func)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.a_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.c_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
            # a can be positive or negative
            self.a_upper = self.a + (Z * self.a_SE)
            self.a_lower = self.a + (-Z * self.a_SE)
            # c is strictly positive
            self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
            self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
            # n can be positive or negative
            self.n_upper = self.n + (Z * self.n_SE)
            self.n_lower = self.n + (-Z * self.n_SE)
            # sigma is strictly positive
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Normal_Power_Exponential model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.c_SE = 0
            self.n_SE = 0
            self.sigma_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.c_upper = self.c
            self.c_lower = self.c
            self.n_upper = self.n
            self.n_lower = self.n
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "n", "sigma"],
            "Point Estimate": [self.a, self.c, self.n, self.sigma],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = self.life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            self.distribution_at_use_stress = Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(self.life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(str(round_and_string(stress[0]) + ", " + round_and_string(stress[1])))
            if use_level_stress is not None:
                AF.append(
                    self.life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / self.life_func(S1=stress[0], S2=stress[1]),
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (common_sigmas[i] - sigmas_for_change_df[i]) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(str("+" + str(round(sigma_diff * 100, 2)) + "%"))
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))
        self.__mus_for_change_df = mus_for_change_df
        self.__sigmas_for_change_df = sigmas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )
        self.__failures = failures
        self.__right_censored = right_censored
        self.__CI = CI
        self.__shape_change_exceeded = shape_change_exceeded
        self.__use_level_stress = use_level_stress

    def print_results(self):
        n = len(self.__failures) + len(self.__right_censored)
        CI_rounded = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored = len(self.__right_censored) / n * 100
        if frac_censored % 1 < 1e-10:
            frac_censored = int(frac_censored)
        colorprint(
            str("Results from Fit_Normal_Power_Exponential (" + str(CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print("Analysis method: Maximum Likelihood Estimation (MLE)")
        if self.optimizer is not None:
            print("Optimizer:", self.optimizer)
        print(
            "Failures / Right censored:",
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.change_of_parameters.to_string(index=False))
        if self.__shape_change_exceeded is True:
            print(
                str(
                    "The sigma parameter has been found to change significantly (>"
                    + str(int(shape_change_threshold * 100))
                    + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate.",
                ),
            )
        print("\n", self.goodness_of_fit.to_string(index=False), "\n")

        if self.__use_level_stress is not None:
            print(
                str(
                    "At the use level stress of "
                    + round_and_string(self.__use_level_stress[0])
                    + ", "
                    + round_and_string(self.__use_level_stress[1])
                    + ", the mean life is "
                    + str(round(self.mean_life, 5))
                    + "\n",
                ),
            )

    def probability_plot(self, ax: bool | Axes = True):
        return ALT_prob_plot(
            dist="Normal",
            model="Power_Exponential",
            stresses_for_groups=self.__stresses_for_groups,
            failure_groups=self.__failure_groups,
            right_censored_groups=self.__right_censored_groups,
            life_func=self.life_func,
            shape=self.sigma,
            scale_for_change_df=self.__mus_for_change_df,
            shape_for_change_df=self.__sigmas_for_change_df,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_stress_plot(self, ax: bool | Axes = True):
        return life_stress_plot(
            dist="Normal",
            model="Power_Exponential",
            life_func=self.life_func,
            failure_groups=self.__failure_groups,
            stresses_for_groups=self.__stresses_for_groups,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_func(self, S1, S2):
        return self.c * (S2**self.n) * np.exp(self.a / S1)

    @staticmethod
    def logf(t, S1, S2, a, c, n, sigma):  # Log PDF
        life = c * S2**n * anp.exp(a / S1)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(sigma * (2 * anp.pi) ** 0.5)

    @staticmethod
    def logR(t, S1, S2, a, c, n, sigma):  # Log SF
        life = c * S2**n * anp.exp(a / S1)
        return anp.log((1 + erf(((life - t) / sigma) / 2**0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Normal_Power_Exponential.logf(t_f, S1_f, S2_f, params[0], params[1], params[2], params[3]).sum()
        # right censored times
        LL_rc = Fit_Normal_Power_Exponential.logR(t_rc, S1_rc, S2_rc, params[0], params[1], params[2], params[3]).sum()
        return -(LL_f + LL_rc)


class Fit_Normal_Dual_Power:
    """This function will Fit the Normal_Dual_Power life-stress model to the data
    provided. Please see the online documentation for the equations of this
    model.

    This model is most appropriate to model a life-stress relationship with two
    non-thermal stresses such as voltage and load.

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress_1 : array, list
        The corresponding stress 1 (such as voltage) at which each failure
        occurred. This must match the length of failures as each failure is tied
        to a failure stress.
    failure_stress_2 : array, list
        The corresponding stress 2 (such as load) at which each failure
        occurred. This must match the length of failures as each failure is tied
        to a failure stress.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    right_censored_stress_1 : array, list, optional
        The corresponding stress 1 (such as voltage) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a right_censored
        stress. Conditionally optional input. This must be provided if
        right_censored is provided.
    right_censored_stress_2 : array, list, optional
        The corresponding stress 1 (such as load) at which each
        right_censored data point was obtained. This must match the length of
        right_censored as each right_censored value is tied to a right_censored
        stress. Conditionally optional input. This must be provided if
        right_censored is provided.
    use_level_stress : array, list optional
        A two element list or array of the use level stresses in the form
        [stress_1, stress_2] at which you want to know the mean life. Optional
        input.
    print_results : bool, optional
        True/False. Default is True. Prints the results to the console.
    show_probability_plot : bool, object, optional
        True/False. Default is True. Provides a probability plot of the fitted
        ALT model. If an axes object is passed it will be used.
    show_life_stress_plot : bool, str, object, optional
        If True the life-stress plot will be shown. To hide the life-stress
        plot use False. To swap the axes and show a stress-life plot use
        'swap'. If an axes handle is passed it will be used. Default is True.
    CI : float, optional
        Confidence interval for estimating confidence limits on parameters. Must
        be between 0 and 1. Default is 0.95 for 95% CI.
    optimizer : str, optional
        The optimization algorithm used to find the solution. Must be either
        'TNC', 'L-BFGS-B', 'nelder-mead', or 'powell'. Specifying the optimizer
        will result in that optimizer being used. To use all of these specify
        'best' and the best result will be returned. The default behaviour is to
        try each optimizer in order ('TNC', 'L-BFGS-B', 'nelder-mead', and
        'powell') and stop once one of the optimizers finds a solution. If the
        optimizer fails, the initial guess will be returned.
        For more detail see the
        `documentation <https://reliability.readthedocs.io/en/latest/Optimizers.html>`_.

    Returns
    -------
    c : float
        The fitted parameter from the Dual_Power model
    m : float
        The fitted parameter from the Dual_Power model
    n : float
        The fitted parameter from the Dual_Power model
    sigma : float
        The fitted Normal_2P sigma parameter
    loglik2 : float
        Log Likelihood*-2 (as used in JMP Pro)
    loglik : float
        Log Likelihood (as used in Minitab and Reliasoft)
    AICc : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    c_SE : float
        The standard error (sqrt(variance)) of the parameter
    m_SE : float
        The standard error (sqrt(variance)) of the parameter
    n_SE : float
        The standard error (sqrt(variance)) of the parameter
    sigma_SE : float
        The standard error (sqrt(variance)) of the parameter
    c_upper : float
        The upper CI estimate of the parameter
    c_lower : float
        The lower CI estimate of the parameter
    m_upper : float
        The upper CI estimate of the parameter
    m_lower : float
        The lower CI estimate of the parameter
    n_upper : float
        The upper CI estimate of the parameter
    n_lower : float
        The lower CI estimate of the parameter
    sigma_upper : float
        The upper CI estimate of the parameter
    sigma_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (mu and sigma) at
        each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    mu_at_use_stress : float
        The equivalent Normal mu parameter at the use level stress (only
        provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Normal distribution at the use level stress (only provided if
        use_level_stress is provided).
    probability_plot : object
        The figure object from the probability plot (only provided if
        show_probability_plot is True).
    life_stress_plot : object
        The figure object from the life-stress plot (only provided if
        show_life_stress_plot is True).

    """

    def __init__(
        self,
        failures,
        failure_stress_1,
        failure_stress_2,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress: npt.NDArray[np.float64] | None = None,
        CI=0.95,
        optimizer=None,
    ):
        inputs = alt_fitters_dual_stress_input_checking(
            dist="Normal",
            life_stress_model="Dual_Power",
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        failures = inputs.failures
        failure_stress_1 = inputs.failure_stress_1
        failure_stress_2 = inputs.failure_stress_2
        right_censored = inputs.right_censored
        right_censored_stress_1 = inputs.right_censored_stress_1
        right_censored_stress_2 = inputs.right_censored_stress_2
        CI = inputs.CI
        optimizer = inputs.optimizer
        use_level_stress = inputs.use_level_stress
        failure_groups = inputs.failure_groups
        right_censored_groups = inputs.right_censored_groups
        stresses_for_groups = inputs.stresses_for_groups
        LL_func = Fit_Normal_Dual_Power.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(
            model="Dual_Power",
            failures=failures,
            stress_1_array=failure_stress_1,
            stress_2_array=failure_stress_2,
        )

        # obtain the common shape parameter
        sigmas = []
        sigmas_for_change_df = []
        mus_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Normal_2P(
                    failures=f,
                    right_censored=rc,
                )
                sigmas.append(fit.sigma)
                sigmas_for_change_df.append(fit.sigma)
                mus_for_change_df.append(fit.mu)
            else:  # 1 failure at this stress
                sigmas_for_change_df.append(0)
                mus_for_change_df.append("")
        common_sigma = float(np.average(sigmas)) if (len(sigmas) > 0) else 1  # guess in the absence of enough points
        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
            common_sigma,
        ]  # c, m, n, sigma

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Dual_Power",
            dist="Normal",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress_1,
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress_1,
            right_censored_stress_2=right_censored_stress_2,
        )
        self.c = MLE_results.c
        self.m = MLE_results.m
        self.n = MLE_results.n
        self.sigma = MLE_results.sigma
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.m, self.n, self.sigma]
        hessian_matrix = hessian(LL_func)(  # type: ignore
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
            np.array(tuple(failure_stress_1)),
            np.array(tuple(failure_stress_2)),
            np.array(tuple(right_censored_stress_1)),
            np.array(tuple(right_censored_stress_2)),
        )
        try:
            covariance_matrix = np.linalg.inv(hessian_matrix)
            self.c_SE = abs(covariance_matrix[0][0]) ** 0.5
            self.m_SE = abs(covariance_matrix[1][1]) ** 0.5
            self.n_SE = abs(covariance_matrix[2][2]) ** 0.5
            self.sigma_SE = abs(covariance_matrix[3][3]) ** 0.5
            # c is strictly positive
            self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
            self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
            # m can be positive or negative
            self.m_upper = self.m + (Z * self.m_SE)
            self.m_lower = self.m + (-Z * self.m_SE)
            # n can be positive or negative
            self.n_upper = self.n + (Z * self.n_SE)
            self.n_lower = self.n + (-Z * self.n_SE)
            # sigma is strictly positive
            self.sigma_upper = self.sigma * (np.exp(Z * (self.sigma_SE / self.sigma)))
            self.sigma_lower = self.sigma * (np.exp(-Z * (self.sigma_SE / self.sigma)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Normal_Dual_Power model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.c_SE = 0
            self.m_SE = 0
            self.n_SE = 0
            self.sigma_SE = 0
            self.c_upper = self.c
            self.c_lower = self.c
            self.m_upper = self.m
            self.m_lower = self.m
            self.n_upper = self.n
            self.n_lower = self.n
            self.sigma_upper = self.sigma
            self.sigma_lower = self.sigma

        # results dataframe
        results_data = {
            "Parameter": ["c", "m", "n", "sigma"],
            "Point Estimate": [self.c, self.m, self.n, self.sigma],
            "Standard Error": [self.c_SE, self.m_SE, self.n_SE, self.sigma_SE],
            "Lower CI": [self.c_lower, self.m_lower, self.n_lower, self.sigma_lower],
            "Upper CI": [self.c_upper, self.m_upper, self.n_upper, self.sigma_upper],
        }
        self.results = pd.DataFrame(
            data=results_data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        # goodness of fit dataframe
        n = len(failures) + len(right_censored)
        k = len(guess)
        LL2 = 2 * LL_func(
            params,
            failures,
            right_censored,
            failure_stress_1,
            failure_stress_2,
            right_censored_stress_1,
            right_censored_stress_2,
        )
        self.loglik2 = LL2
        self.loglik = LL2 * -0.5
        if n - k - 1 > 0:
            self.AICc = 2 * k + LL2 + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            self.AICc = np.inf
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        # use level stress calculations
        if use_level_stress is not None:
            self.mu_at_use_stress = self.life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            self.distribution_at_use_stress = Normal_Distribution(mu=self.mu_at_use_stress, sigma=self.sigma)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_mus = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_mus.append(self.life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(str(round_and_string(stress[0]) + ", " + round_and_string(stress[1])))
            if use_level_stress is not None:
                AF.append(
                    self.life_func(S1=use_level_stress[0], S2=use_level_stress[1])
                    / self.life_func(S1=stress[0], S2=stress[1]),
                )
        common_sigmas = np.ones(len(stresses_for_groups)) * self.sigma
        sigma_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if sigmas_for_change_df[i] == 0:
                sigmas_for_change_df[i] = ""  # replace with space
                sigma_differences.append("")
            else:
                sigma_diff = (common_sigmas[i] - sigmas_for_change_df[i]) / sigmas_for_change_df[i]
                if abs(sigma_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if sigma_diff > 0:
                    sigma_differences.append(str("+" + str(round(sigma_diff * 100, 2)) + "%"))
                else:
                    sigma_differences.append(str(str(round(sigma_diff * 100, 2)) + "%"))
        self.__mus_for_change_df = mus_for_change_df
        self.__sigmas_for_change_df = sigmas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "original mu": mus_for_change_df,
                "original sigma": sigmas_for_change_df,
                "new mu": new_mus,
                "common sigma": common_sigmas,
                "sigma change": sigma_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )
        self.__failures = failures
        self.__right_censored = right_censored
        self.__CI = CI
        self.__shape_change_exceeded = shape_change_exceeded
        self.__use_level_stress = use_level_stress

    def print_results(self):
        n = len(self.__failures) + len(self.__right_censored)
        CI_rounded = self.__CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(self.__CI * 100)
        frac_censored = len(self.__right_censored) / n * 100
        if frac_censored % 1 < 1e-10:
            frac_censored = int(frac_censored)
        colorprint(
            str("Results from Fit_Normal_Dual_Power (" + str(CI_rounded) + "% CI):"),
            bold=True,
            underline=True,
        )
        print("Analysis method: Maximum Likelihood Estimation (MLE)")
        if self.optimizer is not None:
            print("Optimizer:", self.optimizer)
        print(
            "Failures / Right censored:",
            str(str(len(self.__failures)) + "/" + str(len(self.__right_censored))),
            str("(" + round_and_string(frac_censored) + "% right censored)"),
            "\n",
        )
        print(self.results.to_string(index=False), "\n")
        print(self.change_of_parameters.to_string(index=False))
        if self.__shape_change_exceeded is True:
            print(
                str(
                    "The sigma parameter has been found to change significantly (>"
                    + str(int(shape_change_threshold * 100))
                    + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Normal distribution may not be appropriate.",
                ),
            )
        print("\n", self.goodness_of_fit.to_string(index=False), "\n")

        if self.__use_level_stress is not None:
            print(
                str(
                    "At the use level stress of "
                    + round_and_string(self.__use_level_stress[0])
                    + ", "
                    + round_and_string(self.__use_level_stress[1])
                    + ", the mean life is "
                    + str(round(self.mean_life, 5))
                    + "\n",
                ),
            )

    def probability_plot(self, ax: bool | Axes = True):
        return ALT_prob_plot(
            dist="Normal",
            model="Dual_Power",
            stresses_for_groups=self.__stresses_for_groups,
            failure_groups=self.__failure_groups,
            right_censored_groups=self.__right_censored_groups,
            life_func=self.life_func,
            shape=self.sigma,
            scale_for_change_df=self.__mus_for_change_df,
            shape_for_change_df=self.__sigmas_for_change_df,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_stress_plot(self, ax: bool | Axes = True):
        return life_stress_plot(
            dist="Normal",
            model="Dual_Power",
            life_func=self.life_func,
            failure_groups=self.__failure_groups,
            stresses_for_groups=self.__stresses_for_groups,
            use_level_stress=self.__use_level_stress,
            ax=ax,
        )

    def life_func(self, S1, S2):
        return self.c * (S1**self.m) * (S2**self.n)

    @staticmethod
    def logf(t, S1, S2, c, m, n, sigma):  # Log PDF
        life = c * (S1**m) * (S2**n)
        return anp.log(anp.exp(-0.5 * (((t - life) / sigma) ** 2))) - anp.log(sigma * (2 * anp.pi) ** 0.5)

    @staticmethod
    def logR(t, S1, S2, c, m, n, sigma):  # Log SF
        life = c * (S1**m) * (S2**n)
        return anp.log((1 + erf(((life - t) / sigma) / 2**0.5)) / 2)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        LL_f = Fit_Normal_Dual_Power.logf(
            t_f,
            S1_f,
            S2_f,
            params[0],
            params[1],
            params[2],
            params[3],
        ).sum()  # failure times
        LL_rc = Fit_Normal_Dual_Power.logR(
            t_rc,
            S1_rc,
            S2_rc,
            params[0],
            params[1],
            params[2],
            params[3],
        ).sum()  # right censored times
        return -(LL_f + LL_rc)

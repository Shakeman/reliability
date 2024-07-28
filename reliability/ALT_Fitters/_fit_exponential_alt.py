import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as ss
from autograd.differential_operators import hessian
from numpy.linalg import LinAlgError

from reliability.Distributions import (
    Exponential_Distribution,
)
from reliability.Fitters import Fit_Weibull_2P
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


class Fit_Exponential_Exponential:
    """This function will fit the Exponential-Exponential life-stress model to the
    data provided. Please see the online documentation for the equations of this
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
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    b_upper : float
        The upper CI estimate of the parameter
    b_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (equivalent Weibull
        alpha and beta) at each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    Lambda_at_use_stress : float
        The equivalent Exponential Lambda parameter at the use level stress
        (only provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Exponential distribution at the use level stress (only provided if
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
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):
        inputs = alt_single_stress_fitters_input_checking(
            dist="Exponential",
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
        LL_func = Fit_Exponential_Exponential.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(model="Exponential", failures=failures, stress_1_array=failure_stress)

        # obtain the common shape parameter
        betas = []  # weibull betas
        betas_for_change_df = []
        alphas_for_change_df = []  # weibull alphas
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1]]  # a, b

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Exponential",
            dist="Exponential",
            LL_func=LL_func,
            initial_guess=guess,
            optimizer=optimizer,
            failures=failures,
            failure_stress_1=failure_stress,
            right_censored=right_censored,
            right_censored_stress_1=right_censored_stress,
        )
        self.a = MLE_results.a
        self.b = MLE_results.b
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b]
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
            self.b_SE = abs(covariance_matrix[1][1]) ** 0.5
            # a can be positive or negative
            self.a_upper = self.a + (Z * self.a_SE)
            self.a_lower = self.a + (-Z * self.a_SE)
            # b is strictly positive
            self.b_upper = self.b * (np.exp(Z * (self.b_SE / self.b)))
            self.b_lower = self.b * (np.exp(-Z * (self.b_SE / self.b)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_Exponential model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.b_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.b_upper = self.b
            self.b_lower = self.b

        # results dataframe
        results_data = {
            "Parameter": ["a", "b"],
            "Point Estimate": [self.a, self.b],
            "Standard Error": [self.a_SE, self.b_SE],
            "Lower CI": [self.a_lower, self.b_lower],
            "Upper CI": [self.a_upper, self.b_upper],
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
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        def life_func(S1):
            return self.b * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Exponential_Distribution(Lambda=self.Lambda_at_use_stress)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))  # 1/lambda
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups)
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (common_betas[i] - betas_for_change_df[i]) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(str("+" + str(round(beta_diff * 100, 2)) + "%"))
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_Exponential (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate.",
                    ),
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")
            print(
                "If this model is being used for the Arrhenius Model, a = Ea/K_B ==> Ea =",
                round(self.a * 8.617333262145 * 10**-5, 5),
                "eV\n",
            )

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + round_and_string(use_level_stress)
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n",
                    ),
                )

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, T, a, b):  # Log PDF
        life = b * anp.exp(a / T)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, T, a, b):  # Log SF
        life = b * anp.exp(a / T)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Exponential.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Exponential.logR(t_rc, T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Eyring:
    """This function will fit the Exponential-Eyring life-stress model to the
    data provided. Please see the online documentation for the equations of this
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
    c : float
        The fitted parameter from the Exponential model
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
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    c_upper : float
        The upper CI estimate of the parameter
    c_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (equivalent Weibull
        alpha and beta) at each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    Lambda_at_use_stress : float
        The equivalent Exponential Lambda parameter at the use level stress
        (only provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Exponential distribution at the use level stress (only provided if
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
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):
        inputs = alt_single_stress_fitters_input_checking(
            dist="Exponential",
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
        LL_func = Fit_Exponential_Eyring.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(model="Eyring", failures=failures, stress_1_array=failure_stress)

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1]]  # a, c

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Eyring",
            dist="Exponential",
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
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c]
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
            # a can be positive or negative
            self.a_upper = self.a + (Z * self.a_SE)
            self.a_lower = self.a + (-Z * self.a_SE)
            # c can be positive or negative
            self.c_upper = self.c + (Z * self.c_SE)
            self.c_lower = self.c + (-Z * self.c_SE)
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_Eyring model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.c_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.c_upper = self.c
            self.c_lower = self.c

        # results dataframe
        results_data = {
            "Parameter": ["a", "c"],
            "Point Estimate": [self.a, self.c],
            "Standard Error": [self.a_SE, self.c_SE],
            "Lower CI": [self.a_lower, self.c_lower],
            "Upper CI": [self.a_upper, self.c_upper],
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
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        def life_func(S1):
            return 1 / S1 * np.exp(-(self.c - self.a / S1))

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Exponential_Distribution(Lambda=self.Lambda_at_use_stress)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups)
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (common_betas[i] - betas_for_change_df[i]) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(str("+" + str(round(beta_diff * 100, 2)) + "%"))
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_Eyring (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The beta parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate.",
                    ),
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + round_and_string(use_level_stress)
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n",
                    ),
                )

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Eyring",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Eyring",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, T, a, c):  # Log PDF
        life = 1 / T * anp.exp(-(c - a / T))
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, T, a, c):  # Log SF
        life = 1 / T * anp.exp(-(c - a / T))
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Eyring.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Eyring.logR(t_rc, T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Power:
    """This function will Fit the Exponential-Power life-stress model to the data
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
    a_upper : float
        The upper CI estimate of the parameter
    a_lower : float
        The lower CI estimate of the parameter
    n_upper : float
        The upper CI estimate of the parameter
    n_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (equivalent Weibull
        alpha and beta) at each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    Lambda_at_use_stress : float
        The equivalent Exponential Lambda parameter at the use level stress
        (only provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Exponential distribution at the use level stress (only provided if
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
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):
        inputs = alt_single_stress_fitters_input_checking(
            dist="Exponential",
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
        LL_func = Fit_Exponential_Power.LL
        self.__failure_groups = inputs.failure_groups
        self.__right_censored_groups = inputs.right_censored_groups
        self.__stresses_for_groups = inputs.stresses_for_groups

        # obtain the initial guess for the life stress model and the life distribution
        life_stress_guess = ALT_least_squares(model="Power", failures=failures, stress_1_array=failure_stress)

        # obtain the common shape parameter
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [life_stress_guess[0], life_stress_guess[1]]  # a, n

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Power",
            dist="Exponential",
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
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.n]
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
            # a is strictly positive
            self.a_upper = self.a * (np.exp(Z * (self.a_SE / self.a)))
            self.a_lower = self.a * (np.exp(-Z * (self.a_SE / self.a)))
            # n can be positive or negative
            self.n_upper = self.n + (Z * self.n_SE)
            self.n_lower = self.n + (-Z * self.n_SE)
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_Power model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.n_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.n_upper = self.n
            self.n_lower = self.n

        # results dataframe
        results_data = {
            "Parameter": ["a", "n"],
            "Point Estimate": [self.a, self.n],
            "Standard Error": [self.a_SE, self.n_SE],
            "Lower CI": [self.a_lower, self.n_lower],
            "Upper CI": [self.a_upper, self.n_upper],
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
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        def life_func(S1):
            return self.a * S1**self.n

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress)
            self.distribution_at_use_stress = Exponential_Distribution(Lambda=self.Lambda_at_use_stress)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress))
            if use_level_stress is not None:
                AF.append(life_func(S1=use_level_stress) / life_func(S1=stress))
        common_betas = np.ones_like(stresses_for_groups)
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (common_betas[i] - betas_for_change_df[i]) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(str("+" + str(round(beta_diff * 100, 2)) + "%"))
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate.",
                    ),
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + round_and_string(use_level_stress)
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n",
                    ),
                )

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Power",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, T, a, n):  # Log PDF
        life = a * T**n
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, T, a, n):  # Log SF
        life = a * T**n
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Power.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Power.logR(t_rc, T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Dual_Exponential:
    """This function will Fit the Exponential_Dual_Exponential life-stress model to
    the data provided. Please see the online documentation for the equations of
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
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (equivalent Weibull
        alpha and beta) at each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    Lambda_at_use_stress : float
        The equivalent Exponential Lambda parameter at the use level stress
        (only provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Exponential distribution at the use level stress (only provided if
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
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):
        inputs = alt_fitters_dual_stress_input_checking(
            dist="Exponential",
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
        LL_func = Fit_Exponential_Dual_Exponential.LL
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
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
        ]  # a, b, c

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Dual_Exponential",
            dist="Exponential",
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
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.b, self.c]
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
            # a can be positive or negative
            self.a_upper = self.a + (Z * self.a_SE)
            self.a_lower = self.a + (-Z * self.a_SE)
            # b can be positive or negative
            self.b_upper = self.b + (Z * self.b_SE)
            self.b_lower = self.b + (-Z * self.b_SE)
            # c is strictly positive
            self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
            self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_Dual_Exponential model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.b_SE = 0
            self.c_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.b_upper = self.b
            self.b_lower = self.b
            self.c_upper = self.c
            self.c_lower = self.c

        # results dataframe
        results_data = {
            "Parameter": ["a", "b", "c"],
            "Point Estimate": [self.a, self.b, self.c],
            "Standard Error": [self.a_SE, self.b_SE, self.c_SE],
            "Lower CI": [self.a_lower, self.b_lower, self.c_lower],
            "Upper CI": [self.a_upper, self.b_upper, self.c_upper],
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
            self.AICc = "Insufficient data"
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
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            self.distribution_at_use_stress = Exponential_Distribution(Lambda=self.Lambda_at_use_stress)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        AF = []
        stresses_for_groups_str = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(str(round_and_string(stress[0]) + ", " + round_and_string(stress[1])))
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1]) / life_func(S1=stress[0], S2=stress[1]),
                )
        common_betas = np.ones(len(stresses_for_groups))
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (common_betas[i] - betas_for_change_df[i]) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(str("+" + str(round(beta_diff * 100, 2)) + "%"))
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_Dual_Exponential (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate.",
                    ),
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + round_and_string(use_level_stress[0])
                        + ", "
                        + round_and_string(use_level_stress[1])
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n",
                    ),
                )

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Dual_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Dual_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, S1, S2, a, b, c):  # Log PDF
        life = c * anp.exp(a / S1 + b / S2)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, S1, S2, a, b, c):  # Log SF
        life = c * anp.exp(a / S1 + b / S2)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Dual_Exponential.logf(t_f, S1_f, S2_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Dual_Exponential.logR(t_rc, S1_rc, S2_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Power_Exponential:
    """This function will Fit the Exponential_Power_Exponential life-stress model
    to the data provided. Please see the online documentation for the equations
    of this model.

    This model is most appropriate to model a life-stress relationship with
    thermal and non-thermal stresses. It is essential that you ensure your
    thermal stress is stress_1 (as it will be modeled by the Exponential) and
    your non-thermal stress is stress_2 (as it will be modeled by the Power).
    Also ensure that your temperature data are in Kelvin.

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress_1 : array, list
        The corresponding stress 1 (thermal stress) at which each failure
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
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (equivalent Weibull
        alpha and beta) at each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    Lambda_at_use_stress : float
        The equivalent Exponential Lambda parameter at the use level stress
        (only provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Exponential distribution at the use level stress (only provided if
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
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):
        inputs = alt_fitters_dual_stress_input_checking(
            dist="Exponential",
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
        LL_func = Fit_Exponential_Power_Exponential.LL
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
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
        ]  # a, c, n

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Power_Exponential",
            dist="Exponential",
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
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.a, self.c, self.n]
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
            # a can be positive or negative
            self.a_upper = self.a + (Z * self.a_SE)
            self.a_lower = self.a + (-Z * self.a_SE)
            # c is strictly positive
            self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
            self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
            # n can be positive or negative
            self.n_upper = self.n + (Z * self.n_SE)
            self.n_lower = self.n + (-Z * self.n_SE)
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_Power_Exponential model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.a_SE = 0
            self.c_SE = 0
            self.n_SE = 0
            self.a_upper = self.a
            self.a_lower = self.a
            self.c_upper = self.c
            self.c_lower = self.c
            self.n_upper = self.n
            self.n_lower = self.n

        # results dataframe
        results_data = {
            "Parameter": ["a", "c", "n"],
            "Point Estimate": [self.a, self.c, self.n],
            "Standard Error": [self.a_SE, self.c_SE, self.n_SE],
            "Lower CI": [self.a_lower, self.c_lower, self.n_lower],
            "Upper CI": [self.a_upper, self.c_upper, self.n_upper],
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
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        def life_func(S1, S2):
            return self.c * (S2**self.n) * np.exp(self.a / S1)

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            self.distribution_at_use_stress = Exponential_Distribution(Lambda=self.Lambda_at_use_stress)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(str(round_and_string(stress[0]) + ", " + round_and_string(stress[1])))
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1]) / life_func(S1=stress[0], S2=stress[1]),
                )
        common_betas = np.ones(len(stresses_for_groups))
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (common_betas[i] - betas_for_change_df[i]) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(str("+" + str(round(beta_diff * 100, 2)) + "%"))
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_Power_Exponential (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate.",
                    ),
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + round_and_string(use_level_stress[0])
                        + ", "
                        + round_and_string(use_level_stress[1])
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n",
                    ),
                )

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Power_Exponential",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Power_Exponential",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, S1, S2, a, c, n):  # Log PDF
        life = c * S2**n * anp.exp(a / S1)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, S1, S2, a, c, n):  # Log SF
        life = c * S2**n * anp.exp(a / S1)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Power_Exponential.logf(t_f, S1_f, S2_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Power_Exponential.logR(t_rc, S1_rc, S2_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)


class Fit_Exponential_Dual_Power:
    """This function will Fit the Exponential_Dual_Power life-stress model to the
    data provided. Please see the online documentation for the equations of
    this model.

    This model is most appropriate to model a life-stress relationship with
    two non-thermal stresses such as voltage and load.

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
    n : float
        The fitted parameter from the Dual_Power model
    m : float
        The fitted parameter from the Dual_Power model
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
    n_SE : float
        The standard error (sqrt(variance)) of the parameter
    m_SE : float
        The standard error (sqrt(variance)) of the parameter
    c_upper : float
        The upper CI estimate of the parameter
    c_lower : float
        The lower CI estimate of the parameter
    n_upper : float
        The upper CI estimate of the parameter
    n_lower : float
        The lower CI estimate of the parameter
    m_upper : float
        The upper CI estimate of the parameter
    m_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    goodness_of_fit : dataframe
        A dataframe of the goodness of fit criterion (Log-likelihood, AICc, BIC)
    change_of_parameters : dataframe
        A dataframe showing the change of the parameters (equivalent Weibull
        alpha and beta) at each stress level.
    mean_life : float
        The mean life at the use_level_stress (only provided if use_level_stress
        is provided).
    Lambda_at_use_stress : float
        The equivalent Exponential Lambda parameter at the use level stress
        (only provided if use_level_stress is provided).
    distribution_at_use_stress : object
        The Exponential distribution at the use level stress (only provided if
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
        show_probability_plot=True,
        show_life_stress_plot=True,
        print_results=True,
    ):
        inputs = alt_fitters_dual_stress_input_checking(
            dist="Exponential",
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
        LL_func = Fit_Exponential_Dual_Power.LL
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
        betas = []
        betas_for_change_df = []
        alphas_for_change_df = []
        for i in range(len(stresses_for_groups)):
            f = failure_groups[i]
            rc = None if right_censored_groups is None else right_censored_groups[i]

            if len(f) > 1:
                fit = Fit_Weibull_2P(
                    failures=f,
                    right_censored=rc,
                    print_results=False,
                    show_probability_plot=False,
                )
                betas.append(fit.beta)
                betas_for_change_df.append(fit.beta)
                alphas_for_change_df.append(fit.alpha)
            else:  # 1 failure at this stress
                betas_for_change_df.append(0)
                alphas_for_change_df.append("")

        # compile the guess for the MLE method
        guess = [
            life_stress_guess[0],
            life_stress_guess[1],
            life_stress_guess[2],
        ]  # c, m, n

        # fit the model using the MLE method
        MLE_results = ALT_MLE_optimization(
            model="Dual_Power",
            dist="Exponential",
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
        self.success = MLE_results.success
        self.optimizer = MLE_results.optimizer

        # confidence interval estimates of parameters
        Z = -ss.norm.ppf((1 - CI) / 2)
        params = [self.c, self.m, self.n]
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
            # c is strictly positive
            self.c_upper = self.c * (np.exp(Z * (self.c_SE / self.c)))
            self.c_lower = self.c * (np.exp(-Z * (self.c_SE / self.c)))
            # m can be positive or negative
            self.m_upper = self.m + (Z * self.m_SE)
            self.m_lower = self.m + (-Z * self.m_SE)
            # n can be positive or negative
            self.n_upper = self.n + (Z * self.n_SE)
            self.n_lower = self.n + (-Z * self.n_SE)
        except LinAlgError:
            # this exception is rare but can occur with some optimizers and small data sets
            colorprint(
                str(
                    "WARNING: The hessian matrix obtained using the "
                    + self.optimizer
                    + " optimizer is non-invertable for the Exponential_Dual_Power model.\n"
                    "Confidence interval estimates of the parameters could not be obtained.\n"
                    "You may want to try fitting the model using a different optimizer.",
                ),
                text_color="red",
            )
            self.c_SE = 0
            self.m_SE = 0
            self.n_SE = 0
            self.c_upper = self.c
            self.c_lower = self.c
            self.m_upper = self.m
            self.m_lower = self.m
            self.n_upper = self.n
            self.n_lower = self.n

        # results dataframe
        results_data = {
            "Parameter": ["c", "m", "n"],
            "Point Estimate": [self.c, self.m, self.n],
            "Standard Error": [self.c_SE, self.m_SE, self.n_SE],
            "Lower CI": [self.c_lower, self.m_lower, self.n_lower],
            "Upper CI": [self.c_upper, self.m_upper, self.n_upper],
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
            self.AICc = "Insufficient data"
        self.BIC = np.log(n) * k + LL2
        GoF_data = {
            "Goodness of fit": ["Log-likelihood", "AICc", "BIC"],
            "Value": [self.loglik, self.AICc, self.BIC],
        }
        self.goodness_of_fit = pd.DataFrame(data=GoF_data, columns=["Goodness of fit", "Value"])

        def life_func(S1, S2):
            return self.c * (S1**self.m) * (S2**self.n)

        # use level stress calculations
        if use_level_stress is not None:
            self.Lambda_at_use_stress = 1 / life_func(S1=use_level_stress[0], S2=use_level_stress[1])
            self.distribution_at_use_stress = Exponential_Distribution(Lambda=self.Lambda_at_use_stress)
            self.mean_life = self.distribution_at_use_stress.mean

        # change of parameters dataframe
        new_alphas = []
        stresses_for_groups_str = []
        AF = []
        for stress in stresses_for_groups:
            new_alphas.append(life_func(S1=stress[0], S2=stress[1]))
            stresses_for_groups_str.append(str(round_and_string(stress[0]) + ", " + round_and_string(stress[1])))
            if use_level_stress is not None:
                AF.append(
                    life_func(S1=use_level_stress[0], S2=use_level_stress[1]) / life_func(S1=stress[0], S2=stress[1]),
                )
        common_betas = np.ones(len(stresses_for_groups))
        beta_differences = []
        shape_change_exceeded = False
        for i in range(len(stresses_for_groups)):
            if betas_for_change_df[i] == 0:
                betas_for_change_df[i] = ""  # replace with space
                beta_differences.append("")
            else:
                beta_diff = (common_betas[i] - betas_for_change_df[i]) / betas_for_change_df[i]
                if abs(beta_diff) > shape_change_threshold:
                    shape_change_exceeded = True
                if beta_diff > 0:
                    beta_differences.append(str("+" + str(round(beta_diff * 100, 2)) + "%"))
                else:
                    beta_differences.append(str(str(round(beta_diff * 100, 2)) + "%"))
        self.__scale_for_change_df = alphas_for_change_df
        self.__shape_for_change_df = betas_for_change_df

        if use_level_stress is not None:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
                "acceleration factor": AF,
            }
        else:
            change_of_parameters_data = {
                "stress": stresses_for_groups_str,
                "weibull alpha": alphas_for_change_df,
                "weibull beta": betas_for_change_df,
                "new 1/Lambda": new_alphas,
                "common shape": common_betas,
                "shape change": beta_differences,
            }
        self.change_of_parameters = pd.DataFrame(
            data=change_of_parameters_data,
            columns=list(change_of_parameters_data.keys()),
        )

        if print_results is True:
            n = len(failures) + len(right_censored)
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            frac_censored = len(right_censored) / n * 100
            if frac_censored % 1 < 1e-10:
                frac_censored = int(frac_censored)
            colorprint(
                str("Results from Fit_Exponential_Dual_Power (" + str(CI_rounded) + "% CI):"),
                bold=True,
                underline=True,
            )
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            if self.optimizer is not None:
                print("Optimizer:", self.optimizer)
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + round_and_string(frac_censored) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")
            print(self.change_of_parameters.to_string(index=False))
            if shape_change_exceeded is True:
                print(
                    str(
                        "The shape parameter has been found to change significantly (>"
                        + str(int(shape_change_threshold * 100))
                        + "%) when fitting the ALT model.\nThis may indicate that a different failure mode is acting at different stress levels or that the Exponential distribution may not be appropriate.",
                    ),
                )
            print("\n", self.goodness_of_fit.to_string(index=False), "\n")

            if use_level_stress is not None:
                print(
                    str(
                        "At the use level stress of "
                        + round_and_string(use_level_stress[0])
                        + ", "
                        + round_and_string(use_level_stress[1])
                        + ", the mean life is "
                        + str(round(self.mean_life, 5))
                        + "\n",
                    ),
                )

        self.probability_plot = ALT_prob_plot(
            dist="Exponential",
            model="Dual_Power",
            stresses_for_groups=stresses_for_groups,
            failure_groups=failure_groups,
            right_censored_groups=right_censored_groups,
            life_func=life_func,
            shape=None,
            scale_for_change_df=alphas_for_change_df,
            shape_for_change_df=betas_for_change_df,
            use_level_stress=use_level_stress,
            ax=show_probability_plot,
        )

        self.life_stress_plot = life_stress_plot(
            dist="Exponential",
            model="Dual_Power",
            life_func=life_func,
            failure_groups=failure_groups,
            stresses_for_groups=stresses_for_groups,
            use_level_stress=use_level_stress,
            ax=show_life_stress_plot,
        )

    @staticmethod
    def logf(t, S1, S2, c, m, n):  # Log PDF
        life = c * (S1**m) * (S2**n)
        return anp.log(1 / life) - 1 / life * t

    @staticmethod
    def logR(t, S1, S2, c, m, n):  # Log SF
        life = c * (S1**m) * (S2**n)
        return -(1 / life * t)

    @staticmethod
    def LL(params, t_f, t_rc, S1_f, S2_f, S1_rc, S2_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Dual_Power.logf(t_f, S1_f, S2_f, params[0], params[1], params[2]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Dual_Power.logR(t_rc, S1_rc, S2_rc, params[0], params[1], params[2]).sum()
        return -(LL_f + LL_rc)

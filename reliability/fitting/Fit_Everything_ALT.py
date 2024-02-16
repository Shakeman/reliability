import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reliability.ALT_fitters import (
    Fit_Exponential_Dual_Exponential,
    Fit_Exponential_Dual_Power,
    Fit_Exponential_Exponential,
    Fit_Exponential_Eyring,
    Fit_Exponential_Power,
    Fit_Exponential_Power_Exponential,
    Fit_Lognormal_Dual_Exponential,
    Fit_Lognormal_Dual_Power,
    Fit_Lognormal_Exponential,
    Fit_Lognormal_Eyring,
    Fit_Lognormal_Power,
    Fit_Lognormal_Power_Exponential,
    Fit_Normal_Dual_Exponential,
    Fit_Normal_Dual_Power,
    Fit_Normal_Exponential,
    Fit_Normal_Eyring,
    Fit_Normal_Power,
    Fit_Normal_Power_Exponential,
    Fit_Weibull_Dual_Exponential,
    Fit_Weibull_Dual_Power,
    Fit_Weibull_Exponential,
    Fit_Weibull_Eyring,
    Fit_Weibull_Power,
    Fit_Weibull_Power_Exponential,
)
from reliability.Distributions import (
    Exponential_Distribution,
    Lognormal_Distribution,
    Normal_Distribution,
    Weibull_Distribution,
)
from reliability.Utils import (
    ALT_fitters_input_checking,
    colorprint,
    round_and_string,
)


class Fit_Everything_ALT:
    """This function will fit all available ALT models for the data you enter,
    which may include right censored data.

    ALT models are either single stress (Exponential, Eyring, Power) or dual
    stress (Dual_Exponential, Power_Exponential, Dual_Power).

    Depending on the data you enter (ie. whether failure_stress_2 is provided),
    the applicable set of ALT models will be fitted.

    Parameters
    ----------
    failures : array, list
        The failure data.
    failure_stress_1 : array, list, optional
        The corresponding stresses (such as temperature or voltage) at which
        each failure occurred. This must match the length of failures as each
        failure is tied to a failure stress. Alternative keyword of
        failure_stress is accepted in place of failure_stress_1.
    failure_stress_2 : array, list, optional
        The corresponding stresses (such as temperature or voltage) at which
        each failure occurred. This must match the length of failures as each
        failure is tied to a failure stress. Optional input. Providing this will
        trigger the use of dual stress models. Leaving this empty will trigger
        the use of single stress models.
    right_censored : array, list, optional
        The right censored failure times. Optional input.
    right_censored_stress_1 : array, list, optional
        The corresponding stresses (such as temperature or voltage) at which
        each right_censored data point was obtained. This must match the length
        of right_censored as each right_censored value is tied to a
        right_censored stress. Conditionally optional input. This must be
        provided if right_censored is provided. Alternative keyword of
        right_censored_stress is accepted in place of right_censored_stress_1.
    right_censored_stress_2 : array, list, optional
        The corresponding stresses (such as temperature or voltage) at which
        each right_censored data point was obtained. This must match the length
        of right_censored as each right_censored value is tied to a
        right_censored stress. Conditionally optional input. This must be
        provided if failure_stress_2 is provided.
    use_level_stress : int, float, list, array, optional
        The use level stress at which you want to know the mean life. Optional
        input. This must be a list or array [stress_1,stress_2] if
        failure_stress_2 is provided and you want to know the mean life.
    print_results : bool, optional
        True/False. Default is True. Prints the results to the console.
    show_probability_plot : bool, optional
        True/False. Default is True. Provides a probability plot of each of the
        fitted ALT model.
    show_best_distribution_probability_plot : bool, optional
        True/False. Defaults to True. Provides a probability plot in a new
        figure of the best ALT model.
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
    sort_by : str, optional
        Goodness of fit test to sort results by. Must be 'BIC','AICc', or
        'Log-likelihood'. Default is 'BIC'.
    exclude : list, array, optional
        A list or array of strings specifying which distributions to exclude.
        Default is None. Options are: Weibull_Exponential, Weibull_Eyring,
        Weibull_Power, Weibull_Dual_Exponential, Weibull_Power_Exponential,
        Weibull_Dual_Power, Lognormal_Exponential, Lognormal_Eyring,
        Lognormal_Power, Lognormal_Dual_Exponential,
        Lognormal_Power_Exponential, Lognormal_Dual_Power, Normal_Exponential,
        Normal_Eyring, Normal_Power, Normal_Dual_Exponential,
        Normal_Power_Exponential, Normal_Dual_Power, Exponential_Exponential,
        Exponential_Eyring, Exponential_Power, Exponential_Dual_Exponential,
        Exponential_Power_Exponential, Exponential_Dual_Power
    kwargs
        Accepts failure_stress and right_censored_stress as alternative keywords
        to failure_stress_1 and right_censored_stress_1. This is used to provide
        consistency with the other functions in ALT_Fitters which also accept
        failure_stress and right_censored_stress.

    Returns
    -------
    results : dataframe
        The dataframe of results. Fitted parameters in this dataframe may be
        accessed by name. See below example.
    best_model_name : str
        The name of the best fitting ALT model. E.g. 'Weibull_Exponential'. See
        above list for exclude.
    best_model_at_use_stress : object
        A distribution object created based on the parameters of the best
        fitting ALT model at the use stress. This is only provided if the
        use_level_stress is provided. This is because use_level_stress is
        required to find the scale parameter.
    parameters and goodness of fit results : float
        This is provided for each fitted model. For example, the
        Weibull_Exponential model values are Weibull_Exponential_a,
        Weibull_Exponential_b, Weibull_Exponential_beta,
        Weibull_Exponential_BIC, Weibull_Exponential_AICc,
        Weibull_Exponential_loglik
    excluded_models : list
        A list of the models which were excluded. This will always include at
        least half the models since only single stress OR dual stress can be
        fitted depending on the data.
    probability_plot : object
        The figure handle from the probability plot (only provided if
        show_probability_plot is True).
    best_distribution_probability_plot : object
        The figure handle from the best distribution probability plot (only
        provided if show_best_distribution_probability_plot is True).

    Notes
    -----
    From the results, the models are sorted based on their goodness of fit test
    results, where the smaller the goodness of fit value, the better the fit of
    the model to the data.

    Example Usage:

    .. code:: python

        failures = [619, 417, 173, 161, 1016, 512, 999, 1131, 1883, 2413, 3105, 2492]
        failure_stresses = [500, 500, 500, 500, 400, 400, 400, 400, 350, 350, 350, 350]
        right_censored = [29, 180, 1341]
        right_censored_stresses = [500, 400, 350]
        use_level_stress = 300
        output = Fit_Everything_ALT(failures=failures,failure_stress_1=failure_stresses,right_censored=right_censored, right_censored_stress_1=right_censored_stresses, use_level_stress=use_level_stress)

        # To extract the parameters of the Weibull_Exponential model from the results dataframe, you may access the parameters by name:
        print('Weibull Exponential beta =',output.Weibull_Exponential_beta)
        >>> Weibull Exponential beta = 3.0807072337386123

    """

    def __init__(
        self,
        failures,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        show_probability_plot=True,
        show_best_distribution_probability_plot=True,
        print_results=True,
        exclude=None,
        sort_by="BIC",
        **kwargs,
    ):
        # check kwargs for failure_stress and right_censored_stress
        if "failure_stress" in kwargs and failure_stress_1 is None:
            failure_stress_1 = kwargs.pop("failure_stress")
        elif "failure_stress" in kwargs and failure_stress_1 is not None:
            colorprint(
                "failure_stress has been ignored because failure_stress_1 was provided.",
                text_color="red",
            )

        if "right_censored_stress" in kwargs and right_censored_stress_1 is None:
            right_censored_stress_1 = kwargs.pop("right_censored_stress")
        elif "right_censored_stress" in kwargs and right_censored_stress_1 is not None:
            colorprint(
                "right_censored_stress has been ignored because right_censored_stress_1 was provided.",
                text_color="red",
            )

        inputs = ALT_fitters_input_checking(
            dist="Everything",
            life_stress_model="Everything",
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

        # these are only here for code formatting reasons. They get redefined later
        self._Fit_Everything_ALT__Weibull_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Eyring_params = None
        self._Fit_Everything_ALT__Weibull_Power_params = None
        self._Fit_Everything_ALT__Lognormal_Exponential_params = None
        self._Fit_Everything_ALT__Lognormal_Eyring_params = None
        self._Fit_Everything_ALT__Lognormal_Power_params = None
        self._Fit_Everything_ALT__Normal_Exponential_params = None
        self._Fit_Everything_ALT__Normal_Eyring_params = None
        self._Fit_Everything_ALT__Normal_Power_params = None
        self._Fit_Everything_ALT__Exponential_Exponential_params = None
        self._Fit_Everything_ALT__Exponential_Eyring_params = None
        self._Fit_Everything_ALT__Exponential_Power_params = None
        self._Fit_Everything_ALT__Weibull_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Power_Exponential_params = None
        self._Fit_Everything_ALT__Weibull_Dual_Power_params = None
        self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Lognormal_Power_Exponential_params = None
        self._Fit_Everything_ALT__Lognormal_Dual_Power_params = None
        self._Fit_Everything_ALT__Normal_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Normal_Power_Exponential_params = None
        self._Fit_Everything_ALT__Normal_Dual_Power_params = None
        self._Fit_Everything_ALT__Exponential_Dual_Exponential_params = None
        self._Fit_Everything_ALT__Exponential_Power_Exponential_params = None
        self._Fit_Everything_ALT__Exponential_Dual_Power_params = None

        # for passing to the probability plot
        self.__use_level_stress = use_level_stress

        if print_results not in [True, False]:
            raise ValueError("print_results must be either True or False. Defaults is True.")
        if show_probability_plot not in [True, False]:
            raise ValueError("show_probability_plot must be either True or False. Default is True.")
        if show_best_distribution_probability_plot not in [True, False]:
            raise ValueError("show_best_distribution_probability_plot must be either True or False. Default is True.")

        single_stress_ALT_models_list = [
            "Weibull_Exponential",
            "Weibull_Eyring",
            "Weibull_Power",
            "Lognormal_Exponential",
            "Lognormal_Eyring",
            "Lognormal_Power",
            "Normal_Exponential",
            "Normal_Eyring",
            "Normal_Power",
            "Exponential_Exponential",
            "Exponential_Eyring",
            "Exponential_Power",
        ]

        dual_stress_ALT_models_list = [
            "Weibull_Dual_Exponential",
            "Weibull_Power_Exponential",
            "Weibull_Dual_Power",
            "Lognormal_Dual_Exponential",
            "Lognormal_Power_Exponential",
            "Lognormal_Dual_Power",
            "Normal_Dual_Exponential",
            "Normal_Power_Exponential",
            "Normal_Dual_Power",
            "Exponential_Dual_Exponential",
            "Exponential_Power_Exponential",
            "Exponential_Dual_Power",
        ]
        all_ALT_models_list = single_stress_ALT_models_list + dual_stress_ALT_models_list

        excluded_models = []
        unknown_exclusions = []
        if exclude is not None:
            for item in exclude:
                if item.title() in all_ALT_models_list:
                    excluded_models.append(item.title())
                else:
                    unknown_exclusions.append(item)
            if len(unknown_exclusions) > 0:
                colorprint(
                    str(
                        "WARNING: The following items were not recognised ALT models to exclude: "
                        + str(unknown_exclusions),
                    ),
                    text_color="red",
                )
                colorprint("Available ALT models to exclude are:", text_color="red")
                for item in all_ALT_models_list:
                    colorprint(item, text_color="red")

        if len(failure_stress_2) > 0:
            dual_stress = True
            excluded_models.extend(single_stress_ALT_models_list)
        else:
            dual_stress = False
            excluded_models.extend(dual_stress_ALT_models_list)
        self.excluded_models = excluded_models

        # create an empty dataframe to append the data from the fitted distributions
        if dual_stress is True:
            df = pd.DataFrame(
                columns=[
                    "ALT_model",
                    "a",
                    "b",
                    "c",
                    "m",
                    "n",
                    "beta",
                    "sigma",
                    "Log-likelihood",
                    "AICc",
                    "BIC",
                    "optimizer",
                ],
            )
        else:  # same df but without column m
            df = pd.DataFrame(
                columns=[
                    "ALT_model",
                    "a",
                    "b",
                    "c",
                    "n",
                    "beta",
                    "sigma",
                    "Log-likelihood",
                    "AICc",
                    "BIC",
                    "optimizer",
                ],
            )

        # Fit the parametric models and extract the fitted parameters
        if "Weibull_Exponential" not in self.excluded_models:
            self.__Weibull_Exponential_params = Fit_Weibull_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Exponential_a = self.__Weibull_Exponential_params.a
            self.Weibull_Exponential_b = self.__Weibull_Exponential_params.b
            self.Weibull_Exponential_beta = self.__Weibull_Exponential_params.beta
            self.Weibull_Exponential_loglik = self.__Weibull_Exponential_params.loglik
            self.Weibull_Exponential_BIC = self.__Weibull_Exponential_params.BIC
            self.Weibull_Exponential_AICc = self.__Weibull_Exponential_params.AICc
            self.Weibull_Exponential_optimizer = self.__Weibull_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Weibull_Exponential"],
                            "a": [self.Weibull_Exponential_a],
                            "b": [self.Weibull_Exponential_b],
                            "c": [""],
                            "n": [""],
                            "beta": [self.Weibull_Exponential_beta],
                            "sigma": [""],
                            "Log-likelihood": [self.Weibull_Exponential_loglik],
                            "AICc": [self.Weibull_Exponential_AICc],
                            "BIC": [self.Weibull_Exponential_BIC],
                            "optimizer": [self.Weibull_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_Eyring" not in self.excluded_models:
            self.__Weibull_Eyring_params = Fit_Weibull_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Eyring_a = self.__Weibull_Eyring_params.a
            self.Weibull_Eyring_c = self.__Weibull_Eyring_params.c
            self.Weibull_Eyring_beta = self.__Weibull_Eyring_params.beta
            self.Weibull_Eyring_loglik = self.__Weibull_Eyring_params.loglik
            self.Weibull_Eyring_BIC = self.__Weibull_Eyring_params.BIC
            self.Weibull_Eyring_AICc = self.__Weibull_Eyring_params.AICc
            self.Weibull_Eyring_optimizer = self.__Weibull_Eyring_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Weibull_Eyring"],
                            "a": [self.Weibull_Eyring_a],
                            "b": [""],
                            "c": [self.Weibull_Eyring_c],
                            "n": [""],
                            "beta": [self.Weibull_Eyring_beta],
                            "sigma": [""],
                            "Log-likelihood": [self.Weibull_Eyring_loglik],
                            "AICc": [self.Weibull_Eyring_AICc],
                            "BIC": [self.Weibull_Eyring_BIC],
                            "optimizer": [self.Weibull_Eyring_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_Power" not in self.excluded_models:
            self.__Weibull_Power_params = Fit_Weibull_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Power_a = self.__Weibull_Power_params.a
            self.Weibull_Power_n = self.__Weibull_Power_params.n
            self.Weibull_Power_beta = self.__Weibull_Power_params.beta
            self.Weibull_Power_loglik = self.__Weibull_Power_params.loglik
            self.Weibull_Power_BIC = self.__Weibull_Power_params.BIC
            self.Weibull_Power_AICc = self.__Weibull_Power_params.AICc
            self.Weibull_Power_optimizer = self.__Weibull_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Weibull_Power"],
                            "a": [self.Weibull_Power_a],
                            "b": [""],
                            "c": [""],
                            "n": [self.Weibull_Power_n],
                            "beta": [self.Weibull_Power_beta],
                            "sigma": [""],
                            "Log-likelihood": [self.Weibull_Power_loglik],
                            "AICc": [self.Weibull_Power_AICc],
                            "BIC": [self.Weibull_Power_BIC],
                            "optimizer": [self.Weibull_Power_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_Exponential" not in self.excluded_models:
            self.__Lognormal_Exponential_params = Fit_Lognormal_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Exponential_a = self.__Lognormal_Exponential_params.a
            self.Lognormal_Exponential_b = self.__Lognormal_Exponential_params.b
            self.Lognormal_Exponential_sigma = self.__Lognormal_Exponential_params.sigma
            self.Lognormal_Exponential_loglik = self.__Lognormal_Exponential_params.loglik
            self.Lognormal_Exponential_BIC = self.__Lognormal_Exponential_params.BIC
            self.Lognormal_Exponential_AICc = self.__Lognormal_Exponential_params.AICc
            self.Lognormal_Exponential_optimizer = self.__Lognormal_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Lognormal_Exponential"],
                            "a": [self.Lognormal_Exponential_a],
                            "b": [self.Lognormal_Exponential_b],
                            "c": [""],
                            "n": [""],
                            "beta": [""],
                            "sigma": [self.Lognormal_Exponential_sigma],
                            "Log-likelihood": [self.Lognormal_Exponential_loglik],
                            "AICc": [self.Lognormal_Exponential_AICc],
                            "BIC": [self.Lognormal_Exponential_BIC],
                            "optimizer": [self.Lognormal_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_Eyring" not in self.excluded_models:
            self.__Lognormal_Eyring_params = Fit_Lognormal_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Eyring_a = self.__Lognormal_Eyring_params.a
            self.Lognormal_Eyring_c = self.__Lognormal_Eyring_params.c
            self.Lognormal_Eyring_sigma = self.__Lognormal_Eyring_params.sigma
            self.Lognormal_Eyring_loglik = self.__Lognormal_Eyring_params.loglik
            self.Lognormal_Eyring_BIC = self.__Lognormal_Eyring_params.BIC
            self.Lognormal_Eyring_AICc = self.__Lognormal_Eyring_params.AICc
            self.Lognormal_Eyring_optimizer = self.__Lognormal_Eyring_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Lognormal_Eyring"],
                            "a": [self.Lognormal_Eyring_a],
                            "b": [""],
                            "c": [self.Lognormal_Eyring_c],
                            "n": [""],
                            "beta": [""],
                            "sigma": [self.Lognormal_Eyring_sigma],
                            "Log-likelihood": [self.Lognormal_Eyring_loglik],
                            "AICc": [self.Lognormal_Eyring_AICc],
                            "BIC": [self.Lognormal_Eyring_BIC],
                            "optimizer": [self.Lognormal_Eyring_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_Power" not in self.excluded_models:
            self.__Lognormal_Power_params = Fit_Lognormal_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Power_a = self.__Lognormal_Power_params.a
            self.Lognormal_Power_n = self.__Lognormal_Power_params.n
            self.Lognormal_Power_sigma = self.__Lognormal_Power_params.sigma
            self.Lognormal_Power_loglik = self.__Lognormal_Power_params.loglik
            self.Lognormal_Power_BIC = self.__Lognormal_Power_params.BIC
            self.Lognormal_Power_AICc = self.__Lognormal_Power_params.AICc
            self.Lognormal_Power_optimizer = self.__Lognormal_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Lognormal_Power"],
                            "a": [self.Lognormal_Power_a],
                            "b": [""],
                            "c": [""],
                            "n": [self.Lognormal_Power_n],
                            "beta": [""],
                            "sigma": [self.Lognormal_Power_sigma],
                            "Log-likelihood": [self.Lognormal_Power_loglik],
                            "AICc": [self.Lognormal_Power_AICc],
                            "BIC": [self.Lognormal_Power_BIC],
                            "optimizer": [self.Lognormal_Power_optimizer],
                        },
                    ),
                ],
            )

        if "Normal_Exponential" not in self.excluded_models:
            self.__Normal_Exponential_params = Fit_Normal_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Exponential_a = self.__Normal_Exponential_params.a
            self.Normal_Exponential_b = self.__Normal_Exponential_params.b
            self.Normal_Exponential_sigma = self.__Normal_Exponential_params.sigma
            self.Normal_Exponential_loglik = self.__Normal_Exponential_params.loglik
            self.Normal_Exponential_BIC = self.__Normal_Exponential_params.BIC
            self.Normal_Exponential_AICc = self.__Normal_Exponential_params.AICc
            self.Normal_Exponential_optimizer = self.__Normal_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Normal_Exponential"],
                            "a": [self.Normal_Exponential_a],
                            "b": [self.Normal_Exponential_b],
                            "c": [""],
                            "n": [""],
                            "beta": [""],
                            "sigma": [self.Normal_Exponential_sigma],
                            "Log-likelihood": [self.Normal_Exponential_loglik],
                            "AICc": [self.Normal_Exponential_AICc],
                            "BIC": [self.Normal_Exponential_BIC],
                            "optimizer": [self.Normal_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Normal_Eyring" not in self.excluded_models:
            self.__Normal_Eyring_params = Fit_Normal_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Eyring_a = self.__Normal_Eyring_params.a
            self.Normal_Eyring_c = self.__Normal_Eyring_params.c
            self.Normal_Eyring_sigma = self.__Normal_Eyring_params.sigma
            self.Normal_Eyring_loglik = self.__Normal_Eyring_params.loglik
            self.Normal_Eyring_BIC = self.__Normal_Eyring_params.BIC
            self.Normal_Eyring_AICc = self.__Normal_Eyring_params.AICc
            self.Normal_Eyring_optimizer = self.__Normal_Eyring_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Normal_Eyring"],
                            "a": [self.Normal_Eyring_a],
                            "b": [""],
                            "c": [self.Normal_Eyring_c],
                            "n": [""],
                            "beta": [""],
                            "sigma": [self.Normal_Eyring_sigma],
                            "Log-likelihood": [self.Normal_Eyring_loglik],
                            "AICc": [self.Normal_Eyring_AICc],
                            "BIC": [self.Normal_Eyring_BIC],
                            "optimizer": [self.Normal_Eyring_optimizer],
                        },
                    ),
                ],
            )

        if "Normal_Power" not in self.excluded_models:
            self.__Normal_Power_params = Fit_Normal_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Power_a = self.__Normal_Power_params.a
            self.Normal_Power_n = self.__Normal_Power_params.n
            self.Normal_Power_sigma = self.__Normal_Power_params.sigma
            self.Normal_Power_loglik = self.__Normal_Power_params.loglik
            self.Normal_Power_BIC = self.__Normal_Power_params.BIC
            self.Normal_Power_AICc = self.__Normal_Power_params.AICc
            self.Normal_Power_optimizer = self.__Normal_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Normal_Power"],
                            "a": [self.Normal_Power_a],
                            "b": [""],
                            "c": [""],
                            "n": [self.Normal_Power_n],
                            "beta": [""],
                            "sigma": [self.Normal_Power_sigma],
                            "Log-likelihood": [self.Normal_Power_loglik],
                            "AICc": [self.Normal_Power_AICc],
                            "BIC": [self.Normal_Power_BIC],
                            "optimizer": [self.Normal_Power_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_Exponential" not in self.excluded_models:
            self.__Exponential_Exponential_params = Fit_Exponential_Exponential(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Exponential_a = self.__Exponential_Exponential_params.a
            self.Exponential_Exponential_b = self.__Exponential_Exponential_params.b
            self.Exponential_Exponential_loglik = self.__Exponential_Exponential_params.loglik
            self.Exponential_Exponential_BIC = self.__Exponential_Exponential_params.BIC
            self.Exponential_Exponential_AICc = self.__Exponential_Exponential_params.AICc
            self.Exponential_Exponential_optimizer = self.__Exponential_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Exponential_Exponential"],
                            "a": [self.Exponential_Exponential_a],
                            "b": [self.Exponential_Exponential_b],
                            "c": [""],
                            "n": [""],
                            "beta": [""],
                            "sigma": [""],
                            "Log-likelihood": [self.Exponential_Exponential_loglik],
                            "AICc": [self.Exponential_Exponential_AICc],
                            "BIC": [self.Exponential_Exponential_BIC],
                            "optimizer": [self.Exponential_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_Eyring" not in self.excluded_models:
            self.__Exponential_Eyring_params = Fit_Exponential_Eyring(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Eyring_a = self.__Exponential_Eyring_params.a
            self.Exponential_Eyring_c = self.__Exponential_Eyring_params.c
            self.Exponential_Eyring_loglik = self.__Exponential_Eyring_params.loglik
            self.Exponential_Eyring_BIC = self.__Exponential_Eyring_params.BIC
            self.Exponential_Eyring_AICc = self.__Exponential_Eyring_params.AICc
            self.Exponential_Eyring_optimizer = self.__Exponential_Eyring_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Exponential_Eyring"],
                            "a": [self.Exponential_Eyring_a],
                            "b": [""],
                            "c": [self.Exponential_Eyring_c],
                            "n": [""],
                            "beta": [""],
                            "sigma": [""],
                            "Log-likelihood": [self.Exponential_Eyring_loglik],
                            "AICc": [self.Exponential_Eyring_AICc],
                            "BIC": [self.Exponential_Eyring_BIC],
                            "optimizer": [self.Exponential_Eyring_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_Power" not in self.excluded_models:
            self.__Exponential_Power_params = Fit_Exponential_Power(
                failures=failures,
                failure_stress=failure_stress_1,
                right_censored=right_censored,
                right_censored_stress=right_censored_stress_1,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Power_a = self.__Exponential_Power_params.a
            self.Exponential_Power_n = self.__Exponential_Power_params.n
            self.Exponential_Power_loglik = self.__Exponential_Power_params.loglik
            self.Exponential_Power_BIC = self.__Exponential_Power_params.BIC
            self.Exponential_Power_AICc = self.__Exponential_Power_params.AICc
            self.Exponential_Power_optimizer = self.__Exponential_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Exponential_Power"],
                            "a": [self.Exponential_Power_a],
                            "b": [""],
                            "c": [""],
                            "n": [self.Exponential_Power_n],
                            "beta": [""],
                            "sigma": [""],
                            "Log-likelihood": [self.Exponential_Power_loglik],
                            "AICc": [self.Exponential_Power_AICc],
                            "BIC": [self.Exponential_Power_BIC],
                            "optimizer": [self.Exponential_Power_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_Dual_Exponential" not in self.excluded_models:
            self.__Weibull_Dual_Exponential_params = Fit_Weibull_Dual_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Dual_Exponential_a = self.__Weibull_Dual_Exponential_params.a
            self.Weibull_Dual_Exponential_b = self.__Weibull_Dual_Exponential_params.b
            self.Weibull_Dual_Exponential_c = self.__Weibull_Dual_Exponential_params.c
            self.Weibull_Dual_Exponential_beta = self.__Weibull_Dual_Exponential_params.beta
            self.Weibull_Dual_Exponential_loglik = self.__Weibull_Dual_Exponential_params.loglik
            self.Weibull_Dual_Exponential_BIC = self.__Weibull_Dual_Exponential_params.BIC
            self.Weibull_Dual_Exponential_AICc = self.__Weibull_Dual_Exponential_params.AICc
            self.Weibull_Dual_Exponential_optimizer = self.__Weibull_Dual_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Weibull_Dual_Exponential"],
                            "a": [self.Weibull_Dual_Exponential_a],
                            "b": [self.Weibull_Dual_Exponential_b],
                            "c": [self.Weibull_Dual_Exponential_c],
                            "m": [""],
                            "n": [""],
                            "beta": [self.Weibull_Dual_Exponential_beta],
                            "sigma": [""],
                            "Log-likelihood": [self.Weibull_Dual_Exponential_loglik],
                            "AICc": [self.Weibull_Dual_Exponential_AICc],
                            "BIC": [self.Weibull_Dual_Exponential_BIC],
                            "optimizer": [self.Weibull_Dual_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_Power_Exponential" not in self.excluded_models:
            self.__Weibull_Power_Exponential_params = Fit_Weibull_Power_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Power_Exponential_a = self.__Weibull_Power_Exponential_params.a
            self.Weibull_Power_Exponential_c = self.__Weibull_Power_Exponential_params.c
            self.Weibull_Power_Exponential_n = self.__Weibull_Power_Exponential_params.n
            self.Weibull_Power_Exponential_beta = self.__Weibull_Power_Exponential_params.beta
            self.Weibull_Power_Exponential_loglik = self.__Weibull_Power_Exponential_params.loglik
            self.Weibull_Power_Exponential_BIC = self.__Weibull_Power_Exponential_params.BIC
            self.Weibull_Power_Exponential_AICc = self.__Weibull_Power_Exponential_params.AICc
            self.Weibull_Power_Exponential_optimizer = self.__Weibull_Power_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Weibull_Power_Exponential"],
                            "a": [self.Weibull_Power_Exponential_a],
                            "b": [""],
                            "c": [self.Weibull_Power_Exponential_c],
                            "m": [""],
                            "n": [self.Weibull_Power_Exponential_n],
                            "beta": [self.Weibull_Power_Exponential_beta],
                            "sigma": [""],
                            "Log-likelihood": [self.Weibull_Power_Exponential_loglik],
                            "AICc": [self.Weibull_Power_Exponential_AICc],
                            "BIC": [self.Weibull_Power_Exponential_BIC],
                            "optimizer": [self.Weibull_Power_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Weibull_Dual_Power" not in self.excluded_models:
            self.__Weibull_Dual_Power_params = Fit_Weibull_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Weibull_Dual_Power_c = self.__Weibull_Dual_Power_params.c
            self.Weibull_Dual_Power_m = self.__Weibull_Dual_Power_params.m
            self.Weibull_Dual_Power_n = self.__Weibull_Dual_Power_params.n
            self.Weibull_Dual_Power_beta = self.__Weibull_Dual_Power_params.beta
            self.Weibull_Dual_Power_loglik = self.__Weibull_Dual_Power_params.loglik
            self.Weibull_Dual_Power_BIC = self.__Weibull_Dual_Power_params.BIC
            self.Weibull_Dual_Power_AICc = self.__Weibull_Dual_Power_params.AICc
            self.Weibull_Dual_Power_optimizer = self.__Weibull_Dual_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Weibull_Dual_Power"],
                            "a": [""],
                            "b": [""],
                            "c": [self.Weibull_Dual_Power_c],
                            "m": [self.Weibull_Dual_Power_m],
                            "n": [self.Weibull_Dual_Power_n],
                            "beta": [self.Weibull_Dual_Power_beta],
                            "sigma": [""],
                            "Log-likelihood": [self.Weibull_Dual_Power_loglik],
                            "AICc": [self.Weibull_Dual_Power_AICc],
                            "BIC": [self.Weibull_Dual_Power_BIC],
                            "optimizer": [self.Weibull_Dual_Power_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_Dual_Exponential" not in self.excluded_models:
            self.__Lognormal_Dual_Exponential_params = Fit_Lognormal_Dual_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Dual_Exponential_a = self.__Lognormal_Dual_Exponential_params.a
            self.Lognormal_Dual_Exponential_b = self.__Lognormal_Dual_Exponential_params.b
            self.Lognormal_Dual_Exponential_c = self.__Lognormal_Dual_Exponential_params.c
            self.Lognormal_Dual_Exponential_sigma = self.__Lognormal_Dual_Exponential_params.sigma
            self.Lognormal_Dual_Exponential_loglik = self.__Lognormal_Dual_Exponential_params.loglik
            self.Lognormal_Dual_Exponential_BIC = self.__Lognormal_Dual_Exponential_params.BIC
            self.Lognormal_Dual_Exponential_AICc = self.__Lognormal_Dual_Exponential_params.AICc
            self.Lognormal_Dual_Exponential_optimizer = self.__Lognormal_Dual_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Lognormal_Dual_Exponential"],
                            "a": [self.Lognormal_Dual_Exponential_a],
                            "b": [self.Lognormal_Dual_Exponential_b],
                            "c": [self.Lognormal_Dual_Exponential_c],
                            "m": [""],
                            "n": [""],
                            "beta": [""],
                            "sigma": [self.Lognormal_Dual_Exponential_sigma],
                            "Log-likelihood": [self.Lognormal_Dual_Exponential_loglik],
                            "AICc": [self.Lognormal_Dual_Exponential_AICc],
                            "BIC": [self.Lognormal_Dual_Exponential_BIC],
                            "optimizer": [self.Lognormal_Dual_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_Power_Exponential" not in self.excluded_models:
            self.__Lognormal_Power_Exponential_params = Fit_Lognormal_Power_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Power_Exponential_a = self.__Lognormal_Power_Exponential_params.a
            self.Lognormal_Power_Exponential_c = self.__Lognormal_Power_Exponential_params.c
            self.Lognormal_Power_Exponential_n = self.__Lognormal_Power_Exponential_params.n
            self.Lognormal_Power_Exponential_sigma = self.__Lognormal_Power_Exponential_params.sigma
            self.Lognormal_Power_Exponential_loglik = self.__Lognormal_Power_Exponential_params.loglik
            self.Lognormal_Power_Exponential_BIC = self.__Lognormal_Power_Exponential_params.BIC
            self.Lognormal_Power_Exponential_AICc = self.__Lognormal_Power_Exponential_params.AICc
            self.Lognormal_Power_Exponential_optimizer = self.__Lognormal_Power_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Lognormal_Power_Exponential"],
                            "a": [self.Lognormal_Power_Exponential_a],
                            "b": [""],
                            "c": [self.Lognormal_Power_Exponential_c],
                            "m": [""],
                            "n": [self.Lognormal_Power_Exponential_n],
                            "beta": [""],
                            "sigma": [self.Lognormal_Power_Exponential_sigma],
                            "Log-likelihood": [self.Lognormal_Power_Exponential_loglik],
                            "AICc": [self.Lognormal_Power_Exponential_AICc],
                            "BIC": [self.Lognormal_Power_Exponential_BIC],
                            "optimizer": [self.Lognormal_Power_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Lognormal_Dual_Power" not in self.excluded_models:
            self.__Lognormal_Dual_Power_params = Fit_Lognormal_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Lognormal_Dual_Power_c = self.__Lognormal_Dual_Power_params.c
            self.Lognormal_Dual_Power_m = self.__Lognormal_Dual_Power_params.m
            self.Lognormal_Dual_Power_n = self.__Lognormal_Dual_Power_params.n
            self.Lognormal_Dual_Power_sigma = self.__Lognormal_Dual_Power_params.sigma
            self.Lognormal_Dual_Power_loglik = self.__Lognormal_Dual_Power_params.loglik
            self.Lognormal_Dual_Power_BIC = self.__Lognormal_Dual_Power_params.BIC
            self.Lognormal_Dual_Power_AICc = self.__Lognormal_Dual_Power_params.AICc
            self.Lognormal_Dual_Power_optimizer = self.__Lognormal_Dual_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Lognormal_Dual_Power"],
                            "a": [""],
                            "b": [""],
                            "c": [self.Lognormal_Dual_Power_c],
                            "m": [self.Lognormal_Dual_Power_m],
                            "n": [self.Lognormal_Dual_Power_n],
                            "beta": [""],
                            "sigma": [self.Lognormal_Dual_Power_sigma],
                            "Log-likelihood": [self.Lognormal_Dual_Power_loglik],
                            "AICc": [self.Lognormal_Dual_Power_AICc],
                            "BIC": [self.Lognormal_Dual_Power_BIC],
                            "optimizer": [self.Lognormal_Dual_Power_optimizer],
                        },
                    ),
                ],
            )

        if "Normal_Dual_Exponential" not in self.excluded_models:
            self.__Normal_Dual_Exponential_params = Fit_Normal_Dual_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Dual_Exponential_a = self.__Normal_Dual_Exponential_params.a
            self.Normal_Dual_Exponential_b = self.__Normal_Dual_Exponential_params.b
            self.Normal_Dual_Exponential_c = self.__Normal_Dual_Exponential_params.c
            self.Normal_Dual_Exponential_sigma = self.__Normal_Dual_Exponential_params.sigma
            self.Normal_Dual_Exponential_loglik = self.__Normal_Dual_Exponential_params.loglik
            self.Normal_Dual_Exponential_BIC = self.__Normal_Dual_Exponential_params.BIC
            self.Normal_Dual_Exponential_AICc = self.__Normal_Dual_Exponential_params.AICc
            self.Normal_Dual_Exponential_optimizer = self.__Normal_Dual_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Normal_Dual_Exponential"],
                            "a": [self.Normal_Dual_Exponential_a],
                            "b": [self.Normal_Dual_Exponential_b],
                            "c": [self.Normal_Dual_Exponential_c],
                            "m": [""],
                            "n": [""],
                            "beta": [""],
                            "sigma": [self.Normal_Dual_Exponential_sigma],
                            "Log-likelihood": [self.Normal_Dual_Exponential_loglik],
                            "AICc": [self.Normal_Dual_Exponential_AICc],
                            "BIC": [self.Normal_Dual_Exponential_BIC],
                            "optimizer": [self.Normal_Dual_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Normal_Power_Exponential" not in self.excluded_models:
            self.__Normal_Power_Exponential_params = Fit_Normal_Power_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Power_Exponential_a = self.__Normal_Power_Exponential_params.a
            self.Normal_Power_Exponential_c = self.__Normal_Power_Exponential_params.c
            self.Normal_Power_Exponential_n = self.__Normal_Power_Exponential_params.n
            self.Normal_Power_Exponential_sigma = self.__Normal_Power_Exponential_params.sigma
            self.Normal_Power_Exponential_loglik = self.__Normal_Power_Exponential_params.loglik
            self.Normal_Power_Exponential_BIC = self.__Normal_Power_Exponential_params.BIC
            self.Normal_Power_Exponential_AICc = self.__Normal_Power_Exponential_params.AICc
            self.Normal_Power_Exponential_optimizer = self.__Normal_Power_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Normal_Power_Exponential"],
                            "a": [self.Normal_Power_Exponential_a],
                            "b": [""],
                            "c": [self.Normal_Power_Exponential_c],
                            "m": [""],
                            "n": [self.Normal_Power_Exponential_n],
                            "beta": [""],
                            "sigma": [self.Normal_Power_Exponential_sigma],
                            "Log-likelihood": [self.Normal_Power_Exponential_loglik],
                            "AICc": [self.Normal_Power_Exponential_AICc],
                            "BIC": [self.Normal_Power_Exponential_BIC],
                            "optimizer": [self.Normal_Power_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Normal_Dual_Power" not in self.excluded_models:
            self.__Normal_Dual_Power_params = Fit_Normal_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Normal_Dual_Power_c = self.__Normal_Dual_Power_params.c
            self.Normal_Dual_Power_m = self.__Normal_Dual_Power_params.m
            self.Normal_Dual_Power_n = self.__Normal_Dual_Power_params.n
            self.Normal_Dual_Power_sigma = self.__Normal_Dual_Power_params.sigma
            self.Normal_Dual_Power_loglik = self.__Normal_Dual_Power_params.loglik
            self.Normal_Dual_Power_BIC = self.__Normal_Dual_Power_params.BIC
            self.Normal_Dual_Power_AICc = self.__Normal_Dual_Power_params.AICc
            self.Normal_Dual_Power_optimizer = self.__Normal_Dual_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Normal_Dual_Power"],
                            "a": [""],
                            "b": [""],
                            "c": [self.Normal_Dual_Power_c],
                            "m": [self.Normal_Dual_Power_m],
                            "n": [self.Normal_Dual_Power_n],
                            "beta": [""],
                            "sigma": [self.Normal_Dual_Power_sigma],
                            "Log-likelihood": [self.Normal_Dual_Power_loglik],
                            "AICc": [self.Normal_Dual_Power_AICc],
                            "BIC": [self.Normal_Dual_Power_BIC],
                            "optimizer": [self.Normal_Dual_Power_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_Dual_Exponential" not in self.excluded_models:
            self.__Exponential_Dual_Exponential_params = Fit_Exponential_Dual_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Dual_Exponential_a = self.__Exponential_Dual_Exponential_params.a
            self.Exponential_Dual_Exponential_b = self.__Exponential_Dual_Exponential_params.b
            self.Exponential_Dual_Exponential_c = self.__Exponential_Dual_Exponential_params.c
            self.Exponential_Dual_Exponential_loglik = self.__Exponential_Dual_Exponential_params.loglik
            self.Exponential_Dual_Exponential_BIC = self.__Exponential_Dual_Exponential_params.BIC
            self.Exponential_Dual_Exponential_AICc = self.__Exponential_Dual_Exponential_params.AICc
            self.Exponential_Dual_Exponential_optimizer = self.__Exponential_Dual_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Exponential_Dual_Exponential"],
                            "a": [self.Exponential_Dual_Exponential_a],
                            "b": [self.Exponential_Dual_Exponential_b],
                            "c": [self.Exponential_Dual_Exponential_c],
                            "m": [""],
                            "n": [""],
                            "beta": [""],
                            "sigma": [""],
                            "Log-likelihood": [self.Exponential_Dual_Exponential_loglik],
                            "AICc": [self.Exponential_Dual_Exponential_AICc],
                            "BIC": [self.Exponential_Dual_Exponential_BIC],
                            "optimizer": [self.Exponential_Dual_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_Power_Exponential" not in self.excluded_models:
            self.__Exponential_Power_Exponential_params = Fit_Exponential_Power_Exponential(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Power_Exponential_a = self.__Exponential_Power_Exponential_params.a
            self.Exponential_Power_Exponential_c = self.__Exponential_Power_Exponential_params.c
            self.Exponential_Power_Exponential_n = self.__Exponential_Power_Exponential_params.n
            self.Exponential_Power_Exponential_loglik = self.__Exponential_Power_Exponential_params.loglik
            self.Exponential_Power_Exponential_BIC = self.__Exponential_Power_Exponential_params.BIC
            self.Exponential_Power_Exponential_AICc = self.__Exponential_Power_Exponential_params.AICc
            self.Exponential_Power_Exponential_optimizer = self.__Exponential_Power_Exponential_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Exponential_Power_Exponential"],
                            "a": [self.Exponential_Power_Exponential_a],
                            "b": [""],
                            "c": [self.Exponential_Power_Exponential_c],
                            "m": [""],
                            "n": [self.Exponential_Power_Exponential_n],
                            "beta": [""],
                            "sigma": [""],
                            "Log-likelihood": [self.Exponential_Power_Exponential_loglik],
                            "AICc": [self.Exponential_Power_Exponential_AICc],
                            "BIC": [self.Exponential_Power_Exponential_BIC],
                            "optimizer": [self.Exponential_Power_Exponential_optimizer],
                        },
                    ),
                ],
            )

        if "Exponential_Dual_Power" not in self.excluded_models:
            self.__Exponential_Dual_Power_params = Fit_Exponential_Dual_Power(
                failures=failures,
                failure_stress_1=failure_stress_1,
                failure_stress_2=failure_stress_2,
                right_censored=right_censored,
                right_censored_stress_1=right_censored_stress_1,
                right_censored_stress_2=right_censored_stress_2,
                use_level_stress=use_level_stress,
                CI=CI,
                optimizer=optimizer,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )
            self.Exponential_Dual_Power_c = self.__Exponential_Dual_Power_params.c
            self.Exponential_Dual_Power_m = self.__Exponential_Dual_Power_params.m
            self.Exponential_Dual_Power_n = self.__Exponential_Dual_Power_params.n
            self.Exponential_Dual_Power_loglik = self.__Exponential_Dual_Power_params.loglik
            self.Exponential_Dual_Power_BIC = self.__Exponential_Dual_Power_params.BIC
            self.Exponential_Dual_Power_AICc = self.__Exponential_Dual_Power_params.AICc
            self.Exponential_Dual_Power_optimizer = self.__Exponential_Dual_Power_params.optimizer

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data={
                            "ALT_model": ["Exponential_Dual_Power"],
                            "a": [""],
                            "b": [""],
                            "c": [self.Exponential_Dual_Power_c],
                            "m": [self.Exponential_Dual_Power_m],
                            "n": [self.Exponential_Dual_Power_n],
                            "beta": [""],
                            "sigma": [""],
                            "Log-likelihood": [self.Exponential_Dual_Power_loglik],
                            "AICc": [self.Exponential_Dual_Power_AICc],
                            "BIC": [self.Exponential_Dual_Power_BIC],
                            "optimizer": [self.Exponential_Dual_Power_optimizer],
                        },
                    ),
                ],
            )

        # change to sorting by BIC if there is insufficient data to get the AICc for everything that was fitted
        if sort_by.upper() in ["AIC", "AICC"] and "Insufficient data" in df["AICc"].values:
            sort_by = "BIC"
        # sort the dataframe by BIC, AICc, or log-likelihood. Smallest AICc, BIC, log-likelihood is better fit
        if not isinstance(sort_by, str):
            raise ValueError(
                "Invalid input to sort_by. Options are 'BIC', 'AICc', or 'Log-likelihood'. Default is 'BIC'.",
            )
        if sort_by.upper() == "BIC":
            df2 = df.sort_values(by="BIC")
        elif sort_by.upper() in ["AICC", "AIC"]:
            df2 = df.sort_values(by="AICc")
        elif sort_by.upper() in [
            "LOGLIK",
            "LOG LIK",
            "LOG-LIKELIHOOD",
            "LL",
            "LOGLIKELIHOOD",
            "LOG LIKELIHOOD",
        ]:
            df["LLabs"] = abs(df["Log-likelihood"])  # need to create a new column for the absolute value before sorting
            df2 = df.sort_values(by="LLabs")
            df2.drop("LLabs", axis=1, inplace=True)  # remove the column created just for sorting
        else:
            raise ValueError(
                "Invalid input to sort_by. Options are 'BIC', 'AICc', or 'Log-likelihood'. Default is 'BIC'.",
            )
        if len(df2.index.values) == 0:
            raise ValueError("You have excluded all available ALT models")
        self.results = df2

        # creates a distribution object of the best fitting distribution and assigns its name
        best_model = self.results["ALT_model"].values[0]
        self.best_model_name = best_model
        if use_level_stress is not None:
            if best_model == "Weibull_Exponential":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Exponential_b * np.exp(self.Weibull_Exponential_a / use_level_stress),
                    beta=self.Weibull_Exponential_beta,
                )
            elif best_model == "Weibull_Eyring":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=(1 / use_level_stress)
                    * np.exp(-(self.Weibull_Eyring_c - self.Weibull_Eyring_a / use_level_stress)),
                    beta=self.Weibull_Eyring_beta,
                )
            elif best_model == "Weibull_Power":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Power_a * use_level_stress**self.Weibull_Power_n,
                    beta=self.Weibull_Power_beta,
                )
            elif best_model == "Lognormal_Exponential":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=np.log(self.Lognormal_Exponential_b * np.exp(self.Lognormal_Exponential_a / use_level_stress)),
                    sigma=self.Lognormal_Exponential_sigma,
                )
            elif best_model == "Lognormal_Eyring":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=np.log(
                        (1 / use_level_stress)
                        * np.exp(-(self.Lognormal_Eyring_c - self.Lognormal_Eyring_a / use_level_stress)),
                    ),
                    sigma=self.Lognormal_Eyring_sigma,
                )
            elif best_model == "Lognormal_Power":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=np.log(self.Lognormal_Power_a * use_level_stress**self.Lognormal_Power_n),
                    sigma=self.Lognormal_Power_sigma,
                )
            elif best_model == "Normal_Exponential":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Exponential_b * np.exp(self.Normal_Exponential_a / use_level_stress),
                    sigma=self.Normal_Exponential_sigma,
                )
            elif best_model == "Normal_Eyring":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=(1 / use_level_stress)
                    * np.exp(-(self.Normal_Eyring_c - self.Normal_Eyring_a / use_level_stress)),
                    sigma=self.Normal_Eyring_sigma,
                )
            elif best_model == "Normal_Power":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Power_a * use_level_stress**self.Normal_Power_n,
                    sigma=self.Normal_Power_sigma,
                )
            elif best_model == "Exponential_Exponential":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=1
                    / (self.Exponential_Exponential_b * np.exp(self.Exponential_Exponential_a / use_level_stress)),
                )
            elif best_model == "Exponential_Eyring":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=1
                    / (
                        (1 / use_level_stress)
                        * np.exp(-(self.Exponential_Eyring_c - self.Exponential_Eyring_a / use_level_stress))
                    ),
                )
            elif best_model == "Exponential_Power":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=1 / (self.Exponential_Power_a * use_level_stress**self.Exponential_Power_n),
                )
            elif best_model == "Weibull_Dual_Exponential":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Dual_Exponential_c
                    * np.exp(
                        self.Weibull_Dual_Exponential_a / use_level_stress[0]
                        + self.Weibull_Dual_Exponential_b / use_level_stress[1],
                    ),
                    beta=self.Weibull_Dual_Exponential_beta,
                )
            elif best_model == "Weibull_Power_Exponential":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Power_Exponential_c
                    * use_level_stress[1] ** self.Weibull_Power_Exponential_n
                    * np.exp(self.Weibull_Power_Exponential_a / use_level_stress[0]),
                    beta=self.Weibull_Power_Exponential_beta,
                )
            elif best_model == "Weibull_Dual_Power":
                self.best_model_at_use_stress = Weibull_Distribution(
                    alpha=self.Weibull_Dual_Power_c
                    * use_level_stress[0] ** self.Weibull_Dual_Power_m
                    * use_level_stress[1] ** self.Weibull_Dual_Power_n,
                    beta=self.Weibull_Dual_Power_beta,
                )
            elif best_model == "Lognormal_Dual_Exponential":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=1
                    / (
                        self.Lognormal_Dual_Exponential_c
                        * np.exp(
                            self.Lognormal_Dual_Exponential_a / use_level_stress[0]
                            + self.Lognormal_Dual_Exponential_b / use_level_stress[1],
                        )
                    ),
                    sigma=self.Lognormal_Dual_Exponential_sigma,
                )
            elif best_model == "Lognormal_Power_Exponential":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=1
                    / (
                        self.Lognormal_Power_Exponential_c
                        * use_level_stress[1] ** self.Lognormal_Power_Exponential_n
                        * np.exp(self.Lognormal_Power_Exponential_a / use_level_stress[0])
                    ),
                    sigma=self.Lognormal_Power_Exponential_sigma,
                )
            elif best_model == "Lognormal_Dual_Power":
                self.best_model_at_use_stress = Lognormal_Distribution(
                    mu=1
                    / (
                        self.Lognormal_Dual_Power_c
                        * use_level_stress[0] ** self.Lognormal_Dual_Power_m
                        * use_level_stress[1] ** self.Lognormal_Dual_Power_n
                    ),
                    sigma=self.Lognormal_Dual_Power_sigma,
                )
            elif best_model == "Normal_Dual_Exponential":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Dual_Exponential_c
                    * np.exp(
                        self.Normal_Dual_Exponential_a / use_level_stress[0]
                        + self.Normal_Dual_Exponential_b / use_level_stress[1],
                    ),
                    sigma=self.Normal_Dual_Exponential_sigma,
                )
            elif best_model == "Normal_Power_Exponential":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Power_Exponential_c
                    * use_level_stress[1] ** self.Normal_Power_Exponential_n
                    * np.exp(self.Normal_Power_Exponential_a / use_level_stress[0]),
                    sigma=self.Normal_Power_Exponential_sigma,
                )
            elif best_model == "Normal_Dual_Power":
                self.best_model_at_use_stress = Normal_Distribution(
                    mu=self.Normal_Dual_Power_c
                    * use_level_stress[0] ** self.Normal_Dual_Power_m
                    * use_level_stress[1] ** self.Normal_Dual_Power_n,
                    sigma=self.Normal_Dual_Power_sigma,
                )
            elif best_model == "Exponential_Dual_Exponential":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=self.Exponential_Dual_Exponential_c
                    * np.exp(
                        self.Exponential_Dual_Exponential_a / use_level_stress[0]
                        + self.Exponential_Dual_Exponential_b / use_level_stress[1],
                    ),
                )
            elif best_model == "Exponential_Power_Exponential":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=self.Exponential_Power_Exponential_c
                    * use_level_stress[1] ** self.Exponential_Power_Exponential_n
                    * np.exp(self.Exponential_Power_Exponential_a / use_level_stress[0]),
                )
            elif best_model == "Exponential_Dual_Power":
                self.best_model_at_use_stress = Exponential_Distribution(
                    Lambda=self.Exponential_Dual_Power_c
                    * use_level_stress[0] ** self.Exponential_Dual_Power_m
                    * use_level_stress[1] ** self.Exponential_Dual_Power_n,
                )

        # print the results
        if print_results is True:  # printing occurs by default
            if len(right_censored) > 0:
                frac_cens = (len(right_censored) / (len(failures) + len(right_censored))) * 100
            else:
                frac_cens = 0
            if frac_cens % 1 < 1e-10:
                frac_cens = int(frac_cens)
            colorprint("Results from Fit_Everything_ALT:", bold=True, underline=True)
            print("Analysis method: Maximum Likelihood Estimation (MLE)")
            print(
                "Failures / Right censored:",
                str(str(len(failures)) + "/" + str(len(right_censored))),
                str("(" + str(frac_cens) + "% right censored)"),
                "\n",
            )
            print(self.results.to_string(index=False), "\n")

            if use_level_stress is not None:
                if type(use_level_stress) not in [list, np.ndarray]:
                    use_level_stress_str = round_and_string(use_level_stress)
                else:
                    use_level_stress_str = str(
                        round_and_string(use_level_stress[0]) + ", " + round_and_string(use_level_stress[1]),
                    )
                print(
                    str(
                        "At the use level stress of "
                        + use_level_stress_str
                        + ", the "
                        + self.best_model_name
                        + " model has a mean life of "
                        + round_and_string(self.best_model_at_use_stress.mean),
                    ),
                )

        if show_probability_plot is True:
            # plotting occurs by default
            self.probability_plot = Fit_Everything_ALT.__probability_plot(self)

        if show_best_distribution_probability_plot is True:
            self.best_distribution_probability_plot = Fit_Everything_ALT.__probability_plot(self, best_only=True)

        if show_probability_plot is True or show_best_distribution_probability_plot is True:
            plt.show()

    def __probability_plot(self, best_only=False):
        from reliability.Utils import ALT_prob_plot

        use_level_stress = self.__use_level_stress
        plt.figure()
        if best_only is False:
            items = len(self.results.index.values)  # number of items that were fitted
            if items in [10, 11, 12]:  # --- w , h
                cols, rows, figsize = 4, 3, (15, 8)
            elif items in [7, 8, 9]:
                cols, rows, figsize = 3, 3, (12.5, 8)
            elif items in [5, 6]:
                cols, rows, figsize = 3, 2, (12.5, 6)
            elif items == 4:
                cols, rows, figsize = 2, 2, (10, 6)
            elif items == 3:
                cols, rows, figsize = 3, 1, (12.5, 5)
            elif items == 2:
                cols, rows, figsize = 2, 1, (10, 4)
            elif items == 1:
                cols, rows, figsize = 1, 1, (7.5, 4)

            # this is the order to plot to match the results dataframe
            plotting_order = self.results["ALT_model"].to_numpy()
            plt.suptitle("Probability plots of each fitted ALT model\n\n")
            subplot_counter = 1
        else:
            # plots the best model only
            plotting_order = [self.results["ALT_model"].values[0]]
            rows, cols, subplot_counter = 1, 1, 1

        for item in plotting_order:
            ax = plt.subplot(rows, cols, subplot_counter)

            if item == "Weibull_Exponential":

                def life_func(S1):
                    return self.Weibull_Exponential_b * np.exp(self.Weibull_Exponential_a / S1)

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Exponential_params._Fit_Weibull_Exponential__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Exponential_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Weibull_Eyring":

                def life_func(S1):
                    return 1 / S1 * np.exp(-(self.Weibull_Eyring_c - self.Weibull_Eyring_a / S1))

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__shape_for_change_df
                )
                failure_groups = self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__failure_groups
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Eyring_params._Fit_Weibull_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Eyring_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Weibull_Power":

                def life_func(S1):
                    return self.Weibull_Power_a * S1**self.Weibull_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__shape_for_change_df
                )
                failure_groups = self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__failure_groups
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Power_params._Fit_Weibull_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Power_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Lognormal_Exponential":

                def life_func(S1):
                    return self.Lognormal_Exponential_b * np.exp(self.Lognormal_Exponential_a / S1)

                stresses_for_groups = self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__shape_for_change_df
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__failure_groups
                )
                right_censored_groups = self._Fit_Everything_ALT__Lognormal_Exponential_params._Fit_Lognormal_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Lognormal_Eyring":

                def life_func(S1):
                    return 1 / S1 * np.exp(-(self.Lognormal_Eyring_c - self.Lognormal_Eyring_a / S1))

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__shape_for_change_df
                )
                failure_groups = self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__failure_groups
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Eyring_params._Fit_Lognormal_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Eyring_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Lognormal_Power":

                def life_func(S1):
                    return self.Lognormal_Power_a * S1**self.Lognormal_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__shape_for_change_df
                )
                failure_groups = self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__failure_groups
                right_censored_groups = (
                    self._Fit_Everything_ALT__Lognormal_Power_params._Fit_Lognormal_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Normal_Exponential":

                def life_func(S1):
                    return self.Normal_Exponential_b * np.exp(self.Normal_Exponential_a / S1)

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Exponential_params._Fit_Normal_Exponential__right_censored_groups
                )

                ALT_prob_plot(
                    dist="Normal",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Normal_Eyring":

                def life_func(S1):
                    return 1 / S1 * np.exp(-(self.Normal_Eyring_c - self.Normal_Eyring_a / S1))

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__shape_for_change_df
                )
                failure_groups = self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__failure_groups
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Eyring_params._Fit_Normal_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Eyring_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Normal_Power":

                def life_func(S1):
                    return self.Normal_Power_a * S1**self.Normal_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__shape_for_change_df
                )
                failure_groups = self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__failure_groups
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Power_params._Fit_Normal_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Exponential_Exponential":

                def life_func(S1):
                    return self.Exponential_Exponential_b * np.exp(self.Exponential_Exponential_a / S1)

                stresses_for_groups = self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Exponential_Exponential_params._Fit_Exponential_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Exponential",
                    model="Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Exponential_Eyring":

                def life_func(S1):
                    return 1 / S1 * np.exp(-(self.Exponential_Eyring_c - self.Exponential_Eyring_a / S1))

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Eyring_params._Fit_Exponential_Eyring__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Eyring",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )
            elif item == "Exponential_Power":

                def life_func(S1):
                    return self.Exponential_Power_a * S1**self.Exponential_Power_n

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Exponential_Power_params._Fit_Exponential_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Exponential",
                    model="Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Weibull_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Weibull_Dual_Exponential_c * np.exp(
                        self.Weibull_Dual_Exponential_a / S1 + self.Weibull_Dual_Exponential_b / S2,
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Weibull_Dual_Exponential_params._Fit_Weibull_Dual_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Weibull",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Dual_Exponential_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Weibull_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Weibull_Power_Exponential_c
                        * (S2**self.Weibull_Power_Exponential_n)
                        * np.exp(self.Weibull_Power_Exponential_a / S1)
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Weibull_Power_Exponential_params._Fit_Weibull_Power_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Weibull",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Power_Exponential_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Weibull_Dual_Power":

                def life_func(S1, S2):
                    return self.Weibull_Dual_Power_c * (S1**self.Weibull_Dual_Power_m) * (S2**self.Weibull_Dual_Power_n)

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Weibull_Dual_Power_params._Fit_Weibull_Dual_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Weibull",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Weibull_Dual_Power_beta,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Lognormal_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Lognormal_Dual_Exponential_c * np.exp(
                        self.Lognormal_Dual_Exponential_a / S1 + self.Lognormal_Dual_Exponential_b / S2,
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Lognormal_Dual_Exponential_params._Fit_Lognormal_Dual_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Dual_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Lognormal_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Lognormal_Power_Exponential_c
                        * (S2**self.Lognormal_Power_Exponential_n)
                        * np.exp(self.Lognormal_Power_Exponential_a / S1)
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Lognormal_Power_Exponential_params._Fit_Lognormal_Power_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Power_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Lognormal_Dual_Power":

                def life_func(S1, S2):
                    return (
                        self.Lognormal_Dual_Power_c
                        * (S1**self.Lognormal_Dual_Power_m)
                        * (S2**self.Lognormal_Dual_Power_n)
                    )

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__failure_groups
                )
                right_censored_groups = self._Fit_Everything_ALT__Lognormal_Dual_Power_params._Fit_Lognormal_Dual_Power__right_censored_groups
                ALT_prob_plot(
                    dist="Lognormal",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Lognormal_Dual_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Normal_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Normal_Dual_Exponential_c * np.exp(
                        self.Normal_Dual_Exponential_a / S1 + self.Normal_Dual_Exponential_b / S2,
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Normal_Dual_Exponential_params._Fit_Normal_Dual_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Normal",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Dual_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Normal_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Normal_Power_Exponential_c
                        * (S2**self.Normal_Power_Exponential_n)
                        * np.exp(self.Normal_Power_Exponential_a / S1)
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Normal_Power_Exponential_params._Fit_Normal_Power_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Normal",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Power_Exponential_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Normal_Dual_Power":

                def life_func(S1, S2):
                    return self.Normal_Dual_Power_c * (S1**self.Normal_Dual_Power_m) * (S2**self.Normal_Dual_Power_n)

                stresses_for_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__stresses_for_groups
                )
                scale_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__scale_for_change_df
                )
                shape_for_change_df = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__shape_for_change_df
                )
                failure_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__failure_groups
                )
                right_censored_groups = (
                    self._Fit_Everything_ALT__Normal_Dual_Power_params._Fit_Normal_Dual_Power__right_censored_groups
                )
                ALT_prob_plot(
                    dist="Normal",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=self.Normal_Dual_Power_sigma,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Exponential_Dual_Exponential":

                def life_func(S1, S2):
                    return self.Exponential_Dual_Exponential_c * np.exp(
                        self.Exponential_Dual_Exponential_a / S1 + self.Exponential_Dual_Exponential_b / S2,
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Exponential_Dual_Exponential_params._Fit_Exponential_Dual_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Exponential",
                    model="Dual_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Exponential_Power_Exponential":

                def life_func(S1, S2):
                    return (
                        self.Exponential_Power_Exponential_c
                        * (S2**self.Exponential_Power_Exponential_n)
                        * np.exp(self.Exponential_Power_Exponential_a / S1)
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__shape_for_change_df
                failure_groups = self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__failure_groups
                right_censored_groups = self._Fit_Everything_ALT__Exponential_Power_Exponential_params._Fit_Exponential_Power_Exponential__right_censored_groups
                ALT_prob_plot(
                    dist="Exponential",
                    model="Power_Exponential",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            elif item == "Exponential_Dual_Power":

                def life_func(S1, S2):
                    return (
                        self.Exponential_Dual_Power_c
                        * (S1**self.Exponential_Dual_Power_m)
                        * (S2**self.Exponential_Dual_Power_n)
                    )

                stresses_for_groups = self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__stresses_for_groups
                scale_for_change_df = self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__scale_for_change_df
                shape_for_change_df = self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__shape_for_change_df
                failure_groups = (
                    self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__failure_groups
                )
                right_censored_groups = self._Fit_Everything_ALT__Exponential_Dual_Power_params._Fit_Exponential_Dual_Power__right_censored_groups
                ALT_prob_plot(
                    dist="Exponential",
                    model="Dual_Power",
                    stresses_for_groups=stresses_for_groups,
                    failure_groups=failure_groups,
                    right_censored_groups=right_censored_groups,
                    life_func=life_func,
                    shape=None,
                    scale_for_change_df=scale_for_change_df,
                    shape_for_change_df=shape_for_change_df,
                    use_level_stress=use_level_stress,
                    ax=ax,
                )

            else:
                raise ValueError("unknown item was fitted")

            if best_only is False:
                plt.title(item)
                ax.set_yticklabels([], minor=False)
                ax.set_xticklabels([], minor=False)
                ax.set_yticklabels([], minor=True)
                ax.set_xticklabels([], minor=True)
                ax.set_ylabel("")
                ax.set_xlabel("")
                ax.get_legend().remove()
                subplot_counter += 1
            else:
                plt.title("Probability plot of best model\n" + item)
        if best_only is False:
            plt.tight_layout()
            plt.gcf().set_size_inches(figsize)
        return plt.gcf()  # return the figure handle

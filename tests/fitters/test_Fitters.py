import warnings

from numpy.testing import assert_allclose

from reliability.Distributions import (
    Beta_Distribution,
)
from reliability.Fitters import (
    Fit_Everything,
)
from reliability.Other_functions import make_right_censored_data

# I would like to make these smaller but the slight differences in different python versions (3.6-3.9) mean that tight tolerances result in test failures
atol = 1e-3
atol_big = 0  # 0 means it will not look at the absolute difference
rtol = 1e-3
rtol_big = 0.1  # 10% variation


def test_Fit_Everything():
    # ignores the runtime warning from scipy when the nelder-mean or powell optimizers are used and jac is not required
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    dist = Beta_Distribution(alpha=5, beta=4)
    rawdata = dist.random_samples(200, seed=5)
    data = make_right_censored_data(data=rawdata, threshold=dist.mean)
    MLE = Fit_Everything(
        failures=data.failures,
        right_censored=data.right_censored,
        method="MLE",
        show_probability_plot=True,
        show_histogram_plot=True,
        show_PP_plot=True,
        show_best_distribution_probability_plot=True,
        print_results=False,
    )
    LS = Fit_Everything(
        failures=data.failures,
        right_censored=data.right_censored,
        method="LS",
        show_probability_plot=True,
        show_histogram_plot=True,
        show_PP_plot=True,
        show_best_distribution_probability_plot=True,
        print_results=True,
    )

    assert_allclose(
        MLE.best_distribution.alpha,
        0.5796887225805948,
        rtol=rtol,
        atol=atol,
    )  # best fit here is a Beta distribution
    assert_allclose(MLE.best_distribution.beta, 4.205258710807067, rtol=rtol, atol=atol)

    assert_allclose(MLE.Weibull_2P_alpha, 0.5796887225805948, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_2P_beta, 4.205258710807067, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_2P_AICc, 22.509958498975394, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_2P_BIC, 29.04567952648771, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_2P_loglik, -9.224522396695818, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_2P_AD, 543.31193295208, rtol=rtol, atol=atol)

    assert_allclose(MLE.Weibull_3P_alpha, 0.5796887225805948, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_3P_beta, 4.205258710807067, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_3P_AICc, 24.571493772983473, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_3P_BIC, 34.343996893035744, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_3P_loglik, -9.224522396695818, rtol=rtol, atol=atol)
    assert_allclose(MLE.Weibull_3P_AD, 543.31193295208, rtol=rtol, atol=atol)

    assert_allclose(MLE.Gamma_2P_alpha, 0.06343366643685251, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_2P_beta, 8.730724670235508, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_2P_AICc, 29.72088918292124, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_2P_BIC, 36.25661021043356, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_2P_loglik, -12.829987738668741, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_2P_AD, 543.5598195358288, rtol=rtol, atol=atol)

    assert_allclose(MLE.Gamma_3P_alpha, 0.06343366643685251, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_3P_beta, 8.730724670235508, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_3P_AICc, 31.78242445692932, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_3P_BIC, 41.55492757698159, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_3P_loglik, -12.829987738668741, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gamma_3P_AD, 543.5598195358288, rtol=rtol, atol=atol)

    assert_allclose(MLE.Loglogistic_2P_alpha, 0.5327695781726263, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_2P_beta, 4.959959950671738, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_2P_AICc, 26.2468431389576, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_2P_BIC, 32.78256416646992, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_2P_loglik, -11.092964716686922, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_2P_AD, 543.3968941075816, rtol=rtol, atol=atol)

    assert_allclose(MLE.Loglogistic_3P_alpha, 0.5327695781726263, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_3P_beta, 4.959959950671738, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_3P_AICc, 28.30837841296568, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_3P_BIC, 38.08088153301795, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_3P_loglik, -11.092964716686922, rtol=rtol, atol=atol)
    assert_allclose(MLE.Loglogistic_3P_AD, 543.3968941075816, rtol=rtol, atol=atol)

    assert_allclose(MLE.Lognormal_2P_mu, -0.6258670209896524, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_2P_sigma, 0.3859306240146529, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_2P_AICc, 36.58934382876143, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_2P_BIC, 43.125064856273745, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_2P_loglik, -16.264215061588835, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_2P_AD, 543.7578077426027, rtol=rtol, atol=atol)

    assert_allclose(MLE.Lognormal_3P_mu, -0.6258670209896524, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_3P_sigma, 0.3859306240146529, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_3P_AICc, 38.65087910276951, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_3P_BIC, 48.42338222282178, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_3P_loglik, -16.264215061588835, rtol=rtol, atol=atol)
    assert_allclose(MLE.Lognormal_3P_AD, 543.7578077426027, rtol=rtol, atol=atol)

    assert_allclose(MLE.Normal_2P_mu, 0.5313204293962966, rtol=rtol, atol=atol)
    assert_allclose(MLE.Normal_2P_sigma, 0.14842166096827056, rtol=rtol, atol=atol)
    assert_allclose(MLE.Normal_2P_AICc, 23.0363966340782, rtol=rtol, atol=atol)
    assert_allclose(MLE.Normal_2P_BIC, 29.572117661590518, rtol=rtol, atol=atol)
    assert_allclose(MLE.Normal_2P_loglik, -9.487741464247222, rtol=rtol, atol=atol)
    assert_allclose(MLE.Normal_2P_AD, 543.3042437249142, rtol=rtol, atol=atol)

    assert_allclose(MLE.Gumbel_2P_mu, 0.5706624792367315, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gumbel_2P_sigma, 0.10182903954122995, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gumbel_2P_AICc, 26.09054970134011, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gumbel_2P_BIC, 32.626270728852425, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gumbel_2P_loglik, -11.014817997878176, rtol=rtol, atol=atol)
    assert_allclose(MLE.Gumbel_2P_AD, 543.3089024789034, rtol=rtol, atol=atol)

    assert_allclose(MLE.Beta_2P_alpha, 5.586642953718748, rtol=rtol, atol=atol)
    assert_allclose(MLE.Beta_2P_beta, 4.950693419749502, rtol=rtol, atol=atol)
    assert_allclose(MLE.Beta_2P_AICc, 24.204124482547897, rtol=rtol, atol=atol)
    assert_allclose(MLE.Beta_2P_BIC, 30.739845510060213, rtol=rtol, atol=atol)
    assert_allclose(MLE.Beta_2P_loglik, -10.07160538848207, rtol=rtol, atol=atol)
    assert_allclose(MLE.Beta_2P_AD, 543.3809275359781, rtol=rtol, atol=atol)

    assert_allclose(MLE.Exponential_2P_lambda, 1.5845505775713558, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_2P_gamma, 0.12428161981215716, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_2P_AICc, 127.11230931613672, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_2P_BIC, 133.64803034364903, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_2P_loglik, -61.52569780527648, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_2P_AD, 548.8966650502098, rtol=rtol, atol=atol)

    assert_allclose(MLE.Exponential_1P_lambda, 1.1776736956890317, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_1P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_1P_AICc, 192.73284561137785, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_1P_BIC, 196.01096095772388, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_1P_loglik, -95.35632179558792, rtol=rtol, atol=atol)
    assert_allclose(MLE.Exponential_1P_AD, 551.326873807673, rtol=rtol, atol=atol)

    assert_allclose(
        LS.best_distribution.mu,
        0.5350756091376212,
        rtol=rtol,
        atol=atol,
    )  # best fit here is a Normal distribution
    assert_allclose(LS.best_distribution.sigma, 0.15352298167936318, rtol=rtol, atol=atol)

    assert_allclose(LS.Weibull_2P_alpha, 0.5948490848650297, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_2P_beta, 3.850985192722524, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_2P_AICc, 24.002343535956285, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_2P_BIC, 30.538064563468602, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_2P_loglik, -9.970714915186264, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_2P_AD, 543.3536598333712, rtol=rtol, atol=atol)

    assert_allclose(LS.Weibull_3P_alpha, 0.5796887225805948, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_3P_beta, 4.205258710807067, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_3P_AICc, 24.571493772983473, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_3P_BIC, 34.343996893035744, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_3P_loglik, -9.224522396695818, rtol=rtol, atol=atol)
    assert_allclose(LS.Weibull_3P_AD, 543.31193295208, rtol=rtol, atol=atol)

    assert_allclose(LS.Gamma_2P_alpha, 0.047474493713487956, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_2P_beta, 11.56120649983023, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_2P_AICc, 34.77520772749797, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_2P_BIC, 41.31092875501029, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_2P_loglik, -15.357147010957107, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_2P_AD, 543.5555679280225, rtol=rtol, atol=atol)

    assert_allclose(LS.Gamma_3P_alpha, 0.06343366643685251, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_3P_beta, 8.730724670235508, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_3P_AICc, 31.78242445692932, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_3P_BIC, 41.55492757698159, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_3P_loglik, -12.829987738668741, rtol=rtol, atol=atol)
    assert_allclose(LS.Gamma_3P_AD, 543.5598195358288, rtol=rtol, atol=atol)

    assert_allclose(LS.Loglogistic_2P_alpha, 0.5489258630949324, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_2P_beta, 4.282869717868545, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_2P_AICc, 29.55884374185365, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_2P_BIC, 36.09456476936597, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_2P_loglik, -12.748965018134946, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_2P_AD, 543.4725652046802, rtol=rtol, atol=atol)

    assert_allclose(LS.Loglogistic_3P_alpha, 0.5327695781726263, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_3P_beta, 4.959959950671738, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_3P_AICc, 28.30837841296568, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_3P_BIC, 38.08088153301795, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_3P_loglik, -11.092964716686922, rtol=rtol, atol=atol)
    assert_allclose(LS.Loglogistic_3P_AD, 543.3968941075816, rtol=rtol, atol=atol)

    assert_allclose(LS.Lognormal_2P_mu, -0.5829545855241497, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_2P_sigma, 0.42938026719038264, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_2P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_2P_AICc, 39.2494098877054, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_2P_BIC, 45.785130915217714, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_2P_loglik, -17.59424809106082, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_2P_AD, 543.6895545238489, rtol=rtol, atol=atol)

    assert_allclose(LS.Lognormal_3P_mu, -0.6258670209896524, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_3P_sigma, 0.3859306240146529, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_3P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_3P_AICc, 38.65087910276951, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_3P_BIC, 48.42338222282178, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_3P_loglik, -16.264215061588835, rtol=rtol, atol=atol)
    assert_allclose(LS.Lognormal_3P_AD, 543.7578077426027, rtol=rtol, atol=atol)

    assert_allclose(LS.Normal_2P_mu, 0.5350756091376212, rtol=rtol, atol=atol)
    assert_allclose(LS.Normal_2P_sigma, 0.15352298167936318, rtol=rtol, atol=atol)
    assert_allclose(LS.Normal_2P_AICc, 23.270071653194492, rtol=rtol, atol=atol)
    assert_allclose(LS.Normal_2P_BIC, 29.80579268070681, rtol=rtol, atol=atol)
    assert_allclose(LS.Normal_2P_loglik, -9.604578973805367, rtol=rtol, atol=atol)
    assert_allclose(LS.Normal_2P_AD, 543.3018089629097, rtol=rtol, atol=atol)

    assert_allclose(LS.Gumbel_2P_mu, 0.5575543755580943, rtol=rtol, atol=atol)
    assert_allclose(LS.Gumbel_2P_sigma, 0.09267958281580514, rtol=rtol, atol=atol)
    assert_allclose(LS.Gumbel_2P_AICc, 28.66352107358925, rtol=rtol, atol=atol)
    assert_allclose(LS.Gumbel_2P_BIC, 35.19924210110157, rtol=rtol, atol=atol)
    assert_allclose(LS.Gumbel_2P_loglik, -12.301303684002747, rtol=rtol, atol=atol)
    assert_allclose(LS.Gumbel_2P_AD, 543.3456378838292, rtol=rtol, atol=atol)

    assert_allclose(LS.Beta_2P_alpha, 6.54242621734743, rtol=rtol, atol=atol)
    assert_allclose(LS.Beta_2P_beta, 5.795236872686599, rtol=rtol, atol=atol)
    assert_allclose(LS.Beta_2P_AICc, 25.745158997195162, rtol=rtol, atol=atol)
    assert_allclose(LS.Beta_2P_BIC, 32.28088002470748, rtol=rtol, atol=atol)
    assert_allclose(LS.Beta_2P_loglik, -10.842122645805702, rtol=rtol, atol=atol)
    assert_allclose(LS.Beta_2P_AD, 543.3718252593867, rtol=rtol, atol=atol)

    assert_allclose(LS.Exponential_2P_lambda, 1.1858797968873822, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_2P_gamma, 0.12338161981215715, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_2P_AICc, 136.25275877909922, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_2P_BIC, 142.78847980661155, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_2P_loglik, -66.09592253675774, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_2P_AD, 546.5849877012892, rtol=rtol, atol=atol)

    assert_allclose(LS.Exponential_1P_lambda, 1.0678223705385204, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_1P_gamma, 0, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_1P_AICc, 193.7910857336068, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_1P_BIC, 197.06920107995282, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_1P_loglik, -95.88544185670239, rtol=rtol, atol=atol)
    assert_allclose(LS.Exponential_1P_AD, 549.85986679373, rtol=rtol, atol=atol)

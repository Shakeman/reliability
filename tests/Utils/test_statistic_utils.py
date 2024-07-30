import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from reliability.Utils._statstic_utils import (
    Beta_2P_guess,
    Exponential_1P_guess,
    Exponential_2P_guess,
    Gamma_2P_guess,
    Gamma_3P_guess,
    Gumbel_2P_guess,
    Loglogistic_2P_guess,
    Loglogistic_3P_guess,
    Lognormal_2P_guess,
    Lognormal_3P_guess,
    Normal_2P_guess,
    Weibull_2P_guess,
    Weibull_3P_guess,
    __beta_2P_CDF,
    __gamma_2P_CDF,
    __gamma_3P_CDF,
    __gamma_optimizer,
    __loglogistic_3P_CDF,
    __normal_2P_CDF,
    non_invertable_handler,
)


def test_Weibull_2P_guess():
    # Test case: Positive slope
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    force_shape = None
    expected_result = [7.073909752122913, 1.1651597484161327]
    assert Weibull_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: Negative slope
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    method = "RRX"
    force_shape = None
    expected_result = [1.039928027981657, -1.2673812106491167]
    assert Weibull_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: Force shape parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    force_shape = 2.0
    expected_result = [4.662007337682776, 2.0]
    assert Weibull_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: Method is "RRY"
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    force_shape = None
    expected_result = [7.087182406412927, 1.1629773507668202]
    assert Weibull_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: Empty arrays
    x = np.array([])
    y = np.array([])
    method = "RRX"
    force_shape = None
    expected_result = [np.nan, np.nan]
    with pytest.raises(ValueError) as e:
        Weibull_2P_guess(x, y, method, force_shape)
    assert (
        str(e.value) == "A minimum of 2 points are required to fit the line when slope or intercept are not specified."
    )


def test_Weibull_3P_guess():
    # Test case: Positive shape parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    force_shape = None
    expected_result = [6.8701030567178645, 1.214464642404451, 5.297345100789064e-10]
    assert Weibull_3P_guess(x, y, method, gamma0, failures, force_shape) == expected_result

    # Test case: Non-linear least squares estimation failure
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    force_shape = -1.0
    expected_result = [0.6145864188690114, -1.0, 0.5]
    assert Weibull_3P_guess(x, y, method, gamma0, failures, force_shape) == expected_result

    # Test case: RRY Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    force_shape = None
    expected_result = [6.870103083574396, 1.2144646336659919, 3.289767595144275e-10]
    assert Weibull_3P_guess(x, y, method, gamma0, failures, force_shape) == expected_result


def test_Exponential_1P_guess():
    # Test case: RRX method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    expected_result = [0.1303526043922722]  # Expected guess value for Exponential_1P model
    assert Exponential_1P_guess(x, y, method) == expected_result

    # Test case: RRY method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    expected_result = [0.12964928814483878]  # Expected guess value for Exponential_1P model
    assert Exponential_1P_guess(x, y, method) == expected_result


def test_Exponential_2P_guess():
    # Test case: Positive shape parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    expected_result = [0.14205458883223968, 0.3497157953954444]
    assert Exponential_2P_guess(x, y, gamma0, failures) == expected_result

    # Test case: RRY Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    expected_result = [0.14205458883223968, 0.3497157953954444]
    assert Exponential_2P_guess(x, y, gamma0, failures) == expected_result

    # Test case: Non-linear least squares estimation failure
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma0 = -0.5
    failures = np.array([1, 2, 3, 4, 5])
    expected_result = [0.1151051050364014, -0.5]  # Fallback to ordinary least squares estimation
    assert Exponential_2P_guess(x, y, gamma0, failures) == expected_result


def test_Normal_2P_guess():
    # Test case: RRX method, no forced shape parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    force_shape = None
    expected_result = [4.816005541001474, 3.130050606758314]
    assert Normal_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: RRY method, forced shape parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    force_shape = 2.0
    expected_result = [4.160368165984542, 2.0]
    assert Normal_2P_guess(x, y, method, force_shape) == expected_result


def test_Gumbel_2P_guess():
    # Test case: RRX Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    expected_result = [5.443576279536926, 2.099480763001191]
    assert Gumbel_2P_guess(x, y, method) == expected_result

    # Test case: RRY Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    expected_result = [5.532452858362983, 2.1758420655270236]
    assert Gumbel_2P_guess(x, y, method) == expected_result


def test_Lognormal_2P_guess():
    # Test case: RRX Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    force_shape = None
    expected_result = [1.6875780741271529, 1.2583587640070946]
    assert Lognormal_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: RRY Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    force_shape = None
    expected_result = [1.6973017023071022, 1.2751183209563315]  # Fallback to least squares estimation
    assert Lognormal_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: Negative shape parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    force_shape = -1.0
    expected_result = [0.3773142655641382, -1.0]
    assert Lognormal_2P_guess(x, y, method, force_shape) == expected_result

    # Test case: RRY Method with force shape
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    force_shape = 1.0
    expected_result = [1.5376824315486803, 1.0]  # Fallback to least squares estimation
    assert Lognormal_2P_guess(x, y, method, force_shape) == expected_result


def test___gamma_optimizer():
    # Test case: Positive correlation coefficient
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma_guess = 0.5
    expected_result = 0.03576390077131164
    assert np.isclose(__gamma_optimizer(gamma_guess, x, y), expected_result)

    # Test case: Negative correlation coefficient
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    gamma_guess = 0.5
    expected_result = 0.15813124779152266
    assert np.isclose(__gamma_optimizer(gamma_guess, x, y), expected_result)

    # Test case: Zero correlation coefficient
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    gamma_guess = 0.5
    expected_result = 1.0
    assert np.isclose(__gamma_optimizer(gamma_guess, x, y), expected_result)


def test_normal_2P_CDF():
    # Test case 1: Standard normal distribution
    t = 0
    mu = 0
    sigma = 1
    expected_result = 0.5
    assert np.isclose(__normal_2P_CDF(t, mu, sigma), expected_result)

    # Test case 2: Positive mean, positive standard deviation
    t = 1
    mu = 2
    sigma = 3
    expected_result = 0.36944134018176367
    assert np.isclose(__normal_2P_CDF(t, mu, sigma), expected_result)

    # Test case 3: Negative mean, negative standard deviation
    t = -1
    mu = -2
    sigma = -3
    expected_result = 0.36944134018176367
    assert np.isclose(__normal_2P_CDF(t, mu, sigma), expected_result)

    # Test case 4: Zero mean, non-zero standard deviation
    t = -2
    mu = 0
    sigma = 4
    expected_result = 0.30853753872598694
    assert np.isclose(__normal_2P_CDF(t, mu, sigma), expected_result)


def test_Lognormal_3P_guess():
    # Test case: Positive shape parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    expected_result = [1.661852414321683, 1.177303513131254, 0.0]
    assert Lognormal_3P_guess(x, y, gamma0, failures) == expected_result


def test_Loglogistic_2P_guess():
    # Test case 1
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    expected_result = [5.313736132570901, 1.3569874666195965]
    assert Loglogistic_2P_guess(x, y, method) == expected_result

    # Test case 2
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    expected_result = [5.3404557038858895, 1.3475053647737885]
    assert Loglogistic_2P_guess(x, y, method) == expected_result


def test_loglogistic_3P_CDF():
    # Test case 1
    t = 1.0
    alpha = 2.0
    beta = 1.5
    gamma = 0.0
    expected_result = 0.2612038749637414
    assert np.isclose(__loglogistic_3P_CDF(t, alpha, beta, gamma), expected_result)

    # Test case 2
    t = 2.5
    alpha = 1.0
    beta = 2.0
    gamma = -1.0
    expected_result = 0.9245283018867924
    assert np.isclose(__loglogistic_3P_CDF(t, alpha, beta, gamma), expected_result)

    # Test case 3
    t = 0.5
    alpha = 0.5
    beta = 0.5
    gamma = 1.0
    expected_result = 0.5 + 0.5j
    assert np.isclose(__loglogistic_3P_CDF(t, alpha, beta, gamma), expected_result)


def test_Loglogistic_3P_guess():
    # Test case:RRX method parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    method = "RRX"
    expected_result = [5.187796669436402, 1.4449857699246145, 5.915989666785221e-14]
    assert Loglogistic_3P_guess(x, y, method, gamma0, failures) == expected_result

    # Test case:RRY method parameter
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    method = "RRY"
    expected_result = [5.187796643455984, 1.4449857936930202, 1.2242313397159331e-12]
    assert Loglogistic_3P_guess(x, y, method, gamma0, failures) == expected_result


def test__gamma_2P_CDF():
    # Test case 1
    t = 1.0
    alpha = 2.0
    beta = 3.0
    expected_result = 0.01438767796697068
    assert np.isclose(__gamma_2P_CDF(t, alpha, beta), expected_result)

    # Test case 2
    t = 2.5
    alpha = 1.5
    beta = 2.5
    expected_result = 0.3512576413324066
    assert np.isclose(__gamma_2P_CDF(t, alpha, beta), expected_result)

    # Test case 3
    t = 0.5
    alpha = 0.8
    beta = 1.2
    expected_result = 0.3728289484897223
    assert np.isclose(__gamma_2P_CDF(t, alpha, beta), expected_result)


def test_Gamma_2P_guess():
    # Test case: RRX Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    failures = np.array([1, 2, 3, 4, 5])
    method = "RRX"
    expected_result = [5.170449644731437, 1.3012814793450251]
    assert_array_almost_equal(Gamma_2P_guess(x, y, method, failures), expected_result)

    # Test case: RRY Method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    failures = np.array([1, 2, 3, 4, 5])
    method = "RRY"
    expected_result = [5.170449661386152, 1.3012814764850034]
    assert_array_almost_equal(Gamma_2P_guess(x, y, method, failures), expected_result)

    # Test case: Large values
    x = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    failures = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
    method = "RRX"
    expected_result = [51704496447.09725, 1.3012814793486678]
    assert_array_almost_equal(Gamma_2P_guess(x, y, method, failures), expected_result)


def test___gamma_3P_CDF():
    # Test case: t = 0, alpha = 1, beta = 1, gamma = 0
    assert __gamma_3P_CDF(0, 1, 1, 0) == 0

    # Test case: t = 1, alpha = 1, beta = 1, gamma = 0
    assert __gamma_3P_CDF(1, 1, 1, 0) == 0.6321205588285577

    # Test case: t = 2, alpha = 2, beta = 2, gamma = 1
    assert __gamma_3P_CDF(2, 2, 2, 1) == 0.09020401043104986

    # Test case: t = 3, alpha = 2, beta = 2, gamma = 1
    assert __gamma_3P_CDF(3, 2, 2, 1) == 0.2642411176571153

    # Test case: t = 4, alpha = 3, beta = 1, gamma = 2
    assert __gamma_3P_CDF(4, 3, 1, 2) == 0.4865828809674077

    # Test case: t = 5, alpha = 3, beta = 1, gamma = 2
    assert __gamma_3P_CDF(5, 3, 1, 2) == 0.6321205588285577


def test_Gamma_3P_guess():
    # Test case: Normal execution RRX method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRX"
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    expected_result = [5.1704489869481804, 1.3012815922228111, 3.223302155028797e-17]
    assert Gamma_3P_guess(x, y, method, gamma0, failures) == expected_result

    # Test case: Normal execution RRY method
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = "RRY"
    gamma0 = 0.5
    failures = np.array([1, 2, 3, 4, 5])
    expected_result = [5.170448986922751, 1.3012815922268393, 3.2238824435138636e-17]
    assert Gamma_3P_guess(x, y, method, gamma0, failures) == expected_result


def test___beta_2P_CDF():
    # Test case: t = 0, alpha = 1, beta = 1
    assert __beta_2P_CDF(0, 1, 1) == 0.0

    # Test case: t = 0.5, alpha = 1, beta = 1
    assert __beta_2P_CDF(0.5, 1, 1) == 0.5

    # Test case: t = 1, alpha = 1, beta = 1
    assert __beta_2P_CDF(1, 1, 1) == 1.0

    # Test case: t = 0.25, alpha = 2, beta = 3
    assert __beta_2P_CDF(0.25, 2, 3) == 0.26171875

    # Test case: t = 0.75, alpha = 2, beta = 3
    assert __beta_2P_CDF(0.75, 2, 3) == 0.94921875


def test_Beta_2P_guess():
    # Test case: Normal data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    failures = [1, 2, 3, 4, 5]
    assert Beta_2P_guess(x, y, failures) == [2.0, 1.0]

    # Test case: Large data
    x = np.arange(1, 10001)
    y = np.random.rand(10000)
    failures = list(range(1, 10001))
    assert Beta_2P_guess(x, y, failures) == [2.0, 1.0]


def test_non_invertable_handler():
    # Test case: Non-invertible matrix
    xx = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    yy = np.array([1, 2, 3])
    model = "Test Model"
    expected_result = [np.float64(-0.328125), np.float64(0.0), np.float64(0.453125)]
    assert_array_almost_equal(non_invertable_handler(xx, yy, model), expected_result)

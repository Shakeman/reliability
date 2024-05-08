import numpy as np

from reliability.Utils._statstic_utils import Weibull_2P_guess


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
    try:
        Weibull_2P_guess(x, y, method, force_shape)
    except ValueError as e:
        assert str(e) == "A minimum of 2 points are required to fit the line when slope or intercept are not specified."

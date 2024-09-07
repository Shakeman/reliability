import numpy as np
from numpy.testing import assert_allclose

from reliability.Distributions import (
    Weibull_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Weibull_Distribution():
    dist = Weibull_Distribution(alpha=5, beta=2, gamma=10)
    dist.plot()
    dist.stats()
    assert_allclose(dist.mean, 14.4311346272637895, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 2.316256875880522, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 5.365045915063796, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 0.6311106578189344, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 3.2450893006876456, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Weibull Distribution (α=5,β=2,γ=10)"
    assert_allclose(dist.quantile(0.2), 12.361903635387193, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 12.9861134604144417, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 1.1316926249544481, rtol=rtol, atol=atol)
    xvals = np.array(
        [
            dist.gamma - 1,
            dist.quantile(0.001),
            dist.quantile(0.01),
            dist.quantile(0.1),
            dist.quantile(0.9),
            dist.quantile(0.99),
            dist.quantile(0.999),
        ],
    )
    assert_allclose(
        dist.PDF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.012639622357755485,
            0.03969953988653618,
            0.11685342455082046,
            0.06069708517540586,
            0.008583864105157392,
            0.0010513043539513882,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CDF(xvals=xvals, show_plot=True),
        [0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.SF(xvals=xvals, show_plot=True),
        [1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.HF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.012652274632387873,
            0.04010054533993554,
            0.12983713838980052,
            0.6069708517540585,
            0.8583864105157389,
            1.0513043539513862,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.0010005003335835354,
            0.010050335853501409,
            0.10536051565782631,
            2.3025850929940455,
            4.605170185988091,
            6.907755278982135,
        ],
        rtol=rtol,
        atol=atol,
    )

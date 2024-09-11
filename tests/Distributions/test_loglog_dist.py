import numpy as np
from numpy.testing import assert_allclose

from reliability.Distributions import (
    Loglogistic_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Loglogistic_Distribution():
    dist = Loglogistic_Distribution(alpha=50, beta=8, gamma=10)
    dist.plot()
    dist.stats()
    assert_allclose(dist.mean, 61.308607648851535, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 12.009521950735257, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 144.228617485192, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 1.2246481827926854, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 8.342064360132765, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Loglogistic Distribution (α=50.0,β=8.0,γ=10.0)"
    assert_allclose(dist.quantile(0.2), 52.044820762685724, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 54.975179587474166, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 41.308716243335226, rtol=rtol, atol=atol)
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
            0.0003789929723245846,
            0.0028132580909498313,
            0.01895146578651591,
            0.010941633873382936,
            0.0008918684027148376,
            6.741239934687115e-05,
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
            0,
            0.000379372344669,
            0.002841674839343,
            0.021057184207240,
            0.109416338733829,
            0.089186840271483,
            0.067412399346865,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.001000500333583622,
            0.010050335853501506,
            0.10536051565782635,
            2.302585092994045,
            4.605170185988085,
            6.907755278982047,
        ],
        rtol=rtol,
        atol=atol,
    )

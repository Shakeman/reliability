import numpy as np
from numpy.testing import assert_allclose

from reliability.Distributions import (
    Normal_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Normal_Distribution():
    dist = Normal_Distribution(mu=5.0, sigma=2.0)
    dist.plot()
    dist.stats()
    assert_allclose(dist.mean, 5, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 2, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 4, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 0, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 3, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Normal Distribution (μ=5.0,σ=2.0)"
    assert_allclose(dist.quantile(0.2), 3.3167575328541714, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 3.9511989745839187, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(10), 0.6454895953278145, rtol=rtol, atol=atol)
    xvals = np.array(
        [
            0,
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
            0.00876415024678427,
            0.001683545038531998,
            0.01332607110172904,
            0.08774916596624342,
            0.08774916596624342,
            0.01332607110172904,
            0.001683545038531998,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CDF(xvals=xvals, show_plot=True),
        [0.006209665325776132, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.SF(xvals=xvals, show_plot=True),
        [0.9937903346742238, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.HF(xvals=xvals, show_plot=True),
        [
            0.00881891274345837,
            0.0016852302688007987,
            0.013460677880534384,
            0.09749907329582604,
            0.8774916596624335,
            1.332607110172904,
            1.6835450385319983,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [
            0.006229025485860027,
            0.0010005003335835344,
            0.01005033585350145,
            0.1053605156578264,
            2.302585092994045,
            4.605170185988091,
            6.907755278982137,
        ],
        rtol=rtol,
        atol=atol,
    )

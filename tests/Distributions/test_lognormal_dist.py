import numpy as np
from numpy.testing import assert_allclose

from reliability.Distributions import (
    Lognormal_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Lognormal_Distribution():
    dist = Lognormal_Distribution(mu=2, sigma=0.8, gamma=10)
    dist.plot()
    dist.stats()
    assert_allclose(dist.mean, 20.175674306073336, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 9.634600550542682, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 92.82552776851736, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 3.689292296091298, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 34.36765343083244, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Lognormal Distribution (μ=2,σ=0.8,γ=10)"
    assert_allclose(dist.quantile(0.2), 13.7685978648453116, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 14.857284757111664, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 9.143537277214762, rtol=rtol, atol=atol)
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
            0.006748891633682291,
            0.028994071579561444,
            0.08276575111567319,
            0.01064970121764939,
            0.0007011277158027589,
            4.807498012690953e-05,
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
            0.006755647280963254,
            0.029286940989456004,
            0.09196194568408134,
            0.10649701217649381,
            0.07011277158027589,
            0.04807498012690954,
        ],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [
            -0.0,
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

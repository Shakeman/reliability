from numpy.testing import assert_allclose

from reliability.Distributions import (
    Exponential_Distribution,
)

atol = 1e-8
rtol = 1e-7


def test_Exponential_Distribution():
    dist = Exponential_Distribution(Lambda=0.2, gamma=10)
    dist.plot()
    dist.stats()
    assert_allclose(dist.mean, 15, rtol=rtol, atol=atol)
    assert_allclose(dist.standard_deviation, 5, rtol=rtol, atol=atol)
    assert_allclose(dist.variance, 25, rtol=rtol, atol=atol)
    assert_allclose(dist.skewness, 2, rtol=rtol, atol=atol)
    assert_allclose(dist.kurtosis, 9, rtol=rtol, atol=atol)
    assert dist.param_title_long == "Exponential Distribution (λ=0.2,γ=10)"
    assert_allclose(dist.quantile(0.2), 11.11571775657105, rtol=rtol, atol=atol)
    assert_allclose(dist.inverse_SF(q=0.7), 11.783374719693661, rtol=rtol, atol=atol)
    assert_allclose(dist.mean_residual_life(20), 5, rtol=rtol, atol=atol)
    xvals = [
        dist.gamma - 1,
        dist.quantile(0.001),
        dist.quantile(0.01),
        dist.quantile(0.1),
        dist.quantile(0.9),
        dist.quantile(0.99),
        dist.quantile(0.999),
    ]
    assert_allclose(
        dist.PDF(xvals=xvals, show_plot=True),
        [0.0, 0.19980000000000003, 0.198, 0.18, 0.019999999999999997, 0.002000000000000001, 0.0002000000000000004],
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
        actual=dist.SF(xvals=xvals, show_plot=True),
        desired=[1.0, 0.999, 0.99, 0.9, 0.1, 0.01, 0.001],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(dist.HF(xvals=xvals, show_plot=True), [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], rtol=rtol, atol=atol)
    assert_allclose(
        dist.CHF(xvals=xvals, show_plot=True),
        [
            0.0,
            0.0010005003335834318,
            0.01005033585350148,
            0.10536051565782643,
            2.3025850929940463,
            4.605170185988091,
            6.907755278982136,
        ],
        rtol=rtol,
        atol=atol,
    )
